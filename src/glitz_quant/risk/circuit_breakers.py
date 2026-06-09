"""
Active monitoring. Runs alongside the orchestrator and trips circuit
breakers when dangerous conditions are met. Tripping = halts new orders
+ alerts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from glitz_quant.data.types import Incident
from glitz_quant.settings import get_risk_config
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

Action = Literal["halt_and_alert", "pause_60min", "pause_30min", "disable_venue", "skip_signal"]


@dataclass
class BreakerEvent:
    name: str
    action: Action
    message: str
    ts: float = field(default_factory=time.time)


class CircuitBreakerEngine:
    """
    Stateful. Holds rolling windows for thresholds. Call .observe_*() from
    appropriate spots in the orchestrator/OMS. Call .status() to know what's
    tripped right now.
    """

    def __init__(self) -> None:
        self.config = get_risk_config().get("circuit_breakers", {})
        self._daily_pnl = Decimal(0)
        self._consecutive_losses = 0
        self._equity_window: deque[tuple[float, Decimal]] = deque()
        self._order_outcomes: deque[tuple[float, bool]] = deque()  # (ts, was_rejected)
        self._venue_last_data: dict[str, float] = {}
        self._tripped: dict[str, BreakerEvent] = {}
        self._paused_until: float = 0.0
        self._disabled_venues: set[str] = set()

    # -------- Observation API --------
    def observe_realized_pnl(self, pnl_delta_usd: Decimal) -> BreakerEvent | None:
        """Call when a trade closes."""
        self._daily_pnl += pnl_delta_usd

        # Consecutive losses
        if pnl_delta_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check daily loss
        cfg = self.config.get("daily_loss", {})
        if cfg.get("enabled") and self._daily_pnl <= -Decimal(str(cfg.get("threshold_usd", 0))):
            return self._trip(
                "daily_loss",
                cfg.get("action", "halt_and_alert"),
                f"daily realized PnL hit {self._daily_pnl}",
            )

        # Check consecutive losses
        cfg = self.config.get("consecutive_losses", {})
        if cfg.get("enabled") and self._consecutive_losses >= int(cfg.get("count", 999)):
            return self._trip(
                "consecutive_losses",
                cfg.get("action", "pause_60min"),
                f"{self._consecutive_losses} consecutive losses",
            )
        return None

    def observe_equity(self, equity_usd: Decimal) -> BreakerEvent | None:
        """Call periodically (every minute or so) with total account equity."""
        now = time.time()
        cfg = self.config.get("rapid_drawdown", {})
        if not cfg.get("enabled"):
            return None
        window_seconds = int(cfg.get("window_minutes", 15)) * 60
        threshold_pct = Decimal(str(cfg.get("threshold_pct", 2.0)))

        self._equity_window.append((now, equity_usd))
        cutoff = now - window_seconds
        while self._equity_window and self._equity_window[0][0] < cutoff:
            self._equity_window.popleft()

        if len(self._equity_window) < 2:
            return None
        peak = max(eq for _, eq in self._equity_window)
        if peak <= 0:
            return None
        dd_pct = (peak - equity_usd) / peak * Decimal(100)
        if dd_pct >= threshold_pct:
            return self._trip(
                "rapid_drawdown",
                cfg.get("action", "halt_and_alert"),
                f"drawdown {dd_pct:.2f}% in last {window_seconds//60}min",
            )
        return None

    def observe_order_outcome(self, rejected: bool) -> BreakerEvent | None:
        cfg = self.config.get("order_rejection_rate", {})
        if not cfg.get("enabled"):
            return None
        now = time.time()
        window = int(cfg.get("window_minutes", 5)) * 60
        threshold_pct = Decimal(str(cfg.get("threshold_pct", 20)))

        self._order_outcomes.append((now, rejected))
        cutoff = now - window
        while self._order_outcomes and self._order_outcomes[0][0] < cutoff:
            self._order_outcomes.popleft()

        if len(self._order_outcomes) < 10:
            return None
        n_rej = sum(1 for _, r in self._order_outcomes if r)
        pct = Decimal(n_rej) / Decimal(len(self._order_outcomes)) * Decimal(100)
        if pct >= threshold_pct:
            return self._trip(
                "order_rejection_rate",
                cfg.get("action", "pause_30min"),
                f"order rejection rate {pct:.1f}% in last {window//60}min",
            )
        return None

    def observe_venue_data(self, venue: str) -> None:
        self._venue_last_data[venue] = time.time()

    def check_data_staleness(self) -> list[BreakerEvent]:
        cfg = self.config.get("data_staleness", {})
        if not cfg.get("enabled"):
            return []
        max_age = int(cfg.get("max_age_seconds", 30))
        action: Action = cfg.get("action", "disable_venue")
        events: list[BreakerEvent] = []
        now = time.time()
        for venue, last in self._venue_last_data.items():
            if now - last > max_age:
                ev = BreakerEvent(
                    name=f"data_staleness:{venue}",
                    action=action,
                    message=f"venue {venue} data is {int(now - last)}s stale",
                )
                if action == "disable_venue":
                    self._disabled_venues.add(venue)
                events.append(ev)
                self._tripped[ev.name] = ev
        return events

    # -------- State --------
    def reset_daily(self) -> None:
        self._daily_pnl = Decimal(0)
        self._consecutive_losses = 0

    def is_halted(self) -> bool:
        if any(ev.action == "halt_and_alert" for ev in self._tripped.values()):
            return True
        if self._paused_until > time.time():
            return True
        return False

    def is_venue_disabled(self, venue: str) -> bool:
        return venue in self._disabled_venues

    def reset_venue(self, venue: str) -> None:
        self._disabled_venues.discard(venue)
        key = f"data_staleness:{venue}"
        self._tripped.pop(key, None)

    def status(self) -> dict[str, str]:
        return {name: ev.message for name, ev in self._tripped.items()}

    # -------- Internal --------
    def _trip(self, name: str, action: str, message: str) -> BreakerEvent:
        action_typed: Action = action  # type: ignore[assignment]
        ev = BreakerEvent(name=name, action=action_typed, message=message)
        self._tripped[name] = ev
        log.error("circuit_breaker_tripped", name=name, action=action, msg=message)

        if action == "pause_60min":
            self._paused_until = max(self._paused_until, time.time() + 60 * 60)
        elif action == "pause_30min":
            self._paused_until = max(self._paused_until, time.time() + 30 * 60)

        return ev

    def to_incident(self, ev: BreakerEvent) -> Incident:
        return Incident(
            severity="critical" if ev.action == "halt_and_alert" else "error",
            kind=f"circuit_breaker:{ev.name}",
            message=ev.message,
            details={"action": ev.action},
        )
