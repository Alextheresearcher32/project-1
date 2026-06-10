"""
Pre-trade risk engine. Every Order must pass `check()` before going to an
exchange adapter. If any check fails, the order is rejected and an
Incident is logged.

This is deterministic, fast, side-effect-free. No LLMs, no network calls,
no surprises. The risk config (config/risk.yaml) is the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from glitz_quant.data.types import Order, OrderType, Position, Side, Venue
from glitz_quant.settings import get_risk_config, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]

    @property
    def rejected(self) -> bool:
        return not self.approved


class RiskEngine:
    """
    Deterministic pre-trade checks. Constructed once at startup, called
    synchronously from the OMS hot path.
    """

    def __init__(self) -> None:
        self.config = get_risk_config()
        self.settings = get_settings()

    # -------- Top-level --------
    def check_order(
        self,
        order: Order,
        current_position: Position | None,
        current_account_notional_usd: Decimal,
        current_daily_pnl_usd: Decimal,
        reference_price_usd: Decimal,
    ) -> RiskDecision:
        reasons: list[str] = []

        if not self._venue_enabled(order.venue):
            reasons.append(f"venue {order.venue.value} not enabled in risk.yaml")

        reasons.extend(self._check_order_type(order))
        reasons.extend(self._check_order_size(order, reference_price_usd))
        reasons.extend(self._check_account_limits(order, reference_price_usd, current_account_notional_usd))
        reasons.extend(self._check_symbol_limits(order, current_position, reference_price_usd))
        reasons.extend(self._check_daily_loss(current_daily_pnl_usd))
        reasons.extend(self._check_live_max_notional(order, reference_price_usd))

        decision = RiskDecision(approved=len(reasons) == 0, reasons=reasons)
        if decision.rejected:
            log.warning(
                "order_rejected_by_risk",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                reasons=reasons,
            )
        return decision

    # -------- Granular checks --------
    def _venue_enabled(self, venue: Venue) -> bool:
        v = self.config.get("venues", {}).get(venue.value, {})
        return bool(v.get("enabled", False))

    def _check_order_type(self, order: Order) -> list[str]:
        out: list[str] = []
        order_cfg: dict[str, Any] = self.config.get("order", {})
        allowed = set(order_cfg.get("allowed_types", []))
        if allowed and order.order_type.value not in allowed:
            out.append(f"order type {order.order_type.value} not in allowed list {sorted(allowed)}")
        if order_cfg.get("forbid_market_orders", True) and order.order_type in (
            OrderType.MARKET,
            OrderType.STOP_MARKET,
        ):
            out.append("market orders are forbidden by risk policy")
        return out

    def _check_order_size(self, order: Order, ref_price: Decimal) -> list[str]:
        out: list[str] = []
        order_cfg: dict[str, Any] = self.config.get("order", {})
        notional = abs(order.size) * (order.price if order.price else ref_price)

        max_notional = Decimal(str(order_cfg.get("max_notional_usd", 0)))
        if max_notional > 0 and notional > max_notional:
            out.append(f"order notional {notional} exceeds per-order max {max_notional}")

        min_notional = Decimal(str(order_cfg.get("min_size_usd", 0)))
        if min_notional > 0 and notional < min_notional:
            out.append(f"order notional {notional} below per-order min {min_notional}")

        max_dev_pct = Decimal(str(order_cfg.get("max_price_deviation_pct", 5.0)))
        if order.price and ref_price > 0:
            dev_pct = abs(order.price - ref_price) / ref_price * Decimal(100)
            if dev_pct > max_dev_pct:
                out.append(f"price deviation {dev_pct:.2f}% from reference exceeds {max_dev_pct}%")
        return out

    def _check_account_limits(
        self, order: Order, ref_price: Decimal, current_notional_usd: Decimal
    ) -> list[str]:
        out: list[str] = []
        acct: dict[str, Any] = self.config.get("account", {})
        max_total = Decimal(str(acct.get("max_total_notional_usd", 0)))
        if max_total > 0:
            order_notional = abs(order.size) * (order.price if order.price else ref_price)
            projected = current_notional_usd + order_notional
            if projected > max_total:
                out.append(f"projected account notional {projected} exceeds cap {max_total}")
        return out

    def _check_symbol_limits(
        self, order: Order, position: Position | None, ref_price: Decimal
    ) -> list[str]:
        out: list[str] = []
        sym_cfg: dict[str, Any] = self.config.get("symbol_limits", {})
        cfg = sym_cfg.get(order.symbol) or sym_cfg.get("default", {})
        cap = Decimal(str(cfg.get("max_position_notional_usd", 0)))
        if cap <= 0:
            return out

        current_notional = position.notional if position else Decimal(0)
        order_notional = abs(order.size) * (order.price if order.price else ref_price)

        # If the order grows position (same side), check projected notional.
        # If it reduces position, allow (closing is always allowed under symbol cap).
        is_growing = (
            position is None
            or position.is_flat
            or (position.is_long and order.side == Side.BUY)
            or (not position.is_long and order.side == Side.SELL)
        )
        if is_growing:
            projected = current_notional + order_notional
            if projected > cap:
                out.append(
                    f"symbol {order.symbol} projected position notional {projected} "
                    f"exceeds cap {cap}"
                )
        return out

    def _check_daily_loss(self, current_daily_pnl_usd: Decimal) -> list[str]:
        out: list[str] = []
        acct = self.config.get("account", {})
        max_loss = Decimal(str(acct.get("max_daily_loss_usd", 0)))
        if max_loss > 0 and current_daily_pnl_usd <= -max_loss:
            out.append(
                f"daily loss {current_daily_pnl_usd} has hit or exceeded limit -{max_loss}"
            )
        return out

    def _check_live_max_notional(self, order: Order, ref_price: Decimal) -> list[str]:
        """In live mode, also enforce LIVE_TRADING_MAX_NOTIONAL_USD from .env."""
        if self.settings.glitz_mode.value != "live":
            return []
        max_live = Decimal(str(self.settings.live_trading_max_notional_usd))
        if max_live <= 0:
            return ["live mode but LIVE_TRADING_MAX_NOTIONAL_USD is 0"]
        notional = abs(order.size) * (order.price if order.price else ref_price)
        if notional > max_live:
            return [f"live order notional {notional} exceeds LIVE_TRADING_MAX_NOTIONAL_USD {max_live}"]
        return []
