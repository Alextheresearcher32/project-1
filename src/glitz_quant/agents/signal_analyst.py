"""
SignalAnalyst — reads historical signal + order data and uses an LLM
(via litellm → OpenRouter) to identify what's working, what isn't, and why.

Runs on a slow loop (every few hours). Output is broadcast via
Telegram/Discord and persisted in agent_runs.

Analysis covers:
- Win rate and average PnL per strategy
- Whether confidence scores are calibrated (high conf = more wins?)
- What indicator conditions (RSI, volume_z from signal metadata) predict wins
- Recommendations for parameter tuning
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from glitz_quant.agents.llm_router import LLMRouter
from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

# litellm model string — routes through OpenRouter, swap model here without code changes
ANALYSIS_MODEL = "openrouter/anthropic/claude-opus-4-7"


class SignalAnalysisReport(BaseModel):
    overall_assessment: str = Field(description="1-2 sentence summary of signal quality")
    win_rate_pct: float = Field(description="Estimated win rate across completed round-trips, 0-100")
    confidence_calibrated: bool = Field(description="True if high-confidence signals outperform low-confidence ones")
    best_conditions: list[str] = Field(max_length=5, description="Conditions that reliably preceded wins")
    worst_conditions: list[str] = Field(max_length=5, description="Conditions that reliably preceded losses")
    recommendations: list[str] = Field(max_length=4, description="Concrete parameter or rule changes to consider")
    risk_flags: list[str] = Field(default_factory=list, max_length=3, description="Anything alarming in the data")


def _pair_round_trips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Match BUY entry orders to SELL exit orders chronologically per strategy+symbol.
    Returns completed round-trips with realized PnL and entry signal metadata.
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["strategy"], r["symbol"])
        buckets.setdefault(key, []).append(r)

    trips: list[dict[str, Any]] = []
    for (strategy, symbol), events in buckets.items():
        events_sorted = sorted(events, key=lambda x: x["signal_ts"])
        pending: dict[str, Any] | None = None
        for ev in events_sorted:
            if ev["order_status"] != "filled" or ev["avg_fill_price"] is None:
                continue
            if ev["order_side"] == "buy" and pending is None:
                pending = ev
            elif ev["order_side"] == "sell" and pending is not None:
                entry_px = Decimal(str(pending["avg_fill_price"]))
                exit_px = Decimal(str(ev["avg_fill_price"]))
                size = Decimal(str(pending["filled_size"] or 0))
                fees = Decimal(str(pending["fees_paid"] or 0)) + Decimal(str(ev["fees_paid"] or 0))
                pnl = (exit_px - entry_px) * size - fees
                trips.append({
                    "strategy": strategy,
                    "symbol": symbol,
                    "entry_ts": str(pending["signal_ts"]),
                    "exit_ts": str(ev["signal_ts"]),
                    "entry_price": float(entry_px),
                    "exit_price": float(exit_px),
                    "realized_pnl": float(pnl),
                    "won": pnl > 0,
                    "entry_confidence": pending.get("confidence"),
                    "entry_reason": pending.get("reason", ""),
                    "entry_metadata": pending.get("metadata") or {},
                })
                pending = None
    return trips


def _format_for_llm(trips: list[dict[str, Any]], raw_signals: list[dict[str, Any]]) -> str:
    if not trips:
        directions: dict[str, int] = {}
        for s in raw_signals:
            d = s.get("direction", "?")
            directions[d] = directions.get(d, 0) + 1
        return (
            f"No completed round-trips yet. {len(raw_signals)} signals emitted "
            f"(breakdown: {directions}). Analyze metadata quality and confidence distribution."
        )

    wins = [t for t in trips if t["won"]]
    losses = [t for t in trips if not t["won"]]
    win_rate = len(wins) / len(trips) * 100
    total_pnl = sum(t["realized_pnl"] for t in trips)

    by_strategy: dict[str, dict[str, Any]] = {}
    for t in trips:
        s = t["strategy"]
        if s not in by_strategy:
            by_strategy[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "confs": []}
        by_strategy[s]["wins" if t["won"] else "losses"] += 1
        by_strategy[s]["pnl"] += t["realized_pnl"]
        if t.get("entry_confidence") is not None:
            by_strategy[s]["confs"].append(t["entry_confidence"])

    strat_lines = []
    for s, d in by_strategy.items():
        avg_c = sum(d["confs"]) / len(d["confs"]) if d["confs"] else None
        conf_str = f"  avg_conf={avg_c:.2f}" if avg_c else ""
        strat_lines.append(f"  {s}: {d['wins']}W/{d['losses']}L  PnL=${d['pnl']:.2f}{conf_str}")

    high_conf = [t for t in trips if (t.get("entry_confidence") or 0) >= 0.75]
    low_conf = [t for t in trips if (t.get("entry_confidence") or 0) < 0.75]
    hc_wr = (sum(1 for t in high_conf if t["won"]) / len(high_conf) * 100) if high_conf else None
    lc_wr = (sum(1 for t in low_conf if t["won"]) / len(low_conf) * 100) if low_conf else None

    lines = [
        f"=== Signal Analysis: {len(trips)} completed round-trips ===",
        f"Win rate: {win_rate:.1f}%  |  Total PnL: ${total_pnl:.2f}",
        f"Wins: {len(wins)}  Losses: {len(losses)}",
        "",
        "By strategy:",
        *strat_lines,
        "",
        f"High-confidence (>=0.75) win rate: {hc_wr:.1f}% ({len(high_conf)} trades)" if hc_wr is not None else "No high-conf trades yet",
        f"Low-confidence (<0.75) win rate:  {lc_wr:.1f}% ({len(low_conf)} trades)" if lc_wr is not None else "No low-conf trades yet",
        "",
        "Sample winning signal metadata (last 10):",
        str([t["entry_metadata"] for t in wins[-10:]]),
        "",
        "Sample losing signal metadata (last 10):",
        str([t["entry_metadata"] for t in losses[-10:]]),
        "",
        "Sample winning reasons:",
        str([t["entry_reason"] for t in wins[-5:]]),
        "",
        "Sample losing reasons:",
        str([t["entry_reason"] for t in losses[-5:]]),
    ]
    return "\n".join(lines)


class SignalAnalyst:
    """
    Slow-loop agent. Reads signal + order history from Supabase, calls
    an LLM via litellm (OpenRouter) to analyze patterns, and returns a
    structured report suitable for broadcasting.
    """
    name = "signal_analyst"

    def __init__(self, llm: LLMRouter, store: SupabaseStore) -> None:
        self.llm = llm
        self.store = store

    async def run(self, lookback_days: int = 30) -> SignalAnalysisReport | None:
        rows = await self.store.fetch_signal_history(days=lookback_days)
        if not rows:
            log.info("signal_analyst_no_data", lookback_days=lookback_days)
            return None

        trips = _pair_round_trips(rows)
        summary = _format_for_llm(trips, rows)

        system = (
            "You are a quantitative analyst reviewing the live trading history of a "
            "crypto algorithmic trading system. You receive a compact summary of signals "
            "and their trade outcomes. Be concrete and data-driven. Do not invent numbers "
            "not present in the data. If sample size is under 10 trades, say so and focus "
            "on metadata quality and confidence distribution instead."
        )
        user = (
            f"Signal and trade history for the past {lookback_days} days:\n\n"
            f"{summary}\n\n"
            "Analyze: win rate and PnL per strategy, confidence calibration "
            "(do higher-confidence signals win more?), which indicator values in entry_metadata "
            "(RSI, volume_z, atr, etc.) correlate with wins vs losses, and give up to 4 "
            "actionable recommendations for improving signal quality."
        )

        report: SignalAnalysisReport = await self.llm.complete_structured(
            agent_name=self.name,
            system=system,
            user=user,
            schema=SignalAnalysisReport,
            model=ANALYSIS_MODEL,
            temperature=0.1,
            max_tokens=1500,
        )

        log.info(
            "signal_analyst_complete",
            win_rate=report.win_rate_pct,
            calibrated=report.confidence_calibrated,
            n_trips=len(trips),
            n_signals=len(rows),
        )
        return report

    @staticmethod
    def format_broadcast(report: SignalAnalysisReport) -> str:
        lines = [
            "<b>Signal Analysis Report</b>",
            "",
            f"<b>Assessment:</b> {report.overall_assessment}",
            f"<b>Win rate:</b> {report.win_rate_pct:.1f}%",
            f"<b>Confidence calibrated:</b> {'Yes' if report.confidence_calibrated else 'No - review thresholds'}",
            "",
            "<b>Best conditions:</b>",
            *[f"  - {c}" for c in report.best_conditions],
            "",
            "<b>Worst conditions:</b>",
            *[f"  - {c}" for c in report.worst_conditions],
            "",
            "<b>Recommendations:</b>",
            *[f"  {i + 1}. {r}" for i, r in enumerate(report.recommendations)],
        ]
        if report.risk_flags:
            lines += ["", "<b>Risk flags:</b>", *[f"  - {f}" for f in report.risk_flags]]
        return "\n".join(lines)
