"""
Funding Rate Carry strategy.

Logic:
- Perpetuals funding rate is paid every 8 hours. When funding is very
  positive, longs pay shorts — the market is overleveraged long and
  historically tends to mean-revert.
- When funding is very negative, shorts pay longs — market is
  overleveraged short, a contrarian long entry on spot.

Signal rules:
  LONG  : funding_rate_8h < -entry_threshold  (shorts squeezed, go long spot)
  FLAT  : funding_rate_8h >  exit_threshold   (longs overextended, close long)
  None  : funding rate is in neutral zone     (no edge, don't trade)

This strategy does NOT size positions directly — target_notional_usd
is fixed in params and the OMS caps it.

Requires extra_data["funding_rate_8h"] to be populated by the runner.
If the value is missing the strategy skips.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class FundingRateCarry(Strategy):
    name = "funding_rate_carry"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # Positive means shorts are paying longs: contrarian long.
        # Threshold in absolute terms per 8h period (e.g., 0.0003 = 0.03%)
        self.entry_threshold: float = float(params.get("entry_threshold", 0.0003))
        # When funding flips strongly positive, longs are overextended — exit.
        self.exit_threshold: float = float(params.get("exit_threshold", 0.0005))
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 50)))
        self.symbol: str = str(params.get("symbol", "BTC-USD"))

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        rate = ctx.extra_data.get("funding_rate_8h")
        if rate is None:
            return None

        rate = float(rate)
        has_position = abs(ctx.open_position_size) > 1e-9

        # Exit: funding strongly positive → longs are being squeezed, close
        if has_position and rate >= self.exit_threshold:
            log.info("funding_carry_exit", rate=rate, threshold=self.exit_threshold)
            return Signal(
                strategy=self.name,
                symbol=self.symbol,
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=0.9,
                reason=f"funding rate {rate:.4%} > exit threshold {self.exit_threshold:.4%} — longs overextended",
                metadata={"funding_rate_8h": rate},
            )

        # Entry: funding strongly negative → shorts overextended, go long spot
        if not has_position and rate <= -self.entry_threshold:
            annualized_carry = rate * 3 * 365  # 3 funding events/day * 365
            confidence = min(0.9, 0.55 + abs(rate) / self.entry_threshold * 0.15)
            log.info("funding_carry_entry", rate=rate, annualized_carry=annualized_carry)
            return Signal(
                strategy=self.name,
                symbol=self.symbol,
                direction=SignalDirection.LONG,
                target_notional_usd=self.target_notional_usd,
                confidence=confidence,
                confidence_label=SignalConfidence.MEDIUM,
                reason=(
                    f"funding rate {rate:.4%} < -{self.entry_threshold:.4%} — "
                    f"shorts overextended, implied carry {annualized_carry:.1%} ann."
                ),
                metadata={"funding_rate_8h": rate, "annualized_carry": annualized_carry},
            )

        return None
