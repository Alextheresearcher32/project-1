"""
Funding Rate Carry strategy — bidirectional.

Logic:
- Perpetuals funding rate is paid every 8 hours. When funding is very
  positive, longs pay shorts — collect carry by going SHORT.
- When funding is very negative, shorts pay longs — collect carry by
  going LONG.

Signal rules:
  LONG  : funding_rate_8h < -entry_threshold  (shorts overextended, collect carry long)
  SHORT : funding_rate_8h >  entry_threshold  (longs overextended, collect carry short)
  FLAT  : funding reverses past exit_threshold (carry has dissipated, close position)
  None  : funding rate is in neutral zone      (no edge, don't trade)

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
        self.entry_threshold: float = float(params.get("entry_threshold", 0.0003))
        self.exit_threshold: float = float(params.get("exit_threshold", 0.0005))
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 50)))
        self.symbol: str = str(params.get("symbol", "BTC-USD"))
        self.bidirectional: bool = bool(params.get("bidirectional", True))

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        rate = ctx.extra_data.get("funding_rate_8h")
        if rate is None:
            return None

        rate = float(rate)
        is_long = ctx.open_position_size > 1e-9
        is_short = ctx.open_position_size < -1e-9
        has_position = is_long or is_short

        annualized_carry = rate * 3 * 365  # 3 funding events/day * 365

        # Exit long: funding has flipped positive → carry reversed, close
        if is_long and rate >= self.exit_threshold:
            log.info("funding_carry_exit_long", rate=rate)
            return Signal(
                strategy=self.name, symbol=self.symbol,
                direction=SignalDirection.FLAT, target_notional_usd=Decimal(0),
                confidence=0.9,
                reason=f"funding {rate:.4%} > {self.exit_threshold:.4%} — long carry exhausted",
                metadata={"funding_rate_8h": rate},
            )

        # Exit short: funding has flipped negative → carry reversed, close
        if is_short and rate <= -self.exit_threshold:
            log.info("funding_carry_exit_short", rate=rate)
            return Signal(
                strategy=self.name, symbol=self.symbol,
                direction=SignalDirection.FLAT, target_notional_usd=Decimal(0),
                confidence=0.9,
                reason=f"funding {rate:.4%} < -{self.exit_threshold:.4%} — short carry exhausted",
                metadata={"funding_rate_8h": rate},
            )

        if has_position:
            return None

        confidence = min(0.9, 0.55 + abs(rate) / self.entry_threshold * 0.15)

        # Entry long: funding strongly negative → shorts overextended, collect carry
        if rate <= -self.entry_threshold:
            log.info("funding_carry_entry_long", rate=rate, annualized_carry=annualized_carry)
            return Signal(
                strategy=self.name, symbol=self.symbol,
                direction=SignalDirection.LONG,
                target_notional_usd=self.target_notional_usd,
                confidence=confidence, confidence_label=SignalConfidence.MEDIUM,
                reason=(
                    f"funding {rate:.4%} < -{self.entry_threshold:.4%} — "
                    f"shorts overextended, carry {annualized_carry:.1%} ann."
                ),
                metadata={"funding_rate_8h": rate, "annualized_carry": annualized_carry},
            )

        # Entry short: funding strongly positive → longs overextended, collect carry
        if self.bidirectional and rate >= self.entry_threshold:
            log.info("funding_carry_entry_short", rate=rate, annualized_carry=annualized_carry)
            return Signal(
                strategy=self.name, symbol=self.symbol,
                direction=SignalDirection.SHORT,
                target_notional_usd=self.target_notional_usd,
                confidence=confidence, confidence_label=SignalConfidence.MEDIUM,
                reason=(
                    f"funding {rate:.4%} > {self.entry_threshold:.4%} — "
                    f"longs overextended, carry {annualized_carry:.1%} ann."
                ),
                metadata={"funding_rate_8h": rate, "annualized_carry": annualized_carry},
            )

        return None
