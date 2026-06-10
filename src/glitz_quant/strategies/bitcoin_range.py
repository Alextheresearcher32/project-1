"""
Bitcoin range-bound momentum strategy.

Per Larissa's documented ruleset:
- Identify range via rolling support (N-bar lows) and resistance (N-bar highs).
- ENTRY (long): price at/near support AND a confirming bullish bounce candle
  AND above-average volume AND RSI < oversold threshold.
- STOP: just below the support zone (configurable buffer in ATR units).
- TAKE PROFIT: at the resistance zone.
- Exit FLAT signal when price hits TP or SL, or when range invalidates.

Constraints honored:
- One concurrent position at a time (max_concurrent_positions = 1 by default).
- Position sizing uses fixed notional from strategy params; OMS enforces caps.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.strategies.indicators import (
    atr,
    bounce_candle,
    rsi,
    support_resistance,
    volume_z,
)
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class BitcoinRangeMomentum(Strategy):
    name = "bitcoin_range"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.support_lookback = int(params.get("support_lookback_bars", 96))
        self.resistance_lookback = int(params.get("resistance_lookback_bars", 96))
        self.rsi_period = int(params.get("rsi_period", 14))
        self.rsi_oversold = float(params.get("rsi_oversold", 30))
        self.rsi_overbought = float(params.get("rsi_overbought", 70))
        self.zone_proximity_pct = float(params.get("zone_proximity_pct", 0.4))  # within 0.4% of support
        self.volume_z_threshold = float(params.get("volume_z_threshold", 0.5))
        self.atr_period = int(params.get("atr_period", 14))
        self.stop_atr_buffer = float(params.get("stop_atr_buffer", 1.0))
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 50)))

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles
        min_required = max(self.support_lookback, self.resistance_lookback, self.rsi_period) + 5
        if len(c) < min_required:
            return None

        # Indicators
        sup, res = support_resistance(c["high"], c["low"], lookback=self.support_lookback)
        rsi_ser = rsi(c["close"], period=self.rsi_period)
        atr_ser = atr(c["high"], c["low"], c["close"], period=self.atr_period)
        vol_z = volume_z(c["volume"], lookback=20)
        bounce = bounce_candle(c["open"], c["high"], c["low"], c["close"])

        # Latest values
        last = c.iloc[-1]
        close = float(last["close"])
        last_sup = float(sup.iloc[-1])
        last_res = float(res.iloc[-1])
        last_rsi = float(rsi_ser.iloc[-1])
        last_atr = float(atr_ser.iloc[-1])
        last_vol_z = float(vol_z.iloc[-1])
        last_bounce = bool(bounce.iloc[-1])

        # If we have an open position, manage exits instead of new entries
        if abs(ctx.open_position_size) > 1e-9:
            return self._maybe_exit(close, last_res, ctx)

        # Entry checks: ALL must be true
        near_support = (close - last_sup) / max(last_sup, 1e-9) <= self.zone_proximity_pct / 100
        oversold = last_rsi < self.rsi_oversold
        vol_ok = last_vol_z >= self.volume_z_threshold
        room_to_resistance = (last_res - close) / max(close, 1e-9) > 0.005  # need >0.5% to TP

        if not (near_support and last_bounce and oversold and vol_ok and room_to_resistance):
            return None

        # Build signal
        stop_price = last_sup - last_atr * self.stop_atr_buffer
        target_price = last_res

        # Confidence is a soft blend; OMS still caps it
        confidence = self._confidence(last_rsi, last_vol_z, near_support, last_atr, close)

        log.info(
            "bitcoin_range_entry",
            close=close,
            support=last_sup,
            resistance=last_res,
            rsi=last_rsi,
            vol_z=last_vol_z,
            confidence=confidence,
        )

        return Signal(
            strategy=self.name,
            symbol="BTC-USD",
            direction=SignalDirection.LONG,
            target_notional_usd=self.target_notional_usd,
            confidence=confidence,
            confidence_label=SignalConfidence.HIGH if confidence > 0.75 else SignalConfidence.MEDIUM,
            stop_loss_price=Decimal(str(round(stop_price, 2))),
            take_profit_price=Decimal(str(round(target_price, 2))),
            reason=(
                f"price {close:.2f} near support {last_sup:.2f}, RSI {last_rsi:.1f} oversold, "
                f"volume z {last_vol_z:.2f}, bullish bounce candle"
            ),
            metadata={
                "support": last_sup,
                "resistance": last_res,
                "atr": last_atr,
                "rsi": last_rsi,
                "volume_z": last_vol_z,
            },
        )

    def _maybe_exit(self, close: float, resistance: float, ctx: StrategyContext) -> Signal | None:
        """Generate FLAT signal if we've hit TP or invalidated."""
        last_sig = next((s for s in reversed(ctx.recent_signals) if s.direction == SignalDirection.LONG), None)
        if last_sig is None:
            return None

        tp = float(last_sig.take_profit_price) if last_sig.take_profit_price else resistance
        sl = float(last_sig.stop_loss_price) if last_sig.stop_loss_price else 0.0

        if close >= tp:
            return Signal(
                strategy=self.name,
                symbol="BTC-USD",
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=1.0,
                reason=f"price {close:.2f} reached TP {tp:.2f}",
            )
        if sl > 0 and close <= sl:
            return Signal(
                strategy=self.name,
                symbol="BTC-USD",
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=1.0,
                reason=f"price {close:.2f} hit SL {sl:.2f}",
            )
        return None

    @staticmethod
    def _confidence(rsi_val: float, vol_z: float, near_support: bool, atr_val: float, close: float) -> float:
        score = 0.5
        # deeper oversold -> more confidence
        if rsi_val < 25:
            score += 0.15
        elif rsi_val < 28:
            score += 0.08
        # higher volume -> more confidence
        if vol_z >= 1.5:
            score += 0.15
        elif vol_z >= 1.0:
            score += 0.08
        # tighter to support -> more confidence
        if near_support:
            score += 0.05
        # too-high vol relative to price -> penalize (risky chop)
        if close > 0 and atr_val / close > 0.03:
            score -= 0.1
        return max(0.0, min(0.95, score))
