"""
BTC Trend-Following Strategy — designed for trending crypto markets.

Unlike range-bound strategies (bitcoin_range, range_trader) that need price
to stay within a channel, this strategy RIDES the trend:

Entry signals:
  LONG  : EMA8 crosses above EMA21 AND close > EMA55 (trend confirmation)
          AND RSI > 50 (momentum confirmation)
  SHORT : EMA8 crosses below EMA21 AND close < EMA55
          AND RSI < 50

Exit:
  TP = 3×ATR above entry (wide — let winners run)
  SL = 1×ATR below entry (2:1 required win rate = 25%)
  Max hold = 96 bars (24h on 15m) — cut stale trends
  Force flat on EMA cross reversal

Regime filter:
  Only trade when ATR/price > 0.3% (trending/volatile session, not dead air)

Cooldown: 4 hours (one trend per session — avoid overtrading whipsaws)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.strategies.indicators import atr, ema, rsi
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class BtcTrendFollow(Strategy):
    name = "btc_trend"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.symbol: str = params.get("symbol", "BTC-USD")
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 500)))
        self.tp_atr_mult: float = float(params.get("tp_atr_mult", 3.0))
        self.sl_atr_mult: float = float(params.get("sl_atr_mult", 1.0))
        self.max_hold_bars: int = int(params.get("max_hold_bars", 96))
        self.atr_regime_min: float = float(params.get("atr_regime_min", 0.003))
        self.cooldown_bars: int = int(params.get("cooldown_bars", 16))  # 4h on 15m
        self.bidirectional: bool = bool(params.get("bidirectional", True))
        self.rsi_long_min: float = float(params.get("rsi_long_min", 50.0))
        self.rsi_short_max: float = float(params.get("rsi_short_max", 50.0))
        self.ema_fast: int = int(params.get("ema_fast", 8))
        self.ema_slow: int = int(params.get("ema_slow", 21))
        self.ema_trend: int = int(params.get("ema_trend", 55))

        self._bars_held: int = 0
        self._bars_since_last: int = 999
        self._entry_direction: SignalDirection | None = None
        self._entry_price: float = 0.0
        self._tp_price: float = 0.0
        self._sl_price: float = 0.0

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles
        if len(c) < self.ema_trend + 10:
            return None

        close = c["close"]
        high = c["high"]
        low = c["low"]

        ema_f = ema(close, self.ema_fast)
        ema_s = ema(close, self.ema_slow)
        ema_t = ema(close, self.ema_trend)
        atr_ser = atr(high, low, close, 14)
        rsi_ser = rsi(close, 14)

        close_now = float(close.iloc[-1])
        atr_now = float(atr_ser.iloc[-1])
        atr_pct = atr_now / (close_now + 1e-9)

        ema8_now = float(ema_f.iloc[-1])
        ema8_prev = float(ema_f.iloc[-2])
        ema21_now = float(ema_s.iloc[-1])
        ema21_prev = float(ema_s.iloc[-2])
        ema55_now = float(ema_t.iloc[-1])
        rsi_now = float(rsi_ser.iloc[-1])

        self._bars_since_last += 1

        # ── In a position: manage exit ────────────────────────────────────────
        if abs(ctx.open_position_size) > 1e-9:
            self._bars_held += 1

            # TP / SL check
            if self._entry_direction == SignalDirection.LONG:
                if close_now >= self._tp_price or close_now <= self._sl_price:
                    return self._flat_signal(ctx, "tp_sl")
                # EMA reversal exit
                if ema8_now < ema21_now and ema8_prev >= ema21_prev:
                    return self._flat_signal(ctx, "ema_cross_reversal")
            elif self._entry_direction == SignalDirection.SHORT:
                if close_now <= self._tp_price or close_now >= self._sl_price:
                    return self._flat_signal(ctx, "tp_sl")
                if ema8_now > ema21_now and ema8_prev <= ema21_prev:
                    return self._flat_signal(ctx, "ema_cross_reversal")

            # Time-based exit
            if self._bars_held >= self.max_hold_bars:
                return self._flat_signal(ctx, "max_hold")
            return None

        self._bars_held = 0

        # ── Regime filter: need volatility to trend ───────────────────────────
        if atr_pct < self.atr_regime_min:
            return None

        # ── Cooldown ──────────────────────────────────────────────────────────
        if self._bars_since_last < self.cooldown_bars:
            return None

        # ── Entry signals ─────────────────────────────────────────────────────
        ema_cross_up = ema8_now > ema21_now and ema8_prev <= ema21_prev
        ema_cross_dn = ema8_now < ema21_now and ema8_prev >= ema21_prev

        if ema_cross_up and close_now > ema55_now and rsi_now > self.rsi_long_min:
            return self._entry_signal(ctx, SignalDirection.LONG, close_now, atr_now)

        if self.bidirectional and ema_cross_dn and close_now < ema55_now and rsi_now < self.rsi_short_max:
            return self._entry_signal(ctx, SignalDirection.SHORT, close_now, atr_now)

        return None

    def _entry_signal(
        self, ctx: StrategyContext, direction: SignalDirection, close: float, atr_val: float
    ) -> Signal:
        self._entry_direction = direction
        self._entry_price = close
        self._bars_since_last = 0

        if direction == SignalDirection.LONG:
            self._tp_price = close + self.tp_atr_mult * atr_val
            self._sl_price = close - self.sl_atr_mult * atr_val
        else:
            self._tp_price = close - self.tp_atr_mult * atr_val
            self._sl_price = close + self.sl_atr_mult * atr_val

        conf = 0.65  # rules-based: high confidence when all filters align

        log.info(
            "btc_trend_entry",
            direction=direction.value,
            close=round(close, 0),
            tp=round(self._tp_price, 0),
            sl=round(self._sl_price, 0),
        )
        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            direction=direction,
            target_notional_usd=self.target_notional_usd,
            confidence=conf,
            confidence_label=SignalConfidence.HIGH,
            stop_loss_price=Decimal(str(round(self._sl_price, 2))),
            take_profit_price=Decimal(str(round(self._tp_price, 2))),
            reason=(
                f"BTC_TREND {direction.value.upper()}: "
                f"EMA8/21 cross, RSI={int(float(rsi(ctx.candles['close'], 14).iloc[-1]))}, "
                f"TP={round(self._tp_price, 0)}, SL={round(self._sl_price, 0)}"
            ),
            metadata={"close": close, "atr": atr_val},
        )

    def _flat_signal(self, ctx: StrategyContext, reason: str) -> Signal:
        self._entry_direction = None
        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            direction=SignalDirection.FLAT,
            target_notional_usd=self.target_notional_usd,
            confidence=1.0,
            confidence_label=SignalConfidence.HIGH,
            reason=f"BTC_TREND EXIT: {reason}",
            metadata={},
        )
