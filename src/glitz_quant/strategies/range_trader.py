"""
Range Trader — pure support/resistance bounce strategy.

Buys at support (rides up to resistance) and sells short at resistance
(rides down to support). No trend-direction filter — works in both
ranging and trending regimes.

Differences from BitcoinRangeMomentum:
- No EMA trend filter. Both long and short fire in any market regime.
- RSI threshold is looser: < 50 (any pullback) for long, > 50 for short.
  We care about RSI direction (turning), not extreme oversold/overbought.
- Candle confirmation (bounce/shooting-star) is a REQUIRED gate, not a bonus.
  It ensures we only enter on actual level-hold candles, not momentum runs.
- Zone proximity is 2% (vs 1.5% in bitcoin_range) — catches more touches.
- ATR stop buffer is 1.5× (vs 1.0×) — wider, because no trend filter
  means we get caught in more false bounces and need breathing room.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.strategies.indicators import (
    atr,
    bounce_candle,
    ema,
    rsi,
    shooting_star_candle,
    support_resistance,
    volume_z,
)
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class RangeTrader(Strategy):
    name = "range_trader"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.support_lookback = int(params.get("support_lookback_bars", 48))
        self.resistance_lookback = int(params.get("resistance_lookback_bars", 48))
        self.rsi_period = int(params.get("rsi_period", 14))
        self.rsi_long_threshold = float(params.get("rsi_long_threshold", 50))   # long if RSI < this
        self.rsi_short_threshold = float(params.get("rsi_short_threshold", 50)) # short if RSI > this
        self.zone_proximity_pct = float(params.get("zone_proximity_pct", 2.0))
        self.volume_z_min = float(params.get("volume_z_min", 0.5))   # require above-avg volume
        self.atr_period = int(params.get("atr_period", 14))
        self.stop_atr_buffer = float(params.get("stop_atr_buffer", 1.5))
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 500)))
        self.max_hold_bars: int = int(params.get("max_hold_bars", 32))
        # Candle pattern is a REQUIRED gate here (unlike bitcoin_range where it's a bonus).
        # Bounce candle = bullish hammer at support; shooting star = bearish at resistance.
        self.require_candle_pattern: bool = bool(params.get("require_candle_pattern", True))
        # Optional trend EMA filter — set to 0 to disable (default: disabled)
        self.trend_ema_bars: int = int(params.get("trend_ema_bars", 0))
        # Regime filter: skip entries when ATR/price > threshold (trending market).
        # 0.012 = 1.2% — consolidation is typically 0.6–0.9%, trending spikes to 2–4%.
        # Set 0.0 to disable.
        self.atr_regime_pct: float = float(params.get("atr_regime_pct", 0.0))

        self._bars_held: int = 0

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        c = ctx.candles
        min_required = max(self.support_lookback, self.resistance_lookback, self.rsi_period) + 5
        if len(c) < min_required:
            return None

        sup, res = support_resistance(c["high"], c["low"], lookback=self.support_lookback)
        rsi_ser = rsi(c["close"], period=self.rsi_period)
        atr_ser = atr(c["high"], c["low"], c["close"], period=self.atr_period)
        vol_z_ser = volume_z(c["volume"], lookback=20)
        bounce = bounce_candle(c["open"], c["high"], c["low"], c["close"])
        star = shooting_star_candle(c["open"], c["high"], c["low"], c["close"])
        trend_ema_ser = ema(c["close"], self.trend_ema_bars) if self.trend_ema_bars > 0 else None

        last = c.iloc[-1]
        close = float(last["close"])
        last_sup = float(sup.iloc[-1])
        last_res = float(res.iloc[-1])
        last_rsi = float(rsi_ser.iloc[-1])
        prev_rsi = float(rsi_ser.iloc[-2]) if len(rsi_ser) >= 2 else last_rsi
        last_atr = float(atr_ser.iloc[-1])
        last_vol_z = float(vol_z_ser.iloc[-1])
        last_bounce = bool(bounce.iloc[-1])
        last_star = bool(star.iloc[-1])

        if abs(ctx.open_position_size) > 1e-9:
            self._bars_held += 1
            return self._maybe_exit(close, last_sup, last_res, ctx)
        else:
            self._bars_held = 0

        # Regime filter: skip when ATR/price is too high (trending, not ranging)
        if self.atr_regime_pct > 0.0 and close > 0 and last_atr / close > self.atr_regime_pct:
            return None

        # Trend filter (optional — disabled by default)
        trend_ema_val = float(trend_ema_ser.iloc[-1]) if trend_ema_ser is not None else None
        allow_long = trend_ema_val is None or close > trend_ema_val
        allow_short = trend_ema_val is None or close < trend_ema_val

        # Shared conditions
        rsi_turning_up = last_rsi > prev_rsi
        rsi_turning_down = last_rsi < prev_rsi
        vol_ok = last_vol_z >= self.volume_z_min
        room_long = (last_res - close) / max(close, 1e-9) > 0.005
        room_short = (close - last_sup) / max(close, 1e-9) > 0.005

        # ── LONG: price at support, RSI pulling back (< threshold), turning up ──
        if allow_long:
            near_support = (close - last_sup) / max(last_sup, 1e-9) <= self.zone_proximity_pct / 100
            rsi_ok_long = last_rsi < self.rsi_long_threshold and rsi_turning_up

            if near_support and rsi_ok_long and vol_ok and room_long:
                if not self.require_candle_pattern or last_bounce:
                    stop_price = last_sup - last_atr * self.stop_atr_buffer
                    confidence = self._long_confidence(last_rsi, last_vol_z, last_atr, close, last_bounce)
                    log.info(
                        "range_trader_long",
                        close=close, support=last_sup, resistance=last_res,
                        rsi=last_rsi, vol_z=last_vol_z, confidence=confidence,
                    )
                    return Signal(
                        strategy=self.name,
                        symbol="BTC-USD",
                        direction=SignalDirection.LONG,
                        target_notional_usd=self.target_notional_usd,
                        confidence=confidence,
                        confidence_label=SignalConfidence.HIGH if confidence > 0.75 else SignalConfidence.MEDIUM,
                        stop_loss_price=Decimal(str(round(stop_price, 2))),
                        take_profit_price=Decimal(str(round(last_res, 2))),
                        reason=(
                            f"LONG: {close:.0f} at support {last_sup:.0f}, "
                            f"RSI {last_rsi:.1f}↑, vol_z {last_vol_z:.2f}"
                        ),
                        metadata={"support": last_sup, "resistance": last_res,
                                  "atr": last_atr, "rsi": last_rsi, "volume_z": last_vol_z},
                    )

        # ── SHORT: price at resistance, RSI extended (> threshold), turning down ─
        if allow_short:
            near_resistance = (last_res - close) / max(last_res, 1e-9) <= self.zone_proximity_pct / 100
            rsi_ok_short = last_rsi > self.rsi_short_threshold and rsi_turning_down

            if near_resistance and rsi_ok_short and vol_ok and room_short:
                if not self.require_candle_pattern or last_star:
                    stop_price = last_res + last_atr * self.stop_atr_buffer
                    confidence = self._short_confidence(last_rsi, last_vol_z, last_atr, close, last_star)
                    log.info(
                        "range_trader_short",
                        close=close, support=last_sup, resistance=last_res,
                        rsi=last_rsi, vol_z=last_vol_z, confidence=confidence,
                    )
                    return Signal(
                        strategy=self.name,
                        symbol="BTC-USD",
                        direction=SignalDirection.SHORT,
                        target_notional_usd=self.target_notional_usd,
                        confidence=confidence,
                        confidence_label=SignalConfidence.HIGH if confidence > 0.75 else SignalConfidence.MEDIUM,
                        stop_loss_price=Decimal(str(round(stop_price, 2))),
                        take_profit_price=Decimal(str(round(last_sup, 2))),
                        reason=(
                            f"SHORT: {close:.0f} at resistance {last_res:.0f}, "
                            f"RSI {last_rsi:.1f}↓, vol_z {last_vol_z:.2f}"
                        ),
                        metadata={"support": last_sup, "resistance": last_res,
                                  "atr": last_atr, "rsi": last_rsi, "volume_z": last_vol_z},
                    )

        return None

    def _maybe_exit(self, close: float, last_sup: float, last_res: float, ctx: StrategyContext) -> Signal | None:
        is_long = ctx.open_position_size > 0
        entry_dir = SignalDirection.LONG if is_long else SignalDirection.SHORT
        last_sig = next((s for s in reversed(ctx.recent_signals) if s.direction == entry_dir), None)

        if last_sig is None:
            tp = last_res if is_long else last_sup
            sl = 0.0 if is_long else float("inf")
        else:
            tp = float(last_sig.take_profit_price) if last_sig.take_profit_price else (last_res if is_long else last_sup)
            sl = float(last_sig.stop_loss_price) if last_sig.stop_loss_price else (0.0 if is_long else float("inf"))

        hit_tp = (close >= tp) if is_long else (close <= tp)
        hit_sl = (sl > 0 and close <= sl) if is_long else (sl < float("inf") and close >= sl)

        reason = None
        if hit_tp:
            reason = f"TP: {close:.0f} {'≥' if is_long else '≤'} {tp:.0f}"
        elif hit_sl:
            reason = f"SL: {close:.0f} {'≤' if is_long else '≥'} {sl:.0f}"
        elif self._bars_held >= self.max_hold_bars:
            reason = f"max hold {self.max_hold_bars} bars at {close:.0f}"

        if reason:
            return Signal(
                strategy=self.name,
                symbol="BTC-USD",
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=1.0,
                reason=reason,
            )
        return None

    @staticmethod
    def _long_confidence(rsi_val: float, vol_z: float, atr_val: float, close: float, bounce: bool) -> float:
        score = 0.45
        # RSI the further below 50 the better (more pullback = stronger bounce candidate)
        if rsi_val < 30:
            score += 0.20
        elif rsi_val < 40:
            score += 0.12
        elif rsi_val < 45:
            score += 0.06
        if vol_z >= 2.0:
            score += 0.18
        elif vol_z >= 1.0:
            score += 0.10
        elif vol_z >= 0.5:
            score += 0.05
        if bounce:
            score += 0.15  # confirmed hammer candle
        if close > 0 and atr_val / close > 0.03:
            score -= 0.10
        return max(0.0, min(0.95, score))

    @staticmethod
    def _short_confidence(rsi_val: float, vol_z: float, atr_val: float, close: float, star: bool) -> float:
        score = 0.45
        # RSI the further above 50 the better
        if rsi_val > 70:
            score += 0.20
        elif rsi_val > 60:
            score += 0.12
        elif rsi_val > 55:
            score += 0.06
        if vol_z >= 2.0:
            score += 0.18
        elif vol_z >= 1.0:
            score += 0.10
        elif vol_z >= 0.5:
            score += 0.05
        if star:
            score += 0.15  # confirmed shooting-star candle
        if close > 0 and atr_val / close > 0.03:
            score -= 0.10
        return max(0.0, min(0.95, score))
