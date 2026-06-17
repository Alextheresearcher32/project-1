"""
LLM Macro Overlay Strategy.

Uses a slow-updating macro bias (set externally by the orchestrator's
periodic LLM call) combined with EMA trend confirmation to emit signals.

Design:
  - `_macro_bias` is a float in [-1.0, +1.0]:
       > +0.3  → bullish (long bias)
       < -0.3  → bearish (short bias)
       else    → neutral (no entry)
  - The runner calls LLMMacroOverlay.update_macro_bias() from its async
    loop (similar to how signal_analyst is scheduled every 4 hours).
  - on_candle() is pure-sync and reads from the class-level cache.
  - Falls back to EMA cross direction if no macro data has arrived yet.

Entry filters:
  - EMA8 > EMA21 (uptrend) required for longs; EMA8 < EMA21 for shorts
  - Macro bias must agree (or no macro data yet, in which case EMA is enough)
  - 8-hour cooldown between signals (same as ml_momentum)

TP/SL: ATR-based (TP=2×ATR, SL=1×ATR)
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

# ── Module-level macro bias cache (thread-safe via GIL for float writes) ────
_bias_lock = threading.Lock()
_macro_bias: float = 0.0          # -1.0 (bearish) … +1.0 (bullish)
_bias_updated_at: float = 0.0     # unix timestamp of last update
_bias_summary: str = ""           # human-readable LLM rationale


def update_macro_bias(bias: float, summary: str = "") -> None:
    """
    Called by the orchestrator's async loop (via asyncio.to_thread or directly).
    Bias must be in [-1.0, +1.0].
    """
    global _macro_bias, _bias_updated_at, _bias_summary
    with _bias_lock:
        _macro_bias = max(-1.0, min(1.0, float(bias)))
        _bias_updated_at = time.time()
        _bias_summary = summary
    log.info("macro_bias_updated", bias=round(bias, 3), summary=summary[:100])


def get_macro_bias() -> tuple[float, float, str]:
    """Returns (bias, age_seconds, summary)."""
    with _bias_lock:
        age = time.time() - _bias_updated_at if _bias_updated_at > 0 else float("inf")
        return _macro_bias, age, _bias_summary


# ── Strategy ────────────────────────────────────────────────────────────────

_EMA_SHORT = 8
_EMA_LONG = 21
_ATR_PERIOD = 14
_BIAS_THRESHOLD = 0.30        # |bias| must exceed this to use macro signal
_BIAS_MAX_AGE_HOURS = 6.0     # ignore bias older than 6 hours
_COOLDOWN_SECONDS = 8 * 3600  # 8-hour cooldown between entries


class LLMMacroOverlay(Strategy):
    """
    Macro-LLM directed strategy. Enters when LLM macro thesis agrees
    with short-term EMA trend. Falls back to EMA only when no fresh bias.
    """

    name = "llm_macro_overlay"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.symbol: str = params.get("symbol", "BTC-USD")
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 500)))
        self.tp_atr_mult: float = float(params.get("tp_atr_mult", 2.0))
        self.sl_atr_mult: float = float(params.get("sl_atr_mult", 1.0))
        self.min_confidence: float = float(params.get("min_confidence", 0.55))
        self._last_signal_ts: float = 0.0
        self._warmup: int = max(_EMA_LONG, _ATR_PERIOD) + 5

    # ── Public entry point ───────────────────────────────────────────────────

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        df = ctx.candles
        if len(df) < self._warmup:
            return None

        if (time.time() - self._last_signal_ts) < _COOLDOWN_SECONDS:
            return None

        if abs(ctx.open_position_size) > 1e-9:
            return self._maybe_exit(ctx)

        ema_short = self._ema(df["close"], _EMA_SHORT)
        ema_long = self._ema(df["close"], _EMA_LONG)
        atr = self._atr(df, _ATR_PERIOD)
        close = float(df["close"].iloc[-1])

        ema_cross = ema_short - ema_long   # >0 uptrend, <0 downtrend

        bias, age_s, _ = get_macro_bias()
        bias_fresh = age_s < _BIAS_MAX_AGE_HOURS * 3600
        bias_bullish = bias_fresh and bias > _BIAS_THRESHOLD
        bias_bearish = bias_fresh and bias < -_BIAS_THRESHOLD

        # ── LONG ────────────────────────────────────────────────────────────
        if ema_cross > 0:
            if bias_bullish or not bias_fresh:
                confidence = self._confidence(ema_cross, atr, close, bias if bias_fresh else 0.5)
                if confidence >= self.min_confidence:
                    self._last_signal_ts = time.time()
                    return self._build(ctx, SignalDirection.LONG, confidence, close, atr)

        # ── SHORT ───────────────────────────────────────────────────────────
        if ema_cross < 0:
            if bias_bearish or not bias_fresh:
                confidence = self._confidence(abs(ema_cross), atr, close, abs(bias) if bias_fresh else 0.5)
                if confidence >= self.min_confidence:
                    self._last_signal_ts = time.time()
                    return self._build(ctx, SignalDirection.SHORT, confidence, close, atr)

        return None

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(series: pd.Series, span: int) -> float:
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> float:
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])

    def _confidence(self, ema_cross: float, atr: float, close: float, bias_strength: float) -> float:
        # Normalize cross by ATR to get signal clarity (0–1 range)
        cross_pct = min(abs(ema_cross) / (atr + 1e-9), 1.0)
        # Blend with macro bias strength
        raw = 0.5 * cross_pct + 0.5 * min(abs(bias_strength), 1.0)
        return round(min(raw, 1.0), 4)

    def _build(
        self, ctx: StrategyContext, direction: SignalDirection, confidence: float, close: float, atr: float
    ) -> Signal:
        if direction == SignalDirection.LONG:
            tp = Decimal(str(round(close + self.tp_atr_mult * atr, 2)))
            sl = Decimal(str(round(close - self.sl_atr_mult * atr, 2)))
        else:
            tp = Decimal(str(round(close - self.tp_atr_mult * atr, 2)))
            sl = Decimal(str(round(close + self.sl_atr_mult * atr, 2)))

        bias, _, summary = get_macro_bias()
        label = (
            SignalConfidence.HIGH if confidence >= 0.75
            else SignalConfidence.MEDIUM if confidence >= 0.55
            else SignalConfidence.LOW
        )
        log.info(
            "llm_macro_signal",
            direction=direction.value,
            close=close,
            confidence=confidence,
            macro_bias=round(bias, 3),
        )
        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            direction=direction,
            target_notional_usd=self.target_notional_usd,
            confidence=confidence,
            confidence_label=label,
            stop_loss_price=sl,
            take_profit_price=tp,
            reason=(
                f"MACRO {direction.value.upper()}: conf={confidence:.3f}, "
                f"bias={bias:+.2f}, close={close:.0f}, TP={float(tp):.0f}, SL={float(sl):.0f}"
            ),
            metadata={"macro_bias": bias, "macro_summary": summary[:200], "close": close, "atr": atr},
        )

    def _maybe_exit(self, ctx: StrategyContext) -> Signal | None:
        is_long = ctx.open_position_size > 0
        last = next(
            (s for s in reversed(ctx.recent_signals)
             if s.direction == (SignalDirection.LONG if is_long else SignalDirection.SHORT)),
            None,
        )
        close = float(ctx.candles["close"].iloc[-1])
        tp = float(last.take_profit_price) if last and last.take_profit_price else (close * 1.02 if is_long else close * 0.98)
        sl = float(last.stop_loss_price) if last and last.stop_loss_price else (close * 0.99 if is_long else close * 1.01)

        hit_tp = close >= tp if is_long else close <= tp
        hit_sl = close <= sl if is_long else close >= sl

        reason = None
        if hit_tp:
            reason = f"MACRO TP: {close:.0f} {'≥' if is_long else '≤'} {tp:.0f}"
        elif hit_sl:
            reason = f"MACRO SL: {close:.0f} {'≤' if is_long else '≥'} {sl:.0f}"

        if reason:
            return Signal(
                strategy=self.name,
                symbol=self.symbol,
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=1.0,
                reason=reason,
            )
        return None
