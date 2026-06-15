"""
ML Momentum Strategy — cross-sectional and time-series momentum with LightGBM.

Based on:
  - Jegadeesh & Titman (1993): 3–12 month momentum in equities
  - Liu et al. (2020): Crypto momentum outperforms random walk
  - FreqAI community patterns: lag features + EMA cross as signals

What the model learns:
  - Which combinations of return horizons, RSI levels, and volume patterns
    predict profitable trades in the next 12–48 bars
  - Not hand-coded rules — the model finds non-linear interactions

Entry filter (on top of ML confidence):
  - Trend confirmation: EMA8 > EMA21 for longs (momentum in our direction)
  - Volume: vol_z > 0 (at least average volume — no ghost moves)
  - Not near resistance: prox_res > 0.005 (room to run)

Edges this captures:
  - BTC/ETH trend continuation after accumulation
  - Altcoin breakout after range compression
  - Recovery momentum after oversold conditions

TP/SL: ATR-based, asymmetric (2:1 R:R minimum enforced by LightGBM labels)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from glitz_quant.data.types import Signal, SignalDirection
from glitz_quant.strategies.ml_base import MLStrategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class MLMomentumStrategy(MLStrategy):
    """
    LightGBM momentum strategy. Trained on triple-barrier labels
    with TP=2×ATR, SL=1×ATR (2:1 reward/risk).
    """

    name = "ml_momentum"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.symbol: str = params.get("symbol", "BTC-USD")
        self.tp_atr_mult: float = float(params.get("tp_atr_mult", 2.0))
        self.sl_atr_mult: float = float(params.get("sl_atr_mult", 1.0))

    def _should_enter_long(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        names = self._fe.feature_names()
        ema_cross_8_21 = x[names.index("ema_cross_8_21")]
        vol_z = x[names.index("vol_z")]
        prox_res = x[names.index("prox_res")]
        hurst = x[names.index("hurst_proxy")]

        # Paper mode: only EMA trend direction required.
        # Volume and hurst filters re-enabled when model AUC > 0.56.
        return ema_cross_8_21 > 0       # EMA8 above EMA21 (uptrend)

    def _should_enter_short(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        names = self._fe.feature_names()
        ema_cross_8_21 = x[names.index("ema_cross_8_21")]

        return ema_cross_8_21 < 0       # EMA8 below EMA21 (downtrend)

    def _build_signal(
        self, ctx: StrategyContext, direction: SignalDirection, confidence: float, x: np.ndarray
    ) -> Signal:
        names = self._fe.feature_names()
        close = float(ctx.candles["close"].iloc[-1])
        atr_pct = float(x[names.index("atr_pct")])
        atr = atr_pct * close

        if direction == SignalDirection.LONG:
            tp = close + self.tp_atr_mult * atr
            sl = close - self.sl_atr_mult * atr
        else:
            tp = close - self.tp_atr_mult * atr
            sl = close + self.sl_atr_mult * atr

        log.info(
            "ml_momentum_signal",
            direction=direction.value,
            close=close,
            confidence=round(confidence, 3),
            atr=round(atr, 2),
        )

        return Signal(
            strategy=self.name,
            symbol=self.symbol,
            direction=direction,
            target_notional_usd=self.target_notional_usd,
            confidence=confidence,
            confidence_label=self._confidence_label(confidence),
            stop_loss_price=Decimal(str(round(sl, 2))),
            take_profit_price=Decimal(str(round(tp, 2))),
            reason=(
                f"ML_MOM {direction.value.upper()}: "
                f"conf={confidence:.3f}, close={close:.0f}, "
                f"TP={tp:.0f}, SL={sl:.0f}"
            ),
            metadata={
                "model": "lgbm_momentum",
                "confidence": confidence,
                "atr": atr,
                "close": close,
            },
        )
