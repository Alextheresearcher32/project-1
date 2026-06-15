"""
ML Meme Coin / Volume Spike Strategy.

Meme coins and altcoin pumps share a common signature:
  1. Volume spike: vol_z > 3 (3σ above average — the crowd is arriving)
  2. Price acceleration: ret_1 and ret_3 both positive and increasing
  3. Volatility expansion: ATR% expanding (the move has energy)
  4. RSI not yet extreme: RSI14 < 75 (still room to run before exhaustion)
  5. Momentum alignment: EMA8 > EMA21 (short-term trend confirmed)

What LightGBM adds:
  - Learns which COMBINATION of these features predicts continuation vs fake pump
  - Learns that RSI 70 + vol_z 5 is very different from RSI 70 + vol_z 1
  - Learns time-of-day patterns in the features (if timestamp features added)
  - Learns when a pump is a front-run opportunity vs exhaustion

Exit rules (faster than momentum — meme moves are short):
  - TP: 1.5× ATR (tighter — meme pumps revert fast)
  - SL: 0.75× ATR (tight stop — false pumps are common)
  - Max hold: 16 bars (4 hours on 15m) — get out early

Risk note: volume spikes can be wash trading. Model is trained to
distinguish real volume (confirmed by multi-bar continuation) from
single-bar anomalies. Vol_z on MULTIPLE bars (1 and 3 bars) is key.

Futures use: same logic applies to perp futures pump detection.
Look for large open interest spikes alongside volume — signals forced
liquidations are incoming or large players are accumulating.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from glitz_quant.data.types import Signal, SignalDirection
from glitz_quant.strategies.ml_base import MLStrategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class MLMemeCoinStrategy(MLStrategy):
    """
    LightGBM pump detection. Trained on triple-barrier labels
    with tight TP=1.5×ATR, SL=0.75×ATR (quick in, quick out).
    """

    name = "ml_meme_coin"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.symbol: str = params.get("symbol", "BTC-USD")
        self.tp_atr_mult: float = float(params.get("tp_atr_mult", 1.5))
        self.sl_atr_mult: float = float(params.get("sl_atr_mult", 0.75))
        self.min_vol_z: float = float(params.get("min_vol_z", 2.0))  # volume spike threshold
        # Override parent — shorter hold for meme moves
        self.max_hold_bars: int = int(params.get("max_hold_bars", 16))

    def _should_enter_long(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        names = self._fe.feature_names()
        vol_z = x[names.index("vol_z")]
        vol_mom = x[names.index("vol_momentum")]
        ret_1 = x[names.index("ret_1")]
        ret_3 = x[names.index("ret_3")]
        rsi_14 = x[names.index("rsi_14")]  # normalized 0-1

        return (
            vol_z >= self.min_vol_z            # volume spike present
            and vol_mom > 0                    # volume is accelerating
            and ret_1 > 0                      # price moving up now
            and ret_3 > 0                      # momentum over last 3 bars
            and rsi_14 < 0.80                  # RSI < 80 (not yet blow-off)
        )

    def _should_enter_short(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        # Meme coins rarely sustain shorts (whale buying support)
        # Only short during genuine distribution: volume spike on red candle
        names = self._fe.feature_names()
        vol_z = x[names.index("vol_z")]
        ret_1 = x[names.index("ret_1")]
        shooting_star = x[names.index("shooting_star")]

        return (
            vol_z >= self.min_vol_z
            and ret_1 < 0                      # price falling on high volume
            and shooting_star > 0.5            # bearish candle pattern confirmed
        )

    def _build_signal(
        self, ctx: StrategyContext, direction: SignalDirection, confidence: float, x: np.ndarray
    ) -> Signal:
        names = self._fe.feature_names()
        close = float(ctx.candles["close"].iloc[-1])
        atr_pct = float(x[names.index("atr_pct")])
        atr = atr_pct * close
        vol_z = float(x[names.index("vol_z")])

        if direction == SignalDirection.LONG:
            tp = close + self.tp_atr_mult * atr
            sl = close - self.sl_atr_mult * atr
        else:
            tp = close - self.tp_atr_mult * atr
            sl = close + self.sl_atr_mult * atr

        log.info(
            "ml_meme_coin_signal",
            direction=direction.value,
            close=close,
            confidence=round(confidence, 3),
            vol_z=round(vol_z, 2),
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
                f"ML_PUMP {direction.value.upper()}: "
                f"conf={confidence:.3f}, vol_z={vol_z:.1f}, close={close:.0f}"
            ),
            metadata={
                "model": "lgbm_meme_coin",
                "confidence": confidence,
                "vol_z": vol_z,
                "atr": atr,
            },
        )
