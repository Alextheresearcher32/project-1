"""
ML Strategy Base — wraps any LGBMSignalModel behind the Strategy interface.

Every ML strategy inherits from this instead of Strategy directly.
Provides:
  - Automatic feature generation via FeatureEngine
  - Confidence score computation (calibrated probability)
  - Model load/save lifecycle
  - Regime filter (skip entries during trending/volatile periods)
  - Certainty bonus: higher conviction = higher score
  - Direction determination: predict long vs short from signal

Subclasses override:
  - _should_enter_long(ctx, confidence) → bool
  - _should_enter_short(ctx, confidence) → bool  [optional]
  - _build_signal(ctx, direction, confidence, X_last) → Signal
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any
from decimal import Decimal

import numpy as np

from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.ml.features.engine import FeatureEngine
from glitz_quant.ml.models.lgbm import LGBMSignalModel
from glitz_quant.ml.models.calibrator import EnsembleConfidenceScorer
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-9


class MLStrategy(Strategy):
    """
    Base class for all ML-driven strategies in WolfPack.

    Confidence score formula:
        base_confidence    = calibrated LightGBM probability
        certainty_bonus    = 0–0.10 based on distance from 0.5
        regime_multiplier  = 1.0 (ranging) or 0.7 (trending) based on ATR
        final_confidence   = min(base + certainty_bonus, 1.0) * regime_multiplier

    Routing:
        ≥ 0.6 → Boardroom LLM review
        < 0.6 → OMS directly
    """

    name = "ml_base"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.target_notional_usd = Decimal(str(params.get("target_notional_usd", 500)))
        self.max_hold_bars: int = int(params.get("max_hold_bars", 32))
        self.min_confidence: float = float(params.get("min_confidence", 0.55))
        self.atr_regime_pct: float = float(params.get("atr_regime_pct", 0.005))
        self.model_path: str = params.get("model_path", "")
        self.bidirectional: bool = bool(params.get("bidirectional", True))

        self._fe = FeatureEngine()
        self._long_model: LGBMSignalModel | None = None
        self._short_model: LGBMSignalModel | None = None
        self._scorer = EnsembleConfidenceScorer()
        self._bars_held: int = 0
        self._last_signal_ts: float = 0.0  # unix timestamp of last signal (cooldown)
        self._warmup = max(
            max(FeatureEngine.EMA_SPANS),
            max(FeatureEngine.RETURN_HORIZONS),
            48,  # support/resistance lookback
        ) + 10

        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        if self.model_path:
            long_path = Path(self.model_path).with_suffix("") / "long.pkl"
            short_path = Path(self.model_path).with_suffix("") / "short.pkl"
            if long_path.exists():
                self._long_model = LGBMSignalModel.load(long_path)
                self._scorer.register_model("long", self._long_model.sharpe, self._long_model._calibrator)
                log.info("ml_model_loaded", path=str(long_path))
            if short_path.exists():
                self._short_model = LGBMSignalModel.load(short_path)
                self._scorer.register_model("short", self._short_model.sharpe, self._short_model._calibrator)
                log.info("ml_model_loaded", path=str(short_path))

    def set_models(
        self,
        long_model: LGBMSignalModel,
        short_model: LGBMSignalModel | None = None,
    ) -> None:
        """Inject trained models directly (used by training script)."""
        self._long_model = long_model
        self._scorer.register_model("long", long_model.sharpe, long_model._calibrator)
        if short_model:
            self._short_model = short_model
            self._scorer.register_model("short", short_model.sharpe, short_model._calibrator)

    # ── Core entry point ─────────────────────────────────────────────────────

    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        import time as _time
        if len(ctx.candles) < self._warmup:
            log.info("ml_warmup", strategy=self.name, bars=len(ctx.candles), need=self._warmup)
            return None

        # Exit logic first (if holding a position)
        if abs(ctx.open_position_size) > _EPS:
            self._bars_held += 1
            return self._maybe_exit(ctx)
        else:
            self._bars_held = 0

        # Cooldown: don't re-enter within max_hold_bars × 15min of the last signal
        cooldown_seconds = self.max_hold_bars * 15 * 60
        if (_time.time() - self._last_signal_ts) < cooldown_seconds:
            return None

        # Need at least one model loaded
        if self._long_model is None and self._short_model is None:
            return None

        # Feature matrix — last row is current bar
        X = self._fe.transform(ctx.candles)
        x_last = X[-1]

        # Regime filter
        atr_pct = float(x_last[self._fe.feature_names().index("atr_pct")])
        if self.atr_regime_pct > 0 and atr_pct > self.atr_regime_pct:
            log.info("ml_regime_filter", strategy=self.name, atr_pct=round(atr_pct, 5), threshold=self.atr_regime_pct)
            return None  # trending/volatile — skip

        # Regime multiplier (reduce confidence in noisy periods)
        regime_mult = 1.0 if atr_pct < self.atr_regime_pct * 0.7 else 0.85

        # ── LONG signal ───────────────────────────────────────────────────────
        if self._long_model is not None:
            raw_long = self._long_model.predict_confidence_single(x_last)
            cal_long = self._scorer.score({"long": raw_long})
            certainty = self._scorer.certainty_bonus(raw_long)
            long_conf = min(cal_long + certainty, 1.0) * regime_mult

            log.info("ml_score", strategy=self.name, direction="long", raw=round(raw_long, 3), conf=round(long_conf, 3), min=self.min_confidence)
            if long_conf >= self.min_confidence and self._should_enter_long(ctx, long_conf, x_last):
                self._last_signal_ts = _time.time()
                return self._build_signal(ctx, SignalDirection.LONG, long_conf, x_last)

        # ── SHORT signal ──────────────────────────────────────────────────────
        if self.bidirectional and self._short_model is not None:
            raw_short = self._short_model.predict_confidence_single(x_last)
            cal_short = self._scorer.score({"short": raw_short})
            certainty = self._scorer.certainty_bonus(raw_short)
            short_conf = min(cal_short + certainty, 1.0) * regime_mult

            log.info("ml_score", strategy=self.name, direction="short", raw=round(raw_short, 3), conf=round(short_conf, 3), min=self.min_confidence)
            if short_conf >= self.min_confidence and self._should_enter_short(ctx, short_conf, x_last):
                self._last_signal_ts = _time.time()
                return self._build_signal(ctx, SignalDirection.SHORT, short_conf, x_last)

        return None

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _should_enter_long(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        """Override to add strategy-specific entry filters for longs."""
        return True

    def _should_enter_short(self, ctx: StrategyContext, confidence: float, x: np.ndarray) -> bool:
        """Override to add strategy-specific entry filters for shorts."""
        return True

    @abstractmethod
    def _build_signal(
        self, ctx: StrategyContext, direction: SignalDirection, confidence: float, x: np.ndarray
    ) -> Signal:
        """Build the Signal object with TP/SL levels."""
        ...

    # ── Exit logic (shared) ───────────────────────────────────────────────────

    def _maybe_exit(self, ctx: StrategyContext) -> Signal | None:
        is_long = ctx.open_position_size > 0
        entry_dir = SignalDirection.LONG if is_long else SignalDirection.SHORT
        last_sig = next(
            (s for s in reversed(ctx.recent_signals) if s.direction == entry_dir), None
        )

        close = float(ctx.candles["close"].iloc[-1])
        tp = float(last_sig.take_profit_price) if last_sig and last_sig.take_profit_price else (close * 1.02 if is_long else close * 0.98)
        sl = float(last_sig.stop_loss_price) if last_sig and last_sig.stop_loss_price else (close * 0.99 if is_long else close * 1.01)

        hit_tp = (close >= tp) if is_long else (close <= tp)
        hit_sl = (close <= sl) if is_long else (close >= sl)

        reason = None
        if hit_tp:
            reason = f"ML TP: {close:.0f} {'≥' if is_long else '≤'} {tp:.0f}"
        elif hit_sl:
            reason = f"ML SL: {close:.0f} {'≤' if is_long else '≥'} {sl:.0f}"
        elif self._bars_held >= self.max_hold_bars:
            reason = f"ML timeout: {self.max_hold_bars} bars at {close:.0f}"

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
    def _confidence_label(c: float) -> SignalConfidence:
        if c >= 0.75:
            return SignalConfidence.HIGH
        if c >= 0.55:
            return SignalConfidence.MEDIUM
        return SignalConfidence.LOW
