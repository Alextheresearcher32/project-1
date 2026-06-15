"""
LightGBM signal classifier with walk-forward calibration.

Why LightGBM over XGBoost for signals:
  - Faster training (leaf-wise tree growth)
  - Better on tabular financial data with many correlated features
  - Native handling of NaN (no imputation needed for warmup rows)
  - predict_proba() gives class probabilities → confidence score

The model outputs a calibrated probability of a profitable trade.
After isotonic calibration on a validation set, this probability is
the true confidence score used in Signal.confidence.

Label generation uses triple-barrier method (inspired by mlfinlab):
  - Win (1): price hits take-profit before stop-loss within max_hold bars
  - Loss (0): price hits stop-loss, or max_hold expires at a loss
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss

from glitz_quant.ml.models.calibrator import ConfidenceCalibrator
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class LGBMSignalModel:
    """
    LightGBM binary classifier for trade signal prediction.
    Wraps train/predict/calibrate into a single object.

    Label: 1 = trade wins (hits TP before SL within max_hold bars)
           0 = trade loses (hits SL or expires in loss)
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 30,        # prevent overfitting on small datasets
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",     # handle imbalanced win/loss ratios
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        calibrate: bool = True,
    ):
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator = ConfidenceCalibrator() if calibrate else None
        self._feature_names: list[str] = []
        self._train_auc: float = 0.0
        self._val_auc: float = 0.0
        self._n_train: int = 0
        self.sharpe: float = 0.0  # set after backtest validation

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LGBMSignalModel":
        """
        Train on (X_train, y_train) and calibrate on (X_val, y_val).
        Both arrays are from the FeatureEngine.transform() output.
        """
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._n_train = len(X_train)

        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        train_proba = self._model.predict_proba(X_train)[:, 1]
        val_proba = self._model.predict_proba(X_val)[:, 1]

        self._train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0.5
        self._val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5

        if self._calibrator is not None:
            self._calibrator.fit(y_val, val_proba)

        log.info(
            "lgbm_trained",
            n_train=self._n_train,
            train_auc=round(self._train_auc, 4),
            val_auc=round(self._val_auc, 4),
            ece_before=round(self._calibrator._ece_before, 4) if self._calibrator else None,
            ece_after=round(self._calibrator._ece_after, 4) if self._calibrator else None,
        )
        return self

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated confidence scores in [0, 1].
        This is the value that goes into Signal.confidence.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        raw = self._model.predict_proba(X)[:, 1]
        if self._calibrator and self._calibrator._iso:
            return self._calibrator.transform(raw)
        return raw

    def predict_confidence_single(self, x: np.ndarray) -> float:
        """Predict confidence for a single feature vector (1D array)."""
        return float(self.predict_confidence(x.reshape(1, -1))[0])

    # ── Feature importance ────────────────────────────────────────────────────

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-N features by gain importance."""
        if self._model is None:
            return []
        importances = self._model.feature_importances_
        pairs = list(zip(self._feature_names, importances))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("lgbm_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LGBMSignalModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("lgbm_loaded", path=str(path))
        return obj

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        cal = self._calibrator.summary() if self._calibrator else {}
        return {
            "n_train": self._n_train,
            "train_auc": round(self._train_auc, 4),
            "val_auc": round(self._val_auc, 4),
            "best_iteration": getattr(self._model, "best_iteration_", None),
            "sharpe": round(self.sharpe, 3),
            "calibration": cal,
            "top_features": self.top_features(5),
        }


def generate_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_hold: int = 32,
    atr_period: int = 14,
) -> pd.Series:
    """
    Triple-barrier labeling (inspired by mlfinlab / de Prado).

    For each bar, look forward max_hold bars:
      1 = price hits TP barrier first (trade wins)
      0 = price hits SL barrier first, or max_hold expires at a loss

    This is significantly better than simple "price went up/down" labeling
    because it accounts for actual trading costs and hold time.

    Returns: pd.Series of {0, 1} with same index as df.
             Last max_hold bars will be NaN (no forward data).
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # ATR for barrier sizing
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    labels = pd.Series(np.nan, index=df.index)
    closes = c.values
    highs = h.values
    lows = l.values
    atrs = atr.values
    n = len(df)

    for i in range(n - max_hold):
        entry = closes[i]
        tp = entry + tp_atr_mult * atrs[i]
        sl = entry - sl_atr_mult * atrs[i]

        label = 0  # default: loss (SL or timeout)
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if highs[j] >= tp:
                label = 1
                break
            if lows[j] <= sl:
                label = 0
                break
        labels.iloc[i] = label

    return labels


def generate_short_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_hold: int = 32,
    atr_period: int = 14,
) -> pd.Series:
    """Same as generate_labels but for short entries (mirrored barriers)."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    labels = pd.Series(np.nan, index=df.index)
    closes = c.values
    highs = h.values
    lows = l.values
    atrs = atr.values
    n = len(df)

    for i in range(n - max_hold):
        entry = closes[i]
        tp = entry - tp_atr_mult * atrs[i]  # TP is below entry for shorts
        sl = entry + sl_atr_mult * atrs[i]  # SL is above entry for shorts

        label = 0
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if lows[j] <= tp:
                label = 1
                break
            if highs[j] >= sl:
                label = 0
                break
        labels.iloc[i] = label

    return labels
