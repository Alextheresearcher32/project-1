"""
Confidence Score Calibration — the most important part of the ML layer.

A raw LightGBM predict_proba() of 0.70 does NOT mean the trade wins 70%
of the time. It means the model is 70% confident in its label — which
may correspond to a 55% actual win rate. Calibration fixes this mismatch.

After calibration, confidence=0.70 actually means the signal wins ~70%
of the time in the validation set. This makes the confidence score
meaningful for position sizing and Boardroom routing.

Pipeline:
  1. Train LightGBM on training set → raw probabilities
  2. Fit IsotonicRegression on validation set → calibrated probabilities
  3. Evaluate ECE (Expected Calibration Error) — want < 0.05
  4. Re-calibrate every 30 days (walk-forward)

Usage:
    cal = ConfidenceCalibrator()
    cal.fit(y_val, raw_proba_val)
    calibrated = cal.transform(raw_proba_new)
    ece = cal.ece(y_val, calibrated)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class ConfidenceCalibrator:
    """
    Isotonic regression calibrator. Maps raw model probability → true win rate.
    Better than Platt scaling for gradient boosting (non-monotonic relationships).
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._iso: IsotonicRegression | None = None
        self._ece_before: float = float("nan")
        self._ece_after: float = float("nan")

    def fit(self, y_true: np.ndarray, raw_proba: np.ndarray) -> "ConfidenceCalibrator":
        """
        y_true: binary labels (1=win, 0=loss)
        raw_proba: raw model predict_proba output for positive class
        """
        self._ece_before = self.ece_score(y_true, raw_proba)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_proba, y_true)
        calibrated = self._iso.predict(raw_proba)
        self._ece_after = self.ece_score(y_true, calibrated)
        return self

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if self._iso is None:
            return raw_proba
        return np.clip(self._iso.predict(raw_proba), 0.0, 1.0)

    def fit_transform(self, y_true: np.ndarray, raw_proba: np.ndarray) -> np.ndarray:
        self.fit(y_true, raw_proba)
        return self.transform(raw_proba)

    def ece_score(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        """
        Expected Calibration Error — mean absolute difference between
        predicted confidence and actual accuracy across n_bins.
        Target: ECE < 0.05.
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (proba >= lo) & (proba < hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = proba[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    def summary(self) -> dict:
        return {
            "ece_before_calibration": round(self._ece_before, 4),
            "ece_after_calibration": round(self._ece_after, 4),
            "improvement": round(self._ece_before - self._ece_after, 4),
            "calibrated": self._iso is not None,
        }

    def reliability_data(
        self, y_true: np.ndarray, proba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (fraction_of_positives, mean_predicted_value) for reliability diagram."""
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=self.n_bins)
        return frac_pos, mean_pred


class EnsembleConfidenceScorer:
    """
    Combines multiple model signals into a single calibrated confidence score.
    Uses Sharpe-weighted voting: models with higher historical Sharpe get more weight.

    This is the final confidence score that goes onto every Signal.
    """

    def __init__(self):
        self._model_sharpes: dict[str, float] = {}
        self._calibrators: dict[str, ConfidenceCalibrator] = {}

    def register_model(
        self,
        name: str,
        sharpe: float,
        calibrator: ConfidenceCalibrator | None = None,
    ) -> None:
        self._model_sharpes[name] = max(sharpe, 0.0)  # zero-floor negative Sharpe
        self._calibrators[name] = calibrator or ConfidenceCalibrator()

    def score(self, model_probas: dict[str, float]) -> float:
        """
        model_probas: {model_name: raw_probability}
        Returns: single confidence score in [0, 1]
        """
        total_weight = 0.0
        weighted_score = 0.0
        for name, raw_p in model_probas.items():
            sharpe = self._model_sharpes.get(name, 0.0)
            if sharpe <= 0:
                continue
            cal = self._calibrators.get(name)
            calibrated = cal.transform(np.array([raw_p]))[0] if cal and cal._iso else raw_p
            weighted_score += sharpe * calibrated
            total_weight += sharpe

        if total_weight == 0:
            return 0.0
        return float(np.clip(weighted_score / total_weight, 0.0, 1.0))

    def certainty_bonus(self, raw_proba: float) -> float:
        """
        Distance from decision boundary (0.5). Closer to 0 or 1 = more certain.
        Adds 0-0.10 to the base confidence score for high-conviction signals.
        """
        return float(min(abs(raw_proba - 0.5) * 0.2, 0.10))
