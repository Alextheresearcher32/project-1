"""
Feature Engineering Engine — the foundation of every ML strategy.

Takes a raw OHLCV DataFrame and produces a normalized feature matrix.
Every ML strategy in WolfPack calls FeatureEngine.transform() to get
its input vector. All features are stationary (returns, ratios, z-scores)
so they generalize across different price regimes.

Feature groups:
  returns    : log-returns at 1/3/6/12/24/48/96 bar horizons
  momentum   : RSI, EMA-cross, ROC at multiple periods
  volatility : ATR/price ratio, Bollinger width, realized vol, vol rank
  volume     : volume z-score, volume momentum, OBV change
  structure  : proximity to support/resistance, range position, VWAP deviation
  regime     : ATR regime, trend strength (ADX proxy), hurst proxy
  htf        : 1h and 4h synthetic trend features (aggregated from 15m bars)
  time       : hour-of-day and day-of-week cyclical encodings (sin/cos)
  candle     : bounce_candle, shooting_star, body_ratio, wick_imbalance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-9


class FeatureEngine:
    """
    Stateless transformer: DataFrame → feature matrix (numpy array).

    Usage:
        fe = FeatureEngine()
        X = fe.transform(candles_df)          # shape (n_bars, n_features)
        feature_names = fe.feature_names()
    """

    # Horizons (in 15m bars) for return and momentum features
    RETURN_HORIZONS = [1, 3, 6, 12, 24, 48, 96]   # up to 24h
    RSI_PERIODS = [7, 14, 21]
    EMA_SPANS = [8, 21, 55, 200]
    VOL_LOOKBACKS = [6, 12, 24]
    # Bars per higher timeframe (on 15m data: 4=1h, 16=4h, 96=1d)
    HTF_BARS = {"1h": 4, "4h": 16, "1d": 96}

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        df: OHLCV DataFrame with columns [open, high, low, close, volume].
            Index should be DatetimeTzAware (UTC) for time features to work.
        Returns: float32 array of shape (len(df), n_features).
                 Rows with NaN (warmup period) are forward-filled then zero-filled.
        """
        feats: dict[str, pd.Series] = {}
        c = df["close"]
        o = df["open"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        # ── Returns ──────────────────────────────────────────────────────────
        log_c = np.log(c + _EPS)
        for n in self.RETURN_HORIZONS:
            feats[f"ret_{n}"] = log_c - log_c.shift(n)

        # ── RSI ───────────────────────────────────────────────────────────────
        for p in self.RSI_PERIODS:
            feats[f"rsi_{p}"] = self._rsi(c, p) / 100.0  # normalized 0-1

        # ── EMA crossovers (fast / slow ratio) ────────────────────────────────
        emas = {s: c.ewm(span=s, adjust=False).mean() for s in self.EMA_SPANS}
        pairs = [(8, 21), (21, 55), (55, 200)]
        for fast, slow in pairs:
            cross = (emas[fast] - emas[slow]) / (emas[slow] + _EPS)
            feats[f"ema_cross_{fast}_{slow}"] = cross

        # Distance from EMA200 (trend direction signal)
        feats["dist_ema200"] = (c - emas[200]) / (emas[200] + _EPS)

        # ── Volatility ────────────────────────────────────────────────────────
        atr_ser = self._atr(h, l, c, 14)
        feats["atr_pct"] = atr_ser / (c + _EPS)  # ATR as % of price (regime)

        for lb in self.VOL_LOOKBACKS:
            rvol = c.pct_change().rolling(lb).std() * np.sqrt(lb)
            feats[f"rvol_{lb}"] = rvol

        # Bollinger Band width (relative)
        bb_mean = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        feats["bb_width"] = (2 * bb_std) / (bb_mean + _EPS)
        feats["bb_pos"] = (c - (bb_mean - 2 * bb_std)) / (4 * bb_std + _EPS)  # 0=lower,1=upper

        # Realized vol rank: where is current vol vs past 30 days (192 bars on 15m)?
        rvol_14 = c.pct_change().rolling(14).std()
        feats["vol_rank"] = rvol_14.rolling(192).rank(pct=True)

        # ATR compression: low ATR regime often precedes breakout
        atr_rank = atr_ser.rolling(192).rank(pct=True)
        feats["atr_rank"] = atr_rank

        # ── Volume ────────────────────────────────────────────────────────────
        vol_mean = v.rolling(20).mean()
        vol_std = v.rolling(20).std()
        feats["vol_z"] = (v - vol_mean) / (vol_std + _EPS)
        feats["vol_momentum"] = np.log(v + 1) - np.log(v.shift(5) + 1)

        # OBV change (volume-weighted direction)
        obv = (np.sign(c.diff()) * v).cumsum()
        feats["obv_ret_12"] = (obv - obv.shift(12)) / (v.rolling(12).sum() + _EPS)

        # Volume trend (is volume growing or shrinking vs 20-bar average?)
        feats["vol_trend"] = (v.rolling(5).mean() - v.rolling(20).mean()) / (v.rolling(20).mean() + _EPS)

        # ── Price structure ───────────────────────────────────────────────────
        sup = l.rolling(48).min()
        res = h.rolling(48).max()
        rng = (res - sup + _EPS)
        feats["range_pos"] = (c - sup) / rng       # 0=near support, 1=near resistance
        feats["prox_sup"] = (c - sup) / (c + _EPS) # % above support
        feats["prox_res"] = (res - c) / (c + _EPS) # % below resistance

        # High-low range of last bar vs ATR
        feats["bar_range_atr"] = (h - l) / (atr_ser + _EPS)

        # VWAP deviation (proxy: typical price vs rolling VWAP over session ~96 bars = 1 day)
        typical = (h + l + c) / 3
        vwap = (typical * v).rolling(96).sum() / (v.rolling(96).sum() + _EPS)
        feats["vwap_dev"] = (c - vwap) / (vwap + _EPS)

        # ── Higher timeframe synthetic features ───────────────────────────────
        for tf_name, bars in self.HTF_BARS.items():
            # Aggregate close by rolling window — approximate HTF close
            htf_close = c.rolling(bars).mean()       # smoothed HTF close
            htf_ret = htf_close / htf_close.shift(bars) - 1
            feats[f"htf_{tf_name}_ret"] = htf_ret

            htf_ema_fast = htf_close.ewm(span=8, adjust=False).mean()
            htf_ema_slow = htf_close.ewm(span=21, adjust=False).mean()
            feats[f"htf_{tf_name}_trend"] = (htf_ema_fast - htf_ema_slow) / (htf_ema_slow + _EPS)

        # ── Regime ────────────────────────────────────────────────────────────
        # ADX proxy: directional movement
        dm_plus = (h - h.shift(1)).clip(lower=0)
        dm_minus = (l.shift(1) - l).clip(lower=0)
        tr = self._tr(h, l, c)
        di_plus = dm_plus.rolling(14).mean() / (tr.rolling(14).mean() + _EPS)
        di_minus = dm_minus.rolling(14).mean() / (tr.rolling(14).mean() + _EPS)
        dx = (di_plus - di_minus).abs() / (di_plus + di_minus + _EPS)
        feats["adx_proxy"] = dx.rolling(14).mean()  # >0.25 = trending

        # Hurst exponent proxy: variance ratio (H>0.5 = trending, H<0.5 = mean-reverting)
        ret1 = c.pct_change(1)
        ret4 = c.pct_change(4)
        var1 = ret1.rolling(32).var()
        var4 = ret4.rolling(32).var()
        feats["hurst_proxy"] = np.log(var4 / (4 * var1 + _EPS) + _EPS) / np.log(4)

        # ── Time-of-day / day-of-week (cyclical encoding) ────────────────────
        # Crypto has strong intraday seasonality: US open volatility, Asian
        # accumulation, weekend patterns. Sin/cos encoding keeps the circular
        # nature of time (23:00 is close to 00:00, not far away).
        idx = df.index
        if hasattr(idx, "hour"):
            hour = pd.Series(idx.hour, index=df.index, dtype=float)
            dow = pd.Series(idx.dayofweek, index=df.index, dtype=float)
        else:
            try:
                hour = pd.Series(idx.to_series().dt.hour.values, index=df.index, dtype=float)
                dow = pd.Series(idx.to_series().dt.dayofweek.values, index=df.index, dtype=float)
            except Exception:
                hour = pd.Series(0.0, index=df.index)
                dow = pd.Series(0.0, index=df.index)

        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feats["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

        # ── Candle patterns (boolean → 0/1) ──────────────────────────────────
        body = (c - o).abs()
        crange = (h - l + _EPS)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)

        feats["bounce_candle"] = (
            (c > o)
            & (lower_wick / crange >= 0.4)
            & (upper_wick / crange <= 0.25)
            & (body / crange >= 0.15)
        ).astype(float)

        feats["shooting_star"] = (
            (c < o)
            & (upper_wick / crange >= 0.4)
            & (lower_wick / crange <= 0.25)
            & (body / crange >= 0.15)
        ).astype(float)

        # Body ratio and wick balance
        feats["body_ratio"] = body / crange
        feats["wick_imbalance"] = (lower_wick - upper_wick) / (crange + _EPS)

        # ── Assemble ──────────────────────────────────────────────────────────
        feat_df = pd.DataFrame(feats, index=df.index)
        feat_df = feat_df.ffill().fillna(0.0)

        return feat_df.values.astype(np.float32)

    def feature_names(self) -> list[str]:
        """Return list of feature names in same order as transform() output."""
        names = []
        for n in self.RETURN_HORIZONS:
            names.append(f"ret_{n}")
        for p in self.RSI_PERIODS:
            names.append(f"rsi_{p}")
        for fast, slow in [(8, 21), (21, 55), (55, 200)]:
            names.append(f"ema_cross_{fast}_{slow}")
        names += [
            "dist_ema200", "atr_pct",
            *[f"rvol_{lb}" for lb in self.VOL_LOOKBACKS],
            "bb_width", "bb_pos",
            "vol_rank", "atr_rank",
            "vol_z", "vol_momentum", "obv_ret_12", "vol_trend",
            "range_pos", "prox_sup", "prox_res", "bar_range_atr", "vwap_dev",
        ]
        for tf_name in self.HTF_BARS:
            names += [f"htf_{tf_name}_ret", f"htf_{tf_name}_trend"]
        names += [
            "adx_proxy", "hurst_proxy",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "bounce_candle", "shooting_star", "body_ratio", "wick_imbalance",
        ]
        return names

    def n_features(self) -> int:
        return len(self.feature_names())

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / (loss + _EPS)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_c = close.shift(1)
        return pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)

    @classmethod
    def _atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        return cls._tr(high, low, close).ewm(alpha=1 / period, adjust=False).mean()
