"""
Pure-Python technical indicators. Numpy/Pandas-based, no native deps.
Easy to test and reason about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def support_resistance(
    high: pd.Series, low: pd.Series, lookback: int = 96, sensitivity: float = 0.002
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling support/resistance via lookback windows.
    Returns (support, resistance) series same length as input.
    `sensitivity`: cluster zones within +/- this fraction.
    """
    sup = low.rolling(window=lookback, min_periods=lookback).min()
    res = high.rolling(window=lookback, min_periods=lookback).max()
    return sup, res


def bounce_candle(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Bullish bounce confirmation: hammer-like + close above open + decent lower wick.
    Returns boolean series.
    """
    body = (close - open_).abs()
    candle_range = (high - low).replace(0, np.nan)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)

    is_bullish = close > open_
    lower_wick_ratio = (lower_wick / candle_range).fillna(0)
    upper_wick_ratio = (upper_wick / candle_range).fillna(0)
    body_ratio = (body / candle_range).fillna(0)

    return (
        is_bullish
        & (lower_wick_ratio >= 0.4)
        & (upper_wick_ratio <= 0.25)
        & (body_ratio >= 0.15)
    )


def shooting_star_candle(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Bearish reversal at resistance: shooting-star / bearish engulfing pattern.
    Mirror of bounce_candle — large upper wick, small body, small lower wick.
    Returns boolean series.
    """
    body = (close - open_).abs()
    candle_range = (high - low).replace(0, np.nan)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)

    is_bearish = close < open_
    upper_wick_ratio = (upper_wick / candle_range).fillna(0)
    lower_wick_ratio = (lower_wick / candle_range).fillna(0)
    body_ratio = (body / candle_range).fillna(0)

    return (
        is_bearish
        & (upper_wick_ratio >= 0.4)
        & (lower_wick_ratio <= 0.25)
        & (body_ratio >= 0.15)
    )


def volume_z(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Z-score of volume vs rolling lookback. > 1 means above average."""
    mean = volume.rolling(window=lookback, min_periods=lookback).mean()
    std = volume.rolling(window=lookback, min_periods=lookback).std()
    return ((volume - mean) / std.replace(0, np.nan)).fillna(0)
