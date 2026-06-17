"""
WolfPack ML Training Script.

Fetches historical BTC/USD 15m candles, engineers features,
labels trades using triple-barrier method, trains LightGBM models
for each strategy, calibrates confidence scores, and saves models.

Run:
    python train_ml_strategies.py
    python train_ml_strategies.py --strategy momentum
    python train_ml_strategies.py --strategy meme_coin
    python train_ml_strategies.py --strategy funding_carry
    python train_ml_strategies.py --optimize    # Optuna hyperparameter search

Output:
    data/models/ml_momentum/long.pkl
    data/models/ml_momentum/short.pkl
    data/models/ml_meme_coin/long.pkl
    data/models/ml_funding_carry/long.pkl   (carry = long when funding < 0)
    data/models/ml_funding_carry/short.pkl  (carry = short when funding > 0)

Confidence score calibration report printed for each model.
ECE target: < 0.05 (well-calibrated)
AUC target: > 0.55 (significantly better than random)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from glitz_quant.settings import get_settings
from glitz_quant.ml.features.engine import FeatureEngine
from glitz_quant.ml.models.lgbm import LGBMSignalModel, generate_labels, generate_short_labels
from glitz_quant.ml.models.calibrator import ConfidenceCalibrator
from glitz_quant.research.backtest.engine import Backtester
from glitz_quant.research.backtest.walk_forward import WalkForwardAnalyzer
from glitz_quant.strategies.ml_momentum import MLMomentumStrategy
from glitz_quant.strategies.ml_meme_coin import MLMemeCoinStrategy
from glitz_quant.strategies.ml_funding_carry import MLFundingCarryStrategy

MODEL_DIR = Path(__file__).parent / "data" / "models"
TRAIN_START = "2022-01-01T00:00:00Z"   # include 2022 bear + 2023 recovery for regime diversity
TRAIN_END = "2026-06-17T00:00:00Z"
VAL_SPLIT = 0.2   # last 20% of data for validation + calibration
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 14

# Tuned LightGBM params — more trees + lower LR to avoid overfitting on noisy labels
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 7,
    "learning_rate": 0.02,
    "n_estimators": 1500,
    "min_child_samples": 50,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def sep(c="═", w=64):
    print(c * w)


async def fetch_candles(start: str, end: str, timeframe: str = "15m") -> pd.DataFrame:
    s = get_settings()
    exchange = ccxt.coinbaseadvanced({
        "apiKey": s.coinbase_api_key.get_secret_value(),
        "secret": s.coinbase_api_secret.get_secret_value(),
        "enableRateLimit": True,
    })
    print(f"\nFetching BTC/USD {timeframe}  {start[:10]} → {end[:10]} ...")
    all_candles: list = []
    since = exchange.parse8601(start)
    end_ts = exchange.parse8601(end)
    for _ in range(500):
        chunk = await exchange.fetch_ohlcv("BTC/USD", timeframe=timeframe, since=since, limit=300)
        if not chunk:
            break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= end_ts:
            break
        since = last_ts + _tf_ms(timeframe)
        await asyncio.sleep(0.3)
    await exchange.close()
    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    print(f"  {len(df):,} candles  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def _tf_ms(tf: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(tf[:-1]) * units[tf[-1]]


def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    params: dict | None = None,
) -> LGBMSignalModel:
    sep()
    print(f"Training: {name}")
    print(f"  Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")
    print(f"  Label balance — Train: {y_train.mean():.1%} wins  |  Val: {y_val.mean():.1%} wins")

    model = LGBMSignalModel(params=params)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    sep("-", 40)
    s = model.summary()
    print(f"  Train AUC    : {s['train_auc']:.4f}")
    print(f"  Val AUC      : {s['val_auc']:.4f}")
    if s.get("calibration"):
        cal = s["calibration"]
        print(f"  ECE before   : {cal['ece_before_calibration']:.4f}")
        print(f"  ECE after    : {cal['ece_after_calibration']:.4f}  (target < 0.05)")
    print(f"  Top features :")
    for feat, imp in s["top_features"]:
        print(f"    {feat:<30} {imp:.0f}")
    return model


def validate_with_backtest(
    model: LGBMSignalModel,
    strategy_cls,
    params: dict,
    df: pd.DataFrame,
) -> float:
    """Run walk-forward on trained model, return avg Sharpe to set model.sharpe."""
    strategy = strategy_cls(params)
    strategy.set_models(model)
    bt = Backtester(10_000.0, 15.0, 5.0)
    wf = WalkForwardAnalyzer(bt, timedelta(days=WF_TRAIN_DAYS), timedelta(days=WF_TEST_DAYS))
    wfr = wf.run(strategy, df)
    sharpe = wfr.aggregate_metrics.get("avg_sharpe", 0.0)
    pf = wfr.aggregate_metrics.get("avg_profit_factor", 0.0)
    pct_pos = wfr.aggregate_metrics.get("pct_windows_positive_sharpe", 0.0) * 100
    print(f"\n  Walk-Forward Validation:")
    print(f"    Avg Sharpe : {sharpe:+.3f}")
    print(f"    Avg PF     : {pf:.2f}")
    print(f"    % positive : {pct_pos:.0f}%")
    return float(sharpe)


def run_momentum(df: pd.DataFrame) -> None:
    sep("═")
    print("STRATEGY: ML MOMENTUM  (BTC/USD 15m)")
    sep("═")

    fe = FeatureEngine()
    X = fe.transform(df)
    feat_names = fe.feature_names()

    # Labels: long (TP=2×ATR, SL=1×ATR, max_hold=32)
    y_long = generate_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold=32)
    y_short = generate_short_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold=32)

    # Remove warmup + NaN labels
    warmup = 210
    valid = ~y_long.isna()
    valid.iloc[:warmup] = False
    idx = valid.values

    X_clean = X[idx]
    y_long_clean = y_long.values[idx].astype(int)
    y_short_clean = y_short.values[idx].astype(int)

    split = int(len(X_clean) * (1 - VAL_SPLIT))
    X_tr, X_val = X_clean[:split], X_clean[split:]
    y_long_tr, y_long_val = y_long_clean[:split], y_long_clean[split:]
    y_short_tr, y_short_val = y_short_clean[:split], y_short_clean[split:]

    long_model = train_model("momentum_long", X_tr, y_long_tr, X_val, y_long_val, feat_names, params=LGBM_PARAMS)
    short_model = train_model("momentum_short", X_tr, y_short_tr, X_val, y_short_val, feat_names, params=LGBM_PARAMS)

    out = MODEL_DIR / "ml_momentum"
    out.mkdir(parents=True, exist_ok=True)

    # Save immediately with placeholder Sharpe so models survive a kill/interrupt.
    # Walk-forward below will update .sharpe and resave.
    long_model.sharpe = 0.3
    short_model.sharpe = 0.3
    long_model.save(out / "long.pkl")
    short_model.save(out / "short.pkl")
    print(f"\n  Saved (pre-WF) → {out}/")

    strat_params = {
        "symbol": "BTC-USD",
        "target_notional_usd": 500,
        "min_confidence": 0.33,
        "atr_regime_pct": 0.005,
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "max_hold_bars": 32,
        "bidirectional": True,
    }
    long_model.sharpe = validate_with_backtest(long_model, MLMomentumStrategy, strat_params, df)
    short_model.sharpe = max(long_model.sharpe, 0.1)

    long_model.save(out / "long.pkl")
    short_model.save(out / "short.pkl")
    print(f"\n  Saved (post-WF) → {out}/")


def run_meme_coin(df: pd.DataFrame) -> None:
    sep("═")
    print("STRATEGY: ML MEME COIN / PUMP DETECTION  (BTC/USD 15m)")
    sep("═")
    print("  Note: For real meme coins, replace df with top-50 altcoin data.")
    print("  Training on BTC as proxy for now.\n")

    fe = FeatureEngine()
    X = fe.transform(df)
    feat_names = fe.feature_names()

    # Tighter labels for fast pump-and-dump (TP=1.5, SL=0.75, 16 bars)
    y_long = generate_labels(df, tp_atr_mult=1.5, sl_atr_mult=0.75, max_hold=16)
    y_short = generate_short_labels(df, tp_atr_mult=1.5, sl_atr_mult=0.75, max_hold=16)

    warmup = 210
    valid = ~y_long.isna()
    valid.iloc[:warmup] = False
    idx = valid.values

    X_clean = X[idx]
    y_long_clean = y_long.values[idx].astype(int)
    y_short_clean = y_short.values[idx].astype(int)

    split = int(len(X_clean) * (1 - VAL_SPLIT))
    X_tr, X_val = X_clean[:split], X_clean[split:]
    y_long_tr, y_long_val = y_long_clean[:split], y_long_clean[split:]
    y_short_tr, y_short_val = y_short_clean[:split], y_short_clean[split:]

    long_model = train_model("meme_long", X_tr, y_long_tr, X_val, y_long_val, feat_names)
    short_model = train_model("meme_short", X_tr, y_short_tr, X_val, y_short_val, feat_names)

    long_model.sharpe = 0.1
    short_model.sharpe = 0.1
    out = MODEL_DIR / "ml_meme_coin"
    out.mkdir(parents=True, exist_ok=True)
    long_model.save(out / "long.pkl")
    short_model.save(out / "short.pkl")
    print(f"\n  Saved → {out}/")

    if args.validate:
        params = {
            "symbol": "BTC-USD",
            "target_notional_usd": 300,
            "min_confidence": 0.58,
            "atr_regime_pct": 0.008,
            "min_vol_z": 2.0,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 0.75,
            "max_hold_bars": 16,
        }
        long_model.sharpe = validate_with_backtest(long_model, MLMemeCoinStrategy, params, df)
        short_model.sharpe = max(long_model.sharpe * 0.7, 0.1)
        long_model.save(out / "long.pkl")
        short_model.save(out / "short.pkl")


def run_funding_carry(df: pd.DataFrame) -> None:
    sep("═")
    print("STRATEGY: ML FUNDING CARRY  (BTC/USD Perpetuals)")
    sep("═")
    print("  Note: Training on 15m OHLCV as proxy for regime detection.")
    print("  In production, add funding_rate + OI to extra_data via CCXT.\n")

    fe = FeatureEngine()
    X = fe.transform(df)
    feat_names = fe.feature_names()

    # Carry uses 8h holding periods = 32 bars on 15m
    y_long = generate_labels(df, tp_atr_mult=1.5, sl_atr_mult=0.75, max_hold=32)
    y_short = generate_short_labels(df, tp_atr_mult=1.5, sl_atr_mult=0.75, max_hold=32)

    warmup = 210
    valid = ~y_long.isna()
    valid.iloc[:warmup] = False
    idx = valid.values

    X_clean = X[idx]
    y_long_clean = y_long.values[idx].astype(int)
    y_short_clean = y_short.values[idx].astype(int)

    split = int(len(X_clean) * (1 - VAL_SPLIT))
    X_tr, X_val = X_clean[:split], X_clean[split:]
    y_long_tr, y_long_val = y_long_clean[:split], y_long_clean[split:]
    y_short_tr, y_short_val = y_short_clean[:split], y_short_clean[split:]

    long_model = train_model("funding_long", X_tr, y_long_tr, X_val, y_long_val, feat_names)
    short_model = train_model("funding_short", X_tr, y_short_tr, X_val, y_short_val, feat_names)

    long_model.sharpe = 0.1
    short_model.sharpe = 0.1
    out = MODEL_DIR / "ml_funding_carry"
    out.mkdir(parents=True, exist_ok=True)
    long_model.save(out / "long.pkl")
    short_model.save(out / "short.pkl")
    print(f"\n  Saved → {out}/")

    if args.validate:
        params = {
            "symbol": "BTC-USD",
            "target_notional_usd": 500,
            "min_confidence": 0.58,
            "atr_regime_pct": 0.006,
            "min_annual_funding": 0.05,
            "max_annual_funding": 2.0,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 0.75,
        }
        long_model.sharpe = validate_with_backtest(long_model, MLFundingCarryStrategy, params, df)
        short_model.sharpe = validate_with_backtest(short_model, MLFundingCarryStrategy, params, df)
        long_model.save(out / "long.pkl")
        short_model.save(out / "short.pkl")


async def main() -> None:
    df = await fetch_candles(TRAIN_START, TRAIN_END)

    if args.strategy in ("all", "momentum"):
        run_momentum(df)

    if args.strategy in ("all", "meme_coin"):
        run_meme_coin(df)

    if args.strategy in ("all", "funding_carry"):
        run_funding_carry(df)

    sep("═")
    print(f"\nAll models saved to {MODEL_DIR}/")
    if not args.validate:
        print("\nRun with --validate to run walk-forward Sharpe validation (slow, ~2h)")
    print("\nNext steps:")
    print("  1. Check ECE after < 0.05  (calibration quality)")
    print("  2. Check Val AUC > 0.55    (edge over random)")
    print("  3. Enable in config/strategies.yaml  (enabled: true)")
    print("  4. Run paper mode: python -m glitz_quant.orchestrator.runner")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["all", "momentum", "meme_coin", "funding_carry"])
    parser.add_argument("--validate", action="store_true", help="Run walk-forward Sharpe validation after training (slow)")
    args = parser.parse_args()
    asyncio.run(main())
