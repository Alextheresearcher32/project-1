"""
Full multi-strategy backtest on 12 years of BTC/USD 15m data.

Flow:
  1. Fetch from Kraken (max history, ~2014-present)
  2. Run baseline walk-forward on all 6 backtestable strategies
  3. Grid-search key params for btc_trend, bitcoin_range, range_trader
  4. Walk-forward on top-3 param combos per rule-based strategy
  5. Print ranked results and show best params to copy into strategies.yaml

Skipped (require live data or LLM calls):
  - funding_rate_carry  (needs real-time funding_rate_8h)
  - llm_macro_overlay   (expensive LLM calls per bar)

Usage:
  cd /Users/larissasylvester/src
  PYTHONPATH=/Users/larissasylvester/src python run_full_backtest.py
"""
from __future__ import annotations

import asyncio
import itertools
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from glitz_quant.research.backtest.engine import Backtester
from glitz_quant.research.backtest.walk_forward import WalkForwardAnalyzer
from glitz_quant.settings import get_settings
from glitz_quant.strategies.bitcoin_range import BitcoinRangeMomentum
from glitz_quant.strategies.btc_trend import BtcTrendFollow
from glitz_quant.strategies.ml_funding_carry import MLFundingCarryStrategy
from glitz_quant.strategies.ml_meme_coin import MLMemeCoinStrategy
from glitz_quant.strategies.ml_momentum import MLMomentumStrategy
from glitz_quant.strategies.range_trader import RangeTrader

# ── Config ────────────────────────────────────────────────────────────────────
STARTING_CASH = 10_000.0
FEE_BPS = 15.0
SLIP_BPS = 5.0
WF_TRAIN = timedelta(days=120)   # 4-month train window
WF_TEST  = timedelta(days=30)    # 1-month test window
GRID_WORKERS = 1                 # single-threaded to avoid rate limit

DATA_CACHE = Path("/tmp/btc_15m_12yr.parquet")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sep(c="═", w=70): print(c * w)

def _fmt(m: dict) -> str:
    sh   = m.get("sharpe", 0)
    ret  = m.get("total_return", 0) * 100
    dd   = m.get("max_drawdown", 0) * 100
    wr   = m.get("win_rate", 0) * 100
    pf   = m.get("profit_factor", 0)
    nt   = int(m.get("num_trades", 0))
    flag = "✓" if sh > 0.5 else ("~" if sh > 0 else "✗")
    return f"{flag} Sharpe {sh:+.3f} | ret {ret:+.1f}% | DD {dd:.1f}% | WR {wr:.0f}% | PF {pf:.2f} | {nt} trades"

def _wf(strategy, df: pd.DataFrame) -> dict:
    bt = Backtester(STARTING_CASH, FEE_BPS, SLIP_BPS)
    wf = WalkForwardAnalyzer(bt, WF_TRAIN, WF_TEST)
    r = wf.run(strategy, df)
    return r.aggregate_metrics

def _sp(strategy, df: pd.DataFrame) -> dict:
    bt = Backtester(STARTING_CASH, FEE_BPS, SLIP_BPS)
    return bt.run(strategy, df).metrics


# ── Data fetch ────────────────────────────────────────────────────────────────
async def fetch_kraken() -> pd.DataFrame:
    if DATA_CACHE.exists():
        df = pd.read_parquet(DATA_CACHE)
        print(f"  [cache] {len(df):,} candles  {df.index[0].date()} → {df.index[-1].date()}")
        return df

    s = get_settings()
    exchange = ccxt.kraken({
        "apiKey":  s.kraken_api_key.get_secret_value() if s.kraken_api_key else "",
        "secret":  s.kraken_api_secret.get_secret_value() if s.kraken_api_secret else "",
        "enableRateLimit": True,
    })

    # Kraken has BTC/USD data from ~Jan 2014
    since = exchange.parse8601("2014-01-01T00:00:00Z")
    all_candles: list[list] = []
    batch = 0
    max_batches = 2000  # safety cap

    print("  Fetching Kraken BTC/USD 15m from 2014 …", flush=True)
    while batch < max_batches:
        try:
            chunk = await exchange.fetch_ohlcv("BTC/USD", timeframe="15m", since=since, limit=720)
        except Exception as e:
            print(f"  [warn] batch {batch}: {e}. Retrying in 2s …")
            await asyncio.sleep(2)
            continue

        if not chunk:
            break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        now_ms = exchange.milliseconds()
        if last_ts >= now_ms - 15 * 60 * 1000:
            break
        since = last_ts + 15 * 60 * 1000
        batch += 1
        if batch % 50 == 0:
            pct = (last_ts - exchange.parse8601("2014-01-01T00:00:00Z")) / (now_ms - exchange.parse8601("2014-01-01T00:00:00Z")) * 100
            print(f"  … batch {batch}  {pd.Timestamp(last_ts, unit='ms', tz='UTC').date()}  ({pct:.0f}%)", flush=True)
        await asyncio.sleep(0.5)

    await exchange.close()

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df.to_parquet(DATA_CACHE)
    print(f"  Saved to {DATA_CACHE}")
    print(f"  {len(df):,} candles  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── Vectorised single-pass backtests (for fast grid search) ──────────────────

def _np_ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _np_wilder(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (used by ATR/RSI)."""
    out = np.zeros_like(x, dtype=float)
    if period >= len(x):
        return out
    out[period - 1] = x[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(x)):
        out[i] = out[i - 1] * (1 - alpha) + x[i] * alpha
    return out


def _np_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    return _np_wilder(tr, period)


def _np_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g = _np_wilder(gains, period)
    avg_l = _np_wilder(losses, period)
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    return 100 - 100 / (1 + rs)


def _np_support_resistance(low: np.ndarray, high: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros_like(low)
    resistance = np.zeros_like(high)
    for i in range(lookback, len(low)):
        support[i] = low[max(0, i - lookback):i].min()
        resistance[i] = high[max(0, i - lookback):i].max()
    return support, resistance


def _sharpe_from_equity(equity: np.ndarray, bars_per_year: int = 365 * 24 * 4) -> float:
    if len(equity) < 2:
        return 0.0
    rets = np.diff(equity) / (equity[:-1] + 1e-9)
    if rets.std() < 1e-10:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))


def _trades_stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0, "total_return": 0.0}
    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9
    return {
        "num_trades": len(pnls),
        "win_rate": len(wins) / len(pnls),
        "profit_factor": gross_profit / gross_loss,
        "total_pnl": arr.sum(),
        "total_return": arr.sum() / STARTING_CASH,
    }


# ── Fast btc_trend (numpy loop, no pandas overhead) ──────────────────────────

def fast_btc_trend(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    n = len(close)

    ef = p.get("ema_fast", 8);  es = p.get("ema_slow", 21);  et = p.get("ema_trend", 55)
    tp = p.get("tp_atr_mult", 3.0);  sl = p.get("sl_atr_mult", 1.0)
    rsi_min = p.get("rsi_long_min", 50.0);  rsi_max = p.get("rsi_short_max", 50.0)
    cooldown = p.get("cooldown_bars", 16);  max_hold = p.get("max_hold_bars", 96)
    atr_min = p.get("atr_regime_min", 0.002);  bidir = p.get("bidirectional", True)

    ema_f  = _np_ema(close, ef)
    ema_s  = _np_ema(close, es)
    ema_t  = _np_ema(close, et)
    atr_v  = _np_atr(high, low, close)
    rsi_v  = _np_rsi(close)

    fee = FEE_BPS / 10000;  slip = SLIP_BPS / 10000
    notional = p.get("target_notional_usd", 500.0)
    warmup = max(et + 10, 220)

    cash = STARTING_CASH
    pos = 0.0;  entry_p = 0.0;  tp_p = 0.0;  sl_p = 0.0
    bars_held = 0;  bars_since = 999
    is_long = False;  is_short = False
    pnls: list[float] = []
    equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i];  a = atr_v[i]
        bars_since += 1

        if pos != 0.0:
            bars_held += 1
            exit_now = False
            if is_long:
                cross_dn = ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]
                exit_now = c >= tp_p or c <= sl_p or cross_dn or bars_held >= max_hold
            elif is_short:
                cross_up = ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]
                exit_now = c <= tp_p or c >= sl_p or cross_up or bars_held >= max_hold

            if exit_now:
                if is_long:
                    fill = c * (1 - slip);  pnl = (fill - entry_p) * pos - (entry_p + fill) * pos * fee
                else:
                    fill = c * (1 + slip);  pnl = (entry_p - fill) * abs(pos) - (entry_p + fill) * abs(pos) * fee
                pnls.append(pnl);  cash += pnl
                pos = 0.0;  is_long = False;  is_short = False;  bars_held = 0

        mtm = cash if pos == 0.0 else (
            cash + (c - entry_p) * pos if is_long else cash + (entry_p - c) * abs(pos)
        )
        equity.append(mtm)

        if pos != 0.0 or bars_since < cooldown:
            continue
        if a / (c + 1e-9) < atr_min:
            continue

        cross_up = ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]
        cross_dn = ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]

        if cross_up and c > ema_t[i] and rsi_v[i] > rsi_min:
            pos = notional / (c + 1e-9);  entry_p = c * (1 + slip)
            tp_p = entry_p + tp * a;  sl_p = entry_p - sl * a
            is_long = True;  bars_since = 0;  bars_held = 0
        elif bidir and cross_dn and c < ema_t[i] and rsi_v[i] < rsi_max:
            pos = -(notional / (c + 1e-9));  entry_p = c * (1 - slip)
            tp_p = entry_p - tp * a;  sl_p = entry_p + sl * a
            is_short = True;  bars_since = 0;  bars_held = 0

    eq_arr = np.array(equity, dtype=float)
    stats = _trades_stats(pnls)
    stats["sharpe"] = _sharpe_from_equity(eq_arr)
    stats["max_drawdown"] = float((eq_arr.cummax() - eq_arr).max() / (eq_arr.cummax().max() + 1e-9))
    return stats


# ── Fast bitcoin_range ────────────────────────────────────────────────────────

def fast_bitcoin_range(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    n = len(close)

    rsi_os  = p.get("rsi_oversold", 42);   rsi_ob  = p.get("rsi_overbought", 58)
    prox    = p.get("zone_proximity_pct", 1.5) / 100.0
    lkb     = p.get("support_lookback_bars", 96)
    sl_buf  = p.get("stop_atr_buffer", 1.0)
    trend_e = p.get("trend_ema_bars", 200)
    max_h   = p.get("max_hold_bars", 32)
    atr_rg  = p.get("atr_regime_pct", 0.0)

    rsi_v  = _np_rsi(close)
    atr_v  = _np_atr(high, low, close)
    ema_t  = _np_ema(close, trend_e) if trend_e > 0 else np.full(n, 0.0)

    fee = FEE_BPS / 10000;  slip = SLIP_BPS / 10000
    notional = p.get("target_notional_usd", 500.0)
    warmup = max(lkb + 20, 220)

    cash = STARTING_CASH
    pos = 0.0;  entry_p = 0.0;  tp_p = 0.0;  sl_p = 0.0
    bars_held = 0;  is_long = False;  is_short = False
    pnls: list[float] = []
    equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i];  a = atr_v[i]

        if pos != 0.0:
            bars_held += 1
            exit_now = False
            if is_long:
                exit_now = c >= tp_p or c <= sl_p or bars_held >= max_h
            elif is_short:
                exit_now = c <= tp_p or c >= sl_p or bars_held >= max_h

            if exit_now:
                if is_long:
                    fill = c * (1 - slip);  pnl = (fill - entry_p) * pos - (entry_p + fill) * pos * fee
                else:
                    fill = c * (1 + slip);  pnl = (entry_p - fill) * abs(pos) - (entry_p + fill) * abs(pos) * fee
                pnls.append(pnl);  cash += pnl
                pos = 0.0;  is_long = False;  is_short = False;  bars_held = 0

        mtm = cash if pos == 0.0 else (
            cash + (c - entry_p) * pos if is_long else cash + (entry_p - c) * abs(pos)
        )
        equity.append(mtm)

        if pos != 0.0:
            continue
        if atr_rg > 0 and a / (c + 1e-9) > atr_rg:
            continue

        sup = low[max(0, i - lkb):i].min()
        res = high[max(0, i - lkb):i].max()
        rsi = rsi_v[i];  rsi_prev = rsi_v[i - 1]

        near_sup = c <= sup * (1 + prox)
        near_res = c >= res * (1 - prox)

        # Long: near support, RSI oversold and turning up
        if near_sup and rsi < rsi_os and rsi > rsi_prev:
            if trend_e == 0 or c > ema_t[i]:
                entry_p = c * (1 + slip);  pos = notional / (entry_p + 1e-9)
                sl_p = entry_p - sl_buf * a;  tp_p = res
                is_long = True;  bars_held = 0

        # Short: near resistance, RSI overbought and turning down
        elif near_res and rsi > rsi_ob and rsi < rsi_prev:
            if trend_e == 0 or c < ema_t[i]:
                entry_p = c * (1 - slip);  pos = -(notional / (entry_p + 1e-9))
                sl_p = entry_p + sl_buf * a;  tp_p = sup
                is_short = True;  bars_held = 0

    eq_arr = np.array(equity, dtype=float)
    stats = _trades_stats(pnls)
    stats["sharpe"] = _sharpe_from_equity(eq_arr)
    stats["max_drawdown"] = float((eq_arr.cummax() - eq_arr).max() / (eq_arr.cummax().max() + 1e-9))
    return stats


# ── Fast range_trader ─────────────────────────────────────────────────────────

def fast_range_trader(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    n = len(close)

    prox   = p.get("zone_proximity_pct", 2.0) / 100.0
    lkb    = p.get("support_lookback_bars", 48)
    sl_buf = p.get("stop_atr_buffer", 1.5)
    max_h  = p.get("max_hold_bars", 32)
    rsi_lt = p.get("rsi_long_threshold", 50)
    rsi_st = p.get("rsi_short_threshold", 50)
    trend_e = p.get("trend_ema_bars", 0)
    atr_rg  = p.get("atr_regime_pct", 0.0)

    rsi_v = _np_rsi(close)
    atr_v = _np_atr(high, low, close)
    ema_t = _np_ema(close, trend_e) if trend_e > 0 else np.full(n, 0.0)

    fee = FEE_BPS / 10000;  slip = SLIP_BPS / 10000
    notional = p.get("target_notional_usd", 500.0)
    warmup = max(lkb + 20, 220)

    cash = STARTING_CASH
    pos = 0.0;  entry_p = 0.0;  tp_p = 0.0;  sl_p = 0.0
    bars_held = 0;  is_long = False;  is_short = False
    pnls: list[float] = []
    equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i];  a = atr_v[i]

        if pos != 0.0:
            bars_held += 1
            exit_now = False
            if is_long:
                exit_now = c >= tp_p or c <= sl_p or bars_held >= max_h
            elif is_short:
                exit_now = c <= tp_p or c >= sl_p or bars_held >= max_h

            if exit_now:
                if is_long:
                    fill = c * (1 - slip);  pnl = (fill - entry_p) * pos - (entry_p + fill) * pos * fee
                else:
                    fill = c * (1 + slip);  pnl = (entry_p - fill) * abs(pos) - (entry_p + fill) * abs(pos) * fee
                pnls.append(pnl);  cash += pnl
                pos = 0.0;  is_long = False;  is_short = False;  bars_held = 0

        mtm = cash if pos == 0.0 else (
            cash + (c - entry_p) * pos if is_long else cash + (entry_p - c) * abs(pos)
        )
        equity.append(mtm)

        if pos != 0.0:
            continue
        if atr_rg > 0 and a / (c + 1e-9) > atr_rg:
            continue

        sup = low[max(0, i - lkb):i].min()
        res = high[max(0, i - lkb):i].max()
        rsi = rsi_v[i];  rsi_prev = rsi_v[i - 1]

        near_sup = c <= sup * (1 + prox)
        near_res = c >= res * (1 - prox)

        if near_sup and rsi < rsi_lt and rsi > rsi_prev:
            if trend_e == 0 or c > ema_t[i]:
                entry_p = c * (1 + slip);  pos = notional / (entry_p + 1e-9)
                sl_p = entry_p - sl_buf * a;  tp_p = res
                is_long = True;  bars_held = 0

        elif near_res and rsi > rsi_st and rsi < rsi_prev:
            if trend_e == 0 or c < ema_t[i]:
                entry_p = c * (1 - slip);  pos = -(notional / (entry_p + 1e-9))
                sl_p = entry_p + sl_buf * a;  tp_p = sup
                is_short = True;  bars_held = 0

    eq_arr = np.array(equity, dtype=float)
    stats = _trades_stats(pnls)
    stats["sharpe"] = _sharpe_from_equity(eq_arr)
    stats["max_drawdown"] = float((eq_arr.cummax() - eq_arr).max() / (eq_arr.cummax().max() + 1e-9))
    return stats


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(name: str, fn, base_params: dict, grid: dict[str, list], df: pd.DataFrame, top_n: int = 5):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  {name}: {len(combos)} combos …", flush=True)
    results = []
    t0 = time.time()
    for i, values in enumerate(combos):
        p = {**base_params, **dict(zip(keys, values))}
        try:
            m = fn(df, p)
            results.append((m["sharpe"], p, m))
        except Exception as e:
            pass
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(combos)}  elapsed {time.time()-t0:.0f}s", flush=True)

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"  Done in {time.time()-t0:.0f}s")

    sep("─", 70)
    print(f"  TOP {top_n} for {name}")
    sep("─", 70)
    for rank, (sh, p, m) in enumerate(results[:top_n], 1):
        wr = m.get("win_rate", 0) * 100
        pf = m.get("profit_factor", 0)
        nt = int(m.get("num_trades", 0))
        ret = m.get("total_return", 0) * 100
        dd = m.get("max_drawdown", 0) * 100
        changed = {k: v for k, v in p.items() if k in grid}
        print(f"  #{rank:2d}  Sharpe {sh:+.3f}  ret {ret:+.1f}%  DD {dd:.1f}%  WR {wr:.0f}%  PF {pf:.2f}  n={nt}")
        print(f"       params: {changed}")
    sep("─", 70)
    return results[:top_n]


# ── ML strategy baseline ──────────────────────────────────────────────────────

def ml_baseline(name: str, klass, params: dict, df: pd.DataFrame) -> dict:
    print(f"  Loading {name} …", end="", flush=True)
    try:
        strat = klass(params=params)
        print(" running WF …", end="", flush=True)
        m = _wf(strat, df)
        print(f" {_fmt(m)}")
        return m
    except Exception as e:
        print(f" FAILED: {e}")
        return {}


# ── Walk-forward on top combos ────────────────────────────────────────────────

def wf_top_combos(name: str, klass, top_combos: list, df: pd.DataFrame):
    sep()
    print(f"WALK-FORWARD: {name} (top {len(top_combos)} param combos)")
    sep()
    best_sh = -999.0
    best_params = None

    for rank, (_, p, _) in enumerate(top_combos[:3], 1):
        print(f"  Combo #{rank}: {p}", flush=True)
        try:
            strat = klass(params=p)
            m = _wf(strat, df)
            windows = m.get("num_windows", 0)
            sh = m.get("avg_sharpe", 0)
            pct_pos = m.get("pct_windows_positive_sharpe", 0) * 100
            print(f"    WF: {_fmt(m)}  |  {windows:.0f} windows  |  {pct_pos:.0f}% positive")
            if sh > best_sh:
                best_sh = sh
                best_params = p
        except Exception as e:
            print(f"    ERROR: {e}")

    return best_params, best_sh


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    sep()
    print("FULL BACKTEST  —  12yr BTC/USD 15m  —  glitz-quant")
    sep()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/4] FETCHING DATA")
    sep("─", 70)
    df = await fetch_kraken()
    print(f"  Total candles: {len(df):,}  |  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Years covered: {(df.index[-1] - df.index[0]).days / 365.25:.1f}")

    # ── 2. Grid search (fast vectorised) ─────────────────────────────────────
    print("\n[2/4] PARAMETER GRID SEARCH (vectorised single-pass)")
    sep("─", 70)

    # btc_trend grid
    btc_trend_base = {
        "target_notional_usd": 500, "bidirectional": True,
        "cooldown_bars": 16, "atr_regime_min": 0.002,
        "max_hold_bars": 96, "rsi_short_max": 50.0,
    }
    btc_trend_grid = {
        "ema_fast":      [8, 10, 13, 21],
        "ema_slow":      [21, 30, 34, 55],
        "ema_trend":     [55, 100, 200],
        "tp_atr_mult":   [2.0, 2.5, 3.0, 4.0],
        "sl_atr_mult":   [0.75, 1.0, 1.5],
        "rsi_long_min":  [45.0, 50.0, 55.0],
    }
    # Filter out combos where ema_fast >= ema_slow or ema_slow >= ema_trend
    ema_combos = [(f, s, t) for f in btc_trend_grid["ema_fast"]
                  for s in btc_trend_grid["ema_slow"]
                  for t in btc_trend_grid["ema_trend"]
                  if f < s < t]
    btc_trend_grid_filtered = {
        "_ema": ema_combos,
        "tp_atr_mult": btc_trend_grid["tp_atr_mult"],
        "sl_atr_mult": btc_trend_grid["sl_atr_mult"],
        "rsi_long_min": btc_trend_grid["rsi_long_min"],
    }

    # Flatten the EMA combo into the param dict manually
    def btc_trend_search(df_: pd.DataFrame, base: dict, ema_c: list, tp_sl: list, rsi_vals: list):
        results = []
        total = len(ema_c) * len(tp_sl) * len(rsi_vals)
        print(f"\n  btc_trend: {total} combos …", flush=True)
        t0 = time.time()
        done = 0
        for (ef, es, et) in ema_c:
            for (tp, sl) in tp_sl:
                for rsi_min in rsi_vals:
                    p = {**base, "ema_fast": ef, "ema_slow": es, "ema_trend": et,
                         "tp_atr_mult": tp, "sl_atr_mult": sl, "rsi_long_min": rsi_min,
                         "rsi_short_max": 100.0 - rsi_min}
                    try:
                        m = fast_btc_trend(df_, p)
                        results.append((m["sharpe"], p, m))
                    except Exception:
                        pass
                    done += 1
                    if done % 50 == 0:
                        print(f"    {done}/{total}  elapsed {time.time()-t0:.0f}s", flush=True)
        results.sort(key=lambda x: x[0], reverse=True)
        print(f"  Done in {time.time()-t0:.0f}s")
        return results[:5]

    tp_sl_combos = [(tp, sl) for tp in [2.0, 2.5, 3.0, 4.0] for sl in [0.75, 1.0, 1.5]]
    btc_top = btc_trend_search(df, btc_trend_base, ema_combos,
                                tp_sl_combos, [45.0, 50.0, 55.0])

    sep("─", 70);  print("  TOP 5 btc_trend combos (single-pass 12yr)");  sep("─", 70)
    for rank, (sh, p, m) in enumerate(btc_top, 1):
        wr = m.get("win_rate", 0) * 100;  pf = m.get("profit_factor", 0)
        nt = int(m.get("num_trades", 0));  ret = m.get("total_return", 0) * 100
        dd = m.get("max_drawdown", 0) * 100
        print(f"  #{rank}  Sharpe {sh:+.3f}  ret {ret:+.1f}%  DD {dd:.1f}%  WR {wr:.0f}%  PF {pf:.2f}  n={nt}")
        print(f"      ema {p['ema_fast']}/{p['ema_slow']}/{p['ema_trend']}  tp {p['tp_atr_mult']}×  sl {p['sl_atr_mult']}×  rsi≥{p['rsi_long_min']}")

    # bitcoin_range grid
    br_base = {
        "target_notional_usd": 500, "rsi_must_turn_up": True,
        "support_lookback_bars": 96, "resistance_lookback_bars": 96,
        "max_hold_bars": 32, "volume_z_threshold": 0.0, "atr_period": 14,
    }
    br_top = grid_search("bitcoin_range", fast_bitcoin_range, br_base, {
        "rsi_oversold":        [30, 35, 38, 42, 45],
        "rsi_overbought":      [55, 58, 62, 65, 70],
        "zone_proximity_pct":  [0.5, 1.0, 1.5, 2.0, 3.0],
        "stop_atr_buffer":     [0.75, 1.0, 1.5, 2.0],
        "trend_ema_bars":      [0, 96, 200],
        "atr_regime_pct":      [0.0, 0.012, 0.02],
    }, df, top_n=5)

    # range_trader grid
    rt_base = {
        "target_notional_usd": 500, "support_lookback_bars": 48,
        "resistance_lookback_bars": 48, "max_hold_bars": 32,
    }
    rt_top = grid_search("range_trader", fast_range_trader, rt_base, {
        "zone_proximity_pct":  [1.0, 1.5, 2.0, 2.5, 3.0],
        "stop_atr_buffer":     [1.0, 1.5, 2.0, 2.5],
        "rsi_long_threshold":  [40, 45, 50, 55],
        "rsi_short_threshold": [45, 50, 55, 60],
        "trend_ema_bars":      [0, 96, 200],
        "atr_regime_pct":      [0.0, 0.012, 0.02],
    }, df, top_n=5)

    # ── 3. Walk-forward on best params ────────────────────────────────────────
    print("\n[3/4] WALK-FORWARD ON BEST PARAMS  (train 4mo / test 1mo)")
    sep("─", 70)

    btc_params_for_wf = [
        ({**btc_trend_base, "ema_fast": p["ema_fast"], "ema_slow": p["ema_slow"],
          "ema_trend": p["ema_trend"], "tp_atr_mult": p["tp_atr_mult"],
          "sl_atr_mult": p["sl_atr_mult"], "rsi_long_min": p["rsi_long_min"],
          "rsi_short_max": 100.0 - p["rsi_long_min"]}, sh, m)
        for sh, p, m in btc_top[:3]
    ]

    best_btc_params, best_btc_sh  = wf_top_combos("btc_trend",     BtcTrendFollow,       [(_, p, m) for p, _, m in btc_params_for_wf], df)
    best_br_params,  best_br_sh   = wf_top_combos("bitcoin_range",  BitcoinRangeMomentum, br_top, df)
    best_rt_params,  best_rt_sh   = wf_top_combos("range_trader",   RangeTrader,          rt_top, df)

    # ── 4. ML strategy baselines ──────────────────────────────────────────────
    print("\n[4/4] ML STRATEGY WALK-FORWARD BASELINES")
    sep("─", 70)

    ml_momentum_params = {
        "symbol": "BTC-USD", "target_notional_usd": 500,
        "min_confidence": 0.33, "atr_regime_pct": 0.005,
        "tp_atr_mult": 2.0, "sl_atr_mult": 1.0,
        "max_hold_bars": 32, "bidirectional": True,
        "model_path": "data/models/ml_momentum",
    }
    ml_meme_params = {
        "symbol": "BTC-USD", "target_notional_usd": 300,
        "min_confidence": 0.45, "atr_regime_pct": 0.008,
        "min_vol_z": 2.0, "tp_atr_mult": 1.5, "sl_atr_mult": 0.75,
        "max_hold_bars": 16, "bidirectional": True,
        "model_path": "data/models/ml_meme_coin",
    }
    ml_carry_params = {
        "symbol": "BTC-USD", "target_notional_usd": 500,
        "min_confidence": 0.45, "atr_regime_pct": 0.006,
        "min_annual_funding": 0.05, "max_annual_funding": 2.0,
        "tp_atr_mult": 1.5, "sl_atr_mult": 0.75,
        "max_hold_bars": 32, "bidirectional": True,
        "model_path": "data/models/ml_funding_carry",
    }

    mm_m  = ml_baseline("ml_momentum",     MLMomentumStrategy,    ml_momentum_params, df)
    mc_m  = ml_baseline("ml_meme_coin",    MLMemeCoinStrategy,    ml_meme_params,     df)
    mfc_m = ml_baseline("ml_funding_carry",MLFundingCarryStrategy, ml_carry_params,   df)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep()
    print("RESULTS SUMMARY  —  Walk-Forward Avg Sharpe")
    sep()

    rows = [
        ("btc_trend (best)",     best_btc_sh),
        ("bitcoin_range (best)", best_br_sh),
        ("range_trader (best)",  best_rt_sh),
        ("ml_momentum",          mm_m.get("avg_sharpe", 0)),
        ("ml_meme_coin",         mc_m.get("avg_sharpe", 0)),
        ("ml_funding_carry",     mfc_m.get("avg_sharpe", 0)),
    ]
    rows.sort(key=lambda x: x[1], reverse=True)

    for name, sh in rows:
        flag = "✓ ENABLE" if sh > 0.5 else ("~ WATCH" if sh > 0.1 else "✗ SKIP")
        bar = "█" * max(0, int((sh + 1) * 10))
        print(f"  {flag:<12}  {name:<26}  {sh:+.3f}  {bar}")

    sep()
    print("\nBEST PARAMS (copy into strategies.yaml):\n")

    if best_btc_params:
        print("  btc_trend:")
        for k, v in best_btc_params.items():
            if k not in ("symbol", "target_notional_usd", "bidirectional"):
                print(f"    {k}: {v}")

    if best_br_params:
        print("\n  bitcoin_range:")
        for k, v in best_br_params.items():
            if k not in ("symbol", "target_notional_usd"):
                print(f"    {k}: {v}")

    if best_rt_params:
        print("\n  range_trader:")
        for k, v in best_rt_params.items():
            if k not in ("symbol", "target_notional_usd"):
                print(f"    {k}: {v}")

    sep()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
