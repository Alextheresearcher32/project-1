"""
Full multi-strategy backtest — 10yr BTC/USD 15m (Coinbase 2016-present).

Flow:
  1. Fetch data (cached to /tmp/btc_15m_12yr.parquet after first run)
  2. Fast parameter grid search on a 4yr representative sample
     (precomputes all indicator arrays once per sample → ~10-20s/strategy)
  3. Walk-forward on full 10yr history for the top-5 combos per strategy
  4. ML strategy walk-forward baselines
  5. Ranked results + best params

Skipped: funding_rate_carry (needs live funding_rate_8h), llm_macro_overlay (LLM cost)

Usage:
  cd /Users/larissasylvester/src
  PYTHONPATH=/Users/larissasylvester/src python run_full_backtest.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path

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

STARTING_CASH = 10_000.0
FEE_BPS       = 15.0
SLIP_BPS      = 5.0
WF_TRAIN      = timedelta(days=120)
WF_TEST       = timedelta(days=30)
DATA_CACHE    = Path("/tmp/btc_15m_12yr.parquet")

# Grid search uses a 4yr sample covering bear + bull + chop regimes
GRID_START = "2019-01-01"
GRID_END   = "2022-12-31"


# ── Formatting ────────────────────────────────────────────────────────────────
def sep(c="═", w=72): print(c * w)

def _fmt(m: dict, label: str = "") -> str:
    sh  = m.get("avg_sharpe", m.get("sharpe", 0))
    ret = m.get("avg_return",  m.get("total_return", 0)) * 100
    dd  = m.get("avg_max_drawdown", m.get("max_drawdown", 0)) * 100
    wr  = m.get("avg_win_rate",  m.get("win_rate", 0)) * 100
    pf  = m.get("avg_profit_factor", m.get("profit_factor", 0))
    nt  = int(m.get("num_trades", m.get("avg_win_rate", 0) and 0))
    flag = "✓" if sh > 0.5 else ("~" if sh > 0.1 else "✗")
    s = f"{flag} Sharpe {sh:+.3f}  ret {ret:+.1f}%  DD {dd:.1f}%  WR {wr:.0f}%  PF {pf:.2f}"
    return f"  {label:<22} {s}" if label else f"  {s}"


# ── Walk-forward helper ───────────────────────────────────────────────────────
def _wf(strategy, df: pd.DataFrame) -> dict:
    bt  = Backtester(STARTING_CASH, FEE_BPS, SLIP_BPS)
    wf  = WalkForwardAnalyzer(bt, WF_TRAIN, WF_TEST)
    return wf.run(strategy, df).aggregate_metrics


# ── Indicator primitives (numpy, computed once per dataset) ───────────────────
def _ema_np(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _wilder(x: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(x), dtype=float)
    if p >= len(x):
        return out
    out[p - 1] = x[:p].mean()
    k = 1.0 / p
    for i in range(p, len(x)):
        out[i] = out[i - 1] * (1 - k) + x[i] * k
    return out


def _atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, p: int = 14) -> np.ndarray:
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _wilder(tr, p)


def _rsi_np(close: np.ndarray, p: int = 14) -> np.ndarray:
    d  = np.diff(close, prepend=close[0])
    ag = _wilder(np.where(d > 0, d, 0.0), p)
    al = _wilder(np.where(d < 0, -d, 0.0), p)
    rs = np.where(al > 1e-12, ag / al, 100.0)
    return 100 - 100 / (1 + rs)


# ── Shared trade accounting ───────────────────────────────────────────────────
def _calc_metrics(pnls: list[float], equity: list[float]) -> dict:
    eq = np.array(equity, dtype=float)
    pnl_arr = np.array(pnls, dtype=float) if pnls else np.zeros(1)
    rets = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(365 * 24 * 4))
    wins = pnl_arr[pnl_arr > 0]; losses = pnl_arr[pnl_arr < 0]
    return {
        "sharpe": sharpe,
        "total_return": (eq[-1] - STARTING_CASH) / STARTING_CASH,
        "max_drawdown": float((eq.cummax() - eq).max() / (eq.cummax().max() + 1e-9)),
        "win_rate": len(wins) / max(len(pnl_arr), 1),
        "profit_factor": wins.sum() / (abs(losses.sum()) + 1e-9),
        "num_trades": len(pnls),
    }


# ── Precomputed indicator cache for a given DataFrame ────────────────────────
class IndicatorCache:
    """Pre-computes all indicators needed for grid search on one dataset."""
    EMA_PERIODS = [8, 10, 13, 21, 30, 34, 50, 55, 96, 100, 200]

    def __init__(self, df: pd.DataFrame):
        self.close = df["close"].values.astype(float)
        self.high  = df["high"].values.astype(float)
        self.low   = df["low"].values.astype(float)
        self.n     = len(self.close)
        print("  precomputing indicators …", end="", flush=True)
        t0 = time.time()
        self.ema   = {p: _ema_np(self.close, p) for p in self.EMA_PERIODS}
        self.atr14 = _atr_np(self.high, self.low, self.close, 14)
        self.rsi14 = _rsi_np(self.close, 14)
        print(f" done ({time.time()-t0:.1f}s)")

    def get_ema(self, p: int) -> np.ndarray:
        if p not in self.ema:
            self.ema[p] = _ema_np(self.close, p)
        return self.ema[p]


# ── Fast btc_trend (uses precomputed indicators) ──────────────────────────────
def sim_btc_trend(ic: IndicatorCache, p: dict) -> dict:
    ef = ic.get_ema(p["ema_fast"]); es = ic.get_ema(p["ema_slow"]); et = ic.get_ema(p["ema_trend"])
    atr = ic.atr14; rsi = ic.rsi14; close = ic.close; n = ic.n

    tp = p["tp_atr_mult"]; sl = p["sl_atr_mult"]
    rsi_min = p["rsi_long_min"]; rsi_max = p.get("rsi_short_max", 100 - rsi_min)
    cool = p.get("cooldown_bars", 16); max_h = p.get("max_hold_bars", 96)
    atr_min = p.get("atr_regime_min", 0.002)
    fee = FEE_BPS / 10000; slip = SLIP_BPS / 10000; notional = p.get("target_notional_usd", 500.0)
    warmup = max(p["ema_trend"] + 10, 220)

    cash = STARTING_CASH; pos = 0.0; ep = 0.0; tp_p = 0.0; sl_p = 0.0
    bars_held = 0; bars_since = 999; is_long = False; is_short = False
    pnls: list = []; equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i]; a = atr[i]; bars_since += 1
        if pos != 0.0:
            bars_held += 1
            done = False
            if is_long:
                done = c >= tp_p or c <= sl_p or (ef[i] < es[i] and ef[i-1] >= es[i-1]) or bars_held >= max_h
            else:
                done = c <= tp_p or c >= sl_p or (ef[i] > es[i] and ef[i-1] <= es[i-1]) or bars_held >= max_h
            if done:
                fill = c*(1-slip) if is_long else c*(1+slip)
                pnl = ((fill-ep)*pos if is_long else (ep-fill)*abs(pos)) - (ep+fill)*abs(pos)*fee
                pnls.append(pnl); cash += pnl; pos = 0.0; is_long = is_short = False; bars_held = 0
        mtm = cash + ((c-ep)*pos if is_long else ((ep-c)*abs(pos) if is_short else 0))
        equity.append(mtm)
        if pos != 0.0 or bars_since < cool or a/(c+1e-9) < atr_min:
            continue
        cup = ef[i] > es[i] and ef[i-1] <= es[i-1]; cdn = ef[i] < es[i] and ef[i-1] >= es[i-1]
        if cup and c > et[i] and rsi[i] > rsi_min:
            ep = c*(1+slip); pos = notional/(ep+1e-9); tp_p = ep+tp*a; sl_p = ep-sl*a
            is_long = True; bars_since = 0; bars_held = 0
        elif cdn and c < et[i] and rsi[i] < rsi_max:
            ep = c*(1-slip); pos = -(notional/(ep+1e-9)); tp_p = ep-tp*a; sl_p = ep+sl*a
            is_short = True; bars_since = 0; bars_held = 0
    return _calc_metrics(pnls, equity)


# ── Fast bitcoin_range (precomputed) ──────────────────────────────────────────
def sim_bitcoin_range(ic: IndicatorCache, p: dict) -> dict:
    close = ic.close; high = ic.high; low = ic.low; atr = ic.atr14; rsi = ic.rsi14; n = ic.n
    et = ic.get_ema(p["trend_ema_bars"]) if p.get("trend_ema_bars", 0) > 0 else None

    rsi_os = p["rsi_oversold"]; rsi_ob = p["rsi_overbought"]
    prox = p["zone_proximity_pct"] / 100.0; lkb = p.get("support_lookback_bars", 96)
    sl_buf = p["stop_atr_buffer"]; max_h = p.get("max_hold_bars", 32)
    atr_rg = p.get("atr_regime_pct", 0.0)
    fee = FEE_BPS / 10000; slip = SLIP_BPS / 10000; notional = p.get("target_notional_usd", 500.0)
    warmup = max(lkb + 20, 220)

    cash = STARTING_CASH; pos = 0.0; ep = 0.0; tp_p = 0.0; sl_p = 0.0
    bars_held = 0; is_long = False; is_short = False
    pnls: list = []; equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i]; a = atr[i]
        if pos != 0.0:
            bars_held += 1
            done = (c >= tp_p or c <= sl_p or bars_held >= max_h) if is_long else (c <= tp_p or c >= sl_p or bars_held >= max_h)
            if done:
                fill = c*(1-slip) if is_long else c*(1+slip)
                pnl = ((fill-ep)*pos if is_long else (ep-fill)*abs(pos)) - (ep+fill)*abs(pos)*fee
                pnls.append(pnl); cash += pnl; pos = 0.0; is_long = is_short = False; bars_held = 0
        mtm = cash + ((c-ep)*pos if is_long else ((ep-c)*abs(pos) if is_short else 0))
        equity.append(mtm)
        if pos != 0.0: continue
        if atr_rg > 0 and a/(c+1e-9) > atr_rg: continue
        sup = low[max(0,i-lkb):i].min(); res = high[max(0,i-lkb):i].max()
        r = rsi[i]; rp = rsi[i-1]
        if c <= sup*(1+prox) and r < rsi_os and r > rp and (et is None or c > et[i]):
            ep = c*(1+slip); pos = notional/(ep+1e-9); sl_p = ep-sl_buf*a; tp_p = res
            is_long = True; bars_held = 0
        elif c >= res*(1-prox) and r > rsi_ob and r < rp and (et is None or c < et[i]):
            ep = c*(1-slip); pos = -(notional/(ep+1e-9)); sl_p = ep+sl_buf*a; tp_p = sup
            is_short = True; bars_held = 0
    return _calc_metrics(pnls, equity)


# ── Fast range_trader (precomputed) ───────────────────────────────────────────
def sim_range_trader(ic: IndicatorCache, p: dict) -> dict:
    close = ic.close; high = ic.high; low = ic.low; atr = ic.atr14; rsi = ic.rsi14; n = ic.n
    et = ic.get_ema(p["trend_ema_bars"]) if p.get("trend_ema_bars", 0) > 0 else None

    prox = p["zone_proximity_pct"] / 100.0; lkb = p.get("support_lookback_bars", 48)
    sl_buf = p["stop_atr_buffer"]; max_h = p.get("max_hold_bars", 32)
    rsi_lt = p.get("rsi_long_threshold", 50); rsi_st = p.get("rsi_short_threshold", 50)
    atr_rg = p.get("atr_regime_pct", 0.0)
    fee = FEE_BPS / 10000; slip = SLIP_BPS / 10000; notional = p.get("target_notional_usd", 500.0)
    warmup = max(lkb + 20, 220)

    cash = STARTING_CASH; pos = 0.0; ep = 0.0; tp_p = 0.0; sl_p = 0.0
    bars_held = 0; is_long = False; is_short = False
    pnls: list = []; equity = [cash] * warmup

    for i in range(warmup, n):
        c = close[i]; a = atr[i]
        if pos != 0.0:
            bars_held += 1
            done = (c >= tp_p or c <= sl_p or bars_held >= max_h) if is_long else (c <= tp_p or c >= sl_p or bars_held >= max_h)
            if done:
                fill = c*(1-slip) if is_long else c*(1+slip)
                pnl = ((fill-ep)*pos if is_long else (ep-fill)*abs(pos)) - (ep+fill)*abs(pos)*fee
                pnls.append(pnl); cash += pnl; pos = 0.0; is_long = is_short = False; bars_held = 0
        mtm = cash + ((c-ep)*pos if is_long else ((ep-c)*abs(pos) if is_short else 0))
        equity.append(mtm)
        if pos != 0.0: continue
        if atr_rg > 0 and a/(c+1e-9) > atr_rg: continue
        sup = low[max(0,i-lkb):i].min(); res = high[max(0,i-lkb):i].max()
        r = rsi[i]; rp = rsi[i-1]
        if c <= sup*(1+prox) and r < rsi_lt and r > rp and (et is None or c > et[i]):
            ep = c*(1+slip); pos = notional/(ep+1e-9); sl_p = ep-sl_buf*a; tp_p = res
            is_long = True; bars_held = 0
        elif c >= res*(1-prox) and r > rsi_st and r < rp and (et is None or c < et[i]):
            ep = c*(1-slip); pos = -(notional/(ep+1e-9)); sl_p = ep+sl_buf*a; tp_p = sup
            is_short = True; bars_held = 0
    return _calc_metrics(pnls, equity)


# ── Grid search engine ────────────────────────────────────────────────────────
def run_grid(name: str, sim_fn, ic: IndicatorCache, combos: list[dict], top_n: int = 5) -> list:
    print(f"\n  {name}: {len(combos)} combos on {GRID_START}–{GRID_END} …", flush=True)
    results = []
    t0 = time.time()
    for i, p in enumerate(combos):
        try:
            m = sim_fn(ic, p)
            results.append((m["sharpe"], p, m))
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(combos)}  {time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"  Finished in {time.time()-t0:.0f}s")
    sep("─", 72)
    print(f"  TOP {top_n} — {name} (single-pass on 4yr sample)")
    sep("─", 72)
    for rank, (sh, p, m) in enumerate(results[:top_n], 1):
        wr = m["win_rate"]*100; pf = m["profit_factor"]; nt = m["num_trades"]
        ret = m["total_return"]*100; dd = m["max_drawdown"]*100
        flag = "✓" if sh > 0.5 else ("~" if sh > 0 else "✗")
        print(f"  {flag} #{rank}  Sharpe {sh:+.3f}  ret {ret:+.1f}%  DD {dd:.1f}%  WR {wr:.0f}%  PF {pf:.2f}  n={nt}")
        # show only the params that are part of the grid (changed from base)
        grid_keys = set(combos[0].keys())
        print(f"       {' | '.join(f'{k}={p[k]}' for k in sorted(grid_keys))}")
    sep("─", 72)
    return results[:top_n]


# ── Walk-forward validation on full dataset ───────────────────────────────────
def wf_validate(name: str, klass, top_combos: list, df_full: pd.DataFrame):
    sep()
    print(f"WALK-FORWARD — {name}  (top {min(3,len(top_combos))} combos × full 10yr)")
    sep()
    best_sh = -999.0; best_params = None; best_metrics = {}

    for rank, (_, p, _) in enumerate(top_combos[:3], 1):
        try:
            strat = klass(params=p)
            m = _wf(strat, df_full)
            sh = m.get("avg_sharpe", 0); wins = m.get("pct_windows_positive_sharpe", 0)*100
            nw = m.get("num_windows", 0)
            print(f"  #{rank}  {_fmt(m)}  | {nw:.0f} windows  {wins:.0f}% pos")
            if sh > best_sh:
                best_sh = sh; best_params = p; best_metrics = m
        except Exception as e:
            print(f"  #{rank}  ERROR: {e}")
    return best_params, best_sh, best_metrics


# ── Data fetch ────────────────────────────────────────────────────────────────
async def fetch_data() -> pd.DataFrame:
    if DATA_CACHE.exists():
        df = pd.read_parquet(DATA_CACHE)
        print(f"  [cache] {len(df):,} candles  {df.index[0].date()} → {df.index[-1].date()}")
        return df

    s = get_settings()
    exchange = ccxt.coinbaseadvanced({
        "apiKey": s.coinbase_api_key.get_secret_value(),
        "secret": s.coinbase_api_secret.get_secret_value(),
        "enableRateLimit": True,
    })

    start_iso = "2016-01-01T00:00:00Z"
    since = exchange.parse8601(start_iso); start_ts = since
    all_candles: list = []; batch = 0

    print(f"  Fetching Coinbase BTC/USD 15m from {start_iso[:10]} …", flush=True)
    while batch < 3000:
        try:
            chunk = await exchange.fetch_ohlcv("BTC/USD", timeframe="15m", since=since, limit=300)
        except Exception as e:
            print(f"  [warn] {e}. Retrying …"); await asyncio.sleep(3); continue
        if not chunk: break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= exchange.milliseconds() - 15*60*1000: break
        since = last_ts + 15*60*1000; batch += 1
        if batch % 100 == 0:
            pct = (last_ts - start_ts) / (exchange.milliseconds() - start_ts) * 100
            print(f"  … batch {batch:>4}  {pd.Timestamp(last_ts,unit='ms',tz='UTC').date()}  ({pct:.0f}%)  {len(all_candles):,} candles", flush=True)
        await asyncio.sleep(0.4)
    await exchange.close()

    df = pd.DataFrame(all_candles, columns=["ts","open","high","low","close","volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df.to_parquet(DATA_CACHE)
    print(f"  Saved → {DATA_CACHE}  ({len(df):,} candles)")
    return df


# ── Build parameter combos ────────────────────────────────────────────────────
def _btc_trend_combos() -> list[dict]:
    base = {"target_notional_usd": 500, "bidirectional": True,
            "cooldown_bars": 16, "atr_regime_min": 0.002, "max_hold_bars": 96}
    combos = []
    for ef, es, et in [(8,21,55),(8,21,100),(8,21,200),(10,30,100),(10,30,200),
                       (13,34,100),(13,34,200),(21,55,200),(8,34,100),(8,34,200)]:
        for tp in [2.0, 2.5, 3.0, 4.0]:
            for sl in [0.75, 1.0, 1.25, 1.5]:
                for rsi in [45.0, 50.0, 55.0]:
                    combos.append({**base, "ema_fast": ef, "ema_slow": es, "ema_trend": et,
                                   "tp_atr_mult": tp, "sl_atr_mult": sl,
                                   "rsi_long_min": rsi, "rsi_short_max": 100-rsi})
    return combos


def _bitcoin_range_combos() -> list[dict]:
    base = {"target_notional_usd": 500, "rsi_must_turn_up": True,
            "support_lookback_bars": 96, "resistance_lookback_bars": 96,
            "max_hold_bars": 32, "volume_z_threshold": 0.0}
    combos = []
    for rsi_os, rsi_ob in [(30,70),(35,65),(38,62),(42,58),(45,55)]:
        for prox in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for sl_buf in [0.75, 1.0, 1.5, 2.0]:
                for trend_e in [0, 200]:
                    for atr_rg in [0.0, 0.015]:
                        combos.append({**base, "rsi_oversold": rsi_os, "rsi_overbought": rsi_ob,
                                       "zone_proximity_pct": prox, "stop_atr_buffer": sl_buf,
                                       "trend_ema_bars": trend_e, "atr_regime_pct": atr_rg})
    return combos


def _range_trader_combos() -> list[dict]:
    base = {"target_notional_usd": 500, "support_lookback_bars": 48,
            "resistance_lookback_bars": 48, "max_hold_bars": 32}
    combos = []
    for prox in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for sl_buf in [1.0, 1.5, 2.0, 2.5]:
            for rsi_lt, rsi_st in [(40,60),(45,55),(50,50),(55,45)]:
                for trend_e in [0, 200]:
                    for atr_rg in [0.0, 0.015]:
                        combos.append({**base, "zone_proximity_pct": prox, "stop_atr_buffer": sl_buf,
                                       "rsi_long_threshold": rsi_lt, "rsi_short_threshold": rsi_st,
                                       "trend_ema_bars": trend_e, "atr_regime_pct": atr_rg})
    return combos


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    sep()
    print("FULL BACKTEST  —  BTC/USD 15m  —  glitz-quant")
    sep()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/4] DATA")
    sep("─", 72)
    df = await fetch_data()
    print(f"  {len(df):,} candles  {df.index[0].date()} → {df.index[-1].date()}  ({(df.index[-1]-df.index[0]).days/365.25:.1f} yrs)")

    df_grid = df[GRID_START:GRID_END].copy()
    print(f"  Grid sample: {len(df_grid):,} candles  {df_grid.index[0].date()} → {df_grid.index[-1].date()}")

    # ── 2. Grid search on sample ──────────────────────────────────────────────
    print("\n[2/4] GRID SEARCH (4yr sample, precomputed indicators)")
    sep("─", 72)

    ic_grid = IndicatorCache(df_grid)

    bt_top = run_grid("btc_trend",     sim_btc_trend,     ic_grid, _btc_trend_combos())
    br_top = run_grid("bitcoin_range", sim_bitcoin_range, ic_grid, _bitcoin_range_combos())
    rt_top = run_grid("range_trader",  sim_range_trader,  ic_grid, _range_trader_combos())

    # ── 3. Walk-forward on full dataset ───────────────────────────────────────
    print("\n[3/4] WALK-FORWARD VALIDATION (full 10yr)")
    sep("─", 72)

    best_bt_p, best_bt_sh, best_bt_m = wf_validate("btc_trend",    BtcTrendFollow,      bt_top, df)
    best_br_p, best_br_sh, best_br_m = wf_validate("bitcoin_range", BitcoinRangeMomentum, br_top, df)
    best_rt_p, best_rt_sh, best_rt_m = wf_validate("range_trader",  RangeTrader,          rt_top, df)

    # ── 4. ML baselines ───────────────────────────────────────────────────────
    print("\n[4/4] ML STRATEGY WALK-FORWARD BASELINES")
    sep("─", 72)

    ml_results = {}
    for ml_name, klass, params in [
        ("ml_momentum", MLMomentumStrategy, {
            "symbol": "BTC-USD", "target_notional_usd": 500, "min_confidence": 0.33,
            "atr_regime_pct": 0.005, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0,
            "max_hold_bars": 32, "bidirectional": True, "model_path": "data/models/ml_momentum"}),
        ("ml_meme_coin", MLMemeCoinStrategy, {
            "symbol": "BTC-USD", "target_notional_usd": 300, "min_confidence": 0.45,
            "atr_regime_pct": 0.008, "min_vol_z": 2.0, "tp_atr_mult": 1.5,
            "sl_atr_mult": 0.75, "max_hold_bars": 16, "bidirectional": True,
            "model_path": "data/models/ml_meme_coin"}),
        ("ml_funding_carry", MLFundingCarryStrategy, {
            "symbol": "BTC-USD", "target_notional_usd": 500, "min_confidence": 0.45,
            "atr_regime_pct": 0.006, "min_annual_funding": 0.05, "max_annual_funding": 2.0,
            "tp_atr_mult": 1.5, "sl_atr_mult": 0.75, "max_hold_bars": 32,
            "bidirectional": True, "model_path": f"data/models/{ml_name}".replace(ml_name,'ml_funding_carry')}),
    ]:
        print(f"  {ml_name} …", end="", flush=True)
        try:
            m = _wf(klass(params=params), df)
            ml_results[ml_name] = m
            print(f" {_fmt(m)}")
        except Exception as e:
            print(f" FAILED: {e}")
            ml_results[ml_name] = {}

    # ── Summary ───────────────────────────────────────────────────────────────
    sep()
    print("FINAL RESULTS  —  Walk-Forward Avg Sharpe  (train 4mo / test 1mo)")
    sep()
    rows = [
        ("btc_trend",        best_bt_sh,                     best_bt_p),
        ("bitcoin_range",    best_br_sh,                     best_br_p),
        ("range_trader",     best_rt_sh,                     best_rt_p),
        ("ml_momentum",      ml_results.get("ml_momentum",   {}).get("avg_sharpe", 0), None),
        ("ml_meme_coin",     ml_results.get("ml_meme_coin",  {}).get("avg_sharpe", 0), None),
        ("ml_funding_carry", ml_results.get("ml_funding_carry",{}).get("avg_sharpe",0), None),
    ]
    rows.sort(key=lambda x: x[1], reverse=True)

    print()
    for name, sh, p in rows:
        flag = "✓ ENABLE" if sh > 0.5 else ("~ WATCH" if sh > 0.1 else "✗ SKIP ")
        bar = "█" * max(0, int((sh + 1) * 12))
        print(f"  {flag}  {name:<22}  Sharpe {sh:+.3f}  {bar}")

    sep()
    print("\nBEST PARAMS (for strategies.yaml):\n")
    for label, p in [("btc_trend", best_bt_p), ("bitcoin_range", best_br_p), ("range_trader", best_rt_p)]:
        if not p: continue
        print(f"  {label}:")
        skip = {"target_notional_usd", "bidirectional", "rsi_must_turn_up", "volume_z_threshold",
                "support_lookback_bars", "resistance_lookback_bars"}
        for k, v in sorted(p.items()):
            if k not in skip:
                print(f"    {k}: {v}")
        print()
    sep()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
