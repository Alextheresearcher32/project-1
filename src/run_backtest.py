"""
Quick-run backtest script — single-pass + walk-forward on BitcoinRangeMomentum.
Data: BTC/USD 15m fetched from Coinbase Advanced Trade (auth'd).
"""
from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ccxt.async_support as ccxt
import pandas as pd

from glitz_quant.settings import get_settings
from glitz_quant.strategies.bitcoin_range import BitcoinRangeMomentum
from glitz_quant.strategies.range_trader import RangeTrader
from glitz_quant.research.backtest.engine import Backtester
from glitz_quant.research.backtest.walk_forward import WalkForwardAnalyzer


# ── Strategy config (mirrors config/strategies.yaml) ─────────────────────────
STRATEGY_PARAMS = {
    "support_lookback_bars": 48,   # 12h low — faster support detection
    "resistance_lookback_bars": 48,
    "rsi_period": 14,
    "rsi_oversold": 42,            # v4: long entry — symmetric with short
    "rsi_overbought": 58,          # v4: short entry — mirror of 42 (100-42=58)
    "rsi_must_turn_up": True,
    "zone_proximity_pct": 1.5,     # v3: within 1.5% of support
    "volume_z_threshold": 0.0,     # v3: any average volume (was 0.5)
    "atr_period": 14,
    "stop_atr_buffer": 1.0,
    "max_hold_bars": 32,
    "trend_ema_bars": 200,         # v3: 50h trend filter — blocks buys in downtrends
    "require_bounce_candle": False, # v3: soft bonus now, not hard gate
    "target_notional_usd": 500,
}

# ── Loosened params for diagnostic run ───────────────────────────────────────
LOOSE_PARAMS = {
    "support_lookback_bars": 48,   # 12h low (was 24h)
    "resistance_lookback_bars": 48,
    "rsi_period": 14,
    "rsi_oversold": 50,            # any pullback (was 38)
    "rsi_overbought": 50,          # symmetric loose overbought
    "rsi_must_turn_up": False,     # removed (was True)
    "zone_proximity_pct": 3.0,     # within 3% of support (was 1.0%)
    "volume_z_threshold": -1.0,    # effectively no filter (was 0.5)
    "atr_period": 14,
    "stop_atr_buffer": 1.0,
    "max_hold_bars": 32,
    "trend_ema_bars": 0,           # disabled (was 48)
    "target_notional_usd": 500,
}


async def fetch_data() -> pd.DataFrame:
    s = get_settings()
    exchange = ccxt.coinbaseadvanced({
        "apiKey": s.coinbase_api_key.get_secret_value(),
        "secret": s.coinbase_api_secret.get_secret_value(),
        "enableRateLimit": True,
    })

    print("Fetching BTC/USD 15m candles from Coinbase...")
    all_candles: list[list] = []
    since = exchange.parse8601("2025-12-01T00:00:00Z")
    batch = 0

    while batch < 60:
        chunk = await exchange.fetch_ohlcv("BTC/USD", timeframe="15m", since=since, limit=300)
        if not chunk:
            break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= exchange.milliseconds() - 15 * 60 * 1000:
            break
        since = last_ts + 15 * 60 * 1000
        batch += 1
        await asyncio.sleep(0.35)

    await exchange.close()

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    print(f"  {len(df)} candles  |  {df.index[0].date()} → {df.index[-1].date()}\n")
    return df


def sep(char: str = "─", w: int = 62) -> None:
    print(char * w)


def run_single_pass(df: pd.DataFrame) -> None:
    sep("═")
    print("SINGLE-PASS BACKTEST  —  BitcoinRangeMomentum")
    sep("═")
    print(f"  Period  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Bars    : {len(df):,}  (15-minute candles)")
    print(f"  Capital : $10,000  |  fee 15 bps  |  slippage 5 bps")
    print()

    strategy = BitcoinRangeMomentum(params=STRATEGY_PARAMS)
    bt = Backtester(starting_cash_usd=10_000.0, fee_bps=15.0, slippage_bps=5.0)
    result = bt.run(strategy, df)

    m = result.metrics
    if not m:
        print("  No data returned from backtest — check warmup / candle count")
        return

    ret_pct = m.get("total_return", 0) * 100
    dd_pct = m.get("max_drawdown", 0) * 100
    wr_pct = m.get("win_rate", 0) * 100
    sharpe = m.get("sharpe", 0)
    pf = m.get("profit_factor", 0)
    n_trades = int(m.get("num_trades", 0))

    final_equity = 10_000 * (1 + m.get("total_return", 0))
    calmar = ret_pct / abs(dd_pct) if dd_pct != 0 else 0.0

    print(f"  {'Total return':<22} {ret_pct:>+.2f}%   (${final_equity:,.2f})")
    print(f"  {'Sharpe ratio':<22} {sharpe:>8.3f}")
    print(f"  {'Max drawdown':<22} {dd_pct:>8.2f}%")
    print(f"  {'Calmar ratio':<22} {calmar:>8.3f}")
    print(f"  {'Win rate':<22} {wr_pct:>8.1f}%")
    print(f"  {'Profit factor':<22} {pf:>8.2f}")
    print(f"  {'Total trades':<22} {n_trades:>8}")

    if not result.trades.empty:
        best = result.trades["realized_pnl"].max()
        worst = result.trades["realized_pnl"].min()
        avg = result.trades["realized_pnl"].mean()
        print(f"  {'Avg trade PnL':<22} ${avg:>+.2f}")
        print(f"  {'Best trade':<22} ${best:>+.2f}")
        print(f"  {'Worst trade':<22} ${worst:>+.2f}")
        print(f"  {'First trade':<22} {result.trades['ts'].iloc[0]}")
        print(f"  {'Last trade':<22} {result.trades['ts'].iloc[-1]}")
    else:
        print()
        print("  No trades executed — check diagnostics output above.")


def run_walk_forward(df: pd.DataFrame) -> None:
    print()
    sep("═")
    print("WALK-FORWARD BACKTEST  —  BitcoinRangeMomentum")
    sep("═")
    print(f"  Train window : 60 days   |  Test window : 14 days")
    print(f"  Step size    : 14 days   |  Capital: $10,000 per window")
    print()

    strategy = BitcoinRangeMomentum(params=STRATEGY_PARAMS)
    bt = Backtester(starting_cash_usd=10_000.0, fee_bps=15.0, slippage_bps=5.0)
    wf = WalkForwardAnalyzer(
        backtester=bt,
        train_window=timedelta(days=60),
        test_window=timedelta(days=14),
    )
    wf_result = wf.run(strategy, df)
    windows = wf_result.windows

    if not windows:
        print("  No walk-forward windows completed.")
        print("  Need ≥ 74 days of data (60 train + 14 test); have",
              f"{(df.index[-1] - df.index[0]).days} days.")
        return

    agg = wf_result.aggregate_metrics
    print(f"  Windows run              : {int(agg.get('num_windows', 0))}")
    print(f"  Avg Sharpe               : {agg.get('avg_sharpe', 0):>8.3f}")
    print(f"  Sharpe std               : {agg.get('std_sharpe', 0):>8.3f}")
    print(f"  Min / Max Sharpe         : {agg.get('min_sharpe', 0):.3f} / {agg.get('max_sharpe', 0):.3f}")
    print(f"  % windows Sharpe > 0     : {agg.get('pct_windows_positive_sharpe', 0)*100:.1f}%")
    print(f"  Avg total return         : {agg.get('avg_return', 0)*100:>+.2f}%")
    print(f"  Avg max drawdown         : {agg.get('avg_max_drawdown', 0)*100:.2f}%")
    print(f"  Worst drawdown (any win) : {agg.get('worst_drawdown', 0)*100:.2f}%")
    print(f"  Avg win rate             : {agg.get('avg_win_rate', 0)*100:.1f}%")
    pf = agg.get('avg_profit_factor', 0)
    print(f"  Avg profit factor        : {pf:.2f}" if pf != float('inf') else "  Avg profit factor        : ∞ (no losses)")
    print()

    sep()
    print(f"  {'#':>3}  {'Test period':<26} {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Trades':>6}")
    sep()
    for i, w in enumerate(windows, 1):
        m = w["metrics"]
        ts = w["test_start"]
        te = w["test_end"]
        period = f"{ts.strftime('%b %d')} → {te.strftime('%b %d, %y')}"
        ret_pct = m.get("total_return", 0) * 100
        sharpe = m.get("sharpe", 0)
        dd_pct = m.get("max_drawdown", 0) * 100
        n = int(m.get("num_trades", 0))
        flag = "+" if sharpe > 0 else "-"
        print(f"  {flag}{i:>3}  {period:<26} {ret_pct:>+7.2f}%  {sharpe:>7.3f}  {dd_pct:>6.2f}%  {n:>6}")
    sep()
    print()
    print(wf_result.summary())


def run_diagnostics(df: pd.DataFrame) -> None:
    """
    Count how often each entry condition individually fires on the full dataset.
    Shows both long and short side condition fire rates.
    """
    import numpy as np
    from glitz_quant.strategies.indicators import (
        atr, bounce_candle, ema, rsi, shooting_star_candle, support_resistance, volume_z,
    )

    p = STRATEGY_PARAMS
    lookback = int(p["support_lookback_bars"])
    warmup = max(lookback, int(p["rsi_period"])) + 5
    n = len(df) - warmup

    sup, res  = support_resistance(df["high"], df["low"], lookback=lookback)
    rsi_s     = rsi(df["close"], period=int(p["rsi_period"]))
    vol_z_s   = volume_z(df["volume"], lookback=20)
    bounce_s  = bounce_candle(df["open"], df["high"], df["low"], df["close"])
    star_s    = shooting_star_candle(df["open"], df["high"], df["low"], df["close"])
    ema_s     = ema(df["close"], int(p["trend_ema_bars"])) if int(p.get("trend_ema_bars", 0)) > 0 else None

    sup_w     = sup.iloc[warmup:].values
    res_w     = res.iloc[warmup:].values
    rsi_w     = rsi_s.iloc[warmup:].values
    rsi_prev  = rsi_s.iloc[warmup - 1:-1].values
    vol_z_w   = vol_z_s.iloc[warmup:].values
    close_w   = df["close"].iloc[warmup:].values
    close_prev = df["close"].iloc[warmup - 1:-1].values
    bounce_w  = bounce_s.iloc[warmup:].values
    star_w    = star_s.iloc[warmup:].values

    uptrend = close_w > ema_s.iloc[warmup:].values if ema_s is not None else np.ones(n, dtype=bool)
    downtrend = ~uptrend if ema_s is not None else np.ones(n, dtype=bool)

    sep("═")
    print("CONDITION DIAGNOSTICS  —  LONG side (tight params)")
    sep("═")
    near_sup   = ((close_w - sup_w) / sup_w.clip(1e-9)) <= p["zone_proximity_pct"] / 100
    oversold   = rsi_w < p["rsi_oversold"]
    turning_up = rsi_w > rsi_prev
    vol_ok     = vol_z_w >= p["volume_z_threshold"]
    room_to_r  = (res_w - close_w) / close_w.clip(1e-9) > 0.005
    prev_ok    = close_prev >= sup_w

    long_conds = {
        f"near support (within {p['zone_proximity_pct']}% of {lookback}-bar low)": near_sup,
        f"RSI < {p['rsi_oversold']}":               oversold,
        "RSI turning up":                            turning_up,
        f"volume z >= {p['volume_z_threshold']}":   vol_ok,
        "room to resistance (>0.5%)":               room_to_r,
        "prev close >= support":                    prev_ok,
        f"in uptrend (close > EMA{p.get('trend_ema_bars',0)})": uptrend,
        "bounce candle (hammer — bonus indicator)": bounce_w,
    }
    all_long = np.ones(n, dtype=bool)
    print(f"  Total bars analysed: {n:,}\n")
    for label, mask in long_conds.items():
        pct = mask.sum() / n * 100
        print(f"  {pct:>5.1f}%  ({int(mask.sum()):>6,} bars)  {label}")
        if label != "bounce candle (hammer — bonus indicator)":
            all_long &= mask
    print(f"\n  ALL long conditions  →  {int(all_long.sum())} bars  ({all_long.sum()/n*100:.3f}%)")

    sep("═")
    print("CONDITION DIAGNOSTICS  —  SHORT side (tight params)")
    sep("═")
    near_res      = ((res_w - close_w) / res_w.clip(1e-9)) <= p["zone_proximity_pct"] / 100
    overbought    = rsi_w > p["rsi_overbought"]
    turning_down  = rsi_w < rsi_prev
    room_to_s     = (close_w - sup_w) / close_w.clip(1e-9) > 0.005
    prev_below_r  = close_prev <= res_w

    short_conds = {
        f"near resistance (within {p['zone_proximity_pct']}% of {lookback}-bar high)": near_res,
        f"RSI > {p['rsi_overbought']}":              overbought,
        "RSI turning down":                          turning_down,
        f"volume z >= {p['volume_z_threshold']}":   vol_ok,
        "room to support (>0.5%)":                  room_to_s,
        "prev close <= resistance":                 prev_below_r,
        f"in downtrend (close < EMA{p.get('trend_ema_bars',0)})": downtrend,
        "shooting star (bearish — bonus indicator)": star_w,
    }
    all_short = np.ones(n, dtype=bool)
    print(f"  Total bars analysed: {n:,}\n")
    for label, mask in short_conds.items():
        pct = mask.sum() / n * 100
        print(f"  {pct:>5.1f}%  ({int(mask.sum()):>6,} bars)  {label}")
        if label != "shooting star (bearish — bonus indicator)":
            all_short &= mask
    print(f"\n  ALL short conditions →  {int(all_short.sum())} bars  ({all_short.sum()/n*100:.3f}%)")
    print()

    print()
    sep("═")
    print("CONDITION DIAGNOSTICS  —  BitcoinRangeMomentum (LOOSE params)")
    sep("═")

    lp = LOOSE_PARAMS
    l_lookback = int(lp["support_lookback_bars"])
    l_warmup   = max(l_lookback, 14) + 5
    l_window_n = len(df) - l_warmup

    l_sup, l_res = support_resistance(df["high"], df["low"], lookback=l_lookback)
    l_rsi        = rsi(df["close"], period=14)
    l_vol        = volume_z(df["volume"], lookback=20)
    l_bounce     = bounce_candle(df["open"], df["high"], df["low"], df["close"])

    l_sup_w  = l_sup.iloc[l_warmup:]
    l_rsi_w  = l_rsi.iloc[l_warmup:]
    l_vol_w  = l_vol.iloc[l_warmup:]
    l_bou_w  = l_bounce.iloc[l_warmup:]
    l_cls_w  = df["close"].iloc[l_warmup:]
    l_res_w  = l_res.iloc[l_warmup:]

    l_near   = ((l_cls_w.values - l_sup_w.values) / l_sup_w.values.clip(1e-9)) <= lp["zone_proximity_pct"] / 100
    l_over   = l_rsi_w.values < lp["rsi_oversold"]
    l_vol_ok = l_vol_w.values >= lp["volume_z_threshold"]
    l_room   = (l_res_w.values - l_cls_w.values) / l_cls_w.values.clip(1e-9) > 0.005
    l_all    = l_near & l_over & l_bou_w.values & l_vol_ok & l_room

    l_conds = {
        f"near support  (within {lp['zone_proximity_pct']}% of {l_lookback}-bar low)": l_near,
        "bounce candle":                                                                  l_bou_w.values,
        f"RSI < {lp['rsi_oversold']}":                                                  l_over,
        f"volume z >= {lp['volume_z_threshold']}":                                      l_vol_ok,
        "room to resistance":                                                            l_room,
    }
    print(f"  Total bars analysed: {l_window_n:,}\n")
    for label, mask in l_conds.items():
        pct = mask.sum() / l_window_n * 100
        print(f"  {pct:>5.1f}%  ({mask.sum():>6,} bars)  {label}")
    print()
    print(f"  {'ALL loose conditions true simultaneously':}  →  {l_all.sum()} bars  ({l_all.sum()/l_window_n*100:.3f}%)")
    print()


def run_loose_backtest(df: pd.DataFrame) -> None:
    sep("═")
    print("SINGLE-PASS BACKTEST  —  BitcoinRangeMomentum  (LOOSE params)")
    sep("═")
    strategy = BitcoinRangeMomentum(params=LOOSE_PARAMS)
    bt = Backtester(starting_cash_usd=10_000.0, fee_bps=15.0, slippage_bps=5.0)
    result = bt.run(strategy, df)
    m = result.metrics
    if not m or int(m.get("num_trades", 0)) == 0:
        print("  Still no trades — bounce candle pattern is the bottleneck.")
        return
    ret_pct = m["total_return"] * 100
    print(f"  Total return     {ret_pct:>+.2f}%   (${10_000*(1+m['total_return']):,.2f})")
    print(f"  Sharpe           {m['sharpe']:>8.3f}")
    print(f"  Max drawdown     {m['max_drawdown']*100:>8.2f}%")
    print(f"  Win rate         {m['win_rate']*100:>8.1f}%")
    print(f"  Profit factor    {m['profit_factor']:>8.2f}")
    print(f"  Total trades     {int(m['num_trades']):>8}")
    if not result.trades.empty:
        print(f"  Avg trade PnL    ${result.trades['realized_pnl'].mean():>+.2f}")
        print(f"  Best trade       ${result.trades['realized_pnl'].max():>+.2f}")
        print(f"  Worst trade      ${result.trades['realized_pnl'].min():>+.2f}")
    print()


RANGE_TRADER_PARAMS = {
    "support_lookback_bars": 48,
    "resistance_lookback_bars": 48,
    "rsi_period": 14,
    "rsi_long_threshold": 50,
    "rsi_short_threshold": 50,
    "zone_proximity_pct": 2.0,
    "volume_z_min": 0.5,
    "atr_period": 14,
    "stop_atr_buffer": 1.5,
    "max_hold_bars": 32,
    "require_candle_pattern": True,
    "trend_ema_bars": 0,
    "target_notional_usd": 500,
}


def run_range_trader(df: pd.DataFrame) -> None:
    print()
    sep("═")
    print("SINGLE-PASS BACKTEST  —  RangeTrader  (long at support / short at resistance)")
    sep("═")
    print(f"  Period  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Capital : $10,000  |  fee 15 bps  |  slippage 5 bps")
    print()

    strategy = RangeTrader(params=RANGE_TRADER_PARAMS)
    bt = Backtester(starting_cash_usd=10_000.0, fee_bps=15.0, slippage_bps=5.0)
    result = bt.run(strategy, df)
    m = result.metrics

    if not m or int(m.get("num_trades", 0)) == 0:
        print("  No trades executed — check candle pattern and volume thresholds.")
        return

    ret_pct = m.get("total_return", 0) * 100
    dd_pct = m.get("max_drawdown", 0) * 100
    wr_pct = m.get("win_rate", 0) * 100
    sharpe = m.get("sharpe", 0)
    pf = m.get("profit_factor", 0)
    n_trades = int(m.get("num_trades", 0))
    final_equity = 10_000 * (1 + m.get("total_return", 0))
    calmar = ret_pct / abs(dd_pct) if dd_pct != 0 else 0.0

    print(f"  {'Total return':<22} {ret_pct:>+.2f}%   (${final_equity:,.2f})")
    print(f"  {'Sharpe ratio':<22} {sharpe:>8.3f}")
    print(f"  {'Max drawdown':<22} {dd_pct:>8.2f}%")
    print(f"  {'Calmar ratio':<22} {calmar:>8.3f}")
    print(f"  {'Win rate':<22} {wr_pct:>8.1f}%")
    print(f"  {'Profit factor':<22} {pf:>8.2f}")
    print(f"  {'Total trades':<22} {n_trades:>8}")

    if not result.trades.empty:
        best = result.trades["realized_pnl"].max()
        worst = result.trades["realized_pnl"].min()
        avg = result.trades["realized_pnl"].mean()
        print(f"  {'Avg trade PnL':<22} ${avg:>+.2f}")
        print(f"  {'Best trade':<22} ${best:>+.2f}")
        print(f"  {'Worst trade':<22} ${worst:>+.2f}")

        if "side" in result.trades.columns:
            long_trades = result.trades[result.trades["side"].str.contains("buy|long", case=False, na=False)]
            short_trades = result.trades[result.trades["side"].str.contains("sell|short|flat", case=False, na=False)]
            print(f"\n  Long entries : {len(long_trades)}  |  Short entries : {len(short_trades)}")

    print()

    # Walk-forward for range_trader
    sep("═")
    print("WALK-FORWARD  —  RangeTrader")
    sep("═")
    print(f"  Train 60d / Test 14d / Step 14d")
    print()
    strategy2 = RangeTrader(params=RANGE_TRADER_PARAMS)
    bt2 = Backtester(starting_cash_usd=10_000.0, fee_bps=15.0, slippage_bps=5.0)
    wf = WalkForwardAnalyzer(bt2, timedelta(days=60), timedelta(days=14))
    wf_result = wf.run(strategy2, df)
    windows = wf_result.windows

    if not windows:
        print("  No walk-forward windows completed.")
        return

    agg = wf_result.aggregate_metrics
    print(f"  Windows        : {int(agg.get('num_windows', 0))}")
    print(f"  Avg Sharpe     : {agg.get('avg_sharpe', 0):>8.3f}")
    print(f"  Min/Max Sharpe : {agg.get('min_sharpe', 0):.3f} / {agg.get('max_sharpe', 0):.3f}")
    print(f"  % windows > 0  : {agg.get('pct_windows_positive_sharpe', 0)*100:.1f}%")
    print(f"  Avg return     : {agg.get('avg_return', 0)*100:>+.2f}%")
    print(f"  Avg max DD     : {agg.get('avg_max_drawdown', 0)*100:.2f}%")
    print(f"  Avg win rate   : {agg.get('avg_win_rate', 0)*100:.1f}%")
    print(f"  Avg profit fac : {agg.get('avg_profit_factor', 0):.2f}")
    print()

    sep()
    print(f"  {'#':>3}  {'Test period':<26} {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Trades':>6}")
    sep()
    for i, w in enumerate(windows, 1):
        wm = w["metrics"]
        ts, te = w["test_start"], w["test_end"]
        period = f"{ts.strftime('%b %d')} → {te.strftime('%b %d, %y')}"
        ret = wm.get("total_return", 0) * 100
        sh = wm.get("sharpe", 0)
        dd = wm.get("max_drawdown", 0) * 100
        n = int(wm.get("num_trades", 0))
        flag = "+" if sh > 0 else "-"
        print(f"  {flag}{i:>3}  {period:<26} {ret:>+7.2f}%  {sh:>7.3f}  {dd:>6.2f}%  {n:>6}")
    sep()
    print()
    print(wf_result.summary())


async def main() -> None:
    df = await fetch_data()
    run_diagnostics(df)
    run_single_pass(df)
    run_loose_backtest(df)
    run_walk_forward(df)
    run_range_trader(df)
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
