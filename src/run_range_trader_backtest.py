"""Backtest RangeTrader — single-pass + walk-forward."""
from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ccxt.async_support as ccxt
import pandas as pd

from glitz_quant.settings import get_settings
from glitz_quant.strategies.range_trader import RangeTrader
from glitz_quant.research.backtest.engine import Backtester
from glitz_quant.research.backtest.walk_forward import WalkForwardAnalyzer

PARAMS = {
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


def sep(c="─", w=62):
    print(c * w)


async def fetch_data() -> pd.DataFrame:
    s = get_settings()
    exchange = ccxt.coinbaseadvanced({
        "apiKey": s.coinbase_api_key.get_secret_value(),
        "secret": s.coinbase_api_secret.get_secret_value(),
        "enableRateLimit": True,
    })
    print("Fetching BTC/USD 15m candles from Coinbase...")
    all_candles: list = []
    since = exchange.parse8601("2025-12-01T00:00:00Z")
    for _ in range(60):
        chunk = await exchange.fetch_ohlcv("BTC/USD", timeframe="15m", since=since, limit=300)
        if not chunk:
            break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= exchange.milliseconds() - 15 * 60 * 1000:
            break
        since = last_ts + 15 * 60 * 1000
        await asyncio.sleep(0.35)
    await exchange.close()
    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    print(f"  {len(df)} candles  |  {df.index[0].date()} -> {df.index[-1].date()}\n")
    return df


def print_metrics(m: dict, trades: pd.DataFrame) -> None:
    ret = m.get("total_return", 0) * 100
    eq = 10_000 * (1 + m.get("total_return", 0))
    print(f"  {'Total return':<22} {ret:>+.2f}%   (${eq:,.2f})")
    print(f"  {'Sharpe ratio':<22} {m.get('sharpe', 0):>8.3f}")
    print(f"  {'Max drawdown':<22} {m.get('max_drawdown', 0)*100:>8.2f}%")
    calmar = ret / abs(m.get("max_drawdown", 0) * 100) if m.get("max_drawdown", 0) != 0 else 0
    print(f"  {'Calmar ratio':<22} {calmar:>8.3f}")
    print(f"  {'Win rate':<22} {m.get('win_rate', 0)*100:>8.1f}%")
    pf = m.get("profit_factor", 0)
    print(f"  {'Profit factor':<22} {pf:>8.2f}" if pf != float("inf") else f"  {'Profit factor':<22} {'inf':>8}")
    print(f"  {'Total trades':<22} {int(m.get('num_trades', 0)):>8}")
    if not trades.empty:
        print(f"  {'Avg trade PnL':<22} ${trades['realized_pnl'].mean():>+.2f}")
        print(f"  {'Best trade':<22} ${trades['realized_pnl'].max():>+.2f}")
        print(f"  {'Worst trade':<22} ${trades['realized_pnl'].min():>+.2f}")
        if "side" in trades.columns:
            longs = trades[trades["side"].str.contains("buy", case=False, na=False)]
            shorts = trades[~trades["side"].str.contains("buy", case=False, na=False)]
            print(f"  {'Long fills':<22} {len(longs):>8}")
            print(f"  {'Short fills':<22} {len(shorts):>8}")


async def main() -> None:
    df = await fetch_data()

    # ── Single-pass ──────────────────────────────────────────────────────────
    sep("═")
    print("SINGLE-PASS BACKTEST  —  RangeTrader")
    sep("═")
    print(f"  Period  : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Capital : $10,000  |  fee 15 bps  |  slippage 5 bps\n")

    strategy = RangeTrader(params=PARAMS)
    bt = Backtester(10_000.0, 15.0, 5.0)
    result = bt.run(strategy, df)
    m = result.metrics

    if not m or int(m.get("num_trades", 0)) == 0:
        print("  No trades executed.")
    else:
        print_metrics(m, result.trades)

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print()
    sep("═")
    print("WALK-FORWARD BACKTEST  —  RangeTrader")
    sep("═")
    print("  Train 60d / Test 14d / Step 14d\n")

    strategy2 = RangeTrader(params=PARAMS)
    bt2 = Backtester(10_000.0, 15.0, 5.0)
    wf = WalkForwardAnalyzer(bt2, timedelta(days=60), timedelta(days=14))
    wfr = wf.run(strategy2, df)
    windows = wfr.windows

    if not windows:
        print("  No windows completed.")
        print("Done.")
        return

    agg = wfr.aggregate_metrics
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
        period = f"{ts.strftime('%b %d')} -> {te.strftime('%b %d, %y')}"
        ret = wm.get("total_return", 0) * 100
        sh = wm.get("sharpe", 0)
        dd = wm.get("max_drawdown", 0) * 100
        n = int(wm.get("num_trades", 0))
        flag = "+" if sh > 0 else "-"
        print(f"  {flag}{i:>3}  {period:<26} {ret:>+7.2f}%  {sh:>7.3f}  {dd:>6.2f}%  {n:>6}")
    sep()
    print()
    print(wfr.summary())
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
