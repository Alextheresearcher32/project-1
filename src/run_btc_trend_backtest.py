"""
Quick backtest for BtcTrendFollow strategy on BTC/USD 15m.
Fetches 6 months of data — enough for ~9 walk-forward windows.
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
from glitz_quant.strategies.btc_trend import BtcTrendFollow
from glitz_quant.research.backtest.engine import Backtester
from glitz_quant.research.backtest.walk_forward import WalkForwardAnalyzer

PARAMS = {
    "symbol": "BTC-USD",
    "target_notional_usd": 500,
    "tp_atr_mult": 3.0,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 96,
    "atr_regime_min": 0.003,
    "cooldown_bars": 16,
    "bidirectional": True,
    "rsi_long_min": 50.0,
    "rsi_short_max": 50.0,
}

def sep(c="═", w=62):
    print(c * w)


async def fetch_data() -> pd.DataFrame:
    s = get_settings()
    exchange = ccxt.coinbaseadvanced({
        "apiKey": s.coinbase_api_key.get_secret_value(),
        "secret": s.coinbase_api_secret.get_secret_value(),
        "enableRateLimit": True,
    })
    print("Fetching BTC/USD 15m  2025-12-01 → now ...")
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
    print(f"  {len(df):,} candles  |  {df.index[0].date()} → {df.index[-1].date()}\n")
    return df


def run(df: pd.DataFrame) -> None:
    sep()
    print("SINGLE-PASS  —  BtcTrendFollow")
    sep()
    strategy = BtcTrendFollow(PARAMS)
    bt = Backtester(10_000.0, fee_bps=15.0, slippage_bps=5.0)
    result = bt.run(strategy, df)
    m = result.metrics
    print(f"  Period       : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total return : {m.get('total_return', 0)*100:+.2f}%  (${10000*(1+m.get('total_return',0)):.0f})")
    print(f"  Sharpe       : {m.get('sharpe', 0):+.3f}")
    print(f"  Max drawdown : {m.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Win rate     : {m.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit factor: {m.get('profit_factor', 0):.2f}")
    print(f"  Trades       : {int(m.get('num_trades', 0))}")
    if not result.trades.empty:
        print(f"  Avg trade PnL: ${result.trades['realized_pnl'].mean():+.2f}")
        print(f"  Best trade   : ${result.trades['realized_pnl'].max():+.2f}")
        print(f"  Worst trade  : ${result.trades['realized_pnl'].min():+.2f}")
    print()

    sep()
    print("WALK-FORWARD  —  BtcTrendFollow  (60d train / 14d test)")
    sep()
    strategy2 = BtcTrendFollow(PARAMS)
    bt2 = Backtester(10_000.0, fee_bps=15.0, slippage_bps=5.0)
    wf = WalkForwardAnalyzer(bt2, timedelta(days=60), timedelta(days=14))
    wfr = wf.run(strategy2, df)
    agg = wfr.aggregate_metrics
    print(f"  Windows       : {int(agg.get('num_windows', 0))}")
    print(f"  Avg Sharpe    : {agg.get('avg_sharpe', 0):+.3f}")
    print(f"  Min/Max Sharpe: {agg.get('min_sharpe', 0):.3f} / {agg.get('max_sharpe', 0):.3f}")
    print(f"  % positive    : {agg.get('pct_windows_positive_sharpe', 0)*100:.1f}%")
    print(f"  Avg return    : {agg.get('avg_return', 0)*100:+.2f}%")
    print(f"  Avg max DD    : {agg.get('avg_max_drawdown', 0)*100:.2f}%")
    print(f"  Avg win rate  : {agg.get('avg_win_rate', 0)*100:.1f}%")
    print(f"  Avg PF        : {agg.get('avg_profit_factor', 0):.2f}")
    print()
    sep("-", 62)
    print(f"  {'#':>3}  {'Test period':<26} {'Return':>7}  {'Sharpe':>7}  {'Trades':>6}")
    sep("-", 62)
    for i, w in enumerate(wfr.windows, 1):
        wm = w["metrics"]
        ts, te = w["test_start"], w["test_end"]
        period = f"{ts.strftime('%b %d')} → {te.strftime('%b %d, %y')}"
        ret = wm.get("total_return", 0) * 100
        sh = wm.get("sharpe", 0)
        n = int(wm.get("num_trades", 0))
        flag = "+" if sh > 0 else "-"
        print(f"  {flag}{i:>3}  {period:<26} {ret:>+6.2f}%  {sh:>7.3f}  {n:>6}")
    sep("-", 62)
    print()
    print(wfr.summary())
    print()
    print("Done.")


async def main() -> None:
    df = await fetch_data()
    run(df)


if __name__ == "__main__":
    asyncio.run(main())
