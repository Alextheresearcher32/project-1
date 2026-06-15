"""
Backtest with ATR-based ranging-regime filter on Dec 2025 → Jun 2026.
Only enters trades when ATR/price < atr_regime_pct (low volatility = ranging).
Skips entries during trending/crash phases that killed previous results.

ATR/price thresholds to test:
  0.010 = 1.0%  — very tight, only deep consolidation
  0.012 = 1.2%  — moderate (baseline)
  0.015 = 1.5%  — looser, more trades
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

BASE_BTC_PARAMS = {
    "support_lookback_bars": 48,
    "resistance_lookback_bars": 48,
    "rsi_period": 14,
    "rsi_oversold": 42,
    "rsi_overbought": 58,
    "rsi_must_turn_up": True,
    "zone_proximity_pct": 1.5,
    "volume_z_threshold": 0.0,
    "atr_period": 14,
    "stop_atr_buffer": 1.0,
    "max_hold_bars": 32,
    "trend_ema_bars": 200,
    "require_bounce_candle": False,
    "target_notional_usd": 500,
}

BASE_RT_PARAMS = {
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
    print("Fetching BTC/USD 15m  Dec 2025 → Jun 2026 ...")
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
    print(f"  {len(df)} candles  |  {df.index[0].date()} → {df.index[-1].date()}\n")
    return df


def run_single(label: str, strategy, df: pd.DataFrame) -> dict:
    bt = Backtester(10_000.0, 15.0, 5.0)
    result = bt.run(strategy, df)
    m = result.metrics or {}
    ret = m.get("total_return", 0) * 100
    sh = m.get("sharpe", 0)
    dd = m.get("max_drawdown", 0) * 100
    wr = m.get("win_rate", 0) * 100
    pf = m.get("profit_factor", 0)
    n = int(m.get("num_trades", 0))
    eq = 10_000 * (1 + m.get("total_return", 0))
    calmar = ret / abs(dd) if dd != 0 else 0.0
    avg_pnl = result.trades["realized_pnl"].mean() if not result.trades.empty else 0.0
    print(f"\n  ── {label}")
    print(f"     Trades {n:>4}  |  Return {ret:>+.2f}% (${eq:,.0f})  |  Sharpe {sh:>+.3f}")
    print(f"     Win% {wr:>4.1f}%  |  PF {pf:.2f}  |  MaxDD {dd:.2f}%  |  Calmar {calmar:.2f}")
    print(f"     Avg PnL ${avg_pnl:>+.2f}")
    return m


def run_wf(label: str, strategy, df: pd.DataFrame) -> None:
    bt = Backtester(10_000.0, 15.0, 5.0)
    wf = WalkForwardAnalyzer(bt, timedelta(days=60), timedelta(days=14))
    wfr = wf.run(strategy, df)
    windows = wfr.windows
    if not windows:
        print(f"  {label}: no windows completed")
        return
    agg = wfr.aggregate_metrics
    pos = agg.get("pct_windows_positive_sharpe", 0) * 100
    print(f"\n  ── WF {label}")
    print(f"     Avg Sharpe {agg.get('avg_sharpe',0):>+.3f}  |  Max {agg.get('max_sharpe',0):>+.3f}  |  {pos:.0f}% profitable")
    print(f"     Avg return {agg.get('avg_return',0)*100:>+.2f}%  |  Avg PF {agg.get('avg_profit_factor',0):.2f}  |  Avg win% {agg.get('avg_win_rate',0)*100:.1f}%")
    sep()
    print(f"     {'#':>2}  {'Period':<26} {'Return':>8}  {'Sharpe':>7}  {'Trades':>6}")
    sep()
    for i, w in enumerate(windows, 1):
        wm = w["metrics"]
        ts, te = w["test_start"], w["test_end"]
        period = f"{ts.strftime('%b %d')} → {te.strftime('%b %d, %y')}"
        r = wm.get("total_return", 0) * 100
        s = wm.get("sharpe", 0)
        n = int(wm.get("num_trades", 0))
        flag = "+" if s > 0 else "-"
        print(f"     {flag}{i:>2}  {period:<26} {r:>+7.2f}%  {s:>7.3f}  {n:>6}")
    sep()


async def main() -> None:
    df = await fetch_data()

    # ── Regime filter comparison ──────────────────────────────────────────────
    # 15m BTC ATR/price: mean=0.43%, p90=0.63%, max=1.05%
    # Filter at p90 blocks the top 10% most volatile bars (crash/breakout periods)
    thresholds = [0.0, 0.004, 0.005, 0.006]

    sep("═")
    print("ATR REGIME FILTER — BitcoinRangeMomentum  (Dec 2025 → Jun 2026)")
    sep("═")
    print("  atr_regime_pct=0.0 means no filter (baseline)\n")
    for t in thresholds:
        params = {**BASE_BTC_PARAMS, "atr_regime_pct": t}
        label = f"atr_regime_pct={t:.3f}" + (" (no filter)" if t == 0 else "")
        run_single(label, BitcoinRangeMomentum(params), df)

    sep("═")
    print("\nATR REGIME FILTER — RangeTrader  (Dec 2025 → Jun 2026)")
    sep("═")
    print("  atr_regime_pct=0.0 means no filter (baseline)\n")
    for t in thresholds:
        params = {**BASE_RT_PARAMS, "atr_regime_pct": t}
        label = f"atr_regime_pct={t:.3f}" + (" (no filter)" if t == 0 else "")
        run_single(label, RangeTrader(params), df)

    # ── Walk-forward with best threshold ────────────────────────────────────
    BEST_THRESHOLD = 0.005  # p90 ATR/price on 15m — blocks top 10% most volatile bars
    print()
    sep("═")
    print(f"WALK-FORWARD WITH REGIME FILTER  (atr_regime_pct={BEST_THRESHOLD})")
    sep("═")

    run_wf(
        f"BitcoinRangeMomentum  atr<{BEST_THRESHOLD}",
        BitcoinRangeMomentum({**BASE_BTC_PARAMS, "atr_regime_pct": BEST_THRESHOLD}),
        df,
    )
    run_wf(
        f"RangeTrader  atr<{BEST_THRESHOLD}",
        RangeTrader({**BASE_RT_PARAMS, "atr_regime_pct": BEST_THRESHOLD}),
        df,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
