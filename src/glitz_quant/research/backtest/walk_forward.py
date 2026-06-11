"""
Walk-forward analyzer. Performs rolling-window backtests to evaluate
strategy robustness over time.

Each window: train on [t-train_window, t), test on [t, t+test_window).
Metrics are computed on the TEST period only — train data is warmup only.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from glitz_quant.research.backtest.engine import Backtester, BacktestResult
from glitz_quant.strategies.base import Strategy
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    windows: list[dict[str, Any]]
    aggregate_metrics: dict[str, float]

    def summary(self) -> str:
        agg = self.aggregate_metrics
        return (
            f"Walk-forward: {int(agg.get('num_windows', 0))} windows | "
            f"avg Sharpe {agg.get('avg_sharpe', 0):.2f} "
            f"(min {agg.get('min_sharpe', 0):.2f} / max {agg.get('max_sharpe', 0):.2f}) | "
            f"consistency {agg.get('pct_windows_positive_sharpe', 0):.0%} | "
            f"avg return {agg.get('avg_return', 0):.2%} | "
            f"avg max DD {agg.get('avg_max_drawdown', 0):.2%}"
        )


class WalkForwardAnalyzer:
    def __init__(
        self,
        backtester: Backtester,
        train_window: timedelta,
        test_window: timedelta,
    ) -> None:
        self.backtester = backtester
        self.train_window = train_window
        self.test_window = test_window

    def run(self, strategy: Strategy, candles: pd.DataFrame) -> WalkForwardResult:
        windows: list[dict[str, Any]] = []
        start_ts = candles.index[0]
        end_ts = candles.index[-1]

        current_ts = start_ts + self.train_window

        while current_ts + self.test_window <= end_ts:
            train_start = current_ts - self.train_window
            test_end = current_ts + self.test_window

            # Run on full window so the strategy has warmup history
            run_candles = candles.loc[train_start:test_end]
            log.info("wf_window", train_start=str(train_start), test_start=str(current_ts), test_end=str(test_end))

            res: BacktestResult = self.backtester.run(strategy, run_candles)

            # Slice equity and trades to test period only
            test_equity = res.equity_curve.loc[current_ts:test_end]
            if not res.trades.empty and "ts" in res.trades.columns:
                test_trades = res.trades[
                    (res.trades["ts"] >= current_ts) & (res.trades["ts"] <= test_end)
                ]
            else:
                test_trades = res.trades

            test_metrics = Backtester._compute_metrics(test_equity, test_trades)

            windows.append({
                "test_start": current_ts,
                "test_end": test_end,
                "metrics": test_metrics,
                "equity_curve": test_equity,
            })

            current_ts += self.test_window

        return WalkForwardResult(
            windows=windows,
            aggregate_metrics=self._compute_aggregate(windows),
        )

    @staticmethod
    def _compute_aggregate(windows: list[dict[str, Any]]) -> dict[str, float]:
        if not windows:
            return {}

        sharpes = [w["metrics"].get("sharpe", 0.0) for w in windows]
        returns = [w["metrics"].get("total_return", 0.0) for w in windows]
        drawdowns = [w["metrics"].get("max_drawdown", 0.0) for w in windows]
        win_rates = [w["metrics"].get("win_rate", 0.0) for w in windows]
        profit_factors = [w["metrics"].get("profit_factor", 0.0) for w in windows if w["metrics"].get("profit_factor", 0.0) != float("inf")]

        n = len(windows)
        positive_sharpe_windows = sum(1 for s in sharpes if s > 0)

        return {
            "num_windows": float(n),
            "avg_sharpe": sum(sharpes) / n,
            "min_sharpe": min(sharpes),
            "max_sharpe": max(sharpes),
            "std_sharpe": statistics.stdev(sharpes) if n > 1 else 0.0,
            "pct_windows_positive_sharpe": positive_sharpe_windows / n,
            "avg_return": sum(returns) / n,
            "min_return": min(returns),
            "max_return": max(returns),
            "avg_max_drawdown": sum(drawdowns) / n,
            "worst_drawdown": min(drawdowns),
            "avg_win_rate": sum(win_rates) / n,
            "avg_profit_factor": sum(profit_factors) / len(profit_factors) if profit_factors else 0.0,
        }
