"""
Walk-forward analyzer. Performs rolling-window backtests to evaluate
strategy robustness over time.
"""

from __future__ import annotations

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
        """
        Run walk-forward analysis on the given candles.
        Note: Current simple implementation treats each window as a fresh backtest.
        """
        windows = []
        start_ts = candles.index[0]
        end_ts = candles.index[-1]
        
        current_ts = start_ts + self.train_window
        
        while current_ts + self.test_window <= end_ts:
            train_start = current_ts - self.train_window
            test_end = current_ts + self.test_window
            
            # test_candles = candles.loc[current_ts:test_end]
            # Since backtester has a warmup, we include some history
            run_candles = candles.loc[train_start:test_end]
            
            log.info("wf_running_window", start=current_ts, end=test_end)
            res = self.backtester.run(strategy, run_candles)
            
            # Filter results to only the test period
            test_equity = res.equity_curve.loc[current_ts:test_end]
            
            windows.append({
                "test_start": current_ts,
                "test_end": test_end,
                "metrics": res.metrics,
                "equity_curve": test_equity
            })
            
            current_ts += self.test_window
            
        agg_metrics = self._compute_aggregate(windows)
        return WalkForwardResult(windows=windows, aggregate_metrics=agg_metrics)

    def _compute_aggregate(self, windows: list[dict[str, Any]]) -> dict[str, float]:
        if not windows:
            return {}
        
        sharpes = [w["metrics"]["sharpe"] for w in windows]
        returns = [w["metrics"]["total_return"] for w in windows]
        
        return {
            "avg_sharpe": sum(sharpes) / len(sharpes),
            "avg_return": sum(returns) / len(returns),
            "num_windows": float(len(windows))
        }
