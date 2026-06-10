"""
Backtest engine. Replays historical candles through a strategy, simulates
fills with the same paper-fill model as live paper trading, computes
Sharpe / max DD / win rate / profit factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from glitz_quant.data.types import Signal, SignalDirection
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


class Backtester:
    def __init__(
        self,
        starting_cash_usd: float = 10_000.0,
        fee_bps: float = 15.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.starting_cash = starting_cash_usd
        self.fee = Decimal(str(fee_bps / 10000))
        self.slippage = Decimal(str(slippage_bps / 10000))

    def run(self, strategy: Strategy, candles: pd.DataFrame) -> BacktestResult:
        """
        candles: DataFrame with index = datetime, columns = open/high/low/close/volume.
        """
        cash = Decimal(str(self.starting_cash))
        position_size = Decimal(0)
        position_avg_price = Decimal(0)
        recent_signals: list[Signal] = []
        trades_log: list[dict[str, Any]] = []
        equity: list[tuple[datetime, float]] = []

        # warmup
        warmup = 200
        for i in range(warmup, len(candles)):
            window = candles.iloc[: i + 1]
            close_price = Decimal(str(window["close"].iloc[-1]))
            ctx = StrategyContext(
                candles=window,
                open_position_size=float(position_size),
                open_position_avg_price=float(position_avg_price),
                cash_usd=float(cash),
                recent_signals=recent_signals[-10:],
            )
            sig = strategy.on_candle(ctx)
            if sig is not None:
                recent_signals.append(sig)
                trade_info = self._execute_signal(
                    sig, close_price, position_size, position_avg_price, cash
                )
                if trade_info is not None:
                    cash = trade_info["cash"]
                    position_size = trade_info["size"]
                    position_avg_price = trade_info["avg"]
                    trades_log.append({
                        "ts": window.index[-1],
                        "side": trade_info["side"],
                        "price": float(trade_info["fill_price"]),
                        "size": float(trade_info["fill_size"]),
                        "fee": float(trade_info["fee"]),
                        "realized_pnl": float(trade_info["realized_pnl"]),
                        "reason": sig.reason,
                    })

            mtm = cash + position_size * close_price
            equity.append((window.index[-1], float(mtm)))

        eq = pd.Series({ts: v for ts, v in equity})
        trades_df = pd.DataFrame(trades_log)
        metrics = self._compute_metrics(eq, trades_df)
        return BacktestResult(equity_curve=eq, trades=trades_df, metrics=metrics)

    def _execute_signal(
        self,
        sig: Signal,
        close_price: Decimal,
        cur_size: Decimal,
        cur_avg: Decimal,
        cash: Decimal,
    ) -> dict[str, Any] | None:
        if sig.direction == SignalDirection.FLAT:
            if cur_size == 0:
                return None
            # close at close_price minus slippage (selling) or plus (buying back short)
            if cur_size > 0:
                fill = close_price * (Decimal(1) - self.slippage)
                fee = abs(cur_size) * fill * self.fee
                realized = (fill - cur_avg) * cur_size - fee
                cash = cash + cur_size * fill - fee
                return {
                    "side": "sell", "fill_price": fill, "fill_size": cur_size, "fee": fee,
                    "realized_pnl": realized, "cash": cash,
                    "size": Decimal(0), "avg": Decimal(0),
                }
            else:
                fill = close_price * (Decimal(1) + self.slippage)
                fee = abs(cur_size) * fill * self.fee
                realized = (cur_avg - fill) * abs(cur_size) - fee
                cash = cash - abs(cur_size) * fill - fee
                return {
                    "side": "buy", "fill_price": fill, "fill_size": abs(cur_size), "fee": fee,
                    "realized_pnl": realized, "cash": cash,
                    "size": Decimal(0), "avg": Decimal(0),
                }

        if sig.direction == SignalDirection.LONG and cur_size <= 0:
            target_size = sig.target_notional_usd / close_price
            fill = close_price * (Decimal(1) + self.slippage)
            fee = target_size * fill * self.fee
            if cash < target_size * fill + fee:
                return None
            cash = cash - target_size * fill - fee
            return {
                "side": "buy", "fill_price": fill, "fill_size": target_size, "fee": fee,
                "realized_pnl": Decimal(0), "cash": cash,
                "size": target_size, "avg": fill,
            }
        if sig.direction == SignalDirection.SHORT and cur_size >= 0:
            target_size = sig.target_notional_usd / close_price
            fill = close_price * (Decimal(1) - self.slippage)
            fee = target_size * fill * self.fee
            cash = cash + target_size * fill - fee
            return {
                "side": "sell", "fill_price": fill, "fill_size": target_size, "fee": fee,
                "realized_pnl": Decimal(0), "cash": cash,
                "size": -target_size, "avg": fill,
            }
        return None

    @staticmethod
    def _compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
        if equity.empty:
            return {}
        rets = equity.pct_change().dropna()
        ann_factor = float(np.sqrt(365 * 24 * 4))  # 15min bars per year approx
        sharpe = float(rets.mean() / rets.std() * ann_factor) if rets.std() else 0.0
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        wins = trades[trades["realized_pnl"] > 0] if not trades.empty else trades
        losses = trades[trades["realized_pnl"] < 0] if not trades.empty else trades
        win_rate = float(len(wins) / max(len(trades), 1)) if not trades.empty else 0.0
        gross_wins = float(wins["realized_pnl"].sum()) if not wins.empty else 0.0
        gross_losses = float(-losses["realized_pnl"].sum()) if not losses.empty else 0.0
        profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": float(len(trades)),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }
