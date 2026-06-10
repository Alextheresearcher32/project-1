"""
Strategy base class. Strategies don't send orders; they emit Signals.
The OMS turns Signals into Orders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from glitz_quant.data.types import Signal


@dataclass
class StrategyContext:
    """Snapshot of state a strategy might need to make decisions."""
    candles: pd.DataFrame                    # OHLCV with datetime index, latest at end
    open_position_size: float                # current position size (signed)
    open_position_avg_price: float
    cash_usd: float
    recent_signals: list[Signal]             # last few signals from this strategy


class Strategy(ABC):
    name: str

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def on_candle(self, ctx: StrategyContext) -> Signal | None:
        """Return a Signal or None. Called once per closed candle."""
        ...
