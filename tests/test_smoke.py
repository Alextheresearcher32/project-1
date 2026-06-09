"""
Smoke tests. No network, no DB — just import + instantiate.
Run:  uv run pytest tests/test_smoke.py -v
"""

from __future__ import annotations

from decimal import Decimal

import pytest


def test_imports() -> None:
    """Every public module imports cleanly."""
    import glitz_quant
    from glitz_quant import settings
    from glitz_quant.data import types
    from glitz_quant.data.store import redis_cache, supabase_store
    from glitz_quant.data.ingest import ccxt_connector, external
    from glitz_quant.risk import engine, circuit_breakers, kill_switch
    from glitz_quant.execution import oms
    from glitz_quant.execution.adapters import base, paper
    from glitz_quant.strategies import base as sbase, indicators, bitcoin_range
    from glitz_quant.research.backtest import engine as bengine
    from glitz_quant.agents import llm_router, agents
    from glitz_quant.monitoring import metrics, alerts

    assert glitz_quant.__version__


def test_settings_load() -> None:
    from glitz_quant.settings import (
        Mode,
        get_app_config,
        get_exchanges_config,
        get_risk_config,
        get_settings,
        get_strategies_config,
    )
    s = get_settings()
    assert s.glitz_mode in Mode
    assert get_app_config()
    assert get_risk_config()
    assert get_exchanges_config()
    assert get_strategies_config()


def test_live_gate_closed_by_default() -> None:
    from glitz_quant.settings import LiveTradingGate
    open_, reasons = LiveTradingGate.check()
    assert open_ is False
    assert reasons


def test_risk_engine_rejects_market_order() -> None:
    from glitz_quant.data.types import Order, OrderType, Side, Venue
    from glitz_quant.risk.engine import RiskEngine
    engine = RiskEngine()
    order = Order(
        venue=Venue.PAPER, symbol="BTC-USD", side=Side.BUY,
        order_type=OrderType.MARKET, size=Decimal("0.01"), price=None,
    )
    pos = None
    decision = engine.check_order(
        order=order, current_position=pos,
        current_account_notional_usd=Decimal(0),
        current_daily_pnl_usd=Decimal(0),
        reference_price_usd=Decimal(60000),
    )
    assert decision.rejected
    assert any("market" in r.lower() for r in decision.reasons)


def test_risk_engine_caps_oversized_order() -> None:
    from glitz_quant.data.types import Order, OrderType, Side, Venue
    from glitz_quant.risk.engine import RiskEngine
    engine = RiskEngine()
    order = Order(
        venue=Venue.PAPER, symbol="BTC-USD", side=Side.BUY,
        order_type=OrderType.LIMIT, size=Decimal("100"), price=Decimal("60000"),
    )
    decision = engine.check_order(
        order=order, current_position=None,
        current_account_notional_usd=Decimal(0),
        current_daily_pnl_usd=Decimal(0),
        reference_price_usd=Decimal(60000),
    )
    assert decision.rejected
    # Should hit per-order max notional
    assert any("notional" in r.lower() for r in decision.reasons)


def test_bitcoin_range_strategy_constructs() -> None:
    from glitz_quant.strategies.bitcoin_range import BitcoinRangeMomentum
    strat = BitcoinRangeMomentum(params={"target_notional_usd": 50})
    assert strat.name == "bitcoin_range"
    assert strat.target_notional_usd == Decimal("50")


def test_indicators_smoke() -> None:
    import numpy as np
    import pandas as pd
    from glitz_quant.strategies.indicators import atr, bounce_candle, rsi, sma, support_resistance, volume_z
    rng = np.random.default_rng(42)
    n = 300
    close = pd.Series(50000 + np.cumsum(rng.normal(0, 200, n)))
    high = close + np.abs(rng.normal(0, 100, n))
    low = close - np.abs(rng.normal(0, 100, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(np.abs(rng.normal(1000, 200, n)))

    r = rsi(close)
    assert 0 <= r.iloc[-1] <= 100
    s = sma(close, 20)
    assert not s.iloc[-1] != s.iloc[-1]  # not NaN
    a = atr(high, low, close)
    assert a.iloc[-1] > 0
    sup, res = support_resistance(high, low, lookback=50)
    assert res.iloc[-1] >= sup.iloc[-1]
    b = bounce_candle(open_, high, low, close)
    assert b.dtype == bool
    z = volume_z(vol, lookback=20)
    assert not z.empty
