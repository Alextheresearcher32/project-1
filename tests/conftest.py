"""Pytest config."""

import pytest


@pytest.fixture(autouse=True)
def _block_real_env(monkeypatch):
    """Default tests to paper, no live, no real keys."""
    monkeypatch.setenv("GLITZ_MODE", "paper")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # Wipe caches so settings reload picks up env
    from glitz_quant.settings import (
        get_app_config,
        get_exchanges_config,
        get_risk_config,
        get_settings,
        get_strategies_config,
    )
    get_settings.cache_clear()
    get_app_config.cache_clear()
    get_risk_config.cache_clear()
    get_exchanges_config.cache_clear()
    get_strategies_config.cache_clear()
    yield
