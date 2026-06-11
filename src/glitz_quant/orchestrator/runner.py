"""
Top-level orchestrator. Pulls data, runs strategies, routes signals to OMS.

Entry point:
  uv run python -m glitz_quant.orchestrator.runner
or after install:
  glitz
"""

from __future__ import annotations

import asyncio
import importlib
import signal as os_signal
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import typer

import ccxt.async_support as ccxt  # type: ignore[import-untyped]

from glitz_quant.data.ingest.ccxt_connector import CCXTIngest
from glitz_quant.data.store.redis_cache import RedisCache
from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.data.types import Venue
from glitz_quant.execution.adapters.ccxt_adapter import CCXTAdapter
from glitz_quant.execution.oms import build_default_oms
from glitz_quant.monitoring import alerts, metrics
from glitz_quant.risk import kill_switch
from glitz_quant.settings import (
    LiveTradingGate,
    Mode,
    get_app_config,
    get_settings,
    get_strategies_config,
)
from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)


VENUE_BY_NAME = {v.value: v for v in Venue}


def _load_strategies() -> dict[str, Strategy]:
    cfg = get_strategies_config().get("strategies", {})
    loaded: dict[str, Strategy] = {}
    for name, sc in cfg.items():
        if not sc.get("enabled"):
            continue
        mod = importlib.import_module(sc["module"])
        klass = getattr(mod, sc["class"])
        loaded[name] = klass(params=sc.get("params", {}))
        log.info("strategy_loaded", strategy=name)
    return loaded


async def _bootstrap() -> tuple[RedisCache, SupabaseStore | None, dict[Venue, CCXTIngest], Any]:
    cache = RedisCache()
    await cache.connect()

    store: SupabaseStore | None = None
    if get_settings().supabase_db_url:
        store = SupabaseStore()
        try:
            await store.connect()
        except Exception as e:
            log.warning("supabase_unavailable", err=str(e))
            store = None

    # Ingest workers — only enabled venues with credentials (or all read-only public)
    ingests: dict[Venue, CCXTIngest] = {}
    for venue in (Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US):
        try:
            ing = CCXTIngest(venue, cache)
            await ing.start()
            ingests[venue] = ing
        except Exception as e:
            log.warning("ingest_init_failed", venue=venue.value, err=str(e))

    oms = build_default_oms(cache=cache, store=store)

    # Register live adapters if live gate is open
    gate_open, _ = LiveTradingGate.check()
    if get_settings().glitz_mode == Mode.LIVE and gate_open:
        for venue in (Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US):
            try:
                adapter = CCXTAdapter(venue)
                await adapter.start()
                oms.register_adapter(adapter)
            except Exception as e:
                log.warning("live_adapter_failed", venue=venue.value, err=str(e))

    return cache, store, ingests, oms


def _candles_from_supabase_or_ingest_sync(ingest: CCXTIngest, symbol: str, tf: str) -> pd.DataFrame:
    """Synchronous wrapper for clarity — called inside an async context already."""
    raise NotImplementedError  # not used; orchestrator uses async path below


async def _fetch_candles_df(
    ingest: CCXTIngest, symbol: str, timeframe: str, limit: int = 500
) -> pd.DataFrame:
    candles = await ingest.fetch_candles(symbol, timeframe=timeframe, limit=limit)
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        [{"ts": c.ts, "open": float(c.open), "high": float(c.high),
          "low": float(c.low), "close": float(c.close), "volume": float(c.volume)}
         for c in candles]
    )
    return df.set_index("ts")


async def _fetch_funding_rate(symbol: str = "BTC/USDT") -> float | None:
    """Fetch perpetuals funding rate from Binance (public, no auth)."""
    try:
        exchange = ccxt.binance()
        data = await exchange.fetch_funding_rate(symbol)
        await exchange.close()
        rate = data.get("fundingRate")
        return float(rate) if rate is not None else None
    except Exception as e:
        log.warning("funding_rate_fetch_failed", symbol=symbol, err=str(e))
        try:
            await exchange.close()
        except Exception:
            pass
        return None


class Runner:
    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self.cache: RedisCache | None = None
        self.store: SupabaseStore | None = None
        self.ingests: dict[Venue, CCXTIngest] = {}
        self.oms: Any = None
        self.strategies: dict[str, Strategy] = {}
        self.recent_signals: dict[str, list] = {}
        self._last_breaker_status: dict[str, str] = {}

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass

    async def run(self) -> None:
        s = get_settings()
        log.info("orchestrator_starting", env=s.glitz_env.value, mode=s.glitz_mode.value)

        # Metrics endpoint
        try:
            mon = get_app_config().get("monitoring", {})
            if mon.get("prometheus_enabled", True):
                metrics.start_metrics_server(int(mon.get("prometheus_port", 9100)))
        except Exception as e:
            log.warning("metrics_server_failed", err=str(e))

        self._install_signal_handlers()
        self.cache, self.store, self.ingests, self.oms = await _bootstrap()
        self.strategies = _load_strategies()

        await alerts.broadcast("info", f"glitz-quant starting in {s.glitz_mode.value} mode")

        strategy_cfg = get_strategies_config().get("strategies", {})

        try:
            while not self._stop.is_set():
                if kill_switch.is_killed():
                    log.error("kill_switch_active_runner_halting")
                    await alerts.broadcast("critical", "kill switch active — runner halting")
                    break

                for name, strat in self.strategies.items():
                    sc = strategy_cfg.get(name, {})
                    symbols = sc.get("symbols", ["BTC-USD"])
                    tf = sc.get("timeframe", "15m")
                    venues = [VENUE_BY_NAME[v] for v in sc.get("venues", ["coinbase"])
                              if VENUE_BY_NAME.get(v) in self.ingests]
                    if not venues:
                        continue
                    ingest = self.ingests[venues[0]]

                    funding_rate = await _fetch_funding_rate()

                    for sym in symbols:
                        df = await _fetch_candles_df(ingest, sym, tf, limit=500)
                        if df.empty:
                            continue
                        # Update ticker cache too (cheap)
                        t = await ingest.fetch_ticker(sym)
                        if t:
                            await self.cache.set_ticker(t)  # type: ignore[union-attr]

                        # Current position
                        venue = venues[0]
                        pos = self.oms.get_position(venue, sym)

                        equity = await self.oms.get_total_equity()

                        extra: dict[str, Any] = {}
                        if funding_rate is not None:
                            extra["funding_rate_8h"] = funding_rate

                        ctx = StrategyContext(
                            candles=df,
                            open_position_size=float(pos.size),
                            open_position_avg_price=float(pos.avg_entry_price),
                            cash_usd=float(equity),
                            recent_signals=self.recent_signals.get(name, [])[-10:],
                            extra_data=extra,
                        )
                        sig = strat.on_candle(ctx)
                        if sig is not None:
                            self.recent_signals.setdefault(name, []).append(sig)
                            metrics.signals_emitted_total.labels(
                                strategy=name, direction=sig.direction.value
                            ).inc()
                            await self.oms.process_signal(sig)

                # Circuit breakers passive checks
                staleness_events = self.oms.breakers.check_data_staleness()
                for ev in staleness_events:
                    await alerts.broadcast("warning", f"data staleness: {ev.message}")

                # Account equity snapshot for breakers + metrics + store
                equity = await self.oms.get_total_equity()
                metrics.account_equity_usd.set(float(equity))
                metrics.account_pnl_daily_usd.set(float(self.oms.daily_pnl_usd))
                self.oms.breakers.observe_equity(equity)

                # Alert on newly-tripped circuit breakers
                current_status = self.oms.breakers.status()
                for name, msg in current_status.items():
                    if name not in self._last_breaker_status:
                        await alerts.broadcast("critical", f"CIRCUIT BREAKER TRIPPED [{name}]: {msg}")
                        metrics.circuit_breaker_active.labels(name=name).set(1)
                for name in list(self._last_breaker_status):
                    if name not in current_status:
                        metrics.circuit_breaker_active.labels(name=name).set(0)
                self._last_breaker_status = current_status

                await asyncio.sleep(int(get_app_config().get("orchestrator", {}).get("tick_interval_seconds", 30)))
        finally:
            await self._teardown()
            await alerts.broadcast("info", "glitz-quant stopped")

    async def _teardown(self) -> None:
        log.info("orchestrator_shutdown_starting")
        for ing in self.ingests.values():
            try:
                await ing.stop()
            except Exception as e:
                log.warning("ingest_stop_failed", err=str(e))
        try:
            for v, a in (self.oms.adapters.items() if self.oms else []):
                await a.stop()
        except Exception as e:
            log.warning("adapter_stop_failed", err=str(e))
        if self.store is not None:
            await self.store.close()
        if self.cache is not None:
            await self.cache.close()
        log.info("orchestrator_shutdown_complete")


@app.command()
def main(
    i_understand_the_risks: bool = typer.Option(
        False, "--i-understand-the-risks", help="Required to run in live mode."
    ),
) -> None:
    s = get_settings()
    if s.glitz_mode == Mode.LIVE and not i_understand_the_risks:
        typer.echo("Refusing to start in live mode without --i-understand-the-risks flag.")
        raise typer.Exit(code=2)
    if s.glitz_mode == Mode.LIVE:
        gate_open, reasons = LiveTradingGate.check()
        if not gate_open:
            typer.echo("Live-trading gate is CLOSED. Cannot start. Blockers:")
            for r in reasons:
                typer.echo(f"  - {r}")
            raise typer.Exit(code=2)

    runner = Runner()
    asyncio.run(runner.run())


if __name__ == "__main__":
    app()
