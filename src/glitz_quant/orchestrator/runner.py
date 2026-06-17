"""
Top-level orchestrator.

New architecture:
  ThetaData/Oanda → Redis binary (0.1ms) + Redis Streams (push)
                  → 6 Wolves (ThreadPoolExecutor, parallel)
                  → Boardroom (asyncio.gather, ~7.7s not 30.8s)
                  → Chairman → Supervisor → ThetaData/Oanda APIs

Entry point:
  uv run python -m glitz_quant.orchestrator.runner
or after install:
  glitz
"""

from __future__ import annotations

import asyncio
import importlib
import signal as os_signal
import time
from decimal import Decimal
from typing import Any

import pandas as pd
import typer

import ccxt.async_support as ccxt  # type: ignore[import-untyped]

from glitz_quant.agents.boardroom import Boardroom, BoardroomContext
from glitz_quant.agents.chairman import ChairmanAgent, Supervisor
from glitz_quant.agents.llm_router import LLMRouter
from glitz_quant.agents.signal_analyst import SignalAnalyst
from glitz_quant.data.ingest.ccxt_connector import CCXTIngest
from glitz_quant.data.ingest.oanda_connector import OandaConnector
from glitz_quant.data.ingest.thetadata_connector import ThetaDataConnector
from glitz_quant.data.store.redis_cache import RedisCache
from glitz_quant.data.store.redis_streams import RedisStreams
from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.data.types import Venue
from glitz_quant.execution.adapters.ccxt_adapter import CCXTAdapter
from glitz_quant.execution.adapters.oanda_adapter import OandaAdapter
from glitz_quant.execution.oms import build_default_oms
from glitz_quant.monitoring import alerts, metrics
from glitz_quant.orchestrator.wolves import WolfJob, WolfPack
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


async def _bootstrap(
    streams: RedisStreams,
) -> tuple[
    RedisCache,
    SupabaseStore | None,
    dict[Venue, CCXTIngest],
    OandaConnector | None,
    ThetaDataConnector | None,
    Any,
]:
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

    # CCXT ingest workers (crypto venues)
    ingests: dict[Venue, CCXTIngest] = {}
    for venue in (Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US):
        try:
            ing = CCXTIngest(venue, cache)
            await ing.start()
            ingests[venue] = ing
        except Exception as e:
            log.warning("ingest_init_failed", venue=venue.value, err=str(e))

    # Oanda connector (forex streaming)
    oanda: OandaConnector | None = None
    s = get_settings()
    if s.oanda_api_key and s.oanda_account_id:
        try:
            oanda = OandaConnector(streams)
            await oanda.start()
        except Exception as e:
            log.warning("oanda_connector_failed", err=str(e))
            oanda = None

    # ThetaData connector (equities/options)
    thetadata: ThetaDataConnector | None = None
    if s.thetadata_api_key:
        try:
            thetadata = ThetaDataConnector(streams)
            await thetadata.start()
        except Exception as e:
            log.warning("thetadata_connector_failed", err=str(e))
            thetadata = None

    # OMS
    oms = build_default_oms(cache=cache, store=store)

    gate_open, _ = LiveTradingGate.check()
    if s.glitz_mode == Mode.LIVE and gate_open:
        # CCXT live adapters
        for venue in (Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US):
            try:
                adapter = CCXTAdapter(venue)
                await adapter.start()
                oms.register_adapter(adapter)
            except Exception as e:
                log.warning("live_adapter_failed", venue=venue.value, err=str(e))
        # Oanda live adapter
        if s.oanda_api_key and s.oanda_account_id:
            try:
                oanda_adapter = OandaAdapter()
                await oanda_adapter.start()
                oms.register_adapter(oanda_adapter)
            except Exception as e:
                log.warning("oanda_adapter_failed", err=str(e))

    return cache, store, ingests, oanda, thetadata, oms


async def _fetch_candles_df(
    ingest: CCXTIngest, symbol: str, timeframe: str, limit: int = 500
) -> pd.DataFrame:
    candles = await ingest.fetch_candles(symbol, timeframe=timeframe, limit=limit)
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "ts": c.ts,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
            for c in candles
        ]
    )
    return df.set_index("ts")


_funding_cache: tuple[float, float | None] = (0.0, None)  # (last_fetch_ts, value)
_FUNDING_CACHE_TTL = 300  # 5 minutes — avoid Binance geo-block spam every 30s


async def _fetch_funding_rate(symbol: str = "BTC/USDT") -> float | None:
    global _funding_cache
    now = time.time()
    if now - _funding_cache[0] < _FUNDING_CACHE_TTL:
        return _funding_cache[1]
    exchange = ccxt.binance()
    try:
        data = await exchange.fetch_funding_rate(symbol)
        await exchange.close()
        rate = data.get("fundingRate")
        result = float(rate) if rate is not None else None
        _funding_cache = (now, result)
        return result
    except Exception as e:
        _funding_cache = (now, None)  # backoff 5 min before next attempt
        log.warning("funding_rate_fetch_failed", symbol=symbol, err=str(e))
        try:
            await exchange.close()
        except Exception:
            pass
        return None


class Runner:
    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self.streams: RedisStreams = RedisStreams()
        self.cache: RedisCache | None = None
        self.store: SupabaseStore | None = None
        self.ingests: dict[Venue, CCXTIngest] = {}
        self.oanda: OandaConnector | None = None
        self.thetadata: ThetaDataConnector | None = None
        self.oms: Any = None
        self.strategies: dict[str, Strategy] = {}
        self.wolf_pack: WolfPack | None = None
        self.boardroom: Boardroom | None = None
        self.chairman: ChairmanAgent | None = None
        self.supervisor: Supervisor | None = None
        self.recent_signals: dict[str, list] = {}
        self._last_breaker_status: dict[str, str] = {}
        self._last_signal_analysis_ts: float = 0.0

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass

    async def _log_connection_status(self) -> None:
        s = get_settings()
        connected, missing = [], []
        checks = {
            "Anthropic": s.anthropic_api_key,
            "OpenRouter": s.openrouter_api_key,
            "Supabase URL": s.supabase_url,
            "Supabase DB": s.supabase_db_url,
            "Coinbase": s.coinbase_api_key,
            "Kraken": s.kraken_api_key,
            "Binance.US": s.binance_us_api_key,
            "Oanda": s.oanda_api_key,
            "ThetaData": s.thetadata_api_key,
            "Telegram": s.telegram_bot_token,
            "Discord": s.discord_webhook_url,
        }
        for name, val in checks.items():
            (connected if val else missing).append(name)
        log.info("api_key_status", connected=connected, missing=missing)
        await alerts.broadcast(
            "info",
            f"glitz-quant starting\n"
            f"Connected: {', '.join(connected) or 'none'}\n"
            f"Missing: {', '.join(missing) or 'none'}",
        )

    async def _run_boardroom_pipeline(
        self, signal_strategy: str, symbol: str, candles: pd.DataFrame, equity: Decimal
    ) -> None:
        """
        Full Boardroom → Chairman → Supervisor pipeline for a given symbol.
        Called when a Wolf produces a high-confidence signal that warrants LLM review.
        """
        if self.boardroom is None or self.chairman is None or self.supervisor is None:
            return

        ctx = BoardroomContext(
            symbol=symbol,
            candles=candles,
            macro_summary="",
            position_context=f"equity={equity:.2f}",
            pnl_summary=f"daily_pnl={self.oms.daily_pnl_usd:.2f}",
            urgency="medium",
        )
        minutes = await self.boardroom.convene(ctx)

        if not minutes.quorum:
            log.info("boardroom_no_quorum", symbol=symbol)
            return

        if minutes.signal and minutes.signal.direction == "skip":
            log.info("boardroom_quant_skip", symbol=symbol)
            return

        decision = await self.chairman.decide(minutes, strategy=signal_strategy)

        pos = self.oms.get_position(Venue.OANDA, symbol) if Venue.OANDA in self.oms.adapters else None
        pos_notional = pos.notional if pos else Decimal(0)
        verdict = self.supervisor.review(decision, pos_notional, equity)

        if not verdict.approved:
            log.info("supervisor_blocked", symbol=symbol, reason=verdict.reason)
            return

        if decision.signal is not None:
            await self.oms.process_signal(decision.signal)

    async def run(self) -> None:
        s = get_settings()
        log.info("orchestrator_starting", env=s.glitz_env.value, mode=s.glitz_mode.value)

        try:
            mon = get_app_config().get("monitoring", {})
            if mon.get("prometheus_enabled", True):
                metrics.start_metrics_server(int(mon.get("prometheus_port", 9100)))
        except Exception as e:
            log.warning("metrics_server_failed", err=str(e))

        self._install_signal_handlers()

        # Connect Redis Streams first (connectors need it)
        await self.streams.connect()

        await self._log_connection_status()
        self.cache, self.store, self.ingests, self.oanda, self.thetadata, self.oms = (
            await _bootstrap(self.streams)
        )
        self.strategies = _load_strategies()

        # 6 Wolves
        wolves_cfg = get_app_config().get("orchestrator", {}).get("wolves", {})
        max_wolves = int(wolves_cfg.get("max_workers", 6))
        self.wolf_pack = WolfPack(self.strategies, max_wolves=max_wolves)

        # Boardroom pipeline (only if an LLM key is configured)
        llm_cfg_ok = bool(s.anthropic_api_key or s.openrouter_api_key)
        boardroom_enabled = get_app_config().get("orchestrator", {}).get("boardroom_enabled", True)
        if llm_cfg_ok and boardroom_enabled:
            llm = LLMRouter(store=self.store)
            self.boardroom = Boardroom(llm, store=self.store)
            self.chairman = ChairmanAgent(llm)
            sup_cfg = get_app_config().get("supervisor", {})
            self.supervisor = Supervisor(
                max_position_notional_usd=float(sup_cfg.get("max_position_notional_usd", 5000)),
                min_equity_floor_usd=float(sup_cfg.get("min_equity_floor_usd", 500)),
            )
            log.info("boardroom_pipeline_enabled")
        else:
            log.warning("boardroom_pipeline_disabled", reason="no LLM key or disabled in config")

        # Recover open positions from Supabase so OMS is not blind after a crash
        await self.oms.recover_positions()

        await alerts.broadcast("info", f"glitz-quant starting in {s.glitz_mode.value} mode")

        strategy_cfg = get_strategies_config().get("strategies", {})
        orchestrator_cfg = get_app_config().get("orchestrator", {})
        tick_interval = int(orchestrator_cfg.get("tick_interval_seconds", 30))
        boardroom_min_confidence = float(orchestrator_cfg.get("boardroom_min_confidence", 0.6))
        analyst_interval = int(orchestrator_cfg.get("signal_analyst_interval_seconds", 14400))

        try:
            tick_count = 0
            while not self._stop.is_set():
                tick_count += 1
                log.info("tick_start", n=tick_count)

                if kill_switch.is_killed():
                    log.error("kill_switch_active_runner_halting")
                    await alerts.broadcast("critical", "kill switch active — runner halting")
                    break

                funding_rate = await _fetch_funding_rate()

                # Build context map for all strategies × symbols
                contexts: dict[tuple[str, str], StrategyContext] = {}
                candles_cache: dict[tuple[str, str], pd.DataFrame] = {}

                for name, strat in self.strategies.items():
                    sc = strategy_cfg.get(name, {})
                    if not sc.get("enabled"):
                        continue
                    symbols = sc.get("symbols", ["BTC-USD"])
                    tf = sc.get("timeframe", "15m")
                    venues = [
                        VENUE_BY_NAME[v]
                        for v in sc.get("venues", ["coinbase"])
                        if VENUE_BY_NAME.get(v) in self.ingests
                    ]
                    if not venues:
                        continue
                    ingest = self.ingests[venues[0]]

                    for sym in symbols:
                        log.info("tick_fetch_candles", strategy=name, sym=sym, venue=venues[0].value)
                        df = await _fetch_candles_df(ingest, sym, tf, limit=500)
                        log.info("tick_candles_received", strategy=name, sym=sym, n=len(df))
                        if df.empty:
                            continue
                        candles_cache[(name, sym)] = df

                        t = await ingest.fetch_ticker(sym)
                        if t:
                            await self.cache.set_ticker(t)  # type: ignore[union-attr]

                        venue = venues[0]
                        pos = self.oms.get_position(venue, sym)
                        equity = await self.oms.get_total_equity()
                        extra: dict[str, Any] = {}
                        if funding_rate is not None:
                            extra["funding_rate_8h"] = funding_rate

                        contexts[(name, sym)] = StrategyContext(
                            candles=df,
                            open_position_size=float(pos.size),
                            open_position_avg_price=float(pos.avg_entry_price),
                            cash_usd=float(equity),
                            recent_signals=self.recent_signals.get(name, [])[-10:],
                            extra_data=extra,
                        )

                # 6 Wolves — all strategies run in parallel
                assert self.wolf_pack is not None
                jobs = self.wolf_pack.build_jobs(strategy_cfg, contexts)
                wolf_signals = await self.wolf_pack.hunt(jobs)

                equity = await self.oms.get_total_equity()

                for sig in wolf_signals:
                    self.recent_signals.setdefault(sig.strategy, []).append(sig)
                    metrics.signals_emitted_total.labels(
                        strategy=sig.strategy, direction=sig.direction.value
                    ).inc()

                    # High-confidence signals go through the Boardroom → Chairman → Supervisor
                    if sig.confidence >= boardroom_min_confidence and self.boardroom is not None:
                        candles = candles_cache.get(
                            (sig.strategy.split(":")[-1], sig.symbol), pd.DataFrame()
                        )
                        await self._run_boardroom_pipeline(sig.strategy, sig.symbol, candles, equity)
                    else:
                        # Low-confidence: send directly to OMS without LLM overhead
                        await self.oms.process_signal(sig)

                # Circuit breakers
                staleness_events = self.oms.breakers.check_data_staleness()
                for ev in staleness_events:
                    await alerts.broadcast("warning", f"data staleness: {ev.message}")

                metrics.account_equity_usd.set(float(equity))
                metrics.account_pnl_daily_usd.set(float(self.oms.daily_pnl_usd))
                self.oms.breakers.observe_equity(equity)

                current_status = self.oms.breakers.status()
                for name, msg in current_status.items():
                    if name not in self._last_breaker_status:
                        await alerts.broadcast("critical", f"CIRCUIT BREAKER TRIPPED [{name}]: {msg}")
                        metrics.circuit_breaker_active.labels(name=name).set(1)
                for name in list(self._last_breaker_status):
                    if name not in current_status:
                        metrics.circuit_breaker_active.labels(name=name).set(0)
                self._last_breaker_status = current_status

                # Signal analyst slow loop (every 4 hours)
                if self.store is not None and (time.time() - self._last_signal_analysis_ts) >= analyst_interval:
                    self._last_signal_analysis_ts = time.time()  # update first to prevent retry-spam on failure
                    try:
                        llm = LLMRouter(store=self.store)
                        analyst = SignalAnalyst(llm=llm, store=self.store)
                        report = await analyst.run(lookback_days=30)
                        if report is not None:
                            await alerts.broadcast("info", analyst.format_broadcast(report))
                    except Exception as e:
                        log.warning("signal_analyst_failed", err=str(e))

                await asyncio.sleep(tick_interval)

        finally:
            await self._teardown()
            await alerts.broadcast("info", "glitz-quant stopped")

    async def _teardown(self) -> None:
        log.info("orchestrator_shutdown_starting")
        if self.wolf_pack:
            self.wolf_pack.shutdown()
        for ing in self.ingests.values():
            try:
                await ing.stop()
            except Exception as e:
                log.warning("ingest_stop_failed", err=str(e))
        if self.oanda:
            try:
                await self.oanda.stop()
            except Exception as e:
                log.warning("oanda_connector_stop_failed", err=str(e))
        if self.thetadata:
            try:
                await self.thetadata.stop()
            except Exception as e:
                log.warning("thetadata_connector_stop_failed", err=str(e))
        try:
            for v, a in (self.oms.adapters.items() if self.oms else []):
                await a.stop()
        except Exception as e:
            log.warning("adapter_stop_failed", err=str(e))
        if self.store is not None:
            await self.store.close()
        if self.cache is not None:
            await self.cache.close()
        await self.streams.close()
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
