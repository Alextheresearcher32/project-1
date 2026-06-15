"""
WolfPack — 6 parallel strategy workers backed by ThreadPoolExecutor.

Each "wolf" is one (strategy, symbol) job dispatched to a thread.
Strategy.on_candle() is synchronous Python, so running 6 at once via
the thread pool gives genuine wall-clock parallelism for I/O-bound
operations (numpy, pandas slicing) and prevents one slow strategy from
blocking the others.

Usage:
    pack = WolfPack(strategies, max_wolves=6)
    signals = await pack.hunt(jobs)
    pack.shutdown()
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from glitz_quant.strategies.base import Strategy, StrategyContext
from glitz_quant.data.types import Signal
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class WolfJob:
    strategy_name: str
    strategy: Strategy
    symbol: str
    context: StrategyContext


def _run_wolf(job: WolfJob) -> Signal | None:
    """Executed inside a thread. Must not call asyncio."""
    try:
        return job.strategy.on_candle(job.context)
    except Exception as e:
        log.warning("wolf_strategy_error", strategy=job.strategy_name, symbol=job.symbol, err=str(e))
        return None


class WolfPack:
    """
    Dispatches strategy jobs to a ThreadPoolExecutor(max_workers=6) and
    awaits all results with asyncio.gather.
    """

    def __init__(
        self,
        strategies: dict[str, Strategy],
        max_wolves: int = 6,
    ) -> None:
        self._strategies = strategies
        self._pool = ThreadPoolExecutor(
            max_workers=max_wolves,
            thread_name_prefix="wolf",
        )

    async def hunt(self, jobs: list[WolfJob]) -> list[Signal]:
        """
        Run all jobs in parallel. Returns list of non-None signals.
        """
        if not jobs:
            return []

        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self._pool, _run_wolf, job)
            for job in jobs
        ]
        results: list[Any] = await asyncio.gather(*futures, return_exceptions=True)

        signals: list[Signal] = []
        for job, result in zip(jobs, results):
            if isinstance(result, Exception):
                log.warning(
                    "wolf_future_error",
                    strategy=job.strategy_name,
                    symbol=job.symbol,
                    err=str(result),
                )
            elif result is not None:
                signals.append(result)
                log.info(
                    "wolf_signal",
                    strategy=job.strategy_name,
                    symbol=job.symbol,
                    direction=result.direction.value,
                    confidence=result.confidence,
                )
        return signals

    def build_jobs(
        self,
        strategy_cfg: dict[str, dict],
        contexts: dict[tuple[str, str], StrategyContext],
    ) -> list[WolfJob]:
        """
        Helper: build the job list for the current tick from the strategy
        config and the pre-fetched context map.

        strategy_cfg: from get_strategies_config()["strategies"]
        contexts: {(strategy_name, symbol): StrategyContext}
        """
        jobs: list[WolfJob] = []
        for name, strat in self._strategies.items():
            sc = strategy_cfg.get(name, {})
            if not sc.get("enabled"):
                continue
            for sym in sc.get("symbols", []):
                ctx = contexts.get((name, sym))
                if ctx is not None:
                    jobs.append(WolfJob(
                        strategy_name=name,
                        strategy=strat,
                        symbol=sym,
                        context=ctx,
                    ))
        return jobs

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        log.info("wolf_pack_shutdown")
