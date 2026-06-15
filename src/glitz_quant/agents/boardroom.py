"""
Boardroom — runs all 4 LLM agents in parallel via asyncio.gather.

Sequential (old):  director(7.7s) + quant(7.7s) + risk(7.7s) + exec(7.7s) = 30.8s
Parallel (new):    asyncio.gather(all 4) ≈ 7.7s (longest single call)

Each agent call that raises is caught and logged; the Boardroom returns
whatever results it has so the Chairman can still decide.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from glitz_quant.agents.agents import (
    DirectorAgent,
    DirectorThesis,
    ExecutionAdvisor,
    ExecutionHint,
    QuantAgent,
    QuantSignal,
    RiskAdvisor,
    RiskAssessment,
    summarize_candles,
)
from glitz_quant.agents.llm_router import LLMRouter
from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.utils.logging import get_logger

import pandas as pd

log = get_logger(__name__)


@dataclass
class BoardroomContext:
    symbol: str
    candles: pd.DataFrame
    macro_summary: str = ""
    position_context: str = ""
    pnl_summary: str = ""
    urgency: str = "medium"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoardroomMinutes:
    symbol: str
    thesis: DirectorThesis | None
    signal: QuantSignal | None
    risk: RiskAssessment | None
    hint: ExecutionHint | None
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def quorum(self) -> bool:
        """True if at least Director + Quant both returned results."""
        return self.thesis is not None and self.signal is not None

    @property
    def vetoed(self) -> bool:
        return self.risk is not None and self.risk.veto


class Boardroom:
    """
    Convenes all 4 agents simultaneously. The Chairman reads the minutes
    and makes the final go/no-go call.
    """

    def __init__(self, llm: LLMRouter, store: SupabaseStore | None = None) -> None:
        self.director = DirectorAgent(llm)
        self.quant = QuantAgent(llm)
        self.risk_advisor = RiskAdvisor(llm)
        self.exec_advisor = ExecutionAdvisor(llm)

    async def convene(self, ctx: BoardroomContext) -> BoardroomMinutes:
        candles_summary = summarize_candles(ctx.candles) if not ctx.candles.empty else "no data"
        market_summary = f"Symbol: {ctx.symbol}\n{candles_summary}"
        signal_summary = f"Symbol: {ctx.symbol}, recent candles: {candles_summary}"

        t0 = time.perf_counter()
        results = await asyncio.gather(
            self.director.run(ctx.macro_summary or market_summary),
            self.quant.run(ctx.symbol, candles_summary, None),
            self.risk_advisor.run(signal_summary, ctx.position_context, ctx.pnl_summary),
            self.exec_advisor.run(market_summary, ctx.urgency),
            return_exceptions=True,
        )
        elapsed = time.perf_counter() - t0

        thesis, signal, risk, hint = results
        errors: list[str] = []

        def _unwrap(val: Any, label: str) -> Any:
            if isinstance(val, Exception):
                errors.append(f"{label}: {val}")
                log.warning("boardroom_agent_failed", agent=label, err=str(val))
                return None
            return val

        minutes = BoardroomMinutes(
            symbol=ctx.symbol,
            thesis=_unwrap(thesis, "director"),
            signal=_unwrap(signal, "quant"),
            risk=_unwrap(risk, "risk_advisor"),
            hint=_unwrap(hint, "exec_advisor"),
            elapsed_s=round(elapsed, 2),
            errors=errors,
        )
        log.info(
            "boardroom_convened",
            symbol=ctx.symbol,
            elapsed_s=minutes.elapsed_s,
            quorum=minutes.quorum,
            vetoed=minutes.vetoed,
            errors=len(errors),
        )
        return minutes
