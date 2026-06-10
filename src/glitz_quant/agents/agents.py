"""
Multi-agent layer. Slow loop (minutes to hours), NOT in execution hot path.

Director: macro thesis. Big picture, daily-ish cadence.
Quant: looks at indicators + recent candles, proposes signals.
RiskAdvisor: ADVISORY ONLY. Real risk decisions are in risk/engine.py.
ExecutionAdvisor: hints on order style (limit vs limit-maker, urgency).

Each agent emits a Pydantic-typed response that the orchestrator can route.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType
from pydantic import BaseModel, Field

from glitz_quant.agents.llm_router import LLMRouter
from glitz_quant.settings import get_app_config
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


# -------- Output schemas --------
class DirectorThesis(BaseModel):
    regime: Literal["risk_on", "risk_off", "range_bound", "trending", "uncertain"]
    macro_bias: Literal["long", "short", "neutral"]
    key_observations: list[str] = Field(min_length=1, max_length=5)
    risk_factors: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class QuantSignal(BaseModel):
    symbol: str
    direction: Literal["long", "short", "flat", "skip"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    suggested_stop_pct: float | None = None
    suggested_target_pct: float | None = None


class RiskAssessment(BaseModel):
    severity: Literal["info", "warning", "error", "critical"]
    veto: bool = Field(description="If true, the OMS should skip this trade despite passing the engine.")
    reason: str
    suggested_size_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)


class ExecutionHint(BaseModel):
    order_style: Literal["aggressive", "neutral", "passive", "iceberg"]
    urgency: Literal["high", "medium", "low"]
    notes: str


# -------- Agents --------
class DirectorAgent:
    """Macro thesis. Called hourly."""
    name = "director"

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        cfg = get_app_config().get("agents", {}).get("director", {})
        self.model = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.2))

    async def run(self, recent_context: str) -> DirectorThesis:
        system = (
            "You are the Director of a small algo crypto trading desk. "
            "Synthesize the macro picture and produce a regime call. "
            "Be honest about uncertainty. Do not invent numbers. "
            "Keep observations concrete and actionable."
        )
        user = f"Recent market context:\n{recent_context}\n\nProvide your thesis."
        return await self.llm.complete_structured(
            agent_name=self.name, system=system, user=user, schema=DirectorThesis,
            model=self.model, temperature=self.temperature,
        )


class QuantAgent:
    """Looks at recent candles + indicators, proposes a directional signal."""
    name = "quant"

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        cfg = get_app_config().get("agents", {}).get("quant", {})
        self.model = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.1))

    async def run(
        self,
        symbol: str,
        candles_summary: str,
        director_thesis: DirectorThesis | None,
    ) -> QuantSignal:
        thesis_text = director_thesis.model_dump_json() if director_thesis else "{}"
        system = (
            "You are a quantitative analyst. Given a summary of recent OHLCV data, "
            "indicators, and the desk's macro thesis, propose a directional signal. "
            "If you don't have a clear edge, output direction='skip' with low confidence. "
            "Never output specific dollar amounts — only direction and percent suggestions."
        )
        user = (
            f"Symbol: {symbol}\n"
            f"Macro thesis: {thesis_text}\n"
            f"Recent data summary:\n{candles_summary}\n\n"
            "Provide your signal."
        )
        return await self.llm.complete_structured(
            agent_name=self.name, system=system, user=user, schema=QuantSignal,
            model=self.model, temperature=self.temperature,
        )


class RiskAdvisor:
    """Advisory commentary. The deterministic risk engine still has final say."""
    name = "risk_advisor"

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        cfg = get_app_config().get("agents", {}).get("risk_advisor", {})
        self.model = cfg.get("model")

    async def run(
        self, proposed_signal: str, position_context: str, recent_pnl: str,
    ) -> RiskAssessment:
        system = (
            "You are a risk advisor. Review the proposed signal against current "
            "position and recent PnL. Flag anything concerning. You can veto "
            "(veto=true) but the real risk engine has final authority. "
            "Recommend a size multiplier (0.0-1.0) if you'd reduce exposure."
        )
        user = (
            f"Proposed signal:\n{proposed_signal}\n\n"
            f"Position context:\n{position_context}\n\n"
            f"Recent PnL:\n{recent_pnl}\n"
        )
        return await self.llm.complete_structured(
            agent_name=self.name, system=system, user=user, schema=RiskAssessment,
            model=self.model, temperature=0.0,
        )


class ExecutionAdvisor:
    """Hints about order placement style."""
    name = "execution_advisor"

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        cfg = get_app_config().get("agents", {}).get("execution_advisor", {})
        self.model = cfg.get("model")

    async def run(self, market_summary: str, urgency_signal: str) -> ExecutionHint:
        system = (
            "You advise on order placement style. Output one of: aggressive (cross spread), "
            "neutral (mid), passive (limit inside), iceberg (split). Match urgency to the signal."
        )
        user = f"Market:\n{market_summary}\n\nSignal urgency:\n{urgency_signal}"
        return await self.llm.complete_structured(
            agent_name=self.name, system=system, user=user, schema=ExecutionHint,
            model=self.model, temperature=0.2,
        )


# -------- Helper to summarize candles for prompts --------
def summarize_candles(df: pd.DataFrame, n: int = 50) -> str:
    """Compact text summary of recent candles for LLM context."""
    tail = df.tail(n)
    lines = [
        f"Last close: {tail['close'].iloc[-1]:.2f}",
        f"Range over last {n} bars: low={tail['low'].min():.2f}, high={tail['high'].max():.2f}",
        f"Mean volume: {tail['volume'].mean():.2f}",
        f"Last 5 closes: {tail['close'].tail(5).round(2).tolist()}",
        f"Last 5 volumes: {tail['volume'].tail(5).round(2).tolist()}",
    ]
    return "\n".join(lines)


class CamelDebate:
    """Uses camel-ai to run a multi-agent debate on a proposed trade."""
    
    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        
    async def run_debate(self, symbol: str, signal: QuantSignal) -> str:
        """Run a debate between a bull and a bear on the proposed signal."""
        # Note: This is a high-level integration. 
        # In a real implementation, we'd use camel's ChatAgent directly.
        sys_msg_bull = BaseMessage.make_assistant_message(
            role_name="Bullish Analyst",
            content=f"Argue why {symbol} is a good buy based on: {signal.rationale}"
        )
        sys_msg_bear = BaseMessage.make_assistant_message(
            role_name="Bearish Analyst",
            content=f"Argue why {symbol} might be a trap or a poor trade right now."
        )
        
        # Simulating debate steps via LLM router for now
        log.info("starting_camel_debate", symbol=symbol)
        
        debate_transcript = []
        # Round 1: Bull
        bull_view = await self.llm.complete(
            agent_name="bull", system=sys_msg_bull.content, user="Start the debate."
        )
        debate_transcript.append(f"BULL: {bull_view}")
        
        # Round 2: Bear
        bear_view = await self.llm.complete(
            agent_name="bear", system=sys_msg_bear.content, user=f"Counter this argument: {bull_view}"
        )
        debate_transcript.append(f"BEAR: {bear_view}")
        
        return "\n\n".join(debate_transcript)
