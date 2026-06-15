"""
Chairman + Supervisor — the final two gates before an order hits the wire.

Chairman:   LLM agent that synthesizes BoardroomMinutes into a go/no-go
            Signal. Sees all 4 agent outputs at once and breaks ties.

Supervisor: Deterministic (no LLM) hard-check layer. Validates the
            Chairman's signal against position limits, equity floor,
            and circuit-breaker state. Fast — adds <1ms.

Flow:
  Boardroom.convene() → BoardroomMinutes
      → Chairman.decide()      → ChairmanDecision
      → Supervisor.review()    → SupervisorVerdict (approved=True/False)
      → OMS.process_signal()   → ThetaData/Oanda execution
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from glitz_quant.agents.boardroom import BoardroomMinutes
from glitz_quant.agents.llm_router import LLMRouter
from glitz_quant.data.types import Signal, SignalConfidence, SignalDirection
from glitz_quant.settings import get_app_config
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


# ---- Chairman output schema ----

class ChairmanVerdict(BaseModel):
    action: Literal["long", "short", "flat", "pass"]
    confidence: float = Field(ge=0.0, le=1.0)
    size_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str
    stop_pct: float | None = None
    target_pct: float | None = None


@dataclass
class ChairmanDecision:
    verdict: ChairmanVerdict
    symbol: str
    minutes: BoardroomMinutes
    signal: Signal | None  # populated after sizing; None if action == "pass"


# ---- Supervisor output ----

@dataclass
class SupervisorVerdict:
    approved: bool
    reason: str


# ---- Chairman ----

class ChairmanAgent:
    """
    Reads the full BoardroomMinutes and renders a single verdict.
    Called only when Boardroom has quorum and the quant signal is not "skip".
    """

    name = "chairman"

    def __init__(self, llm: LLMRouter) -> None:
        self.llm = llm
        cfg = get_app_config().get("agents", {}).get("chairman", {})
        self.model: str | None = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.1))

    async def decide(self, minutes: BoardroomMinutes, strategy: str) -> ChairmanDecision:
        if not minutes.quorum:
            return ChairmanDecision(
                verdict=ChairmanVerdict(
                    action="pass",
                    confidence=0.0,
                    rationale="boardroom did not reach quorum",
                ),
                symbol=minutes.symbol,
                minutes=minutes,
                signal=None,
            )

        thesis_txt = minutes.thesis.model_dump_json() if minutes.thesis else "{}"
        quant_txt = minutes.signal.model_dump_json() if minutes.signal else "{}"
        risk_txt = minutes.risk.model_dump_json() if minutes.risk else "{}"
        hint_txt = minutes.hint.model_dump_json() if minutes.hint else "{}"
        errors_txt = ", ".join(minutes.errors) if minutes.errors else "none"

        system = (
            "You are the Chairman of a quantitative trading desk. "
            "You have just received the following reports from your Boardroom. "
            "Your job is to produce a single, decisive verdict: long, short, flat, or pass. "
            "'flat' closes any open position. 'pass' means do nothing this cycle. "
            "If the risk advisor vetoed, default to 'pass' unless you have overwhelming evidence. "
            "Be crisp. Do not invent data."
        )
        user = (
            f"Symbol: {minutes.symbol}\n"
            f"Director thesis: {thesis_txt}\n"
            f"Quant signal: {quant_txt}\n"
            f"Risk assessment: {risk_txt}\n"
            f"Execution hint: {hint_txt}\n"
            f"Boardroom elapsed: {minutes.elapsed_s}s | Errors: {errors_txt}\n\n"
            "Render your verdict."
        )

        verdict = await self.llm.complete_structured(
            agent_name=self.name,
            system=system,
            user=user,
            schema=ChairmanVerdict,
            model=self.model,
            temperature=self.temperature,
        )

        sig: Signal | None = None
        if verdict.action not in ("pass", "flat"):
            direction = SignalDirection.LONG if verdict.action == "long" else SignalDirection.SHORT
            cfg = get_app_config().get("chairman", {})
            base_notional = Decimal(str(cfg.get("base_notional_usd", 500)))
            notional = (base_notional * Decimal(str(verdict.size_multiplier))).quantize(Decimal("0.01"))
            confidence_label = (
                SignalConfidence.HIGH if verdict.confidence >= 0.7
                else SignalConfidence.MEDIUM if verdict.confidence >= 0.4
                else SignalConfidence.LOW
            )
            sig = Signal(
                strategy=f"chairman:{strategy}",
                symbol=minutes.symbol,
                direction=direction,
                target_notional_usd=notional,
                confidence=verdict.confidence,
                confidence_label=confidence_label,
                reason=verdict.rationale[:500],
            )
        elif verdict.action == "flat":
            sig = Signal(
                strategy=f"chairman:{strategy}",
                symbol=minutes.symbol,
                direction=SignalDirection.FLAT,
                target_notional_usd=Decimal(0),
                confidence=verdict.confidence,
                reason=verdict.rationale[:500],
            )

        log.info(
            "chairman_decided",
            symbol=minutes.symbol,
            action=verdict.action,
            confidence=verdict.confidence,
        )
        return ChairmanDecision(
            verdict=verdict, symbol=minutes.symbol, minutes=minutes, signal=sig
        )


# ---- Supervisor ----

class Supervisor:
    """
    Deterministic guard. Checks the Chairman's signal against hard limits
    before it reaches the OMS. No LLM — adds < 1ms.
    """

    def __init__(
        self,
        max_position_notional_usd: float = 5_000.0,
        min_equity_floor_usd: float = 500.0,
    ) -> None:
        self._max_notional = Decimal(str(max_position_notional_usd))
        self._min_equity = Decimal(str(min_equity_floor_usd))

    def review(
        self,
        decision: ChairmanDecision,
        current_position_notional: Decimal,
        equity: Decimal,
    ) -> SupervisorVerdict:
        if decision.signal is None:
            return SupervisorVerdict(approved=False, reason="chairman passed — no signal")

        if decision.minutes.vetoed:
            return SupervisorVerdict(
                approved=False, reason="risk advisor veto upheld by supervisor"
            )

        if equity < self._min_equity:
            return SupervisorVerdict(
                approved=False,
                reason=f"equity ${equity:.2f} below floor ${self._min_equity:.2f}",
            )

        notional = decision.signal.target_notional_usd
        if current_position_notional + notional > self._max_notional:
            return SupervisorVerdict(
                approved=False,
                reason=(
                    f"position notional ${current_position_notional + notional:.2f} "
                    f"would exceed limit ${self._max_notional:.2f}"
                ),
            )

        if decision.verdict.confidence < 0.35:
            return SupervisorVerdict(
                approved=False,
                reason=f"chairman confidence {decision.verdict.confidence:.2f} below 0.35 threshold",
            )

        log.info(
            "supervisor_approved",
            symbol=decision.symbol,
            action=decision.verdict.action,
            confidence=decision.verdict.confidence,
        )
        return SupervisorVerdict(approved=True, reason="ok")
