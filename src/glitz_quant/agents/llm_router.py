"""
LLM router. Single interface for Anthropic, Google, Groq, xAI, OpenRouter.
Uses litellm for provider abstraction and instructor for structured output.

All calls are logged to the agent_runs table (latency, tokens, cost, output).
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

import litellm  # type: ignore[import-untyped]
from pydantic import BaseModel

from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.settings import LLMProvider, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# Provider-specific model defaults. Override in code or settings.yaml.
DEFAULT_MODELS = {
    LLMProvider.ANTHROPIC: "anthropic/claude-opus-4-7",
    LLMProvider.GOOGLE: "gemini/gemini-2.0-flash",
    LLMProvider.GROQ: "groq/llama-3.3-70b-versatile",
    LLMProvider.XAI: "xai/grok-2",
    LLMProvider.OPENROUTER: "openrouter/anthropic/claude-opus-4-7",
}


def _configure_litellm() -> None:
    s = get_settings()
    if s.anthropic_api_key:
        litellm.anthropic_key = s.anthropic_api_key.get_secret_value()
    if s.openai_api_key:
        litellm.openai_key = s.openai_api_key.get_secret_value()
    if s.google_api_key:
        litellm.google_key = s.google_api_key.get_secret_value()
    if s.groq_api_key:
        litellm.groq_key = s.groq_api_key.get_secret_value()
    if s.xai_api_key:
        litellm.xai_key = s.xai_api_key.get_secret_value()
    if s.openrouter_api_key:
        litellm.openrouter_key = s.openrouter_api_key.get_secret_value()


_configure_litellm()


class LLMRouter:
    """Routes to a primary provider with optional fallback."""

    def __init__(self, store: SupabaseStore | None = None) -> None:
        self.store = store
        s = get_settings()
        self.primary = s.llm_primary
        self.fallback = s.llm_fallback

    async def complete(
        self,
        agent_name: str,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Plain text completion. Returns (text, metadata)."""
        for provider in [self.primary, self.fallback]:
            model_name = model or DEFAULT_MODELS[provider]
            t0 = time.perf_counter()
            try:
                resp = await litellm.acompletion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if json_mode else None,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                cost = getattr(resp, "_response_cost", None)
                if self.store is not None:
                    await self.store.log_agent_run(
                        agent=agent_name, provider=provider.value, model=model_name,
                        input_={"system": system[:1000], "user": user[:2000]},
                        output={"content": content[:3000]},
                        prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
                        latency_ms=latency_ms,
                    )
                return content, {"provider": provider.value, "model": model_name,
                                 "prompt_tokens": pt, "completion_tokens": ct, "latency_ms": latency_ms}
            except Exception as e:
                log.warning("llm_call_failed", provider=provider.value, agent=agent_name, err=str(e))
                if self.store is not None:
                    try:
                        await self.store.log_agent_run(
                            agent=agent_name, provider=provider.value, model=model_name,
                            input_={"system": system[:1000], "user": user[:2000]},
                            error=str(e),
                        )
                    except Exception:
                        pass
                continue

        raise RuntimeError(f"all LLM providers failed for agent={agent_name}")

    async def complete_structured(
        self,
        agent_name: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> T:
        """Get a typed Pydantic response. Two-step: text completion + parse."""
        schema_json = schema.model_json_schema()
        sys = (
            system
            + "\n\nRespond ONLY with valid JSON matching this schema. "
            + f"No prose, no markdown. Schema:\n{schema_json}"
        )
        text, _meta = await self.complete(
            agent_name=agent_name, system=sys, user=user, model=model,
            temperature=temperature, max_tokens=max_tokens, json_mode=True,
        )
        # Strip ``` fences just in case
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return schema.model_validate_json(cleaned)
