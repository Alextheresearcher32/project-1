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
    """Push secrets from Settings into both litellm attrs and os.environ.

    litellm reads some providers from its own attrs, others directly from
    os.environ. Setting both guarantees coverage regardless of litellm version.
    """
    import os
    s = get_settings()

    _env_map = {
        "ANTHROPIC_API_KEY": s.anthropic_api_key,
        "OPENAI_API_KEY": s.openai_api_key,
        "GOOGLE_API_KEY": s.google_api_key,
        "GROQ_API_KEY": s.groq_api_key,
        "XAI_API_KEY": s.xai_api_key,
        "OPENROUTER_API_KEY": s.openrouter_api_key,
    }
    for var, secret in _env_map.items():
        if secret:
            os.environ[var] = secret.get_secret_value()  # force-set, don't skip if already empty

    # litellm provider-specific attrs (legacy path — keep both)
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
        fallback_model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Plain text completion. Returns (text, metadata).

        If fallback_model is given, it is used on the second provider attempt
        instead of repeating the primary model.
        """
        s = get_settings()

        for i, provider in enumerate([self.primary, self.fallback]):
            model_name = (fallback_model if (i > 0 and fallback_model) else model) or DEFAULT_MODELS[provider]
            t0 = time.perf_counter()
            try:
                # Derive the correct API key and base URL from the model string
                api_key: str | None = None
                api_base: str | None = None
                if model_name.startswith("ollama/"):
                    api_base = "http://localhost:11434"
                elif model_name.startswith("openrouter/") and s.openrouter_api_key:
                    api_key = s.openrouter_api_key.get_secret_value()
                elif model_name.startswith("anthropic/") and s.anthropic_api_key:
                    api_key = s.anthropic_api_key.get_secret_value()
                elif model_name.startswith("groq/") and s.groq_api_key:
                    api_key = s.groq_api_key.get_secret_value()
                elif model_name.startswith("gemini/") and s.google_api_key:
                    api_key = s.google_api_key.get_secret_value()
                elif model_name.startswith("xai/") and s.xai_api_key:
                    api_key = s.xai_api_key.get_secret_value()

                resp = await litellm.acompletion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if json_mode else None,
                    **({"api_key": api_key} if api_key else {}),
                    **({"api_base": api_base} if api_base else {}),
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                cost = getattr(resp, "_response_cost", None)
                if self.store is not None:
                    try:
                        await self.store.log_agent_run(
                            agent=agent_name, provider=provider.value, model=model_name,
                            input_={"system": system[:1000], "user": user[:2000]},
                            output={"content": content[:3000]},
                            prompt_tokens=pt, completion_tokens=ct, cost_usd=cost,
                            latency_ms=latency_ms,
                        )
                    except Exception as log_err:
                        log.warning("agent_run_log_failed", agent=agent_name, err=str(log_err))
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
        fallback_model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> T:
        """Get a typed Pydantic response. Two-step: text completion + parse."""
        import json as _json

        # Build an explicit key template so local models use exact field names
        fields = schema.model_fields
        template: dict[str, Any] = {}
        for name, field in fields.items():
            ann = str(field.annotation)
            if "list" in ann.lower():
                template[name] = ["..."]
            elif "bool" in ann.lower():
                template[name] = True
            elif "float" in ann.lower() or "int" in ann.lower():
                template[name] = 0.0
            else:
                template[name] = "..."
        template_str = _json.dumps(template, indent=2)

        sys = (
            system
            + f"\n\nRespond ONLY with valid JSON using EXACTLY these field names "
            f"(do not rename, translate, or add keys):\n{template_str}\n"
            f"No prose, no markdown, no wrapper object."
        )
        text, _meta = await self.complete(
            agent_name=agent_name, system=sys, user=user, model=model,
            fallback_model=fallback_model,
            temperature=temperature, max_tokens=max_tokens, json_mode=True,
        )
        # Strip ``` fences just in case
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        # Some local models wrap the output in {"SchemaName": {...}} — unwrap it
        try:
            parsed = _json.loads(cleaned)
            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = next(iter(parsed.values()))
                if isinstance(inner, dict):
                    cleaned = _json.dumps(inner)
        except Exception:
            pass
        return schema.model_validate_json(cleaned)
