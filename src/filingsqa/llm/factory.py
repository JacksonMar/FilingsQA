from typing import Literal

from filingsqa.config import Settings
from filingsqa.llm.anthropic_client import AnthropicLLM
from filingsqa.llm.base import LLMClient
from filingsqa.llm.cache import CachingLLM


def make_llm(settings: Settings, role: Literal["generator", "judge"] = "generator") -> LLMClient:
    """Build the generator or the judge LLM. They are configured independently (ADR-003)."""
    if role == "generator":
        provider, model, effort = settings.llm_provider, settings.llm_model, settings.llm_effort
    else:
        provider, model, effort = (
            settings.judge_provider,
            settings.judge_model,
            settings.judge_effort,
        )
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None

    if provider != "anthropic":  # pragma: no cover - Literal type already restricts this
        raise ValueError(f"unsupported provider {provider!r}")
    llm: LLMClient = AnthropicLLM(model=model, effort=effort, api_key=key)
    if settings.llm_cache_enabled:
        llm = CachingLLM(llm, settings.llm_cache_dir)
    return llm
