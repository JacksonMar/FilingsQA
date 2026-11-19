"""Provider-agnostic LLM interface. The RAG pipeline and the eval judge depend only on this."""

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


class LLMRefusalError(LLMError):
    """The model declined to answer (stop_reason == "refusal")."""


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    # Structured output: the reply must be JSON matching this schema. None = free text
    # (used by the DeepEval judge, whose prompts ask for JSON and which parses it itself).
    json_schema: dict[str, Any] | None
    max_tokens: int = 4096

    def cache_payload(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "json_schema": self.json_schema,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str  # raw JSON text returned by the model
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    @property
    def identity(self) -> dict[str, str]:
        """Provider/model/settings that influence the output; part of the cache key."""
        ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...
