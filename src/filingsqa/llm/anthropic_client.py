"""Anthropic implementation of LLMClient, using structured outputs (JSON schema)."""

from typing import Any

import anthropic

from filingsqa.llm.base import LLMError, LLMRefusalError, LLMRequest, LLMResponse


class AnthropicLLM:
    def __init__(self, model: str, effort: str, api_key: str | None = None) -> None:
        # api_key=None lets the SDK resolve credentials from the environment.
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._effort = effort

    @property
    def identity(self) -> dict[str, str]:
        return {"provider": "anthropic", "model": self._model, "effort": self._effort}

    def complete(self, request: LLMRequest) -> LLMResponse:
        output_config: Any = {"effort": self._effort}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config=output_config,
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"model refused (stop_details={response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError(f"no text block in response (stop_reason={response.stop_reason})")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
