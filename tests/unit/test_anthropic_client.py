"""AnthropicLLM with the SDK call stubbed: checks what we send and how we read replies."""

from types import SimpleNamespace

import pytest

from filingsqa.llm.anthropic_client import AnthropicLLM
from filingsqa.llm.base import LLMError, LLMRefusalError, LLMRequest

REQ = LLMRequest(system="sys", user="hello", json_schema={"type": "object"}, max_tokens=500)


def _response(stop_reason="end_turn", text='{"ok": true}'):
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=content if text is not None else [],
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )


@pytest.fixture
def llm_and_calls(monkeypatch):
    llm = AnthropicLLM(model="claude-haiku-5-5", effort="low", api_key="test-key")
    calls: list[dict] = []
    replies: list = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(llm._client.messages, "create", fake_create)
    return llm, calls, replies


def test_sends_structured_output_and_effort_but_no_sampling_params(llm_and_calls):
    llm, calls, replies = llm_and_calls
    replies.append(_response())

    resp = llm.complete(REQ)

    sent = calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert sent["max_tokens"] == 500
    assert "temperature" not in sent
    assert resp.text == '{"ok": true}'  # thinking block skipped
    assert (resp.input_tokens, resp.output_tokens) == (10, 3)


def test_refusal_raises_typed_error(llm_and_calls):
    llm, _, replies = llm_and_calls
    replies.append(_response(stop_reason="refusal", text=None))

    with pytest.raises(LLMRefusalError):
        llm.complete(REQ)


def test_truncation_raises(llm_and_calls):
    llm, _, replies = llm_and_calls
    replies.append(_response(stop_reason="max_tokens"))

    with pytest.raises(LLMError, match="max_tokens"):
        llm.complete(REQ)


def test_identity_feeds_cache_key():
    assert AnthropicLLM("m", "low", api_key="k").identity == {
        "provider": "anthropic",
        "model": "m",
        "effort": "low",
    }


def test_free_text_request_has_no_format(llm_and_calls):
    llm, calls, replies = llm_and_calls
    replies.append(_response(text="plain"))

    resp = llm.complete(LLMRequest(system="s", user="u", json_schema=None))

    assert calls[0]["output_config"] == {"effort": "low"}
    assert resp.text == "plain"
