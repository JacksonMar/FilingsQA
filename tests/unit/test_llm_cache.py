import json

from filingsqa.llm.base import LLMRequest
from filingsqa.llm.cache import CachingLLM, cache_key
from tests.fakes import FakeLLM

REQ = LLMRequest(system="sys", user="question", json_schema={"type": "object"})


def test_second_identical_call_is_served_from_disk(tmp_path):
    inner = FakeLLM(json.dumps({"a": 1}))
    llm = CachingLLM(inner, tmp_path)

    first = llm.complete(REQ)
    second = llm.complete(REQ)

    assert len(inner.requests) == 1
    assert (llm.hits, llm.misses) == (1, 1)
    assert not first.cached
    assert second.cached
    assert second.text == first.text


def test_cache_survives_new_instance(tmp_path):
    CachingLLM(FakeLLM("x"), tmp_path).complete(REQ)
    inner = FakeLLM("y")

    assert CachingLLM(inner, tmp_path).complete(REQ).text == "x"
    assert inner.requests == []


def test_key_changes_with_prompt_model_or_schema():
    base = cache_key({"model": "a"}, REQ)

    assert cache_key({"model": "b"}, REQ) != base
    assert cache_key({"model": "a"}, LLMRequest("sys2", "question", {"type": "object"})) != base
    assert cache_key({"model": "a"}, LLMRequest("sys", "question", {"type": "array"})) != base
    assert cache_key({"model": "a"}, REQ) == base
