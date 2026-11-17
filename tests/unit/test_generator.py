import json

import pytest

from filingsqa.llm.base import LLMError
from filingsqa.rag.generator import Generator
from filingsqa.rag.prompts import ANSWER_SCHEMA
from tests.fakes import FakeLLM, make_chunk, make_hit

HITS = [make_hit(make_chunk("NVDA", f"chunk {i}", index=i)) for i in range(3)]


def _reply(answer: str, cited: list[int], answer_type: str = "answered") -> str:
    return json.dumps({"answer_type": answer_type, "answer": answer, "cited_documents": cited})


def test_parses_answer_and_citations():
    llm = FakeLLM(_reply("Revenue was $130,497 million in fiscal 2025 [2].", [2]))

    gen = Generator(llm).generate("q", HITS)

    assert gen.answer_type == "answered"
    assert gen.cited_doc_ids == [2]
    assert gen.invalid_doc_ids == []


def test_inline_markers_and_field_are_merged_in_order():
    gen = Generator(FakeLLM(_reply("A [3]. B [1].", [1, 2]))).generate("q", HITS)

    assert gen.cited_doc_ids == [3, 1, 2]


def test_out_of_range_citations_are_reported_not_trusted():
    gen = Generator(FakeLLM(_reply("A [7].", [0, 1]))).generate("q", HITS)

    assert gen.cited_doc_ids == [1]
    assert gen.invalid_doc_ids == [7, 0]


def test_request_uses_structured_output_schema():
    llm = FakeLLM(_reply("x", []))

    Generator(llm, max_tokens=1234).generate("What was revenue?", HITS)

    req = llm.requests[0]
    assert req.json_schema == ANSWER_SCHEMA
    assert req.max_tokens == 1234
    assert "What was revenue?" in req.user


@pytest.mark.parametrize(
    "bad", ["not json", json.dumps({"answer": "x"}), _reply("x", [1], answer_type="maybe")]
)
def test_invalid_llm_output_raises(bad):
    with pytest.raises(LLMError):
        Generator(FakeLLM(bad)).generate("q", HITS)
