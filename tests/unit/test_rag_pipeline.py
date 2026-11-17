import json

from filingsqa.llm.base import LLMRefusalError
from filingsqa.rag.generator import Generator
from filingsqa.rag.pipeline import NO_CONTEXT_ANSWER, RAGPipeline
from filingsqa.rag.prompts import PROMPT_VERSION
from filingsqa.rag.retriever import Retriever
from filingsqa.rag.schemas import AskRequest
from tests.fakes import FakeLLM, FakeSearchBackend, make_chunk, make_hit

HITS = [
    make_hit(
        make_chunk(
            "NVDA", "Revenue was $130,497 million.", section_key="item_8", content_type="table"
        )
    ),
    make_hit(make_chunk("AAPL", "Net sales were $391,035 million.", index=1)),
]


def _pipeline(llm, hits=HITS):
    backend = FakeSearchBackend(hits)
    return RAGPipeline(Retriever(backend, top_k=5), Generator(llm)), backend


def _reply(answer, cited, answer_type="answered"):
    return json.dumps({"answer_type": answer_type, "answer": answer, "cited_documents": cited})


def test_answer_with_citation_mapped_to_chunk_and_url():
    pipeline, _ = _pipeline(
        FakeLLM(_reply("NVIDIA revenue was $130,497 million in FY2025 [1].", [1]))
    )

    resp = pipeline.ask(AskRequest(question="What was NVIDIA's revenue?"))

    assert resp.answer_type == "answered"
    assert resp.applied_tickers == ["NVDA"]
    assert [c.chunk_id for c in resp.citations] == [HITS[0].chunk_id]
    assert resp.citations[0].source_url.startswith("https://www.sec.gov/")
    assert resp.citations[0].period_label == "FY2025"
    assert resp.meta.prompt_version == PROMPT_VERSION
    assert resp.meta.model == "fake-model"


def test_detected_ticker_becomes_filter():
    pipeline, backend = _pipeline(FakeLLM(_reply("x", [])))

    pipeline.ask(AskRequest(question="What was Apple's net sales?"))

    assert backend.calls[0][1].tickers == ("AAPL",)


def test_explicit_tickers_override_detection():
    pipeline, backend = _pipeline(FakeLLM(_reply("x", [])))

    pipeline.ask(AskRequest(question="What was Apple's net sales?", tickers=["nvda"]))

    assert backend.calls[0][1].tickers == ("NVDA",)


def test_unknown_explicit_ticker_is_not_dropped_and_yields_no_context():
    llm = FakeLLM(_reply("should not be called", []))
    pipeline, backend = _pipeline(llm)

    resp = pipeline.ask(AskRequest(question="What was revenue?", tickers=["GOOG"]))

    assert backend.calls[0][1].tickers == ("GOOG",)
    assert resp.answer_type == "insufficient_context"
    assert resp.answer == NO_CONTEXT_ANSWER
    assert llm.requests == []  # no LLM call without context


def test_fiscal_filters_are_passed_only_when_explicit():
    pipeline, backend = _pipeline(FakeLLM(_reply("x", [])))

    pipeline.ask(AskRequest(question="NVIDIA revenue in fiscal 2025?"))
    pipeline.ask(AskRequest(question="NVIDIA revenue?", fiscal_year=2025, fiscal_period="FY"))

    assert backend.calls[0][1].fiscal_years == ()
    assert backend.calls[1][1].fiscal_years == (2025,)
    assert backend.calls[1][1].fiscal_periods == ("FY",)


def test_provider_refusal_is_a_typed_answer_not_a_crash():
    pipeline, _ = _pipeline(FakeLLM(LLMRefusalError("refused")))

    resp = pipeline.ask(AskRequest(question="NVIDIA revenue?"))

    assert resp.answer_type == "model_refusal"
    assert resp.citations == []


def test_retrieved_chunks_are_returned_for_evaluation():
    pipeline, _ = _pipeline(FakeLLM(_reply("x", [])))

    resp = pipeline.ask(AskRequest(question="revenue?"))

    assert [c.doc_id for c in resp.retrieved_chunks] == [1, 2]
    assert resp.retrieved_chunks[0].content_type == "table"


def test_invalid_citations_surface_in_response():
    pipeline, _ = _pipeline(FakeLLM(_reply("x [5]", [5])))

    resp = pipeline.ask(AskRequest(question="revenue?"))

    assert resp.citations == []
    assert resp.invalid_citation_ids == [5]
