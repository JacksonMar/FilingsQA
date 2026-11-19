import json

import pytest
from fastapi.testclient import TestClient

from filingsqa.api.app import create_app
from filingsqa.llm.base import LLMError
from filingsqa.rag.generator import Generator
from filingsqa.rag.pipeline import RAGPipeline
from filingsqa.rag.retriever import Retriever
from tests.fakes import FakeLLM, FakeSearchBackend, make_chunk, make_hit


def _client(llm):
    backend = FakeSearchBackend([make_hit(make_chunk("NVDA", "Revenue was $130,497 million."))])
    app = create_app(lambda: RAGPipeline(Retriever(backend), Generator(llm)))
    return TestClient(app)


@pytest.fixture
def ok_llm():
    return FakeLLM(
        json.dumps(
            {
                "answer_type": "answered",
                "answer": "It was $130,497 million [1].",
                "cited_documents": [1],
            }
        )
    )


def test_health(ok_llm):
    with _client(ok_llm) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_ask_returns_structured_answer(ok_llm):
    with _client(ok_llm) as client:
        resp = client.post("/ask", json={"question": "What was NVIDIA's revenue?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_type"] == "answered"
    assert body["citations"][0]["chunk_id"].startswith("NVDA-")
    assert body["retrieved_chunks"][0]["text"] == "Revenue was $130,497 million."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "ok?", "fiscal_period": "H1"},
        {"question": "ok?", "top_k": 0},
    ],
)
def test_invalid_requests_are_rejected_with_422(ok_llm, payload):
    with _client(ok_llm) as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_maps_to_502():
    with _client(FakeLLM(LLMError("boom"))) as client:
        resp = client.post("/ask", json={"question": "NVIDIA revenue?"})

    assert resp.status_code == 502
