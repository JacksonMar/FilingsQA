"""Hybrid search against a real Qdrant (`make up`), with deterministic fake embedders.

The embedders are fakes, so these tests check *our* use of Qdrant (collection layout, payload
filters, RRF fusion, ID handling), not embedding quality.
"""

import uuid

import pytest
from qdrant_client import QdrantClient

from filingsqa.index.qdrant_store import QdrantStore, SearchFilter
from filingsqa.rag.retriever import Retriever
from tests.fakes import HashingDense, HashingSparse, make_chunk

pytestmark = pytest.mark.integration

CHUNKS = [
    make_chunk("NVDA", "Revenue | $130,497 | $60,922", section_key="item_8", content_type="table"),
    make_chunk(
        "NVDA",
        "Export controls on data center products could hurt demand.",
        section_key="item_1a",
        index=1,
    ),
    make_chunk(
        "NVDA",
        "Quarterly revenue was $35,082 million.",
        section_key="part1_item_2",
        fiscal_year=2026,
        fiscal_period="Q3",
    ),
    make_chunk(
        "AAPL", "Net sales | $391,035 | $383,285", section_key="item_8", content_type="table"
    ),
    make_chunk(
        "AAPL",
        "The Company depends on single-source component suppliers.",
        section_key="item_1a",
        index=1,
    ),
    make_chunk("MSFT", "Azure and other cloud services revenue grew 33%.", section_key="item_7"),
]


@pytest.fixture(scope="module")
def store():
    client = QdrantClient(url="http://localhost:6333", timeout=5)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant is not running (make up)")
    s = QdrantStore(client, f"test_{uuid.uuid4().hex[:8]}", HashingDense(), HashingSparse())
    s.recreate_collection()
    s.index(CHUNKS, batch_size=4)
    yield s
    client.delete_collection(s.collection)


def _ids(hits):
    return [h.chunk_id for h in hits]


def test_all_chunks_indexed_and_reindex_is_idempotent(store):
    store.index(CHUNKS)  # same IDs -> upsert, not duplicates

    assert store.client.count(store.collection).count == len(CHUNKS)


def test_ticker_filter_is_strict(store):
    hits = store.hybrid_search(
        "revenue", SearchFilter(tickers=("AAPL",)), limit=10, prefetch_limit=20
    )

    assert hits
    assert {h.payload["ticker"] for h in hits} == {"AAPL"}


def test_lexical_match_ranks_first(store):
    hits = store.hybrid_search(
        "single-source component suppliers", SearchFilter(), limit=3, prefetch_limit=20
    )

    assert hits[0].chunk_id == CHUNKS[4].chunk_id


def test_fiscal_period_filter(store):
    hits = store.hybrid_search(
        "revenue", SearchFilter(tickers=("NVDA",), fiscal_periods=("Q3",)), 10, 20
    )

    assert _ids(hits) == [CHUNKS[2].chunk_id]


def test_unknown_ticker_returns_nothing(store):
    assert store.hybrid_search("revenue", SearchFilter(tickers=("GOOG",)), 10, 20) == []


def test_comparison_retrieval_returns_both_companies(store):
    hits = Retriever(store, top_k=4, prefetch_k=20).retrieve(
        "net sales revenue", SearchFilter(tickers=("AAPL", "NVDA"))
    )

    assert {h.payload["ticker"] for h in hits} == {"AAPL", "NVDA"}


def test_payload_round_trips(store):
    hit = store.hybrid_search("Azure cloud", SearchFilter(tickers=("MSFT",)), 1, 20)[0]

    assert hit.payload["text"] == CHUNKS[5].text
    assert hit.payload["source_url"] == CHUNKS[5].filing.source_url
