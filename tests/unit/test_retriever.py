from filingsqa.index.qdrant_store import SearchFilter
from filingsqa.rag.retriever import Retriever
from tests.fakes import FakeSearchBackend, make_chunk, make_hit


def _backend():
    hits = [make_hit(make_chunk("AAPL", f"apple {i}", index=i)) for i in range(6)]
    hits += [make_hit(make_chunk("MSFT", f"msft {i}", index=i)) for i in range(6)]
    return FakeSearchBackend(hits)


def test_single_ticker_is_one_search_with_filter():
    backend = _backend()

    hits = Retriever(backend, top_k=4).retrieve("q", SearchFilter(tickers=("AAPL",)))

    assert len(backend.calls) == 1
    assert {h.payload["ticker"] for h in hits} == {"AAPL"}
    assert len(hits) == 4


def test_multi_ticker_searches_each_company_and_interleaves():
    backend = _backend()

    hits = Retriever(backend, top_k=4).retrieve("q", SearchFilter(tickers=("MSFT", "AAPL")))

    assert [c[1].tickers for c in backend.calls] == [("MSFT",), ("AAPL",)]
    assert [h.payload["ticker"] for h in hits] == ["MSFT", "AAPL", "MSFT", "AAPL"]


def test_multi_ticker_keeps_other_filters():
    backend = _backend()

    Retriever(backend).retrieve("q", SearchFilter(tickers=("MSFT", "AAPL"), fiscal_years=(2025,)))

    assert all(c[1].fiscal_years == (2025,) for c in backend.calls)


def test_explicit_top_k_overrides_default():
    backend = _backend()

    hits = Retriever(backend, top_k=8).retrieve("q", SearchFilter(), top_k=3)

    assert len(hits) == 3
