"""Retrieval: hybrid search with metadata filters, balanced across companies."""

import math
from typing import Protocol

from filingsqa.index.qdrant_store import SearchFilter, SearchHit


class SearchBackend(Protocol):
    def hybrid_search(
        self, query: str, flt: SearchFilter, limit: int, prefetch_limit: int
    ) -> list[SearchHit]: ...


class Retriever:
    def __init__(self, backend: SearchBackend, top_k: int = 8, prefetch_k: int = 40) -> None:
        self.backend = backend
        self.top_k = top_k
        self.prefetch_k = prefetch_k

    def retrieve(self, query: str, flt: SearchFilter, top_k: int | None = None) -> list[SearchHit]:
        """With several tickers, search each one separately and interleave the results.

        A single search over two companies can return only one company's chunks (e.g. the one
        whose filing uses the query's wording), which silently breaks comparison questions.
        """
        k = top_k or self.top_k
        if len(flt.tickers) <= 1:
            return self.backend.hybrid_search(query, flt, k, self.prefetch_k)

        per_ticker = math.ceil(k / len(flt.tickers))
        results = [
            self.backend.hybrid_search(
                query,
                SearchFilter(
                    tickers=(ticker,),
                    fiscal_years=flt.fiscal_years,
                    fiscal_periods=flt.fiscal_periods,
                    forms=flt.forms,
                    sections=flt.sections,
                ),
                per_ticker,
                self.prefetch_k,
            )
            for ticker in flt.tickers
        ]
        merged: list[SearchHit] = []
        for rank in range(per_ticker):
            merged.extend(hits[rank] for hits in results if rank < len(hits))
        return merged[:k]
