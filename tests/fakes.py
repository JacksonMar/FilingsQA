"""Deterministic test doubles: no network, no model downloads, no API calls."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import date

from filingsqa.index.embeddings import SparseVec
from filingsqa.index.qdrant_store import SearchFilter, SearchHit
from filingsqa.ingest.models import Chunk, FilingMeta
from filingsqa.llm.base import LLMRequest, LLMResponse

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bucket(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % size


class HashingDense:
    """Bag-of-words hashed into a small normalized vector: similar words -> similar vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokens(text):
            vec[_bucket(tok, self._dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashingSparse:
    def _embed(self, text: str) -> SparseVec:
        counts = Counter(_bucket(tok, 2**20) for tok in _tokens(text))
        indices = sorted(counts)
        return SparseVec(indices=indices, values=[float(counts[i]) for i in indices])

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> SparseVec:
        return self._embed(text)


class FakeLLM:
    """Returns scripted JSON text (or raises); records every request."""

    def __init__(
        self, reply: str | Callable[[LLMRequest], str] | Exception, model: str = "fake-model"
    ):
        self.reply = reply
        self.model = model
        self.requests: list[LLMRequest] = []

    @property
    def identity(self) -> dict[str, str]:
        return {"provider": "fake", "model": self.model, "effort": "low"}

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        text = self.reply(request) if callable(self.reply) else self.reply
        return LLMResponse(text=text, model=self.model, input_tokens=100, output_tokens=20)


class FakeSearchBackend:
    """Returns preset hits per ticker (or for any query); records the filters it received."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, SearchFilter, int]] = []

    def hybrid_search(
        self, query: str, flt: SearchFilter, limit: int, prefetch_limit: int
    ) -> list[SearchHit]:
        self.calls.append((query, flt, limit))
        selected = [h for h in self.hits if not flt.tickers or h.payload["ticker"] in flt.tickers]
        return selected[:limit]


def make_meta(
    ticker: str = "NVDA", fiscal_year: int = 2025, fiscal_period: str = "FY"
) -> FilingMeta:
    return FilingMeta(
        ticker=ticker,
        company=f"{ticker} Corp",
        cik=1,
        form="10-K" if fiscal_period == "FY" else "10-Q",
        accession=f"0000000001-25-{ticker}{fiscal_year}{fiscal_period}",
        filing_date=date(fiscal_year, 2, 26),
        period_end_date=date(fiscal_year, 1, 26),
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        source_url=f"https://www.sec.gov/Archives/edgar/data/1/{ticker}.htm",
    )


def make_chunk(
    ticker: str,
    text: str,
    section_key: str = "item_7",
    index: int = 0,
    content_type: str = "text",
    fiscal_year: int = 2025,
    fiscal_period: str = "FY",
) -> Chunk:
    meta = make_meta(ticker, fiscal_year, fiscal_period)
    return Chunk(
        chunk_id=f"{ticker}-{meta.accession}-{section_key}-{index:03d}",
        text=text,
        context=f"{meta.company} ({ticker}) | {meta.form} | {meta.period_label} | {section_key}",
        content_type=content_type,
        section_key=section_key,
        section_title=section_key.replace("_", " ").title(),
        chunk_index=index,
        word_count=len(text.split()),
        filing=meta,
    )


def make_hit(chunk: Chunk, score: float = 0.5) -> SearchHit:
    from filingsqa.index.qdrant_store import chunk_payload

    return SearchHit(chunk_id=chunk.chunk_id, score=score, payload=chunk_payload(chunk))
