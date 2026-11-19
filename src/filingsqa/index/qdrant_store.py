"""Qdrant collection layout, indexing and hybrid search. See ADR-014.

One point per chunk with two named vectors:
- "dense": bge-small embedding of `context + text`, cosine distance;
- "bm25": sparse term vector, IDF applied by Qdrant at query time.
Hybrid search = two prefetch queries (dense, sparse) fused with Reciprocal Rank Fusion inside
Qdrant. Metadata filters (ticker, fiscal year/period, form, section) are payload filters with
payload indexes so they stay fast.
"""

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from filingsqa.index.embeddings import DenseEmbedder, SparseEmbedder
from filingsqa.ingest.models import Chunk

DENSE = "dense"
SPARSE = "bm25"
KEYWORD_INDEXES = ("ticker", "form", "fiscal_period", "section_key", "accession", "chunk_id")
INTEGER_INDEXES = ("fiscal_year",)
_ID_NAMESPACE = uuid.UUID("6f1c3a2e-6d0b-4f43-9a55-5b0f3f2d9a10")


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUID/int IDs; derive a stable UUID from the human-readable chunk ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    f = chunk.filing
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "context": chunk.context,
        "content_type": chunk.content_type,
        "section_key": chunk.section_key,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "ticker": f.ticker,
        "company": f.company,
        "form": f.form,
        "accession": f.accession,
        "filing_date": f.filing_date.isoformat(),
        "period_end_date": f.period_end_date.isoformat(),
        "fiscal_year": f.fiscal_year,
        "fiscal_period": f.fiscal_period,
        "period_label": f.period_label,
        "source_url": f.source_url,
    }


@dataclass(frozen=True)
class SearchFilter:
    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()

    def to_qdrant(self) -> models.Filter | None:
        conditions: list[models.Condition] = []
        for key, values in (
            ("ticker", self.tickers),
            ("fiscal_year", self.fiscal_years),
            ("fiscal_period", self.fiscal_periods),
            ("form", self.forms),
            ("section_key", self.sections),
        ):
            if values:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
                )
        return models.Filter(must=conditions) if conditions else None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def _batches(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense = dense
        self.sparse = sparse

    def recreate_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dense.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        for field in INTEGER_INDEXES:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.INTEGER
            )

    def index(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        count = 0
        for batch in _batches(list(chunks), batch_size):
            texts = [c.embedding_text for c in batch]
            dense = self.dense.embed_documents(texts)
            sparse = self.sparse.embed_documents(texts)
            points = [
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={
                        DENSE: d,
                        SPARSE: models.SparseVector(indices=s.indices, values=s.values),
                    },
                    payload=chunk_payload(c),
                )
                for c, d, s in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            count += len(points)
        return count

    def hybrid_search(
        self, query: str, flt: SearchFilter, limit: int, prefetch_limit: int
    ) -> list[SearchHit]:
        qfilter = flt.to_qdrant()
        sparse = self.sparse.embed_query(query)
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(
                    query=self.dense.embed_query(query),
                    using=DENSE,
                    limit=prefetch_limit,
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=prefetch_limit,
                    filter=qfilter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(chunk_id=str(p.payload["chunk_id"]), score=p.score, payload=dict(p.payload))
            for p in result.points
            if p.payload is not None
        ]
