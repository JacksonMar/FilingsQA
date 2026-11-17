"""Embedding interfaces and the fastembed implementation (local models, no API cost).

Dense: BAAI/bge-small-en-v1.5 (384-dim). Queries and documents are embedded differently:
bge expects an instruction prefix on queries, which fastembed's `query_embed` adds.
Sparse: Qdrant/bm25 produces term-frequency vectors; Qdrant applies IDF at query time
(`Modifier.IDF`), which together gives BM25 scoring. See ADR-014.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class DenseEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...


class FastEmbedDense:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # heavy import, only when actually used

        self._model = TextEmbedding(model_name)
        self._dim = len(next(iter(self._model.embed(["probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


class FastEmbedSparse:
    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)

    @staticmethod
    def _convert(emb: object) -> SparseVec:
        return SparseVec(
            indices=[int(i) for i in emb.indices],  # type: ignore[attr-defined]
            values=[float(v) for v in emb.values],  # type: ignore[attr-defined]
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVec]:
        return [self._convert(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> SparseVec:
        return self._convert(next(iter(self._model.query_embed(text))))
