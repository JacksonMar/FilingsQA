"""Wire the production pipeline from settings (real Qdrant, fastembed models, Anthropic LLM)."""

from qdrant_client import QdrantClient

from filingsqa.config import Settings
from filingsqa.index.embeddings import FastEmbedDense, FastEmbedSparse
from filingsqa.index.qdrant_store import QdrantStore
from filingsqa.llm.factory import make_llm
from filingsqa.rag.generator import Generator
from filingsqa.rag.pipeline import RAGPipeline
from filingsqa.rag.retriever import Retriever


def build_store(settings: Settings) -> QdrantStore:
    return QdrantStore(
        QdrantClient(url=settings.qdrant_url),
        settings.qdrant_collection,
        FastEmbedDense(settings.embedding_model),
        FastEmbedSparse(settings.sparse_model),
    )


def build_pipeline(settings: Settings) -> RAGPipeline:
    retriever = Retriever(
        build_store(settings), settings.retrieval_top_k, settings.retrieval_prefetch_k
    )
    generator = Generator(make_llm(settings, "generator"), settings.llm_max_tokens)
    return RAGPipeline(retriever, generator)
