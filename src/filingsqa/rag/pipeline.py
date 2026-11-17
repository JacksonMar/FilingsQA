"""The RAG pipeline: question -> filters -> hybrid retrieval -> generation -> AskResponse."""

import logging
import time

from filingsqa.index.qdrant_store import SearchFilter, SearchHit
from filingsqa.llm.base import LLMRefusalError
from filingsqa.rag.generator import Generation, Generator
from filingsqa.rag.prompts import PROMPT_VERSION
from filingsqa.rag.query_analysis import detect_tickers
from filingsqa.rag.retriever import Retriever
from filingsqa.rag.schemas import (
    AnswerMeta,
    AskRequest,
    AskResponse,
    Citation,
    RetrievedChunk,
)

log = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the indexed SEC filings for this question."
)
MODEL_REFUSAL_ANSWER = "I can't help with this request."


def _retrieved(hits: list[SearchHit]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id=i,
            chunk_id=h.chunk_id,
            score=h.score,
            ticker=h.payload["ticker"],
            form=h.payload["form"],
            period_label=h.payload["period_label"],
            section_key=h.payload["section_key"],
            section_title=h.payload["section_title"],
            content_type=h.payload["content_type"],
            source_url=h.payload["source_url"],
            text=h.payload["text"],
        )
        for i, h in enumerate(hits, start=1)
    ]


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def build_filter(self, request: AskRequest) -> SearchFilter:
        # Explicit tickers are kept even if unknown: filtering on an unknown ticker returns no
        # chunks (-> "insufficient_context"), whereas dropping it would silently widen the
        # search to every company.
        if request.tickers:
            tickers = [t.strip().upper() for t in request.tickers]
        else:
            tickers = detect_tickers(request.question)
        return SearchFilter(
            tickers=tuple(dict.fromkeys(tickers)),
            fiscal_years=(request.fiscal_year,) if request.fiscal_year else (),
            fiscal_periods=(request.fiscal_period,) if request.fiscal_period else (),
        )

    def ask(self, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        flt = self.build_filter(request)
        hits = self.retriever.retrieve(request.question, flt, request.top_k)
        chunks = _retrieved(hits)
        meta = AnswerMeta(prompt_version=PROMPT_VERSION)

        if not hits:
            # Nothing to ground an answer on: do not even ask the LLM.
            generation = Generation(NO_CONTEXT_ANSWER, "insufficient_context", [])
        else:
            try:
                generation = self.generator.generate(request.question, hits)
            except LLMRefusalError as exc:
                log.warning("LLM refusal: %s", exc)
                generation = Generation(MODEL_REFUSAL_ANSWER, "model_refusal", [])

        if generation.llm is not None:
            meta.model = generation.llm.model
            meta.llm_cached = generation.llm.cached
            meta.input_tokens = generation.llm.input_tokens
            meta.output_tokens = generation.llm.output_tokens
        meta.latency_ms = int((time.perf_counter() - started) * 1000)

        citations = [
            Citation(
                doc_id=d,
                chunk_id=chunks[d - 1].chunk_id,
                ticker=chunks[d - 1].ticker,
                form=chunks[d - 1].form,
                period_label=chunks[d - 1].period_label,
                section_title=chunks[d - 1].section_title,
                source_url=chunks[d - 1].source_url,
            )
            for d in generation.cited_doc_ids
        ]
        return AskResponse(
            answer=generation.answer,
            answer_type=generation.answer_type,
            citations=citations,
            retrieved_chunks=chunks,
            applied_tickers=list(flt.tickers),
            invalid_citation_ids=generation.invalid_doc_ids,
            meta=meta,
        )
