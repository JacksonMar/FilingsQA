"""Public request/response models of the RAG pipeline (also the FastAPI contract)."""

from typing import Literal

from pydantic import BaseModel, Field

AnswerType = Literal[
    "answered",  # answered from the documents
    "insufficient_context",  # documents do not contain the answer
    "declined_investment_advice",  # asked for buy/sell/hold, price predictions, etc.
    "model_refusal",  # the LLM provider refused (stop_reason == "refusal")
]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    tickers: list[str] | None = Field(
        default=None, description="Restrict retrieval to these tickers; auto-detected if omitted."
    )
    fiscal_year: int | None = None
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3", "Q4"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    doc_id: int  # the [n] marker used in the answer
    chunk_id: str
    ticker: str
    form: str
    period_label: str
    section_title: str
    source_url: str


class RetrievedChunk(BaseModel):
    doc_id: int
    chunk_id: str
    score: float
    ticker: str
    form: str
    accession: str
    period_label: str
    section_key: str
    section_title: str
    content_type: str
    source_url: str
    text: str


class AnswerMeta(BaseModel):
    model: str | None = None
    prompt_version: str
    llm_cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class AskResponse(BaseModel):
    answer: str
    answer_type: AnswerType
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    applied_tickers: list[str]
    invalid_citation_ids: list[int] = Field(
        default_factory=list,
        description="Citation markers the model produced that match no retrieved document.",
    )
    meta: AnswerMeta
