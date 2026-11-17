"""Answer generation: prompt -> LLM (structured JSON) -> validated answer with citations."""

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from filingsqa.index.qdrant_store import SearchHit
from filingsqa.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse
from filingsqa.rag.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_message
from filingsqa.rag.schemas import AnswerType

_MARKER = re.compile(r"\[(\d{1,3})\]")


class _LLMAnswer(BaseModel):
    answer_type: Literal["answered", "insufficient_context", "declined_investment_advice"]
    answer: str
    cited_documents: list[int]


@dataclass
class Generation:
    answer: str
    answer_type: AnswerType
    cited_doc_ids: list[int]  # valid ids only, in order of first appearance
    invalid_doc_ids: list[int] = field(default_factory=list)
    llm: LLMResponse | None = None


def collect_citations(answer: _LLMAnswer, n_docs: int) -> tuple[list[int], list[int]]:
    """Union of inline [n] markers and the cited_documents field, split into valid/invalid."""
    ordered: list[int] = []
    for doc_id in [int(m) for m in _MARKER.findall(answer.answer)] + answer.cited_documents:
        if doc_id not in ordered:
            ordered.append(doc_id)
    valid = [d for d in ordered if 1 <= d <= n_docs]
    invalid = [d for d in ordered if not 1 <= d <= n_docs]
    return valid, invalid


class Generator:
    def __init__(self, llm: LLMClient, max_tokens: int = 4096) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[SearchHit]) -> Generation:
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_user_message(question, hits),
            json_schema=ANSWER_SCHEMA,
            max_tokens=self.max_tokens,
        )
        response = self.llm.complete(request)
        try:
            parsed = _LLMAnswer.model_validate(json.loads(response.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(f"LLM returned invalid JSON answer: {exc}") from exc
        valid, invalid = collect_citations(parsed, len(hits))
        return Generation(parsed.answer, parsed.answer_type, valid, invalid, response)
