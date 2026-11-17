"""Golden dataset schema, loading and saving. See ADR-019.

One JSONL line = one `GoldenItem`. Items have stable IDs and are never edited in place in a
released version: a change produces a new dataset version (golden_v2.jsonl, ...).
"""

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["numeric", "qualitative", "comparison", "temporal", "adversarial"]
Difficulty = Literal["easy", "medium", "hard"]
# What a correct system does. Mirrors AskResponse.answer_type of the bot.
ExpectedBehavior = Literal["answer", "insufficient_context", "decline_advice"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class XbrlRef(_Strict):
    """Where an expected number comes from, so every value is traceable to SEC data."""

    taxonomy: str  # "us-gaap"
    tag: str  # "Revenues"
    unit: str  # "USD", "USD/shares"
    accession: str  # filing that reported the fact
    start: date | None  # None for instant facts (balance sheet)
    end: date


class ExpectedNumber(_Strict):
    label: str  # "revenue"
    ticker: str
    value: float  # in base units: USD, USD/share, shares
    unit: str
    fiscal_year: int
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3", "Q4"]
    period_end: date
    tolerance_rel: float = Field(default=0.01, ge=0)  # 1%: allows "$130.5 billion" for 130,497M
    tolerance_abs: float = Field(default=0.0, ge=0)  # e.g. 0.01 for EPS
    xbrl: XbrlRef


class ExpectedSource(_Strict):
    """A filing section that should be retrieved/cited. Accession is resolved from the manifest."""

    ticker: str
    form: Literal["10-K", "10-Q"]
    section_keys: tuple[str, ...]  # any of these counts as a hit
    accession: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None


class InjectedDocument(_Strict):
    """For prompt-injection items: a poisoned chunk the eval runner puts into the context."""

    ticker: str
    section_key: str = "item_7"
    text: str


class AskParams(_Strict):
    tickers: tuple[str, ...] | None = None
    fiscal_year: int | None = None
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3", "Q4"] | None = None


class GoldenItem(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    category: Category
    subcategory: str
    difficulty: Difficulty
    question: str = Field(min_length=5)
    expected_behavior: ExpectedBehavior
    expected_answer: str | None = None  # reference answer for LLM-judged metrics
    expected_numbers: tuple[ExpectedNumber, ...] = ()
    expected_winner: str | None = None  # comparison items: ticker that should win
    key_facts: tuple[str, ...] = ()  # facts a good answer must state (qualitative)
    forbidden_phrases: tuple[str, ...] = ()  # e.g. injected payloads that must not appear
    expected_sources: tuple[ExpectedSource, ...] = ()
    ask: AskParams = AskParams()
    injected_document: InjectedDocument | None = None
    verified: bool = False  # hand-written content checked against the real filing text
    source: Literal["generated", "handwritten"]
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if self.expected_behavior == "answer" and not (
            self.expected_numbers or self.key_facts or self.expected_answer
        ):
            raise ValueError(f"{self.id}: an answerable item needs numbers, key facts or an answer")
        if self.subcategory == "prompt_injection" and self.injected_document is None:
            raise ValueError(f"{self.id}: prompt_injection items need injected_document")
        if self.expected_winner and self.category != "comparison":
            raise ValueError(f"{self.id}: expected_winner only makes sense for comparisons")
        if self.source == "generated" and not self.verified:
            raise ValueError(f"{self.id}: generated items come from XBRL and must be verified")
        return self


class DatasetMeta(BaseModel):
    version: str
    created: date
    manifest_as_of: date | None
    n_items: int
    by_category: dict[str, int]
    by_behavior: dict[str, int]
    sha256: str  # of the JSONL file; detects silent edits of a released version


def load_items(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items


def dump_items(items: list[GoldenItem], path: Path) -> str:
    """Write JSONL deterministically (sorted by id) and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item.model_dump(mode="json", exclude_defaults=True), sort_keys=True)
        for item in sorted(items, key=lambda i: i.id)
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_meta(
    items: list[GoldenItem], version: str, sha: str, manifest_as_of: date | None
) -> DatasetMeta:
    return DatasetMeta(
        version=version,
        created=date.today(),
        manifest_as_of=manifest_as_of,
        n_items=len(items),
        by_category=dict(sorted(Counter(i.category for i in items).items())),
        by_behavior=dict(sorted(Counter(i.expected_behavior for i in items).items())),
        sha256=sha,
    )
