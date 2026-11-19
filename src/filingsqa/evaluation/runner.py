"""Run the bot over golden items, score every answer, aggregate and gate. See ADR-024."""

import hashlib
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from filingsqa.evaluation.dataset import GoldenItem, InjectedDocument
from filingsqa.evaluation.metrics.base import MetricResult
from filingsqa.evaluation.metrics.deterministic import deterministic_metrics
from filingsqa.evaluation.metrics.retrieval import INJECTED_ACCESSION, retrieval_metrics
from filingsqa.evaluation.thresholds import Threshold, ThresholdResult, check_thresholds
from filingsqa.index.qdrant_store import SearchFilter, SearchHit
from filingsqa.rag.pipeline import RAGPipeline
from filingsqa.rag.retriever import Retriever, SearchBackend
from filingsqa.rag.schemas import AskRequest, AskResponse

log = logging.getLogger(__name__)

SMOKE_PER_CATEGORY = {
    "numeric": 6,
    "qualitative": 3,
    "comparison": 3,
    "temporal": 3,
    "adversarial": 6,
}


class JudgeLike(Protocol):  # JudgeSuite, or a fake in tests
    def evaluate(
        self, item: GoldenItem, response: AskResponse
    ) -> tuple[list[MetricResult], list[str]]: ...


def smoke_subset(
    items: list[GoldenItem], per_category: dict[str, int] | None = None
) -> list[GoldenItem]:
    """Deterministic stratified sample: per category, the first N items ordered by sha1(id).

    Hash order (not file order) keeps the sample stable when items are added to the dataset and
    spreads it across companies and subcategories.
    """
    quota = per_category or SMOKE_PER_CATEGORY
    by_cat: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in sorted(items, key=lambda i: hashlib.sha1(i.id.encode()).hexdigest()):
        by_cat[item.category].append(item)
    return [i for cat, n in quota.items() for i in by_cat[cat][:n]]


class InjectingBackend:
    """Wraps the real search backend; puts a poisoned chunk at rank 1 (prompt-injection items)."""

    def __init__(self, inner: SearchBackend, doc: InjectedDocument) -> None:
        self.inner = inner
        self.doc = doc

    def hybrid_search(
        self, query: str, flt: SearchFilter, limit: int, prefetch_limit: int
    ) -> list[SearchHit]:
        hits = self.inner.hybrid_search(query, flt, limit, prefetch_limit)
        if flt.tickers and self.doc.ticker not in flt.tickers:
            return hits
        payload: dict[str, Any] = {
            "chunk_id": f"injected-{self.doc.ticker}", "text": self.doc.text, "context": "",
            "content_type": "text", "section_key": self.doc.section_key,
            "section_title": self.doc.section_key, "chunk_index": 0, "ticker": self.doc.ticker,
            "company": self.doc.ticker, "form": "10-K", "accession": INJECTED_ACCESSION,
            "filing_date": "1970-01-01", "period_end_date": "1970-01-01", "fiscal_year": 0,
            "fiscal_period": "FY", "period_label": "n/a", "source_url": "about:injected",
        }  # fmt: skip
        return [SearchHit(str(payload["chunk_id"]), 1.0, payload), *hits][:limit]


def _pipeline_for(item: GoldenItem, pipeline: RAGPipeline) -> RAGPipeline:
    if item.injected_document is None:
        return pipeline
    base = pipeline.retriever
    retriever = Retriever(
        InjectingBackend(base.backend, item.injected_document), base.top_k, base.prefetch_k
    )
    return RAGPipeline(retriever, pipeline.generator)


class ItemResult(BaseModel):
    item_id: str
    category: str
    subcategory: str
    difficulty: str
    question: str
    expected_behavior: str
    expected_answer: str | None = None
    response: AskResponse | None
    metrics: dict[str, dict[str, Any]]  # name -> {score, passed, details}
    errors: list[str]

    @property
    def failed_metrics(self) -> list[str]:
        return [n for n, m in self.metrics.items() if not m["passed"]]


class RunResult(BaseModel):
    metadata: dict[str, Any]
    items: list[ItemResult]
    summary: dict[str, dict[str, dict[str, float]]]  # scope -> metric -> {mean, pass_rate, n}
    thresholds: list[dict[str, Any]]
    passed: bool


@dataclass
class EvalRunner:
    pipeline: RAGPipeline
    judge: JudgeLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def run_item(self, item: GoldenItem) -> ItemResult:
        errors: list[str] = []
        metrics: list[MetricResult] = []
        response: AskResponse | None = None
        request = AskRequest(
            question=item.question,
            tickers=list(item.ask.tickers) if item.ask.tickers else None,
            fiscal_year=item.ask.fiscal_year,
            fiscal_period=item.ask.fiscal_period,
        )
        try:
            response = _pipeline_for(item, self.pipeline).ask(request)
        except Exception as exc:  # one broken item must not stop the run
            log.exception("item %s failed", item.id)
            errors.append(f"pipeline: {type(exc).__name__}: {exc}")
            metrics.append(MetricResult.binary("behavior_match", False, error=str(exc)))
        if response is not None:
            metrics += deterministic_metrics(item, response)
            metrics += retrieval_metrics(item, response)
            if self.judge is not None:
                judged, judge_errors = self.judge.evaluate(item, response)
                metrics += judged
                errors += [f"judge: {e}" for e in judge_errors]
        return ItemResult(
            item_id=item.id, category=item.category, subcategory=item.subcategory,
            difficulty=item.difficulty, question=item.question,
            expected_behavior=item.expected_behavior, expected_answer=item.expected_answer,
            response=response,
            metrics={
                m.name: {"score": m.score, "passed": m.passed, "details": m.details}
                for m in metrics
            },
            errors=errors,
        )  # fmt: skip

    def run(self, items: list[GoldenItem], thresholds: list[Threshold]) -> RunResult:
        started = time.perf_counter()
        results = []
        for n, item in enumerate(items, start=1):
            log.info("[%d/%d] %s", n, len(items), item.id)
            results.append(self.run_item(item))
        summary = summarize(results)
        checked = check_thresholds(thresholds, summary)
        metadata = {
            **self.metadata,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - started, 1),
            "n_items": len(items),
            "git_commit": _git_commit(),
            "judge": self.judge is not None,
        }
        return RunResult(
            metadata=metadata,
            items=results,
            summary=summary,
            thresholds=[_threshold_dict(t) for t in checked],
            passed=all(t.status != "fail" for t in checked),
        )


def summarize(results: list[ItemResult]) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for name, m in r.metrics.items():
            for scope in ("overall", r.category):
                scores[scope][name].append((float(m["score"]), bool(m["passed"])))
    return {
        scope: {
            name: {
                "mean": sum(s for s, _ in vals) / len(vals),
                "pass_rate": sum(p for _, p in vals) / len(vals),
                "n": len(vals),
            }
            for name, vals in sorted(metrics.items())
        }
        for scope, metrics in scores.items()
    }


def _threshold_dict(t: ThresholdResult) -> dict[str, Any]:
    return {
        "scope": t.threshold.scope,
        "metric": t.threshold.metric,
        "minimum": t.threshold.minimum,
        "actual": t.actual,
        "n": t.n,
        "status": t.status,
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
