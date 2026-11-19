"""Deterministic (no-LLM) checks of a bot answer against a golden item. See ADR-022.

They are cheap, reproducible and explainable, so they run on every item and gate CI.
"""

import re

from filingsqa.evaluation.answer_numbers import AnswerNumber, extract_numbers
from filingsqa.evaluation.dataset import ExpectedNumber, GoldenItem
from filingsqa.evaluation.metrics.base import MetricResult
from filingsqa.evaluation.numbers import number_variants
from filingsqa.rag.schemas import AskResponse

BEHAVIOR_TO_ANSWER_TYPE = {
    "answer": "answered",
    "insufficient_context": "insufficient_context",
    "decline_advice": "declined_investment_advice",
}
QUARTER_PATTERNS = {
    "Q1": r"\bQ1\b|first quarter|first fiscal quarter",
    "Q2": r"\bQ2\b|second quarter|second fiscal quarter",
    "Q3": r"\bQ3\b|third quarter|third fiscal quarter",
    "Q4": r"\bQ4\b|fourth quarter|fourth fiscal quarter",
}
_WORD = re.compile(r"[a-z0-9&]+")


def _within(candidate: float, expected: ExpectedNumber) -> bool:
    allowed = max(expected.tolerance_rel * abs(expected.value), expected.tolerance_abs)
    return abs(abs(candidate) - abs(expected.value)) <= allowed + 1e-9


def match_number(expected: ExpectedNumber, found: list[AnswerNumber]) -> tuple[str, str | None]:
    """Return (status, raw answer text). status: matched | unit_error | missing.

    Signs are compared on absolute values: "net loss of $1.2 billion" has no minus sign.
    """
    candidates = [n for n in found if not n.percent]
    for n in candidates:
        values = [n.value]
        if not n.scaled and expected.unit == "USD":
            values.append(n.value * 1e6)  # table value copied without its "(in millions)" caption
        if any(_within(v, expected) for v in values):
            return "matched", n.raw
    if expected.unit == "USD":
        for n in candidates:
            if n.scaled and any(_within(n.value * f, expected) for f in (1e3, 1e-3)):
                return "unit_error", n.raw  # e.g. "$130,497 billion" or "$130.5 million"
    return "missing", None


def behavior_match(item: GoldenItem, response: AskResponse) -> MetricResult:
    expected = BEHAVIOR_TO_ANSWER_TYPE[item.expected_behavior]
    return MetricResult.binary(
        "behavior_match", response.answer_type == expected,
        expected=expected, actual=response.answer_type,
    )  # fmt: skip


def numeric_accuracy(item: GoldenItem, response: AskResponse) -> MetricResult:
    found = extract_numbers(response.answer)
    statuses = []
    for exp in item.expected_numbers:
        status, raw = match_number(exp, found)
        statuses.append({"label": exp.label, "ticker": exp.ticker, "expected": exp.value,
                         "status": status, "answer_text": raw})  # fmt: skip
    matched = sum(s["status"] == "matched" for s in statuses)
    score = matched / len(statuses)
    return MetricResult("numeric_accuracy", score, score == 1.0, {
        "numbers": statuses,
        "unit_errors": sum(s["status"] == "unit_error" for s in statuses),
    })  # fmt: skip


def period_mentioned(item: GoldenItem, response: AskResponse) -> MetricResult:
    """Cheap baseline for "states the fiscal period": year (and quarter) of every number."""
    answer = response.answer
    checks = []
    for exp in item.expected_numbers:
        ok = str(exp.fiscal_year) in answer
        if exp.fiscal_period != "FY":
            ok = ok and re.search(QUARTER_PATTERNS[exp.fiscal_period], answer, re.I) is not None
        checks.append(ok)
    score = sum(checks) / len(checks)
    return MetricResult("period_mentioned", score, score == 1.0, {"per_number": checks})


def citation_validity(item: GoldenItem, response: AskResponse) -> MetricResult:
    """Every citation points to a retrieved chunk, and correct numbers are backed by a cited chunk.

    "Backed" = the expected value occurs (in any table format) in the text of a cited chunk.
    """
    problems: list[str] = []
    if response.invalid_citation_ids:
        problems.append(f"invalid citation ids {response.invalid_citation_ids}")
    if not response.citations:
        problems.append("answer has no citations")
    cited_text = "\n".join(
        c.text
        for c in response.retrieved_chunks
        if c.chunk_id in {x.chunk_id for x in response.citations}
    )
    found = extract_numbers(response.answer)
    for exp in item.expected_numbers:
        status, _ = match_number(exp, found)
        if status == "matched" and not any(
            v in cited_text for v in number_variants(exp.value, exp.unit)
        ):
            problems.append(f"{exp.ticker} {exp.label} not found in any cited chunk")
    return MetricResult.binary("citation_validity", not problems, problems=problems)


def forbidden_phrases(item: GoldenItem, response: AskResponse) -> MetricResult:
    lowered = response.answer.lower()
    hits = [p for p in item.forbidden_phrases if p.lower() in lowered]
    return MetricResult.binary("forbidden_phrases", not hits, found=hits)


def key_facts_coverage(item: GoldenItem, response: AskResponse) -> MetricResult:
    """Share of key facts whose words all appear in the answer (rough; the LLM judge is finer)."""
    words = set(_WORD.findall(response.answer.lower()))
    covered = [f for f in item.key_facts if set(_WORD.findall(f.lower())) <= words]
    score = len(covered) / len(item.key_facts)
    return MetricResult("key_facts_coverage", score, score >= 0.5, {
        "missing": [f for f in item.key_facts if f not in covered],
    })  # fmt: skip


def deterministic_metrics(item: GoldenItem, response: AskResponse) -> list[MetricResult]:
    results = [behavior_match(item, response)]
    answering = item.expected_behavior == "answer"
    if answering and item.expected_numbers:
        results.append(numeric_accuracy(item, response))
        results.append(period_mentioned(item, response))
    if response.answer_type == "answered":
        results.append(citation_validity(item, response))
    if item.forbidden_phrases:
        results.append(forbidden_phrases(item, response))
    if answering and item.key_facts:
        results.append(key_facts_coverage(item, response))
    return results
