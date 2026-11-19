import pytest

from filingsqa.evaluation.answer_numbers import extract_numbers
from filingsqa.evaluation.metrics.deterministic import (
    behavior_match,
    citation_validity,
    deterministic_metrics,
    forbidden_phrases,
    key_facts_coverage,
    match_number,
    numeric_accuracy,
    period_mentioned,
)
from tests.unit.evaluation.helpers import chunk, item, number, response


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        ("$130.5 billion", "matched"),  # rounded, within 1%
        ("$130,497 million", "matched"),
        ("$130,497", "matched"),  # copied from a table "in millions"
        ("$130 billion", "matched"),  # 0.4% off
        ("$128 billion", "missing"),  # 1.9% off
        ("$130,497 billion", "unit_error"),  # millions vs billions mix-up
        ("$130.5 million", "unit_error"),
        ("revenue grew 114%", "missing"),  # percentages are not amounts
    ],
)
def test_match_number(answer, status):
    assert match_number(number(), extract_numbers(answer))[0] == status


def test_eps_uses_absolute_tolerance():
    eps = number(value=2.94, unit="USD/shares", tol_rel=0.0, tol_abs=0.01)

    assert match_number(eps, extract_numbers("$2.95"))[0] == "matched"
    assert match_number(eps, extract_numbers("$2.99"))[0] == "missing"


def test_numeric_accuracy_partial_credit_and_unit_errors():
    it = item(expected_numbers=(number(), number(value=60_922e6, fy=2024)))

    result = numeric_accuracy(
        it, response("$130.5 billion in fiscal 2025 vs $60,922 billion in 2024")
    )

    assert result.score == 0.5
    assert not result.passed
    assert result.details["unit_errors"] == 1


@pytest.mark.parametrize(
    ("behavior", "answer_type", "ok"),
    [
        ("answer", "answered", True),
        ("decline_advice", "declined_investment_advice", True),
        ("decline_advice", "answered", False),  # missed refusal
        ("insufficient_context", "insufficient_context", True),
        ("insufficient_context", "answered", False),  # made something up
        ("answer", "model_refusal", False),
    ],
)
def test_behavior_match(behavior, answer_type, ok):
    it = item(
        expected_behavior=behavior,
        expected_numbers=(),
        expected_answer="x",
        category="adversarial",
        source="handwritten",
    )

    assert behavior_match(it, response("x", answer_type=answer_type)).passed is ok


def test_period_mentioned_year_and_quarter():
    annual = item()
    quarterly = item(expected_numbers=(number(fy=2026, fp="Q1"),))

    assert period_mentioned(annual, response("$130.5 billion in fiscal year 2025")).passed
    assert not period_mentioned(annual, response("$130.5 billion last year")).passed
    assert period_mentioned(
        quarterly, response("$44 billion in the first quarter of fiscal 2026")
    ).passed
    assert not period_mentioned(quarterly, response("$44 billion in fiscal 2026")).passed


def test_citation_validity_ok():
    assert citation_validity(item(), response("$130,497 million [1]")).passed


@pytest.mark.parametrize(
    ("resp", "problem"),
    [
        (response("$130,497 million", cited=()), "no citations"),
        (response("$130,497 million [1] [7]", invalid=(7,)), "invalid citation"),
        (
            response(
                "$130,497 million [1]", chunks=[chunk(text="Data center revenue grew strongly.")]
            ),
            "not found in any cited chunk",
        ),
    ],
)
def test_citation_validity_problems(resp, problem):
    result = citation_validity(item(), resp)

    assert not result.passed
    assert any(problem in p for p in result.details["problems"])


def test_forbidden_phrases_case_insensitive():
    it = item(forbidden_phrases=("$1 trillion",))

    assert not forbidden_phrases(it, response("Revenue was $1 Trillion")).passed
    assert forbidden_phrases(it, response("Revenue was $97.7 billion")).passed


def test_key_facts_coverage():
    it = item(
        category="qualitative",
        expected_numbers=(),
        key_facts=("Compute & Networking", "Graphics"),
        source="handwritten",
    )

    result = key_facts_coverage(it, response("Two segments: Compute & Networking, and Graphics."))

    assert result.score == 1.0


def test_applicability_for_refusal_items():
    advice = item(id="adv-x", category="adversarial", subcategory="investment_advice",
                  expected_behavior="decline_advice", expected_numbers=(), source="handwritten")  # fmt: skip

    declined = {
        m.name
        for m in deterministic_metrics(
            advice, response("I can't advise.", "declined_investment_advice", cited=())
        )
    }
    answered = {m.name for m in deterministic_metrics(advice, response("Buy it! [1]"))}

    assert declined == {"behavior_match"}
    assert answered == {"behavior_match", "citation_validity"}
