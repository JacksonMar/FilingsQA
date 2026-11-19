"""Judge metrics with a scripted fake judge: checks wiring and score direction, not judgement."""

import json

import pytest

from filingsqa.evaluation.metrics.judge import JudgeAdapter, JudgeSuite
from tests.fakes import FakeLLM
from tests.unit.evaluation.helpers import item, response


def scripted(agree: bool = True, geval_score: int = 9):
    verdict = "yes" if agree else "no"

    def reply(request):
        p = request.user
        if '"truths"' in p:
            return json.dumps({"truths": ["Revenue was $130,497 million in fiscal 2025."]})
        if '"claims"' in p:
            return json.dumps({"claims": ["Revenue was $130,497 million in fiscal 2025."]})
        if '"statements"' in p:
            return json.dumps({"statements": ["Revenue was $130,497 million."]})
        if '"verdicts"' in p:
            return json.dumps({"verdicts": [{"verdict": verdict, "reason": "r"}]})
        if '"score"' in p:
            return json.dumps({"score": geval_score, "reason": "r"})
        return json.dumps({"reason": "r"})

    return FakeLLM(reply, model="fake-judge")


ANSWER = "NVIDIA's total revenue for fiscal year 2025 was $130,497 million [1]."


def test_adapter_sends_free_text_requests():
    llm = scripted()

    JudgeAdapter(llm).generate("Return JSON please")

    assert llm.requests[0].json_schema is None
    assert llm.requests[0].user == "Return JSON please"


def test_all_metrics_run_on_numeric_answer():
    results, errors = JudgeSuite(scripted()).evaluate(
        item(expected_answer=ANSWER), response(ANSWER)
    )

    names = {r.name for r in results}
    assert errors == []
    assert names == {
        "faithfulness",
        "answer_relevancy",
        "hallucination_free",
        "fiscal_period_correct",
        "answer_correctness",
    }


def test_hallucination_direction_is_higher_is_better():
    """Regression guard: DeepEval changed this metric's polarity between major versions."""
    good, _ = JudgeSuite(scripted(agree=True)).evaluate(
        item(expected_answer=ANSWER), response(ANSWER)
    )
    bad, _ = JudgeSuite(scripted(agree=False)).evaluate(
        item(expected_answer=ANSWER), response(ANSWER)
    )

    good_h = next(r for r in good if r.name == "hallucination_free")
    bad_h = next(r for r in bad if r.name == "hallucination_free")
    assert (good_h.score, good_h.passed) == (1.0, True)
    assert (bad_h.score, bad_h.passed) == (0.0, False)


def test_geval_score_is_normalized():
    results, _ = JudgeSuite(scripted(geval_score=4)).evaluate(
        item(expected_answer=ANSWER), response(ANSWER)
    )

    fiscal = next(r for r in results if r.name == "fiscal_period_correct")
    assert fiscal.score == pytest.approx(0.4)
    assert not fiscal.passed


def test_no_judge_calls_for_refusals():
    llm = scripted()

    results, _ = JudgeSuite(llm).evaluate(
        item(), response("I can't help", answer_type="insufficient_context")
    )

    assert results == []
    assert llm.requests == []


def test_unparsable_judge_output_is_recorded_not_raised():
    results, errors = JudgeSuite(FakeLLM("not json at all")).evaluate(
        item(expected_answer=ANSWER), response(ANSWER)
    )

    assert errors
    assert all(e.split(":")[0] in {"faithfulness", "answer_relevancy", "hallucination_free",
               "fiscal_period_correct", "answer_correctness"} for e in errors)  # fmt: skip
    assert len(results) + len(errors) == 5
