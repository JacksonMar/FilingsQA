import pytest

from filingsqa.evaluation.judge_agreement import (
    Agreement,
    LabelRow,
    answer_sha,
    compute_agreement,
    label_sample,
)
from filingsqa.evaluation.runner import ItemResult, RunResult
from tests.unit.evaluation.helpers import response


def _run(verdicts: dict[str, bool]) -> RunResult:
    items = [
        ItemResult(
            item_id=i,
            category="numeric",
            subcategory="annual",
            difficulty="easy",
            question="q",
            expected_behavior="answer",
            response=response(f"answer {i}"),
            metrics={
                "fiscal_period_correct": {"score": 1.0 if v else 0.0, "passed": v, "details": {}}
            },
            errors=[],
        )
        for i, v in verdicts.items()
    ]
    return RunResult(metadata={}, items=items, summary={}, thresholds=[], passed=True)


def _label(item_id, human, answer=None):
    return LabelRow(item_id=item_id, metric="fiscal_period_correct", question="q", answer="a",
                    answer_sha=answer_sha(answer or f"answer {item_id}"), human_pass=human)  # fmt: skip


def test_confusion_counts():
    run = _run({"a": True, "b": True, "c": False, "d": False})
    labels = [_label("a", True), _label("b", False), _label("c", False), _label("d", True)]

    agr = compute_agreement(run, labels)

    assert (agr.tp, agr.fp, agr.tn, agr.fn) == (1, 1, 1, 1)
    assert agr.accuracy == 0.5
    assert agr.kappa == pytest.approx(0.0)  # coin-flip agreement


def test_kappa_known_value():
    # 20 items: judge/human pass-pass 9, fail-fail 7, pass-fail 3, fail-pass 1
    agr = Agreement(n=20, tp=9, tn=7, fp=3, fn=1, skipped=0)

    # p_o = 0.8, p_e = 0.6*0.5 + 0.4*0.5 = 0.5 -> kappa = 0.6
    assert agr.kappa == pytest.approx(0.6)


def test_labels_for_changed_answers_or_unlabeled_rows_are_skipped():
    run = _run({"a": True, "b": True})
    labels = [_label("a", True, answer="an older answer"), _label("b", None), _label("zzz", True)]

    agr = compute_agreement(run, labels)

    assert agr.n == 0
    assert agr.skipped == 2  # changed answer + unknown item; unlabeled row is ignored


def test_label_sample_hides_verdict_and_is_deterministic():
    run = _run({f"i{n}": n % 2 == 0 for n in range(10)})

    rows = label_sample(run, "fiscal_period_correct", n=4)

    assert rows == label_sample(run, "fiscal_period_correct", n=4)
    assert len(rows) == 4
    assert all(r.human_pass is None for r in rows)
    assert "passed" not in rows[0].model_dump()
