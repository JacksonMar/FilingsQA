"""How much can we trust the LLM judge? Compare its verdicts with human labels. See ADR-025.

Workflow: export a sample of judged answers *without* the judge's verdict (to avoid anchoring),
a human fills `human_pass`, then agreement and Cohen's kappa are computed. Each label is tied to
a hash of the exact answer text, so a label never silently applies to a changed answer.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from filingsqa.evaluation.runner import RunResult


def answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]


class LabelRow(BaseModel):
    item_id: str
    metric: str
    answer_sha: str
    question: str
    answer: str
    expected: str | None = None
    human_pass: bool | None = None  # filled in by a human
    note: str = ""


def label_sample(run: RunResult, metric: str, n: int, seed: int = 13) -> list[LabelRow]:
    rows = [
        LabelRow(
            item_id=r.item_id,
            metric=metric,
            answer_sha=answer_sha(r.response.answer),
            question=r.question,
            answer=r.response.answer,
            expected=r.expected_answer,
        )
        for r in run.items
        if r.response is not None and metric in r.metrics
    ]
    random.Random(seed).shuffle(rows)
    return sorted(rows[:n], key=lambda row: row.item_id)


@dataclass(frozen=True)
class Agreement:
    n: int
    tp: int  # judge pass, human pass
    tn: int  # judge fail, human fail
    fp: int  # judge pass, human fail  <- judge too lenient
    fn: int  # judge fail, human pass  <- judge too strict
    skipped: int  # labels whose answer changed or item missing

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def kappa(self) -> float:
        """Cohen's kappa: agreement corrected for chance (1 = perfect, 0 = chance level)."""
        if not self.n:
            return 0.0
        p_o = self.accuracy
        judge_pass, human_pass = (self.tp + self.fp) / self.n, (self.tp + self.fn) / self.n
        p_e = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
        return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def compute_agreement(run: RunResult, labels: list[LabelRow]) -> Agreement:
    by_id = {r.item_id: r for r in run.items}
    tp = tn = fp = fn = skipped = 0
    for label in labels:
        if label.human_pass is None:
            continue
        result = by_id.get(label.item_id)
        if (
            result is None
            or result.response is None
            or label.metric not in result.metrics
            or answer_sha(result.response.answer) != label.answer_sha
        ):
            skipped += 1
            continue
        judge = bool(result.metrics[label.metric]["passed"])
        if judge and label.human_pass:
            tp += 1
        elif not judge and not label.human_pass:
            tn += 1
        elif judge:
            fp += 1
        else:
            fn += 1
    return Agreement(tp + tn + fp + fn, tp, tn, fp, fn, skipped)


def load_labels(path: Path) -> list[LabelRow]:
    return [
        LabelRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def save_labels(rows: list[LabelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in rows))
