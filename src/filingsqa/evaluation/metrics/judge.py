"""LLM-as-judge metrics via DeepEval, using our own judge LLM (Sonnet by default). See ADR-023.

DeepEval calls `generate(prompt)` on a custom model and parses the JSON it asked for itself,
so the adapter only has to forward prompts to our LLMClient (free-text mode, cached on disk).
All judge scores are "higher is better". Note: in DeepEval >= 4 HallucinationMetric scores the
share of contexts the answer *agrees* with (higher = less hallucination); older versions scored
the opposite way. A unit test pins this direction so a library upgrade cannot flip it silently.
"""

import logging
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from filingsqa.evaluation.dataset import GoldenItem
from filingsqa.evaluation.metrics.base import MetricResult
from filingsqa.llm.base import LLMClient, LLMRequest
from filingsqa.rag.schemas import AskResponse

log = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are a strict, careful evaluator. Follow the instructions exactly."

FISCAL_PERIOD_STEPS = [
    "Identify every financial figure stated in the actual output.",
    "For each figure, check that the actual output states which fiscal period it refers to "
    "(fiscal year, and quarter if applicable), not only a calendar year.",
    "Compare the stated fiscal periods with the periods in the expected output. A figure "
    "attributed to the wrong fiscal year or quarter is a serious error.",
    "Give a high score only if every figure has an explicit and correct fiscal period.",
]
CORRECTNESS_STEPS = [
    "Compare the facts in the actual output with the expected output.",
    "Penalize heavily any statement that contradicts the expected output.",
    "Penalize missing key facts from the expected output; do not penalize extra correct detail.",
    "Ignore style and wording differences.",
]


class JudgeAdapter(DeepEvalBaseLLM):  # type: ignore[misc]  # DeepEval is untyped for mypy
    def __init__(self, llm: LLMClient, max_tokens: int = 4096) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        identity = llm.identity
        super().__init__(model=f"{identity.get('provider')}:{identity.get('model')}")

    def load_model(self) -> "JudgeAdapter":
        return self

    def generate(self, prompt: str, schema: object = None) -> str:
        # `schema` is ignored: DeepEval parses the JSON it asked for from our text reply.
        request = LLMRequest(
            system=JUDGE_SYSTEM, user=prompt, json_schema=None, max_tokens=self._max_tokens
        )
        return self._llm.complete(request).text

    async def a_generate(self, prompt: str, schema: object = None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return str(self.name)


def _expected_text(item: GoldenItem) -> str:
    parts = [item.expected_answer or ""]
    if item.key_facts:
        parts.append("Key facts: " + "; ".join(item.key_facts))
    return "\n".join(p for p in parts if p)


class JudgeSuite:
    """Builds the DeepEval metrics once and applies the relevant ones to an item."""

    def __init__(self, llm: LLMClient) -> None:
        judge = JudgeAdapter(llm)
        common: dict[str, Any] = {"model": judge, "async_mode": False}
        self.faithfulness = FaithfulnessMetric(threshold=0.8, **common)
        self.relevancy = AnswerRelevancyMetric(threshold=0.7, **common)
        self.hallucination = HallucinationMetric(threshold=0.7, **common)
        self.fiscal_period = GEval(
            name="fiscal_period_correct",
            evaluation_steps=FISCAL_PERIOD_STEPS,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
            **common,
        )
        self.correctness = GEval(
            name="answer_correctness",
            evaluation_steps=CORRECTNESS_STEPS,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
            **common,
        )

    def evaluate(
        self, item: GoldenItem, response: AskResponse
    ) -> tuple[list[MetricResult], list[str]]:
        """Judge metrics apply only to real answers grounded in retrieved context."""
        if response.answer_type != "answered" or not response.retrieved_chunks:
            return [], []
        context = [c.text for c in response.retrieved_chunks]
        expected = _expected_text(item)
        case = LLMTestCase(
            input=item.question,
            actual_output=response.answer,
            expected_output=expected or None,
            retrieval_context=context,
            context=context,
        )
        plan: list[tuple[str, Any]] = [
            ("faithfulness", self.faithfulness),
            ("answer_relevancy", self.relevancy),
            ("hallucination_free", self.hallucination),
        ]
        if item.expected_numbers and expected:
            plan.append(("fiscal_period_correct", self.fiscal_period))
        if expected and item.subcategory != "prompt_injection":
            plan.append(("answer_correctness", self.correctness))

        results: list[MetricResult] = []
        errors: list[str] = []
        for name, metric in plan:
            try:
                score = float(metric.measure(case, _show_indicator=False))
            except Exception as exc:  # judge output unparsable, API error, ...
                log.warning("judge metric %s failed on %s: %s", name, item.id, exc)
                errors.append(f"{name}: {exc}")
                continue
            reason = getattr(metric, "reason", None)
            results.append(MetricResult(name, score, bool(metric.success), {"reason": reason}))
        return results, errors
