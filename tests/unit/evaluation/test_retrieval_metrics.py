import pytest

from filingsqa.evaluation.dataset import ExpectedSource
from filingsqa.evaluation.metrics.retrieval import INJECTED_ACCESSION, retrieval_metrics
from tests.unit.evaluation.helpers import ACC, chunk, item, response


def _metrics(it, chunks):
    return {m.name: m for m in retrieval_metrics(it, response("x", chunks=chunks))}


def test_relevant_first():
    m = _metrics(item(), [chunk(1, section="item_8"), chunk(2, section="item_1a")])

    assert m["recall_at_k"].score == 1.0
    assert m["mrr"].score == 1.0
    assert m["contextual_precision"].score == 1.0
    assert m["context_has_numbers"].score == 1.0


def test_relevant_at_rank_three():
    chunks = [
        chunk(1, section="item_1a", text="x"),
        chunk(2, section="item_1", text="y"),
        chunk(3, section="item_7"),
    ]

    m = _metrics(item(), chunks)

    assert m["mrr"].score == pytest.approx(1 / 3)
    assert m["contextual_precision"].score == pytest.approx(1 / 3)
    assert m["context_has_numbers"].score == 1.0


def test_wrong_filing_same_section_is_not_relevant():
    m = _metrics(item(), [chunk(1, section="item_8", accession="0001045810-24-000029")])

    assert m["recall_at_k"].score == 0.0
    assert m["mrr"].score == 0.0


def test_two_sources_partial_recall():
    it = item(expected_sources=(
        ExpectedSource(ticker="NVDA", form="10-K", section_keys=("item_8",), accession=ACC),
        ExpectedSource(ticker="AAPL", form="10-K", section_keys=("item_8",), accession="AAPL-ACC"),
    ))  # fmt: skip

    assert _metrics(it, [chunk(1)])["recall_at_k"].score == 0.5


def test_injected_chunk_is_ignored():
    m = _metrics(item(), [chunk(1, accession=INJECTED_ACCESSION), chunk(2)])

    assert m["mrr"].score == 1.0  # rank counted without the poisoned chunk


def test_not_applicable_without_sources_or_for_refusals():
    assert retrieval_metrics(item(expected_sources=()), response("x")) == []
    advice = item(
        expected_behavior="decline_advice",
        expected_numbers=(),
        category="adversarial",
        source="handwritten",
    )
    assert retrieval_metrics(advice, response("x")) == []
