from collections import Counter
from pathlib import Path

import pytest

from filingsqa.evaluation.builder import build_dataset, load_handwritten, resolve_source
from filingsqa.evaluation.dataset import ExpectedSource, GoldenItem
from tests.unit.evaluation.conftest import AAPL_K, AAPL_Q1

HANDWRITTEN = Path(__file__).resolve().parents[3] / "eval/datasets/handwritten"


def test_resolve_10k_and_latest_10q(manifest):
    k = resolve_source(
        ExpectedSource(ticker="AAPL", form="10-K", section_keys=("item_1a",)), manifest
    )
    q = resolve_source(
        ExpectedSource(ticker="AAPL", form="10-Q", section_keys=("part1_item_2",)), manifest
    )

    assert k.accession == AAPL_K
    assert q.accession == AAPL_Q1


def test_resolve_unknown_filing_fails(manifest):
    with pytest.raises(ValueError, match="no 10-Q for MSFT"):
        resolve_source(ExpectedSource(ticker="MSFT", form="10-Q", section_keys=("x",)), manifest)


def test_committed_handwritten_files_are_valid():
    items = load_handwritten(HANDWRITTEN)
    counts = Counter(i.category for i in items)

    assert counts["qualitative"] >= 30
    assert counts["adversarial"] >= 20
    assert {i.subcategory for i in items if i.category == "adversarial"} >= {
        "investment_advice", "unknown_company", "made_up_number", "prompt_injection", "out_of_scope",
    }  # fmt: skip


def test_build_merges_generated_and_handwritten(manifest, facts_by_ticker):
    hw = GoldenItem(
        id="qual-aapl-x", category="qualitative", subcategory="risk", difficulty="easy",
        question="What risks?", expected_behavior="answer", key_facts=("risk",), source="handwritten",
        expected_sources=(ExpectedSource(ticker="AAPL", form="10-K", section_keys=("item_1a",)),),
    )  # fmt: skip

    items, _ = build_dataset(manifest, facts_by_ticker, [hw])

    built = {i.id: i for i in items}
    assert built["qual-aapl-x"].expected_sources[0].accession == AAPL_K
    assert "num-aapl-revenue-2025fy" in built


def test_duplicate_ids_fail(manifest, facts_by_ticker):
    items, _ = build_dataset(manifest, facts_by_ticker, [])

    with pytest.raises(ValueError, match="duplicate"):
        build_dataset(
            manifest, facts_by_ticker, [items[0].model_copy(update={"source": "handwritten"})]
        )
