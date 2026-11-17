import pytest
from pydantic import ValidationError

from filingsqa.evaluation.dataset import GoldenItem, build_meta, dump_items, file_sha256, load_items

BASE = {
    "id": "adv-x", "category": "adversarial", "subcategory": "investment_advice", "difficulty": "easy",
    "question": "Should I buy?", "expected_behavior": "decline_advice", "source": "handwritten",
}  # fmt: skip


def test_minimal_item_is_valid():
    assert GoldenItem.model_validate(BASE).verified is False


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"expected_behavior": "answer"}, "needs numbers"),
        ({"subcategory": "prompt_injection"}, "injected_document"),
        ({"expected_winner": "AAPL"}, "expected_winner"),
        ({"source": "generated"}, "must be verified"),
        ({"id": "Bad ID"}, "pattern"),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_invalid_items_are_rejected(patch, message):
    with pytest.raises(ValidationError, match=message):
        GoldenItem.model_validate({**BASE, **patch})


def test_dump_is_sorted_deterministic_and_round_trips(tmp_path):
    items = [
        GoldenItem.model_validate({**BASE, "id": i, "question": f"Should I buy {i}?"})
        for i in ("b", "a")
    ]

    sha1 = dump_items(items, tmp_path / "d.jsonl")
    sha2 = dump_items(list(reversed(items)), tmp_path / "e.jsonl")

    assert sha1 == sha2 == file_sha256(tmp_path / "d.jsonl")
    assert [i.id for i in load_items(tmp_path / "d.jsonl")] == ["a", "b"]


def test_load_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n')

    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_items(path)


def test_meta_counts():
    items = [
        GoldenItem.model_validate({**BASE, "id": i, "question": f"Should I buy {i}?"})
        for i in ("a", "b")
    ]

    meta = build_meta(items, "v1", "abc", None)

    assert meta.by_category == {"adversarial": 2}
    assert meta.by_behavior == {"decline_advice": 2}
