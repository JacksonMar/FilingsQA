"""Hybrid section resolution, with hand-made oracle sections (edgartools not called)."""

import pytest

from filingsqa.ingest.html_to_text import Block
from filingsqa.ingest.oracle_sections import (
    OracleSection,
    core_disagreement,
    resolve_sections,
    signature,
    sizes_disagree,
    split_with_oracle,
)
from filingsqa.ingest.sections import split_sections


def _blocks(*texts):
    return [Block("table" if "|" in t else "text", t) for t in texts]


# A JPM-like 10-K: formal Items first (short, cross-references), real MD&A and statements later.
JPM_LIKE = _blocks(
    "Form 10-K cover",
    "Item 1. Business.", "We are a bank. " * 40,
    "Item 7. Management's Discussion and Analysis.", "Refer to pages 50-160.",
    "Item 8. Financial Statements.", "Refer to pages 161-300.",
    "Item 15. Exhibits.", "Exhibit list",
    "Annual report",
    "THREE-YEAR SUMMARY OF CONSOLIDATED FINANCIAL HIGHLIGHTS", "Net revenue grew. " * 300,
    "Year ended December 31, (in millions) | 2025 | 2024", "Total net revenue | 182,447 | 177,556",
    "Year ended December 31, (in millions) | 2025 | 2024", "Consolidated statements of income | x | y", "Interest income. " * 400,
)  # fmt: skip
ORACLE = [
    OracleSection("item_1", 600, ("Item 1. Business.", "We are a bank.")),
    OracleSection("item_7", 4800, ("THREE-YEAR SUMMARY OF CONSOLIDATED FINANCIAL HIGHLIGHTS", "Net revenue grew.")),
    OracleSection("item_8", 6000, ("Year ended December 31, (in millions) │ 2025 │ 2024", "Consolidated statements of income")),
    OracleSection("item_15", 20, ("Item 15. Exhibits.", "Exhibit list")),
]  # fmt: skip


def test_signature_ignores_table_rendering():
    assert signature("Total │ 1,234 │") == signature("Total | 1,234") == "total1234"


@pytest.mark.parametrize(
    ("ours", "oracle", "disagree"),
    [
        (1000, 1100, False),
        (100, 5000, True),
        (5000, 100, True),
        (300, 50, False),
        (None, 4000, True),
        (None, None, False),
    ],
)
def test_sizes_disagree(ours, oracle, disagree):
    assert sizes_disagree(ours, oracle) is disagree


def test_own_parser_misses_jpm_core_sections():
    own = split_sections(JPM_LIKE, "10-K")

    assert core_disagreement(own, ORACLE, "10-K") == ["item_7", "item_8"]


def test_split_with_oracle_anchors_real_mda_and_statements():
    split = split_with_oracle(JPM_LIKE, ORACLE, "10-K")

    sections = {s.key: s for s in split.sections}
    assert sections["item_7"].blocks[0].text.startswith("THREE-YEAR SUMMARY")
    # the table header repeats; the second-line check picks the statements, not the MD&A table
    assert sections["item_8"].blocks[1].text.startswith("Consolidated statements of income")
    assert sections["item_15"].blocks[0].text == "Item 15. Exhibits."
    assert [b for s in split.sections for b in s.blocks] == JPM_LIKE  # nothing lost


def test_missing_core_anchor_returns_none():
    oracle = [*ORACLE[:2], OracleSection("item_8", 6000, ("Text that is nowhere in the filing",))]

    assert split_with_oracle(JPM_LIKE, oracle, "10-K") is None


def test_resolve_keeps_own_split_when_core_agrees(monkeypatch):
    blocks = _blocks(
        "Item 7. MD&A", "Revenue grew. " * 100, "Item 8. Financial Statements", "Numbers. " * 100
    )
    own = split_sections(blocks, "10-K")
    sizes = {s.key: len(signature(" ".join(b.text for b in s.blocks))) for s in own.sections}
    agreeing = [OracleSection(k, v, ()) for k, v in sizes.items()]
    monkeypatch.setattr(
        "filingsqa.ingest.oracle_sections.oracle_sections", lambda html, form: agreeing
    )

    split, source = resolve_sections("<html/>", blocks, own, "10-K")

    assert source == "own"
    assert split is own


def test_resolve_uses_oracle_on_core_mismatch(monkeypatch):
    monkeypatch.setattr(
        "filingsqa.ingest.oracle_sections.oracle_sections", lambda html, form: ORACLE
    )

    split, source = resolve_sections("<html/>", JPM_LIKE, split_sections(JPM_LIKE, "10-K"), "10-K")

    assert source == "edgartools"
    assert any("disagreed" in w for w in split.warnings)


def test_library_failure_keeps_own_split(monkeypatch):
    def boom(html, form):
        raise RuntimeError("parser crashed")

    monkeypatch.setattr("filingsqa.ingest.oracle_sections.oracle_sections", boom)
    own = split_sections(JPM_LIKE, "10-K")

    assert resolve_sections("<html/>", JPM_LIKE, own, "10-K") == (own, "own")
