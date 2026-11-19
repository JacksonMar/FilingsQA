import pytest

from filingsqa.ingest.html_to_text import Block, parse_filing_html
from filingsqa.ingest.sections import FRONT_MATTER, FULL_DOCUMENT, _match_heading, split_sections


def _text(*lines: str) -> list[Block]:
    return [Block("text", line) for line in lines]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Item 1A. Risk Factors", ("item", "1A")),
        ("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", ("item", "7")),
        ("Item 7A: Quantitative and Qualitative Disclosures", ("item", "7A")),
        ("Item 1.", ("item", "1")),
        ("Item 9C - Disclosure Regarding Foreign Jurisdictions", ("item", "9C")),
        ("Item 6. [Reserved]", ("item", "6")),
        ("PART II", ("part", "II")),
        ("PART I - FINANCIAL INFORMATION", ("part", "I")),
        ("Part IV", ("part", "IV")),
    ],
)
def test_heading_is_recognized(line, expected):
    assert _match_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Item 7 of this report should be read with the financial statements.",
        "See Item 1A, Risk Factors, for more information.",
        "Items 1 and 2 are not applicable.",
        "Particularly important risks are described below.",
        "Item 1A. Risk Factors " + "x" * 200,  # too long to be a heading
    ],
)
def test_reference_or_prose_is_not_a_heading(line):
    assert _match_heading(line) is None


def test_10k_fixture_sections(sample_10k_html):
    blocks = parse_filing_html(sample_10k_html).blocks
    split = split_sections(blocks, "10-K")

    assert [s.key for s in split.sections] == [
        FRONT_MATTER,
        "item_1",
        "item_1a",
        "item_7",
        "item_8",
    ]
    assert split.warnings == []
    risk = next(s for s in split.sections if s.key == "item_1a")
    assert risk.title == "Risk Factors"
    assert "single-source providers" in " ".join(b.text for b in risk.blocks)


def test_10q_parts_disambiguate_item_numbers(sample_10q_html):
    blocks = parse_filing_html(sample_10q_html).blocks
    keys = [s.key for s in split_sections(blocks, "10-Q").sections]

    assert keys == [
        FRONT_MATTER,
        "part1_item_1",
        "part1_item_2",
        "part2_item_1",
        "part2_item_1a",
        "part2_item_6",
    ]


@pytest.mark.parametrize("fixture", ["sample_10k_html", "sample_10q_html"])
def test_no_block_is_lost_or_duplicated(fixture, request):
    """Invariant: concatenating all sections gives back exactly the input blocks."""
    blocks = parse_filing_html(request.getfixturevalue(fixture)).blocks
    form = "10-K" if "10k" in fixture else "10-Q"

    split = split_sections(blocks, form)

    assert [b for s in split.sections for b in s.blocks] == blocks


def test_text_toc_entries_lose_to_the_real_heading():
    """A plain-text TOC repeats every heading; the real one is followed by much more text."""
    blocks = _text(
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 1. Business",
        "We make widgets. " * 50,
        "Item 1A. Risk Factors",
        "Many risks exist. " * 50,
    )

    split = split_sections(blocks, "10-K")

    assert [s.key for s in split.sections] == [FRONT_MATTER, "item_1", "item_1a"]
    assert split.sections[1].blocks[0] is blocks[2]
    assert split.sections[2].blocks[0] is blocks[4]


def test_no_headings_falls_back_to_full_document_with_warning():
    split = split_sections(_text("Annual report", "Some text"), "10-K")

    assert [s.key for s in split.sections] == [FULL_DOCUMENT]
    assert split.warnings


def test_out_of_order_items_produce_a_warning():
    split = split_sections(
        _text("Item 7. MD&A", "text " * 20, "Item 1. Business", "text " * 20), "10-K"
    )

    assert any("order" in w for w in split.warnings)


def test_item_number_not_valid_for_form_is_ignored():
    split = split_sections(_text("Item 1. Business", "body", "Item 17. Something", "more"), "10-K")

    assert [s.key for s in split.sections] == ["item_1"]


def test_data_tables_are_never_headings():
    blocks = [Block("table", "Item 1A. Risk Factors"), *_text("body")]

    assert split_sections(blocks, "10-K").sections[0].key == FULL_DOCUMENT


def test_running_page_headers_do_not_split_a_section():
    """Regression (BUG-2): MSFT repeats "Item 7" as a page header on every page of MD&A."""
    blocks = _text(
        "Item 6", "ITEM 6. [RESERVED]", "PART II",
        "Item 7", "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", "Overview " * 30,
        "Item 7", "Metrics " * 30,
        "Item 7", "Liquidity " * 80,  # largest single span: used to win and split MD&A
        "Item 7A", "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES", "Risk " * 30,
    )  # fmt: skip

    sections = {s.key: s for s in split_sections(blocks, "10-K").sections}

    assert sections["item_6"].blocks[0].text == "Item 6"
    assert len(sections["item_6"].blocks) == 3
    md_a = " ".join(b.text for b in sections["item_7"].blocks)
    assert "Overview" in md_a
    assert "Liquidity" in md_a
    assert sections["item_7a"].blocks[0].text == "Item 7A"


@pytest.mark.parametrize(("form", "last_item"), [("10-K", "item_16"), ("10-Q", "part2_item_6")])
def test_signatures_block_is_its_own_section(form, last_item):
    """Regression (BUG-3): signatures and power of attorney were glued onto the last Item."""
    heading = "Item 16. Form 10-K Summary" if form == "10-K" else "Item 6. Exhibits"
    blocks = _text(
        "PART II", heading, "None.", "SIGNATURES", "Pursuant to the requirements ...", "By: /s/ CEO"
    )

    sections = {s.key: s for s in split_sections(blocks, form).sections}

    assert [b.text for b in sections[last_item].blocks] == [heading, "None."]
    assert sections["signatures"].blocks[0].text == "SIGNATURES"


def test_word_signature_inside_prose_is_not_a_heading():
    assert _match_heading("Signature pages are attached.") is None
