import pytest

from filingsqa.ingest.html_to_text import normalize_row, normalize_text, parse_filing_html


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        (["Net sales", "$", "410,235", "", "$", "391,035"], ["Net sales", "$410,235", "$391,035"]),
        (["Other", "(", "321", ")"], ["Other", "(321)"]),
        (["Other", "$", "(", "321", ")"], ["Other", "$(321)"]),
        (["Margin", "46.9", "%"], ["Margin", "46.9%"]),
        (["Change", "(4.1", ")%"], ["Change", "(4.1)%"]),
        (["", " ", "\xa0"], []),
    ],
    ids=["dollar-cell", "paren-negative", "dollar-paren", "percent", "paren-percent", "empty"],
)
def test_normalize_row_glues_split_cells(cells, expected):
    assert normalize_row(cells) == expected


def test_normalize_text_maps_unicode_punctuation_to_ascii():
    raw = "Management\N{RIGHT SINGLE QUOTATION MARK}s\N{NO-BREAK SPACE}Discussion \N{EM DASH} 2025"
    assert normalize_text(raw) == ("Management's Discussion - 2025")


@pytest.fixture
def parsed(sample_10k_html):
    return parse_filing_html(sample_10k_html)


@pytest.fixture
def full_text(parsed):
    return "\n".join(b.text for b in parsed.blocks)


def test_dei_facts_are_read_from_hidden_xbrl_header(parsed):
    assert parsed.dei["DocumentFiscalYearFocus"] == "2025"
    assert parsed.dei["DocumentFiscalPeriodFocus"] == "FY"
    assert parsed.dei["DocumentType"] == "10-K"
    assert parsed.dei["DocumentPeriodEndDate"] == "September 27, 2025"


@pytest.mark.parametrize(
    "forbidden",
    [
        "AmendmentFlag",
        "false",
        "hidden text",
        "shouldNotAppear",
        "a comment",
        "acme-20250927",
        "color:red",
    ],
)
def test_hidden_and_non_content_markup_is_dropped(full_text, forbidden):
    assert forbidden not in full_text


@pytest.mark.parametrize(
    "artifact", ["Acme Widgets, Inc. | 2025 Form 10-K | 1", "2", "TABLE OF CONTENTS"]
)
def test_page_furniture_is_removed(parsed, artifact):
    assert artifact not in [b.text for b in parsed.blocks]


def test_inline_xbrl_values_stay_in_text(full_text):
    assert "$410,235" in full_text


def test_layout_table_heading_is_flattened_to_text(parsed):
    assert any(b.kind == "text" and b.text == "Item 1A. Risk Factors" for b in parsed.blocks)


def test_financial_table_is_kept_as_rows(parsed):
    tables = [b for b in parsed.blocks if b.kind == "table"]
    financial = tables[-1].text.split("\n")

    assert "Net sales | $410,235 | $391,035" in financial
    assert "Other income/(expense), net | (321) | 269" in financial
    assert "Gross margin percentage | 46.9% | 46.2%" in financial
    assert "" not in financial  # empty rows dropped


def test_toc_table_stays_a_table_so_it_cannot_become_a_heading(parsed):
    toc = next(b for b in parsed.blocks if b.kind == "table")
    assert "Item 1A. | Risk Factors | 4" in toc.text


def test_block_elements_split_and_inline_elements_join():
    parsed = parse_filing_html("<div>Hello <b>big</b><span> world</span></div><p>Next</p>")
    assert [b.text for b in parsed.blocks] == ["Hello big world", "Next"]


# --- Regression (BUG-1): inline elements must be joined like a browser renders them ---


def test_dei_value_split_across_inline_spans():
    """MSFT 10-Q 0001193125-26-191507 writes the document type as <span>10-</span><span>Q</span>."""
    html = (
        '<html><body><ix:nonNumeric name="dei:DocumentType">'
        '<span style="font-weight:bold">10-</span><span style="text-transform:uppercase">Q</span>'
        "</ix:nonNumeric></body></html>"
    )

    assert parse_filing_html(html).dei["DocumentType"] == "10-Q"


def test_table_cell_inline_spans_are_not_split():
    html = (
        "<table><tr><td><span>Net</span> <span>sales</span></td><td><span>$</span><span>391,035</span></td></tr>"
        "<tr><td>Other</td><td>(<span>321</span>)</td></tr></table>"
    )

    rows = parse_filing_html(html).blocks[0].text.split("\n")

    assert rows == ["Net sales | $391,035", "Other | (321)"]


def test_table_cell_block_children_stay_separated():
    html = "<table><tr><td><p>Total</p><p>revenue</p></td><td>1<br/>2</td></tr><tr><td>a</td><td>b</td></tr></table>"

    rows = parse_filing_html(html).blocks[0].text.split("\n")

    assert rows[0] == "Total revenue | 1 2"
