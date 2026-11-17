from datetime import date
from itertools import pairwise

import pytest

from filingsqa.ingest.chunking import (
    ChunkingConfig,
    chunk_sections,
    split_long_text,
    split_table,
    word_count,
)
from filingsqa.ingest.html_to_text import Block
from filingsqa.ingest.models import FilingMeta
from filingsqa.ingest.sections import Section


@pytest.fixture
def meta() -> FilingMeta:
    return FilingMeta(
        ticker="NVDA",
        company="NVIDIA Corporation",
        cik=1045810,
        form="10-K",
        accession="0001045810-25-000023",
        filing_date=date(2025, 2, 26),
        period_end_date=date(2025, 1, 26),
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
    )


def _para(n_words: int, tag: str) -> Block:
    return Block("text", " ".join(f"{tag}{i}" for i in range(n_words)) + ".")


CFG = ChunkingConfig(max_words=100, overlap_words=30, max_table_words=50, max_caption_words=15)


def test_chunks_never_cross_section_boundaries(meta):
    sections = [
        Section("item_1", "Business", [_para(60, "a")]),
        Section("item_1a", "Risk Factors", [_para(60, "b")]),
    ]

    chunks = chunk_sections(sections, meta, CFG)

    assert [c.section_key for c in chunks] == ["item_1", "item_1a"]
    assert "b0" not in chunks[0].text
    assert "a0" not in chunks[1].text


def test_text_chunks_respect_max_words_and_overlap(meta):
    paragraphs = [_para(25, f"p{i}x") for i in range(10)]  # 250 words in 25-word paragraphs

    chunks = chunk_sections([Section("item_7", "MD&A", paragraphs)], meta, CFG)

    assert len(chunks) > 1
    assert all(c.word_count <= CFG.max_words for c in chunks)
    for prev, nxt in pairwise(chunks):
        last_paragraph = prev.text.split("\n\n")[-1]
        assert nxt.text.startswith(last_paragraph)  # overlap = whole trailing paragraph


def test_every_paragraph_is_covered(meta):
    paragraphs = [_para(25, f"p{i}x") for i in range(10)]

    chunks = chunk_sections([Section("item_7", "MD&A", paragraphs)], meta, CFG)

    joined = "\n".join(c.text for c in chunks)
    assert all(p.text in joined for p in paragraphs)


def test_long_paragraph_is_split_on_sentences():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(40))

    pieces = split_long_text(text, max_words=50)

    assert all(word_count(p) <= 50 for p in pieces)
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


def test_giant_sentence_without_punctuation_is_hard_split():
    pieces = split_long_text("word " * 250, max_words=100)

    assert [word_count(p) for p in pieces] == [100, 100, 50]


def test_table_chunk_includes_caption_with_units(meta):
    section = Section(
        "item_8",
        "Financial Statements",
        [
            Block("text", "(In millions)"),
            Block("table", "| 2025 | 2024\nRevenue | $130,497 | $60,922"),
        ],
    )

    chunks = chunk_sections([section], meta, CFG)

    table = next(c for c in chunks if c.content_type == "table")
    assert table.text.startswith("(In millions)\n")
    assert "Revenue | $130,497" in table.text


def test_long_paragraph_is_not_used_as_caption(meta):
    section = Section("item_8", "FS", [_para(40, "w"), Block("table", "a | 1\nb | 2")])

    table = next(c for c in chunk_sections([section], meta, CFG) if c.content_type == "table")

    assert table.text == "a | 1\nb | 2"


def test_big_table_is_split_by_rows_and_repeats_header():
    rows = ["Header | 2025 | 2024"] + [f"Line item {i} | {i},000 | {i},500" for i in range(40)]

    pieces = split_table("\n".join(rows), max_words=50)

    assert len(pieces) > 1
    assert all(p.split("\n")[0] == rows[0] for p in pieces)
    body = [r for p in pieces for r in p.split("\n")[1:]]
    assert body == rows[1:]


def test_chunk_metadata_and_context(meta):
    chunks = chunk_sections([Section("item_1a", "Risk Factors", [_para(10, "r")])], meta, CFG)

    chunk = chunks[0]
    assert chunk.chunk_id == "NVDA-0001045810-25-000023-item_1a-000"
    assert chunk.filing.fiscal_year == 2025
    assert "FY2025 (period ended 2025-01-26)" in chunk.context
    assert "Risk Factors" in chunk.context
    assert chunk.embedding_text.startswith(chunk.context)


def test_chunking_is_deterministic(meta):
    sections = [Section("item_7", "MD&A", [_para(25, f"p{i}x") for i in range(10)])]

    assert chunk_sections(sections, meta, CFG) == chunk_sections(sections, meta, CFG)


def test_empty_section_produces_no_chunks(meta):
    assert chunk_sections([Section("item_1b", "Unresolved Staff Comments", [])], meta, CFG) == []
