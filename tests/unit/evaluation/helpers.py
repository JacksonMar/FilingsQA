"""Builders for golden items and bot responses used by the eval-metric tests."""

from datetime import date

from filingsqa.evaluation.dataset import ExpectedNumber, ExpectedSource, GoldenItem, XbrlRef
from filingsqa.rag.schemas import AnswerMeta, AskResponse, Citation, RetrievedChunk

ACC = "0001045810-25-000023"


def number(value=130_497e6, unit="USD", fy=2025, fp="FY", ticker="NVDA", tol_rel=0.01, tol_abs=0.0):
    return ExpectedNumber(
        label="revenue", ticker=ticker, value=value, unit=unit, fiscal_year=fy, fiscal_period=fp,
        period_end=date(2025, 1, 26), tolerance_rel=tol_rel, tolerance_abs=tol_abs,
        xbrl=XbrlRef(taxonomy="us-gaap", tag="Revenues", unit=unit, accession=ACC, start=None, end=date(2025, 1, 26)),
    )  # fmt: skip


def item(**kw):
    base = dict(
        id="num-nvda-revenue-2025fy", category="numeric", subcategory="annual", difficulty="easy",
        question="What was NVIDIA's total revenue for fiscal year 2025?", expected_behavior="answer",
        expected_numbers=(number(),), source="generated", verified=True,
        expected_sources=(ExpectedSource(ticker="NVDA", form="10-K", section_keys=("item_8", "item_7"), accession=ACC),),
    )  # fmt: skip
    base.update(kw)
    return GoldenItem(**base)


def chunk(
    doc_id=1, text="Revenue | $130,497 | $60,922", section="item_8", accession=ACC, ticker="NVDA"
):
    return RetrievedChunk(
        doc_id=doc_id, chunk_id=f"{ticker}-{accession}-{section}-{doc_id:03d}", score=0.5, ticker=ticker,
        form="10-K", accession=accession, period_label="FY2025", section_key=section,
        section_title=section, content_type="table", source_url="https://www.sec.gov/x", text=text,
    )  # fmt: skip


def response(answer, answer_type="answered", chunks=None, cited=(1,), invalid=()):
    chunks = chunks if chunks is not None else [chunk()]
    citations = [
        Citation(doc_id=d, chunk_id=chunks[d - 1].chunk_id, ticker=chunks[d - 1].ticker, form="10-K",
                 period_label="FY2025", section_title="x", source_url="https://www.sec.gov/x")
        for d in cited if 1 <= d <= len(chunks)
    ]  # fmt: skip
    return AskResponse(
        answer=answer, answer_type=answer_type, citations=citations, retrieved_chunks=chunks,
        applied_tickers=["NVDA"], invalid_citation_ids=list(invalid), meta=AnswerMeta(prompt_version="v1"),
    )  # fmt: skip
