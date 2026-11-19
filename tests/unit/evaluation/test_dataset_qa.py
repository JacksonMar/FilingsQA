from datetime import date

import pytest

from filingsqa.evaluation.dataset_qa import check_dataset
from filingsqa.evaluation.generators import generate_all
from filingsqa.ingest.models import Chunk, FilingMeta
from tests.unit.evaluation.conftest import AAPL_K

NO_MIN = {}


def _chunk(text, section="item_8", accession=AAPL_K, fy=2025, fp="FY"):
    meta = FilingMeta(
        ticker="AAPL", company="AAPL", cik=1, form="10-K", accession=accession,
        filing_date=date(2025, 10, 31), period_end_date=date(2025, 9, 27), fiscal_year=fy,
        fiscal_period=fp, source_url="https://x",
    )  # fmt: skip
    return Chunk(chunk_id=f"c-{section}-{hash(text)}", text=text, context="", content_type="table",
                 section_key=section, section_title=section, chunk_index=0, word_count=1, filing=meta)  # fmt: skip


@pytest.fixture
def revenue_item(manifest, facts_by_ticker):
    items, _ = generate_all(manifest, facts_by_ticker)
    return next(i for i in items if i.id == "num-aapl-revenue-2025fy")


def test_generated_dataset_is_structurally_clean(manifest, facts_by_ticker):
    items, _ = generate_all(manifest, facts_by_ticker)

    report = check_dataset(items, manifest, min_per_category=NO_MIN)

    assert report.errors == []


def test_duplicate_question_is_an_error(manifest, revenue_item):
    twin = revenue_item.model_copy(update={"id": "num-twin"})

    report = check_dataset([revenue_item, twin], manifest, min_per_category=NO_MIN)

    assert any("duplicate question" in e for e in report.errors)


def test_number_present_in_expected_section_passes(manifest, revenue_item):
    chunks = [_chunk("Total net sales | $400,000 | $390,000")]

    report = check_dataset([revenue_item], manifest, chunks, NO_MIN)

    assert report.errors == []
    assert report.warnings == []


def test_number_only_elsewhere_in_filing_is_a_warning(manifest, revenue_item):
    chunks = [_chunk("Total net sales were $400,000 million.", section="item_1")]

    report = check_dataset([revenue_item], manifest, chunks, NO_MIN)

    assert report.errors == []
    assert any("found elsewhere" in w for w in report.warnings)


def test_number_missing_from_corpus_is_an_error(manifest, revenue_item):
    report = check_dataset([revenue_item], manifest, [_chunk("Total net sales | $1 | $2")], NO_MIN)

    assert any("nor anywhere" in e for e in report.errors)


def test_dei_vs_xbrl_fiscal_period_mismatch_is_an_error(manifest, revenue_item):
    chunks = [_chunk("Total net sales | $400,000", fy=2026)]

    report = check_dataset([revenue_item], manifest, chunks, NO_MIN)

    assert any("fiscal period mismatch" in e for e in report.errors)


def test_number_outside_manifest_is_an_error(manifest, revenue_item):
    other = manifest.model_copy(
        update={"filings": [f for f in manifest.filings if f.accession != AAPL_K]}
    )

    report = check_dataset([revenue_item], other, min_per_category=NO_MIN)

    assert any("outside manifest" in e for e in report.errors)


def test_category_coverage_warning(manifest, revenue_item):
    report = check_dataset([revenue_item], manifest)

    assert any("category qualitative" in w for w in report.warnings)
