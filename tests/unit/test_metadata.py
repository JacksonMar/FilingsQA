from datetime import date

import pytest

from filingsqa.ingest.metadata import MetadataError, build_filing_meta, parse_sec_date

DEI = {
    "DocumentType": "10-K",
    "DocumentFiscalYearFocus": "2025",
    "DocumentFiscalPeriodFocus": "FY",
    "DocumentPeriodEndDate": "September 27, 2025",
}


@pytest.mark.parametrize(
    "value",
    [
        "September 27, 2025",
        "Sep 27, 2025",
        "Sept. 27, 2025",
        "2025-09-27",
        "09/27/2025",
        "27 September 2025",
    ],
)
def test_parse_sec_date_formats(value):
    assert parse_sec_date(value) == date(2025, 9, 27)


def test_parse_sec_date_returns_none_for_garbage():
    assert parse_sec_date("sometime in autumn") is None


def test_meta_uses_fiscal_not_calendar_period(ref_10q):
    dei = {
        "DocumentType": "10-Q",
        "DocumentFiscalYearFocus": "2026",
        "DocumentFiscalPeriodFocus": "Q1",
        "DocumentPeriodEndDate": "December 27, 2025",
    }

    meta = build_filing_meta(ref_10q, dei)

    # Calendar 2025 quarter, but fiscal Q1 2026: the bot must say "Q1 FY2026".
    assert meta.period_end_date.year == 2025
    assert meta.period_label == "Q1 FY2026"


def test_meta_contains_source_url(ref_10k):
    meta = build_filing_meta(ref_10k, DEI)

    assert meta.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000090/acme-20250927.htm"
    )


def test_document_type_mismatch_fails_loudly(ref_10k):
    with pytest.raises(MetadataError, match="DocumentType"):
        build_filing_meta(ref_10k, {**DEI, "DocumentType": "10-Q"})


@pytest.mark.parametrize(
    "broken",
    [
        {"DocumentFiscalYearFocus": None},
        {"DocumentFiscalYearFocus": "FY25"},
        {"DocumentFiscalPeriodFocus": None},
        {"DocumentFiscalPeriodFocus": "H1"},
    ],
)
def test_missing_or_invalid_fiscal_facts_fail_loudly(ref_10k, broken):
    dei = {k: v for k, v in {**DEI, **broken}.items() if v is not None}

    with pytest.raises(MetadataError):
        build_filing_meta(ref_10k, dei)


def test_unparseable_period_end_falls_back_to_manifest(ref_10k):
    meta = build_filing_meta(ref_10k, {**DEI, "DocumentPeriodEndDate": "n/a"})

    assert meta.period_end_date == ref_10k.report_date
