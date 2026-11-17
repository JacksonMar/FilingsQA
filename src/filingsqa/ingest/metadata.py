"""Build per-filing metadata from the manifest entry and the filing's own DEI facts.

The fiscal year/period come from the inline XBRL `dei:` cover-page facts written by the
company itself, not from calendar dates. This matters: NVIDIA's fiscal 2025 ended in
January 2025, Apple's fiscal Q1 ends in December. See ADR-009.
"""

import logging
from datetime import date, datetime

from filingsqa.edgar.manifest import FilingRef
from filingsqa.ingest.models import FilingMeta

log = logging.getLogger(__name__)

_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%b. %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d %B %Y")
_VALID_PERIODS = {"FY", "Q1", "Q2", "Q3", "Q4"}


class MetadataError(ValueError):
    pass


def parse_sec_date(value: str) -> date | None:
    cleaned = value.replace("Sept.", "Sep.").replace("Sept ", "Sep ").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def build_filing_meta(ref: FilingRef, dei: dict[str, str]) -> FilingMeta:
    """Combine manifest data with DEI facts. Fails loudly on missing or contradictory data."""
    doc_type = dei.get("DocumentType")
    if doc_type and doc_type != ref.form:
        raise MetadataError(
            f"{ref.accession}: DEI DocumentType {doc_type!r} != manifest form {ref.form!r}"
        )

    try:
        fiscal_year = int(dei["DocumentFiscalYearFocus"])
        fiscal_period = dei["DocumentFiscalPeriodFocus"].upper()
    except (KeyError, ValueError) as exc:
        raise MetadataError(f"{ref.accession}: missing/invalid DEI fiscal year or period") from exc
    if fiscal_period not in _VALID_PERIODS:
        raise MetadataError(f"{ref.accession}: unexpected fiscal period {fiscal_period!r}")

    period_end = parse_sec_date(dei.get("DocumentPeriodEndDate", ""))
    if period_end is None:
        log.warning(
            "%s: unparseable DocumentPeriodEndDate, using manifest reportDate", ref.accession
        )
        period_end = ref.report_date
    elif period_end != ref.report_date:
        log.warning(
            "%s: DEI period end %s != manifest reportDate %s",
            ref.accession,
            period_end,
            ref.report_date,
        )

    return FilingMeta(
        ticker=ref.ticker,
        company=ref.company,
        cik=ref.cik,
        form=ref.form,
        accession=ref.accession,
        filing_date=ref.filing_date,
        period_end_date=period_end,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        source_url=ref.url,
    )
