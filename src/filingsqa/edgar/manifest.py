"""Filing selection and the frozen filings manifest.

The manifest pins exact accession numbers so that the corpus (and therefore the golden
dataset) does not change when a company files a new report. See ADR-005.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from filingsqa.companies import Company
from filingsqa.edgar.client import document_url

ANNUAL_FORM = "10-K"
QUARTERLY_FORM = "10-Q"


class FilingRef(BaseModel):
    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    filing_date: date
    report_date: date
    primary_document: str

    @property
    def url(self) -> str:
        return document_url(self.cik, self.accession, self.primary_document)

    @property
    def raw_filename(self) -> str:
        return f"{self.ticker}_{self.form}_{self.accession}.htm"


class Manifest(BaseModel):
    as_of: date
    filings: list[FilingRef]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def parse_submissions(company: Company, submissions: dict[str, Any]) -> list[FilingRef]:
    """Turn the column-oriented `filings.recent` block of the submissions API into rows."""
    recent = submissions["filings"]["recent"]
    refs: list[FilingRef] = []
    for i, form in enumerate(recent["form"]):
        if form not in (ANNUAL_FORM, QUARTERLY_FORM):
            continue  # also excludes amendments such as 10-K/A
        report_date = recent["reportDate"][i]
        if not report_date:
            continue
        refs.append(
            FilingRef(
                ticker=company.ticker,
                cik=company.cik,
                company=company.name,
                form=form,
                accession=recent["accessionNumber"][i],
                filing_date=date.fromisoformat(recent["filingDate"][i]),
                report_date=date.fromisoformat(report_date),
                primary_document=recent["primaryDocument"][i],
            )
        )
    return refs


def select_filings(
    refs: list[FilingRef], as_of: date, n_annual: int = 1, n_quarterly: int = 2
) -> list[FilingRef]:
    """Latest `n_annual` 10-K and latest `n_quarterly` 10-Q filed on or before `as_of`."""
    eligible = sorted(
        (r for r in refs if r.filing_date <= as_of),
        key=lambda r: (r.filing_date, r.accession),
        reverse=True,
    )
    annual = [r for r in eligible if r.form == ANNUAL_FORM][:n_annual]
    quarterly = [r for r in eligible if r.form == QUARTERLY_FORM][:n_quarterly]
    if len(annual) < n_annual or len(quarterly) < n_quarterly:
        ticker = refs[0].ticker if refs else "?"
        raise ValueError(
            f"{ticker}: expected {n_annual}x10-K + {n_quarterly}x10-Q before {as_of}, "
            f"found {len(annual)} + {len(quarterly)}"
        )
    return sorted(annual + quarterly, key=lambda r: r.filing_date)
