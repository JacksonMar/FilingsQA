"""Data models produced by ingestion."""

from datetime import date
from typing import Literal

from pydantic import BaseModel

FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class FilingMeta(BaseModel):
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: date
    period_end_date: date
    fiscal_year: int
    fiscal_period: FiscalPeriod
    source_url: str

    @property
    def period_label(self) -> str:
        """Human-readable period, e.g. 'FY2025' or 'Q2 FY2026' (fiscal, not calendar)."""
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"{self.fiscal_period} FY{self.fiscal_year}"


class Chunk(BaseModel):
    chunk_id: str
    text: str
    context: str  # short header with company / form / period / section, prepended for embedding
    content_type: Literal["text", "table"]
    section_key: str
    section_title: str
    chunk_index: int  # position within the section
    word_count: int
    filing: FilingMeta

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n\n{self.text}"
