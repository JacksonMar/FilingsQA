"""The fixed company universe. Chosen to span different sectors and fiscal calendars."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    fiscal_year_end: (
        str  # informational; the authoritative period comes from each filing's DEI tags
    )

    @property
    def cik10(self) -> str:
        """CIK zero-padded to 10 digits, as used by data.sec.gov URLs."""
        return f"{self.cik:010d}"


COMPANIES: dict[str, Company] = {
    c.ticker: c
    for c in (
        Company("AAPL", 320193, "Apple Inc.", "late September"),
        Company("MSFT", 789019, "Microsoft Corporation", "June 30"),
        Company("NVDA", 1045810, "NVIDIA Corporation", "late January"),
        Company("JPM", 19617, "JPMorgan Chase & Co.", "December 31"),
        Company("TSLA", 1318605, "Tesla, Inc.", "December 31"),
    )
}


def get_company(ticker: str) -> Company:
    try:
        return COMPANIES[ticker.upper()]
    except KeyError:
        raise KeyError(f"Unknown ticker {ticker!r}; supported: {sorted(COMPANIES)}") from None
