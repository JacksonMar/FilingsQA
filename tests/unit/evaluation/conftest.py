"""Synthetic companyfacts + manifest. Values are invented; the structure mirrors the SEC API."""

from datetime import date

import pytest

from filingsqa.edgar.manifest import FilingRef, Manifest

AAPL_K, AAPL_K_OLD, AAPL_Q1, AAPL_Q3 = (
    "0000320193-25-000079",
    "0000320193-24-000123",
    "0000320193-26-000006",
    "0000320193-25-000073",
)
MSFT_K = "0000950170-25-100235"


def e(val, end, accn, fy, fp, start=None, form="10-K", filed="2025-10-31"):
    entry = {"val": val, "end": end, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        entry["start"] = start
    return entry


def facts(tags: dict[str, dict[str, list]]) -> dict:
    return {"facts": {"us-gaap": {tag: {"units": units} for tag, units in tags.items()}}}


@pytest.fixture
def aapl_facts():
    k, q1, q3 = AAPL_K, AAPL_Q1, AAPL_Q3
    return facts(
        {
            "Revenues": {
                "USD": [
                    e(400e9, "2025-09-27", k, 2025, "FY", "2024-09-29"),
                    e(
                        390e9, "2024-09-28", k, 2025, "FY", "2023-10-01"
                    ),  # comparative in the same 10-K
                    e(380e9, "2023-09-30", k, 2025, "FY", "2022-10-02"),
                    e(
                        390e9, "2024-09-28", AAPL_K_OLD, 2024, "FY", "2023-10-01"
                    ),  # older 10-K, not in manifest
                    e(120e9, "2025-12-27", q1, 2026, "Q1", "2025-09-28", "10-Q"),
                    e(110e9, "2024-12-28", q1, 2026, "Q1", "2024-09-29", "10-Q"),
                    e(90e9, "2025-06-28", q3, 2025, "Q3", "2025-03-30", "10-Q"),
                    e(
                        300e9, "2025-06-28", q3, 2025, "Q3", "2024-09-29", "10-Q"
                    ),  # 9-month YTD: must be ignored
                ]
            },
            "NetIncomeLoss": {
                "USD": [
                    e(100e9, "2025-09-27", k, 2025, "FY", "2024-09-29"),
                    e(95e9, "2024-09-28", k, 2025, "FY", "2023-10-01"),
                    e(30e9, "2025-12-27", q1, 2026, "Q1", "2025-09-28", "10-Q"),
                ]
            },
            "EarningsPerShareDiluted": {
                "USD/shares": [e(6.5, "2025-09-27", k, 2025, "FY", "2024-09-29")]
            },
            "OperatingIncomeLoss": {
                "USD": [  # two values for one period in one filing -> ambiguous
                    e(120e9, "2025-09-27", k, 2025, "FY", "2024-09-29"),
                    e(121e9, "2025-09-27", k, 2025, "FY", "2024-09-29"),
                ]
            },
            "Assets": {
                "USD": [
                    e(360e9, "2025-09-27", k, 2025, "FY"),
                    e(350e9, "2024-09-28", k, 2025, "FY"),
                ]
            },
        }
    )


@pytest.fixture
def msft_facts():
    k = MSFT_K
    return facts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "USD": [  # fallback tag
                    e(280e9, "2025-06-30", k, 2025, "FY", "2024-07-01"),
                    e(245e9, "2024-06-30", k, 2025, "FY", "2023-07-01"),
                ]
            },
            "NetIncomeLoss": {
                "USD": [
                    e(101e9, "2025-06-30", k, 2025, "FY", "2024-07-01"),
                    e(88e9, "2024-06-30", k, 2025, "FY", "2023-07-01"),
                ]
            },
        }
    )


def ref(ticker, form, accn, filed, report):
    return FilingRef(ticker=ticker, cik=1, company=ticker, form=form, accession=accn,
                     filing_date=date.fromisoformat(filed), report_date=date.fromisoformat(report),
                     primary_document="x.htm")  # fmt: skip


@pytest.fixture
def manifest():
    return Manifest(
        as_of=date(2026, 2, 15),
        filings=[
            ref("AAPL", "10-Q", AAPL_Q3, "2025-08-01", "2025-06-28"),
            ref("AAPL", "10-K", AAPL_K, "2025-10-31", "2025-09-27"),
            ref("AAPL", "10-Q", AAPL_Q1, "2026-01-30", "2025-12-27"),
            ref("MSFT", "10-K", MSFT_K, "2025-07-30", "2025-06-30"),
        ],
    )


@pytest.fixture
def facts_by_ticker(aapl_facts, msft_facts):
    return {"AAPL": aapl_facts, "MSFT": msft_facts}
