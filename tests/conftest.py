from datetime import date
from pathlib import Path

import pytest

from filingsqa.config import get_settings
from filingsqa.edgar.manifest import FilingRef

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Keep tests independent of the developer's real `.env` and environment."""
    monkeypatch.chdir(tmp_path)  # no .env in tmp_path
    for var in ("ANTHROPIC_API_KEY", "LLM_MODEL", "JUDGE_MODEL", "SEC_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_10k_html() -> str:
    return (FIXTURES / "sample_10k.html").read_text(encoding="utf-8")


@pytest.fixture
def sample_10q_html() -> str:
    return (FIXTURES / "sample_10q.html").read_text(encoding="utf-8")


@pytest.fixture
def ref_10k() -> FilingRef:
    return FilingRef(
        ticker="AAPL",
        cik=320193,
        company="Acme Widgets, Inc.",
        form="10-K",
        accession="0000320193-25-000090",
        filing_date=date(2025, 10, 31),
        report_date=date(2025, 9, 27),
        primary_document="acme-20250927.htm",
    )


@pytest.fixture
def ref_10q() -> FilingRef:
    return FilingRef(
        ticker="AAPL",
        cik=320193,
        company="Acme Widgets, Inc.",
        form="10-Q",
        accession="0000320193-26-000005",
        filing_date=date(2026, 1, 30),
        report_date=date(2025, 12, 27),
        primary_document="acme-20251227.htm",
    )
