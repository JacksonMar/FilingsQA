import json
from datetime import date

import pytest

from filingsqa.companies import get_company
from filingsqa.edgar.manifest import Manifest, parse_submissions, select_filings
from tests.conftest import FIXTURES


@pytest.fixture
def refs():
    data = json.loads((FIXTURES / "submissions_sample.json").read_text())
    return parse_submissions(get_company("AAPL"), data)


def test_only_10k_and_10q_are_kept(refs):
    assert {r.form for r in refs} == {"10-K", "10-Q"}  # 10-K/A and 8-K excluded


def test_select_latest_10k_and_two_10q_as_of_date(refs):
    selected = select_filings(refs, as_of=date(2026, 3, 1))

    assert [(r.form, r.report_date.isoformat()) for r in selected] == [
        ("10-Q", "2025-06-28"),
        ("10-K", "2025-09-27"),
        ("10-Q", "2025-12-27"),
    ]


def test_filings_after_as_of_are_ignored(refs):
    selected = select_filings(refs, as_of=date(2026, 3, 1))

    assert all(r.filing_date <= date(2026, 3, 1) for r in selected)


def test_not_enough_filings_raises(refs):
    with pytest.raises(ValueError, match="expected"):
        select_filings(refs, as_of=date(2025, 6, 1))  # no 10-K filed yet


def test_manifest_roundtrip(refs, tmp_path):
    manifest = Manifest(as_of=date(2026, 3, 1), filings=select_filings(refs, date(2026, 3, 1)))
    path = tmp_path / "m.json"

    manifest.save(path)

    assert Manifest.load(path) == manifest


def test_unknown_ticker_has_helpful_error():
    with pytest.raises(KeyError, match="supported"):
        get_company("ZZZZ")


def test_cik_is_zero_padded():
    assert get_company("aapl").cik10 == "0000320193"
