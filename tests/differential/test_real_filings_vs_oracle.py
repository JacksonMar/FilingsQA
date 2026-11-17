"""Differential test on real downloaded filings (`make ingest` first). Run with `make test-diff`.

Writes a JSON report of all disagreements; each one is a candidate for docs/BUGS_FOUND.md.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from filingsqa.edgar.manifest import Manifest
from tests.differential.oracle import compare

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifest/filings.json"
RAW_DIR = ROOT / "data/raw/filings"
REPORT = ROOT / "data/processed/differential_report.json"


def _filings():
    if not MANIFEST.exists():
        return []
    return [f for f in Manifest.load(MANIFEST).filings if (RAW_DIR / f.raw_filename).exists()]


FILINGS = _filings()


@pytest.fixture(scope="module")
def report():
    results: dict[str, dict] = {}
    yield results
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))


@pytest.mark.differential
@pytest.mark.skipif(not FILINGS, reason="no downloaded filings; run `make ingest` first")
@pytest.mark.parametrize("ref", FILINGS, ids=lambda r: f"{r.ticker}-{r.form}-{r.accession}")
def test_sections_match_oracle(ref, report):
    html = (RAW_DIR / ref.raw_filename).read_text(encoding="utf-8")

    result = compare(html, ref.form)

    report[f"{ref.ticker} {ref.form} {ref.accession}"] = asdict(result)
    assert result.agrees, result
