"""Assemble a golden dataset version from generated (XBRL) and hand-written items."""

import json
from pathlib import Path

from filingsqa.edgar.manifest import Manifest
from filingsqa.evaluation.dataset import ExpectedSource, GoldenItem, load_items
from filingsqa.evaluation.generators import CompanyFacts, generate_all


def load_handwritten(directory: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for path in sorted(directory.glob("*.jsonl")):
        items += load_items(path)
    return items


def load_companyfacts(directory: Path, tickers: set[str]) -> dict[str, CompanyFacts]:
    out: dict[str, CompanyFacts] = {}
    for ticker in sorted(tickers):
        path = directory / f"{ticker}.json"
        if path.exists():
            out[ticker] = json.loads(path.read_text(encoding="utf-8"))
    return out


def resolve_source(source: ExpectedSource, manifest: Manifest) -> ExpectedSource:
    """Hand-written items say "AAPL 10-K Item 1A"; pin that to the accession in the manifest.

    For 10-Q without an explicit accession, the latest 10-Q is used.
    """
    if source.accession is not None:
        return source
    candidates = sorted(
        (f for f in manifest.filings if f.ticker == source.ticker and f.form == source.form),
        key=lambda f: f.report_date,
    )
    if not candidates:
        raise ValueError(f"no {source.form} for {source.ticker} in manifest")
    return source.model_copy(update={"accession": candidates[-1].accession})


def resolve_item(item: GoldenItem, manifest: Manifest) -> GoldenItem:
    if not item.expected_sources:
        return item
    sources = tuple(resolve_source(s, manifest) for s in item.expected_sources)
    return item.model_copy(update={"expected_sources": sources})


def build_dataset(
    manifest: Manifest,
    facts_by_ticker: dict[str, CompanyFacts],
    handwritten: list[GoldenItem],
) -> tuple[list[GoldenItem], list[str]]:
    generated, warnings = generate_all(manifest, facts_by_ticker)
    items = generated + [resolve_item(i, manifest) for i in handwritten]
    ids = [i.id for i in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate item ids: {dupes}")
    return items, warnings
