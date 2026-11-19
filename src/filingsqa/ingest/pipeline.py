"""Ingestion pipeline: manifest -> raw HTML (cached on disk) -> blocks -> sections -> chunks."""

import gzip
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from filingsqa.companies import COMPANIES
from filingsqa.edgar.client import EdgarClient, companyfacts_url
from filingsqa.edgar.manifest import FilingRef, Manifest
from filingsqa.ingest.chunking import ChunkingConfig, chunk_sections
from filingsqa.ingest.html_to_text import parse_filing_html
from filingsqa.ingest.metadata import build_filing_meta
from filingsqa.ingest.models import Chunk
from filingsqa.ingest.oracle_sections import resolve_sections
from filingsqa.ingest.sections import split_sections

log = logging.getLogger(__name__)


@dataclass
class FilingReport:
    ticker: str
    form: str
    accession: str
    period: str
    section_source: str = "own"  # "own" parser or "edgartools" boundaries (ADR-027)
    sections: dict[str, int] = field(default_factory=dict)  # section_key -> chunk count
    n_chunks: int = 0
    warnings: list[str] = field(default_factory=list)


def process_filing(
    ref: FilingRef, html: str, config: ChunkingConfig | None = None, use_oracle: bool = True
) -> tuple[list[Chunk], FilingReport]:
    """Pure function: no I/O. The whole parse -> chunk path is testable on a fixture."""
    parsed = parse_filing_html(html)
    meta = build_filing_meta(ref, parsed.dei)
    split = split_sections(parsed.blocks, ref.form)
    source = "own"
    if use_oracle:
        split, source = resolve_sections(html, parsed.blocks, split, ref.form)
    chunks = chunk_sections(split.sections, meta, config)

    report = FilingReport(ref.ticker, ref.form, ref.accession, meta.period_label, source)
    for chunk in chunks:
        report.sections[chunk.section_key] = report.sections.get(chunk.section_key, 0) + 1
    report.n_chunks = len(chunks)
    report.warnings = split.warnings
    return chunks, report


def download_filings(manifest: Manifest, raw_dir: Path, client: EdgarClient) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ref in manifest.filings:
        path = raw_dir / ref.raw_filename
        if path.exists():
            continue
        log.info("Downloading %s %s %s", ref.ticker, ref.form, ref.accession)
        path.write_text(client.get_text(ref.url), encoding="utf-8")


def download_companyfacts(tickers: list[str], out_dir: Path, client: EdgarClient) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        company = COMPANIES[ticker]
        path = out_dir / f"{ticker}.json"
        if path.exists():
            continue
        log.info("Downloading companyfacts for %s", ticker)
        path.write_text(
            json.dumps(client.get_json(companyfacts_url(company.cik10))), encoding="utf-8"
        )


def build_chunks(
    manifest: Manifest, raw_dir: Path, out_path: Path, config: ChunkingConfig | None = None
) -> list[FilingReport]:
    all_chunks: list[Chunk] = []
    reports: list[FilingReport] = []
    for ref in manifest.filings:
        html = (raw_dir / ref.raw_filename).read_text(encoding="utf-8")
        chunks, report = process_filing(ref, html, config)
        for warning in report.warnings:
            log.warning("%s %s %s: %s", ref.ticker, ref.form, ref.accession, warning)
        all_chunks.extend(chunks)
        reports.append(report)

    ids = [c.chunk_id for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate chunk IDs produced")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(chunk.model_dump_json() + "\n")
    report_path = out_path.with_name("ingest_report.json")
    report_path.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    return reports


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks from JSONL, or gzip-compressed JSONL when the name ends with .gz."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(c.model_dump_json() + "\n" for c in chunks).encode("utf-8")
    if path.suffix == ".gz":
        # mtime=0 -> byte-identical output for identical input (clean git diffs)
        path.write_bytes(gzip.compress(lines, mtime=0))
    else:
        path.write_bytes(lines)
