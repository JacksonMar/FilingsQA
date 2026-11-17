"""Command-line interface: `filingsqa --help`."""

import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from filingsqa.companies import COMPANIES
from filingsqa.config import get_settings
from filingsqa.edgar.client import EdgarClient, submissions_url
from filingsqa.edgar.manifest import Manifest, parse_submissions, select_filings
from filingsqa.ingest.pipeline import (
    build_chunks,
    download_companyfacts,
    download_filings,
    load_chunks,
)

app = typer.Typer(help="FilingsQA: question answering over SEC filings.", no_args_is_help=True)

MANIFEST_PATH = Path("data/manifest/filings.json")
RAW_DIR = Path("data/raw/filings")
FACTS_DIR = Path("data/raw/companyfacts")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def _client() -> EdgarClient:
    s = get_settings()
    return EdgarClient(s.sec_user_agent, s.sec_max_requests_per_second)


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("build-manifest")
def build_manifest(
    as_of: Annotated[str, typer.Option(help="Freeze date, YYYY-MM-DD")],
    out: Path = MANIFEST_PATH,
) -> None:
    """Select the latest 10-K + two 10-Q per company as of a date and pin them in a manifest."""
    cutoff = date.fromisoformat(as_of)
    filings = []
    with _client() as client:
        for company in COMPANIES.values():
            refs = parse_submissions(company, client.get_json(submissions_url(company.cik10)))
            filings.extend(select_filings(refs, cutoff))
    Manifest(as_of=cutoff, filings=filings).save(out)
    typer.echo(f"Wrote {len(filings)} filings to {out}")


@app.command()
def ingest(
    manifest: Path = MANIFEST_PATH,
    raw_dir: Path = RAW_DIR,
    out: Path = CHUNKS_PATH,
    download: Annotated[bool, typer.Option(help="Download missing raw files")] = True,
) -> None:
    """Download pinned filings + XBRL companyfacts, then parse and chunk into JSONL."""
    m = Manifest.load(manifest)
    if download:
        with _client() as client:
            download_filings(m, raw_dir, client)
            download_companyfacts(sorted({f.ticker for f in m.filings}), FACTS_DIR, client)
    reports = build_chunks(m, raw_dir, out)
    for r in reports:
        flag = " WARN" if r.warnings else ""
        typer.echo(f"{r.ticker:5} {r.form:5} {r.period:12} {r.n_chunks:5} chunks{flag}")
    typer.echo(f"Total {sum(r.n_chunks for r in reports)} chunks -> {out}")


@app.command()
def index(chunks: Path = CHUNKS_PATH) -> None:
    """(Re)create the Qdrant collection and index all chunks (dense + BM25)."""
    from filingsqa.rag.factory import build_store

    store = build_store(get_settings())
    store.recreate_collection()
    n = store.index(load_chunks(chunks))
    typer.echo(f"Indexed {n} chunks into collection {store.collection!r}")


@app.command()
def ask(
    question: str,
    ticker: Annotated[list[str] | None, typer.Option("--ticker", "-t")] = None,
    show_chunks: bool = False,
) -> None:
    """Ask a question from the command line."""
    from filingsqa.rag.factory import build_pipeline
    from filingsqa.rag.schemas import AskRequest

    response = build_pipeline(get_settings()).ask(AskRequest(question=question, tickers=ticker))
    typer.echo(f"[{response.answer_type}] {response.answer}\n")
    for c in response.citations:
        typer.echo(f"  [{c.doc_id}] {c.ticker} {c.form} {c.period_label} - {c.section_title}")
        typer.echo(f"       {c.source_url}")
    if response.invalid_citation_ids:
        typer.echo(f"  ! invalid citation ids: {response.invalid_citation_ids}")
    if show_chunks:
        for ch in response.retrieved_chunks:
            typer.echo(f"\n--- [{ch.doc_id}] {ch.chunk_id} score={ch.score:.4f}\n{ch.text[:500]}")
    m = response.meta
    typer.echo(
        f"\n(model={m.model} prompt={m.prompt_version} cached={m.llm_cached} "
        f"tokens={m.input_tokens}/{m.output_tokens} latency={m.latency_ms}ms)"
    )


if __name__ == "__main__":
    app()
