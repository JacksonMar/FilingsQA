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


dataset_app = typer.Typer(help="Golden dataset: build, validate, stats.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

DATASETS_DIR = Path("eval/datasets")
HANDWRITTEN_DIR = DATASETS_DIR / "handwritten"


def _dataset_path(version: str) -> Path:
    return DATASETS_DIR / f"golden_{version}.jsonl"


@dataset_app.command("build")
def dataset_build(
    version: Annotated[str, typer.Option(help="Dataset version, e.g. v1")],
    force: Annotated[bool, typer.Option(help="Overwrite an existing version")] = False,
) -> None:
    """Generate XBRL items, merge hand-written ones, write golden_<version>.jsonl + meta."""
    from filingsqa.evaluation.builder import build_dataset, load_companyfacts, load_handwritten
    from filingsqa.evaluation.dataset import build_meta, dump_items

    out = _dataset_path(version)
    if out.exists() and not force:
        raise typer.BadParameter(f"{out} exists; released versions are immutable (use --force)")
    manifest = Manifest.load(MANIFEST_PATH)
    facts = load_companyfacts(FACTS_DIR, {f.ticker for f in manifest.filings})
    items, warnings = build_dataset(manifest, facts, load_handwritten(HANDWRITTEN_DIR))
    for w in warnings:
        typer.echo(f"WARN {w}")
    sha = dump_items(items, out)
    meta = build_meta(items, version, sha, manifest.as_of)
    out.with_suffix(".meta.json").write_text(meta.model_dump_json(indent=2) + "\n")
    typer.echo(f"Wrote {len(items)} items to {out}: {meta.by_category}")


@dataset_app.command("validate")
def dataset_validate(
    version: Annotated[str, typer.Option()] = "v1",
    with_chunks: Annotated[bool, typer.Option(help="Also check answerability vs chunks")] = True,
) -> None:
    """Dataset QA: structure, traceability to the manifest, answerability in the corpus."""
    from filingsqa.evaluation.dataset import DatasetMeta, file_sha256, load_items
    from filingsqa.evaluation.dataset_qa import check_dataset

    path = _dataset_path(version)
    meta = DatasetMeta.model_validate_json(path.with_suffix(".meta.json").read_text())
    if file_sha256(path) != meta.sha256:
        typer.echo(f"ERROR {path} was modified after release (sha256 mismatch)")
        raise typer.Exit(1)
    chunks = load_chunks(CHUNKS_PATH) if with_chunks and CHUNKS_PATH.exists() else None
    report = check_dataset(load_items(path), Manifest.load(MANIFEST_PATH), chunks)
    for w in report.warnings:
        typer.echo(f"WARN  {w}")
    for e in report.errors:
        typer.echo(f"ERROR {e}")
    typer.echo(f"{len(report.errors)} errors, {len(report.warnings)} warnings")
    if not report.ok:
        raise typer.Exit(1)


@dataset_app.command("check-handwritten")
def dataset_check_handwritten() -> None:
    """Schema-validate the hand-written source files (works without SEC data)."""
    from filingsqa.evaluation.builder import load_handwritten
    from filingsqa.evaluation.dataset_qa import check_dataset

    items = load_handwritten(HANDWRITTEN_DIR)
    empty = Manifest(as_of=date.today(), filings=[])
    report = check_dataset(items, empty, min_per_category={})
    errors = [e for e in report.errors if "source" not in e]  # sources resolve only at build
    for e in errors:
        typer.echo(f"ERROR {e}")
    typer.echo(f"{len(items)} hand-written items OK" if not errors else f"{len(errors)} errors")
    if errors:
        raise typer.Exit(1)


eval_app = typer.Typer(help="Run the evaluation suite.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
THRESHOLDS_PATH = Path("eval/thresholds.yaml")
REPORTS_DIR = Path("eval/reports")


@eval_app.command("run")
def eval_run(
    version: Annotated[str, typer.Option(help="Golden dataset version")] = "v1",
    subset: Annotated[str, typer.Option(help="full | smoke")] = "full",
    judge: Annotated[bool, typer.Option(help="Also run LLM-judge metrics")] = True,
    limit: Annotated[int | None, typer.Option(help="Only the first N items (debugging)")] = None,
    category: Annotated[str | None, typer.Option(help="Only this category")] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir (default eval/reports/<ts>)")] = None,
) -> None:
    """Ask every golden question, score the answers, write results.json + report.md, gate."""
    from datetime import datetime

    from filingsqa.evaluation.dataset import DatasetMeta, load_items
    from filingsqa.evaluation.report import save_run
    from filingsqa.evaluation.runner import EvalRunner, smoke_subset
    from filingsqa.evaluation.thresholds import load_thresholds
    from filingsqa.llm.cache import CachingLLM
    from filingsqa.llm.factory import make_llm
    from filingsqa.rag.factory import build_pipeline
    from filingsqa.rag.prompts import PROMPT_VERSION

    settings = get_settings()
    path = _dataset_path(version)
    items = load_items(path)
    if subset == "smoke":
        items = smoke_subset(items)
    if category:
        items = [i for i in items if i.category == category]
    if limit:
        items = items[:limit]

    pipeline = build_pipeline(settings)
    judge_suite = None
    if judge:
        from filingsqa.evaluation.metrics.judge import JudgeSuite

        judge_suite = JudgeSuite(make_llm(settings, "judge"))
    meta = DatasetMeta.model_validate_json(path.with_suffix(".meta.json").read_text())
    runner = EvalRunner(
        pipeline,
        judge_suite,
        metadata={
            "dataset_version": version,
            "dataset_sha256": meta.sha256[:12],
            "subset": subset,
            "prompt_version": PROMPT_VERSION,
            "generator": f"{settings.llm_model} (effort={settings.llm_effort})",
            "judge_model": f"{settings.judge_model} (effort={settings.judge_effort})"
            if judge
            else None,
        },
    )
    run = runner.run(items, load_thresholds(THRESHOLDS_PATH, subset))
    llm = pipeline.generator.llm
    if isinstance(llm, CachingLLM):
        run.metadata["llm_cache"] = f"hits={llm.hits} misses={llm.misses}"
    out_dir = out or REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{version}-{subset}"
    json_path, md_path = save_run(run, out_dir)
    typer.echo(md_path.read_text().split("## Mean score")[0])
    typer.echo(f"Results: {json_path}\nReport:  {md_path}")
    if not run.passed:
        raise typer.Exit(1)


@eval_app.command("label-sample")
def eval_label_sample(
    run: Annotated[Path, typer.Option(help="results.json of a judged run")],
    metric: Annotated[str, typer.Option()] = "fiscal_period_correct",
    n: Annotated[int, typer.Option()] = 30,
) -> None:
    """Export answers for human labeling (judge verdict hidden)."""
    from filingsqa.evaluation.judge_agreement import label_sample, save_labels
    from filingsqa.evaluation.runner import RunResult

    rows = label_sample(RunResult.model_validate_json(run.read_text()), metric, n)
    out = Path("eval/judge_agreement") / f"labels_{metric}.jsonl"
    if out.exists():
        raise typer.BadParameter(f"{out} exists; refusing to overwrite human labels")
    save_labels(rows, out)
    typer.echo(f"Wrote {len(rows)} rows to {out}; fill human_pass true/false")


@eval_app.command("judge-agreement")
def eval_judge_agreement(
    run: Annotated[Path, typer.Option()],
    labels: Annotated[Path, typer.Option()],
) -> None:
    """Agreement and Cohen's kappa between the LLM judge and human labels."""
    from filingsqa.evaluation.judge_agreement import compute_agreement, load_labels
    from filingsqa.evaluation.runner import RunResult

    a = compute_agreement(RunResult.model_validate_json(run.read_text()), load_labels(labels))
    typer.echo(f"n={a.n} accuracy={a.accuracy:.2f} kappa={a.kappa:.2f} skipped={a.skipped}")
    typer.echo(f"judge too lenient (false pass)={a.fp}  judge too strict (false fail)={a.fn}")


if __name__ == "__main__":
    app()
