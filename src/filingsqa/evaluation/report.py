"""Render a run as Markdown (for humans / PR summaries) and save JSON (for machines)."""

from pathlib import Path

from filingsqa.evaluation.runner import RunResult

CATEGORY_ORDER = ["numeric", "qualitative", "comparison", "temporal", "adversarial"]
METRIC_ORDER = [
    "behavior_match", "numeric_accuracy", "period_mentioned", "citation_validity",
    "forbidden_phrases", "key_facts_coverage", "recall_at_k", "mrr", "contextual_precision",
    "context_has_numbers", "faithfulness", "answer_relevancy", "hallucination_free",
    "fiscal_period_correct", "answer_correctness",
]  # fmt: skip
ICON = {"pass": "PASS", "fail": "FAIL", "skipped": "skip"}


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_markdown(run: RunResult, max_failures: int = 50) -> str:
    md = run.metadata
    lines = [
        f"# FilingsQA eval report - {'PASSED' if run.passed else 'FAILED'}",
        "",
        "| | |",
        "|---|---|",
    ]
    for key in ("dataset_version", "dataset_sha256", "subset", "n_items", "prompt_version",
                "generator", "judge_model", "git_commit", "finished_at", "duration_s",
                "llm_cache"):  # fmt: skip
        if key in md:
            lines.append(f"| {key} | `{md[key]}` |")

    lines += ["", "## Quality gates", "", "| scope | metric | min | actual | n | status |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for t in run.thresholds:
        lines.append(
            f"| {t['scope']} | {t['metric']} | {t['minimum']:.2f} | {_fmt(t['actual'])} | "
            f"{t['n']} | {ICON[t['status']]} |"
        )

    scopes = ["overall"] + [c for c in CATEGORY_ORDER if c in run.summary]
    metrics = [m for m in METRIC_ORDER if any(m in run.summary.get(s, {}) for s in scopes)]
    lines += ["", "## Mean score per category", "", "| metric | " + " | ".join(scopes) + " |",
              "|---|" + "---|" * len(scopes)]  # fmt: skip
    for m in metrics:
        row = []
        for s in scopes:
            stats = run.summary.get(s, {}).get(m)
            row.append(f"{stats['mean']:.2f} (n={int(stats['n'])})" if stats else "-")
        lines.append(f"| {m} | " + " | ".join(row) + " |")

    unit_errors = [
        (r.item_id, n)
        for r in run.items
        for n in r.metrics.get("numeric_accuracy", {}).get("details", {}).get("numbers", [])
        if n["status"] == "unit_error"
    ]
    if unit_errors:
        lines += ["", "## Unit mix-ups (millions vs billions)", ""]
        lines += [
            f"- `{i}`: expected {n['expected']:,.0f}, answer said `{n['answer_text']}`"
            for i, n in unit_errors
        ]

    failures = [r for r in run.items if r.failed_metrics or r.errors]
    lines += ["", f"## Failures ({len(failures)} of {len(run.items)} items)", ""]
    for r in failures[:max_failures]:
        got = r.response.answer_type if r.response else "error"
        answer = (r.response.answer if r.response else "").replace("\n", " ")[:300]
        lines += [
            f"### `{r.item_id}` ({r.category}/{r.subcategory}, {r.difficulty})",
            f"- **Q:** {r.question}",
            f"- **Expected:** {r.expected_behavior} · **Got:** {got}",
            f"- **Failed:** {', '.join(r.failed_metrics) or '-'}",
            f"- **Answer:** {answer}",
        ]
        lines += [f"- **Error:** {e}" for e in r.errors]
        lines.append("")
    if len(failures) > max_failures:
        lines.append(f"... and {len(failures) - max_failures} more (see results.json)")
    return "\n".join(lines) + "\n"


def save_run(run: RunResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "results.json", out_dir / "report.md"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, md_path
