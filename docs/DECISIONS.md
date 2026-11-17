# Architecture Decision Records

Short ADRs: context → decision → alternatives considered.

---

## ADR-001: Plain Python, no LLM framework
**Context.** The project is a portfolio piece; every part must be explainable in an interview.
**Decision.** Build the pipeline with plain Python, `qdrant-client`, `fastembed` and the provider SDK.
`src/` layout with package `filingsqa`.
**Alternatives.** LangChain / LlamaIndex — faster to start, but hide retrieval and prompt
details behind abstractions and add churn-heavy dependencies.

## ADR-002: `uv` + `ruff` + `mypy --strict`
**Context.** Need reproducible installs and fast feedback.
**Decision.** `uv` with a committed `uv.lock`; `ruff` for lint + format (one tool instead of
flake8/isort/black); `mypy --strict` on `src/` only (tests stay light).
**Alternatives.** Poetry / pip-tools — slower, more moving parts.

## ADR-003: Generator = Claude Haiku, Judge = Claude Sonnet
**Context.** The judge must be configured separately from the generator. A judge from the same
family may be biased toward its own family's answers (self-preference bias).
**Decision.** Generator: Haiku (cheap, fast — realistic production choice). Judge: Sonnet
(stronger model judging a weaker one). Both via env (`LLM_*`, `JUDGE_*`). The bias risk is
measured, not assumed away: judge agreement vs. hand labels (Phase 4).
**Alternatives.** Judge from another vendor (OpenAI) — less self-preference risk but a second
API key and cost. May be added later as a second judge for comparison.

## ADR-004: Live LLM calls in the CI smoke gate, path-filtered
**Context.** A smoke eval on replayed (recorded) answers would test the eval code, not the bot.
**Decision.** `eval-smoke` calls the real generator on a small subset (~15–20 items), only when
`src/`, prompts or `eval/datasets/` change (paths filter). Temperature 0 and thresholds with
margin to avoid flaky gates. Unit tests always mock the LLM.
**Alternatives.** Replay cassettes — free and deterministic, but cannot catch prompt/model
regressions.

## ADR-005: What data is committed
**Context.** Filings change over time (new 10-Q every quarter); CI must not depend on SEC uptime.
**Decision.** Commit: filings manifest (pinned accession numbers), golden dataset (with XBRL
expected values embedded), and a small chunk snapshot (JSONL) covering only the smoke subset.
Do not commit raw HTML; `make ingest` downloads it.
**Alternatives.** Live "latest filings" — silently breaks the golden dataset. Committing all
raw HTML — tens of MB of repo bloat.

## ADR-006: Own Item parser, `edgartools` as test oracle
**Context.** Section-aware chunking needs 10-K/10-Q Item boundaries. Libraries exist but are
black boxes.
**Decision.** Own parser (lxml/BeautifulSoup + Item heading detection) with fixture-based unit
tests. `edgartools` is a dev-only dependency used as an oracle in differential tests;
disagreements become `BUGS_FOUND.md` candidates. Timebox: 2 evenings, then switch to the library.
**Alternatives.** Library-only — less to explain, less to learn from. Own parser without an
oracle — no independent check of correctness.
