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

## ADR-007: HTML → ordered text/table blocks
**Context.** Filings are inline-XBRL HTML. Financial tables matter, and SEC tables put `$`, `(`,
`)` and `%` in their own cells, plus many empty spacer cells.
**Decision.** BeautifulSoup + lxml walk: block-level elements become text blocks, inline elements
are joined. Data tables stay tables and are rendered row by row as `cell | cell | cell` after
dropping empty cells and gluing split fragments (`$` + `410,235` → `$410,235`, `(` `321` `)` →
`(321)`). Single-row tables with ≤3 cells are layout tables and are flattened to text, so headings
like `Item 1A. | Risk Factors` are still detected. `ix:header`, `display:none`, scripts and page
furniture (page numbers, "Form 10-K | 12" footers) are dropped; `dei:*` facts are read first.
**Alternatives.** Markdown tables with alignment (colspan makes column alignment unreliable;
pipe rows are good enough for an LLM). `html2text`/`trafilatura` — generic, mangle financial tables.
**Known limitation.** Dropping empty cells can shift columns when a row has a missing value.

## ADR-008: Item detection by "largest following span"
**Context.** Item headings appear several times: TOC, real heading, cross-references.
**Decision.** Candidates are short text blocks (≤150 chars, never data tables) matching
`Item N[A-C]` with a capitalized title (rejects "Item 7 of this report…"). For 10-Q the current
`PART I/II` disambiguates `Item 1`. Per Item, keep the candidate followed by the most text before
the next candidate (TOC entries are followed by almost nothing). Text before the first Item is
`front_matter`; no headings at all → one `full_document` section + warning. Invariant (tested):
every block belongs to exactly one section.
**Alternatives.** "Skip the TOC by position" (fragile), "last occurrence wins" (breaks on
cross-references at the end of the document), library-only (ADR-006).
**Known risk.** Filings that use a cross-reference index instead of Item headings (large banks,
e.g. JPM's 10-K wraps the annual report) will fall back to `full_document` or be mis-split.
The differential test against edgartools is designed to surface exactly this.

## ADR-009: Fiscal period comes from the filing's DEI facts
**Context.** Fiscal ≠ calendar: NVIDIA's FY2025 ended Jan 2025; Apple's fiscal Q1 ends in December.
Getting the period wrong is the most likely kind of "correct number, wrong answer".
**Decision.** `fiscal_year`, `fiscal_period` and `period_end_date` come from the company's own
`dei:DocumentFiscalYearFocus / DocumentFiscalPeriodFocus / DocumentPeriodEndDate` tags. Missing or
contradictory data (e.g. DEI DocumentType ≠ manifest form) fails ingestion loudly instead of
guessing. Chunks carry a label like `Q1 FY2026 (period ended 2025-12-27)`.
**Alternatives.** Derive from `reportDate` + a per-company fiscal calendar table (duplicated
knowledge, easy to get wrong); companyfacts `fy`/`fp` (an extra request; and facts can be reported
in later filings with that later filing's `fy`).

## ADR-010: Chunking rules and chunk identity
**Context.** Retrieval quality and citation checks both depend on chunk boundaries.
**Decision.** Chunks never cross sections. Text: paragraphs packed up to 350 words, overlap of up
to 60 words made of whole trailing paragraphs, long paragraphs split on sentences. Tables are
separate chunks (≤800 words, larger tables split by rows with the first row repeated) and carry
the preceding short paragraph as a caption, because it usually holds the units ("In millions").
Every chunk has a `context` header (company, ticker, form, fiscal period, section) that is
prepended for embedding. Chunk ID = `{ticker}-{accession}-{section}-{index}` (deterministic).
The golden dataset will reference expected sources as (ticker, accession, section), **not** chunk
IDs, so changing chunk sizes does not invalidate the dataset.
**Alternatives.** Fixed token windows (cut tables and sentences in half), semantic chunking
(needs embeddings at ingest time, harder to explain and to test).
Word counts are used instead of tokens: simpler and model-agnostic; bge-small's 512-token limit
is respected with margin at 350 words + header.

## ADR-011: Ingestion output is JSONL; indexing is a separate step
**Context.** Qdrant indexing and the CI smoke snapshot both need the chunks.
**Decision.** `make ingest` stops at `data/processed/chunks.jsonl` + `ingest_report.json`
(chunk counts per section, warnings). Indexing into Qdrant is Phase 2 and reads the JSONL. The
smoke snapshot (ADR-005) is a filtered copy of this file.
**Alternatives.** Ingest straight into Qdrant — couples parsing to the vector store and makes the
parsing output impossible to diff or review.

## ADR-012: Synthetic fixtures first, real snippets later
**Context.** Unit tests must not hit the network. During Phase 1 development the cloud sandbox
could not reach sec.gov at all.
**Decision.** Hand-written fixtures that mimic real structures (hidden XBRL header, TOC table,
layout-table headings, split `$`/`)` cells, page footers). Each quirk found in real filings gets
a minimal real-derived fixture + regression test.
**Alternatives.** Commit full real filings as fixtures (MBs of HTML, slow tests, hard to see what
is being tested).
