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

## ADR-013: No `temperature`; managing non-determinism differently (amends ADR-004)
**Context.** The plan was temperature 0 for reproducible answers and non-flaky CI gates. Current
Claude models (Haiku 5.5 as generator, Sonnet 5.5 as judge) reject non-default sampling
parameters with HTTP 400, and API version `2026-10-21` rejects `temperature` on every model.
**Decision.** No sampling parameters anywhere. Variance is controlled by: (1) structured output
(JSON schema) so the *shape* of the answer is fixed; (2) `effort` set explicitly (`low` for the
generator, `medium` for the judge) and logged; (3) a disk cache keyed by provider, model, effort,
prompts and schema, so repeated dev/eval runs replay identical answers; (4) CI thresholds with
margin and per-category aggregates rather than per-item gates; (5) if flakiness shows up, run the
smoke subset N times and gate on the mean — measured, not assumed.
**Alternatives.** Pin an older model that still accepts temperature (stale, and API version
`2026-10-21` removes the parameter anyway). Replay-only CI (rejected in ADR-004).
**Interview note.** Even temperature 0 was never fully deterministic on hosted LLMs; designing the
gate for variance is the robust approach either way.

## ADR-014: Qdrant layout and hybrid retrieval
**Context.** Financial questions mix semantic phrasing ("how much did sales grow") with exact
tokens (line-item names, "Item 1A", tickers, numbers) that dense embeddings handle poorly.
**Decision.** One Qdrant point per chunk with two named vectors: `dense` (bge-small, cosine) and
`bm25` (fastembed `Qdrant/bm25` term frequencies, IDF applied by Qdrant via `Modifier.IDF`).
Both embed `context + text`, so company/period/section words are searchable. A query runs two
prefetches (dense, sparse; 40 candidates each) fused with Reciprocal Rank Fusion in Qdrant, then
returns top 8. Point IDs are `uuid5(chunk_id)` → re-indexing is an idempotent upsert. Payload
indexes on ticker, fiscal_year, fiscal_period, form, section_key, accession, chunk_id.
**Alternatives.** Dense only (misses exact terms); weighted score sum (scores of different
retrievers are not on the same scale, weights need tuning — RRF only uses ranks); a cross-encoder
re-ranker (better quality, more latency/complexity — a candidate experiment for the eval suite).
**Note.** Embedding models are downloaded from Hugging Face on first use; tests never need them.

## ADR-015: Which metadata filters are automatic
**Context.** Filters reduce noise, but a wrong filter silently hides the right chunk.
**Decision.** Tickers are auto-detected from the question (tickers and company aliases, word
boundaries) and become a filter. Fiscal year/period are **not** auto-detected: a FY2025 10-K also
contains FY2024/FY2023 comparatives, and "fiscal 2024" ≠ calendar 2024. They are filters only when
the caller passes them explicitly. Explicit unknown tickers are kept in the filter (→ no hits →
"insufficient context" without an LLM call) instead of being dropped, which would silently search
every company. With several tickers, retrieval runs once per ticker and interleaves results, so a
comparison question always sees both companies.
**Alternatives.** LLM-based query parsing (extra latency, cost and a new failure mode to test);
regex year filters (wrong on comparatives and fiscal calendars).

## ADR-016: Structured answer + numbered documents for citations
**Context.** Citations must be machine-checkable (Phase 4: "cited chunk exists and contains the
number").
**Decision.** Retrieved chunks are sent as `<document id="n" ...>` with ticker, form, fiscal
period, period end and section as attributes. The model returns JSON `{answer_type, answer,
cited_documents}` (structured outputs) and puts `[n]` markers after claims. Code maps `n` back to
chunk IDs and URLs; out-of-range ids are returned in `invalid_citation_ids` instead of being
trusted. `<`/`>` in document text and in the question are escaped so content cannot close our tags
(prompt-injection hardening). `PROMPT_VERSION` is logged with every answer and is part of the cache
key.
**Alternatives.** Anthropic's native citations feature (cites character spans, nice, but is
incompatible with structured outputs and harder to compare across providers); free text with
regex parsing of "Sources:" (fragile).

## ADR-017: Guardrails live in the prompt first; code guardrails only when the eval proves a need
**Context.** Investment-advice refusals, "I don't know" behavior and injection resistance could be
enforced with deterministic pre-filters.
**Decision.** For v1 these behaviors are prompt rules only, so the eval suite measures the model's
real behavior. A code guardrail is added only after the eval shows a failure; each such fix is
logged in `docs/BUGS_FOUND.md` with before/after numbers. Exception: when retrieval returns
nothing, the pipeline answers "insufficient context" without calling the LLM (no grounding = no
generation). Provider refusals (`stop_reason == "refusal"`) become `answer_type="model_refusal"`.
**Alternatives.** Keyword blocklist for advice up front (hides what the model would do; also
over-blocks factual questions like "what did management say about buybacks").

## ADR-018: Test doubles for embeddings and LLM
**Context.** Unit tests must not download models or call APIs; Qdrant behavior must still be
tested for real.
**Decision.** `tests/fakes.py`: hashing bag-of-words dense and sparse embedders (deterministic,
lexical similarity), a scripted `FakeLLM` and a `FakeSearchBackend`. Unit tests use fakes only.
Integration tests (`-m integration`, separate CI job with a Qdrant service container) run the real
Qdrant collection layout, filters and RRF with the fake embedders. The Anthropic client is tested
by stubbing `messages.create` and asserting on the request we send (no `temperature`, structured
output schema, effort) and on refusal/truncation handling.
**Alternatives.** HTTP-level mocking of the Anthropic API (couples tests to SDK internals);
in-memory Qdrant (`:memory:` mode does not implement everything the server does, e.g. it is not
the thing we deploy).
