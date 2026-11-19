# CLAUDE.md — rules for working on FilingsQA

## What this project is
FilingsQA is a RAG chatbot answering questions about US public companies from their SEC filings.
**The chatbot is the system under test; the evaluation suite is the main product.**
QA = Question Answering + Quality Assurance. Portfolio project for AI/ML QA roles.

## Working style
- Talk to the user in **Ukrainian**. Everything in the repo (code, comments, commits, docs) in **English**.
- Work **phase by phase**. At the end of each phase: stop, summarize in simple words, explain key
  decisions ("why X and not Y"), list what the user should be able to explain in an interview,
  and wait for OK.
- Clarity beats cleverness. The user must be able to defend every line in an interview.
  No LangChain/LlamaIndex: plain Python + `qdrant-client` + `fastembed`.
- Record every non-trivial decision in `docs/DECISIONS.md` (ADR: context → decision → alternatives).
- Log every real defect the eval suite finds in `docs/BUGS_FOUND.md` (from Phase 2 onward):
  symptom, how it was detected, root cause, fix, regression test.

## Hard rules
- Never commit secrets. Config only via `.env` (gitignored) + `.env.example`. Use placeholders
  like `your.name@example.com`; never write the user's real email anywhere in the repo.
- SEC EDGAR: real `User-Agent` with contact email from `.env`, max 10 requests/sec.
- Unit tests: no network, LLM always mocked. Network tests are marked `network`,
  Qdrant tests are marked `integration`.
- Keep LLM costs low: disk cache for LLM responses in dev, small eval subsets in CI,
  LLM response cache, thresholds with margin to avoid flaky gates. No `temperature`: current
  Claude models reject non-default sampling params (ADR-013).
- Generator and judge LLMs are configured separately (generator: Haiku, judge: Sonnet).
- Filings and ground truth are **frozen**: accession numbers in a committed manifest, XBRL
  expected values embedded in the golden dataset. Raw HTML is never committed.

## Stack
Python 3.12, `uv`, `ruff`, `mypy --strict` on `src/`, `pytest`. Qdrant (Docker), hybrid retrieval
(bge-small dense + BM25 sparse, RRF fusion). FastAPI + typer CLI. DeepEval + own deterministic
metrics. GitHub Actions.

## Commands
- `make install` — deps + pre-commit hooks
- `make lint` / `make format` / `make typecheck`
- `make test` — unit tests; `make test-integration` — needs `make up`
- `make ingest`, `make serve`, `make eval`, `make eval-smoke`
- `make dataset`, `make snapshot` — golden dataset and the CI smoke chunk snapshot
- CI: `.github/workflows/ci.yml` (PR gate incl. path-filtered `eval-smoke`), `eval-full.yml`
  (nightly/manual). See `docs/CI.md`. `make lint` also runs actionlint on workflows.

## Layout
- `src/filingsqa/` — application (edgar, ingest, index, rag, llm, api, cli)
- `src/filingsqa/evaluation/` — eval code: dataset schema, XBRL generators, dataset QA, metrics, runner
- `eval/` — data only: `datasets/` (hand-written sources + immutable `golden_vN.jsonl`), thresholds, reports
- `tests/` — unit / integration / fixtures (small HTML/JSON snippets)
- `docs/` — DECISIONS.md, BUGS_FOUND.md, ARCHITECTURE.md

## Parsing timebox
Own 10-K/10-Q Item parser, with `edgartools` (dev dependency) as an oracle for differential tests.
Timebox for Item parsing: 2 evenings; if not good enough, switch to the library and record an ADR.
