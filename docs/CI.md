# CI/CD

| Workflow | Trigger | Jobs | Cost |
|----------|---------|------|------|
| `ci.yml` | every PR, push to `main` | `lint-and-test` (ruff, actionlint, mypy, unit tests, hand-written dataset schema), `integration` (real Qdrant, fake embedders), `changes` (path filter), `eval-smoke` | smoke: ~20 Haiku calls when relevant paths change |
| `eval-full.yml` | nightly 03:17 UTC, manual | ingest pinned filings → dataset QA → full eval with LLM judge → report artifact | full dataset × (Haiku + Sonnet judge) |

## eval-smoke (quality gate on PRs)
Runs only if the PR touches `src/` (code and prompts), `eval/datasets/`, `eval/snapshots/`,
`eval/thresholds.yaml`, `pyproject.toml`, `uv.lock` or the workflows themselves. It indexes the committed chunk snapshot
(`eval/snapshots/smoke_chunks.jsonl.gz`, no SEC access), asks the ~20 stratified smoke items with
the real generator (LLM cache off), computes deterministic metrics only and fails on the `smoke`
thresholds. The report is in the job summary and as an artifact.

It is skipped with a warning (not failed) when the golden dataset/snapshot are not committed yet,
or when `ANTHROPIC_API_KEY` is unavailable (PRs from forks get no secrets).

## eval-full (nightly / manual)
Downloads the pinned filings (cached by manifest hash), indexes the full corpus, runs
`dataset validate` (including answerability) and the full eval with the LLM judge. Fails when a
`full` threshold fails, so a red nightly run means a quality regression. LLM responses are cached
between nightly runs; the cache key includes model, effort and prompt, so only changed requests
cost tokens. Manual runs can disable the judge or pick one category.

## Setup checklist
1. Repository secrets: `ANTHROPIC_API_KEY`, `SEC_USER_AGENT` (e.g. `FilingsQA you@example.com`).
2. Commit `data/manifest/filings.json` (`make manifest`), `eval/datasets/golden_v1.jsonl` +
   `.meta.json` (`make dataset`), and `eval/snapshots/smoke_chunks.jsonl.gz` (`make snapshot`).
3. Branch protection on `main`: require `lint-and-test`, `integration` and `eval-smoke`.
   A path-filtered-out `eval-smoke` shows as *skipped*, which GitHub treats as passing.
