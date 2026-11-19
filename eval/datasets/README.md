# Golden datasets

| File | What it is |
|------|------------|
| `handwritten/*.jsonl` | Hand-written items (qualitative, adversarial, some temporal). Source of truth for those categories. |
| `golden_vN.jsonl` | A released dataset version: generated (XBRL) + hand-written items, sources pinned to the manifest. **Immutable.** |
| `golden_vN.meta.json` | Counts per category/behavior, manifest date and the sha256 of the JSONL. |

## Item format
One JSON object per line, schema `filingsqa.evaluation.dataset.GoldenItem`. Key fields:

- `category`: numeric | qualitative | comparison | temporal | adversarial; plus `subcategory`, `difficulty`.
- `expected_behavior`: `answer` | `insufficient_context` | `decline_advice` (mirrors the bot's `answer_type`).
- `expected_numbers`: value in base units (USD, USD/share), tolerance, fiscal period and an `xbrl`
  reference (tag, accession, start/end) so every number is traceable to SEC data.
- `expected_sources`: ticker, form, accepted section keys, accession (pinned at build time).
- `key_facts`: facts a good answer must state (qualitative). `forbidden_phrases`: must never appear
  (prompt-injection payloads). `injected_document`: poisoned chunk the eval runner adds to the context.
- `verified`: hand-written content checked by a human against the real filing text.

## Workflow
```bash
make ingest                     # filings + companyfacts + chunks (needs SEC access)
make dataset VERSION=v1         # build golden_v1.jsonl + meta, then run dataset QA
uv run filingsqa dataset check-handwritten   # schema-check hand-written files (offline)
```
`dataset validate` fails if a released file was edited (sha256), if a number or source points
outside the manifest, or if an expected number does not occur anywhere in the filing's chunks.
Unverified hand-written items and key facts not found in the expected sections are warnings to
review by hand; set `verified: true` once checked.

## Versioning
Never edit a released `golden_vN.jsonl`. Fix the generator or a hand-written file, build `vN+1`, and
note the change below. Eval reports always record which version they used.

## Changelog
- v1 — 2026-11-19, manifest as of 2026-11-17 (15 filings). 214 items: numeric 106, comparison 30,
  qualitative 30, adversarial 28, temporal 20. Dataset QA: 0 errors; the 30 hand-written
  qualitative items are still `verified: false` (key facts were checked against the chunks, a
  human read-through is pending). JPM quarterly revenue comes from `RevenuesNetOfInterestExpense`
  (banks report total net revenue under that tag in 10-Qs).
