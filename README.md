# FilingsQA

> **QA** = Question Answering + Quality Assurance.

A RAG chatbot that answers questions about US public companies using their SEC filings —
and, more importantly, an evaluation and test suite that measures how much you can trust it.
Numeric answers are checked automatically against official SEC XBRL data.

**Status:** Phase 4 (evaluation suite). See `docs/DECISIONS.md` for design decisions.

## Quick start

```bash
cp .env.example .env   # set SEC_USER_AGENT and ANTHROPIC_API_KEY
make install
make test

make manifest AS_OF=2026-11-17   # pin filings (once; commit data/manifest/filings.json)
make ingest                      # download + parse + chunk -> data/processed/chunks.jsonl
make test-diff                   # compare our Item parser with edgartools on real filings

make up                          # start Qdrant
make ingest                      # ... then index into Qdrant
uv run filingsqa ask "What was NVIDIA's revenue in fiscal 2025?"
make serve                       # POST /ask on http://localhost:8000 (OpenAPI docs at /docs)

make dataset VERSION=v1          # build + validate the golden dataset
make eval-smoke                  # ~20 items, deterministic metrics, quality gate
make eval                        # full run with LLM judge -> eval/reports/<run>/report.md
```

## Disclaimer
Educational project. Nothing here is investment advice.
