# FilingsQA

> **QA** = Question Answering + Quality Assurance.

A RAG chatbot that answers questions about US public companies using their SEC filings —
and, more importantly, an evaluation and test suite that measures how much you can trust it.
Numeric answers are checked automatically against official SEC XBRL data.

**Status:** Phase 0 (skeleton). See `docs/DECISIONS.md` for design decisions.

## Quick start

```bash
cp .env.example .env   # set SEC_USER_AGENT and ANTHROPIC_API_KEY
make install
make test
```

## Disclaimer
Educational project. Nothing here is investment advice.
