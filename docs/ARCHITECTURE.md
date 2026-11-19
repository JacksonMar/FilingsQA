# Architecture

Three parts with clear interfaces between them:

1. **Ingestion** turns pinned filings into chunks (JSONL).
2. **RAG pipeline** (the system under test) turns a question into an `AskResponse`.
3. **Evaluation suite** turns golden items and `AskResponse`s into scores, gates and reports.

The eval suite talks to the bot only through `RAGPipeline.ask()` and the `AskResponse` schema,
the same contract the HTTP API exposes.

## Ingestion

```mermaid
flowchart TD
    M[data/manifest/filings.json<br/>pinned accessions] --> DL[EdgarClient<br/>User-Agent, 10 req/s, retries]
    DL --> RAW[data/raw/filings/*.htm]
    DL --> CF[data/raw/companyfacts/*.json]
    RAW --> H2T[html_to_text<br/>blocks: text / table<br/>drop ix:header, hidden, page furniture]
    H2T --> DEI[DEI facts<br/>fiscal year, period, end date]
    H2T --> SEC[sections<br/>Item headings, TOC-robust]
    DEI --> META[FilingMeta]
    SEC --> CH[chunking<br/>350 words, overlap, tables separate<br/>with caption + context header]
    META --> CH
    CH --> J[data/processed/chunks.jsonl<br/>+ ingest_report.json]
    J --> IDX[filingsqa index<br/>dense + BM25 into Qdrant]
```

Invariants (all unit-tested): no block is lost or duplicated by section splitting; chunks never
cross sections; chunk IDs are deterministic; fiscal metadata fails loudly when missing or
contradictory.

## Request flow

```mermaid
sequenceDiagram
    participant C as Client (API / CLI / eval runner)
    participant P as RAGPipeline
    participant R as Retriever
    participant Q as Qdrant
    participant L as LLM (Haiku)
    C->>P: AskRequest(question, tickers?, fiscal_year?, fiscal_period?)
    P->>P: detect tickers (fiscal period is never auto-filtered)
    loop per ticker (comparisons stay balanced)
        P->>R: query + SearchFilter
        R->>Q: prefetch dense + prefetch BM25, fused with RRF
        Q-->>R: top-k chunks + payload
    end
    alt no chunks
        P-->>C: insufficient_context (no LLM call)
    else chunks found
        P->>L: system prompt v1 + numbered <document>s (escaped) + question
        L-->>P: JSON {answer_type, answer, cited_documents}
        P->>P: map [n] to chunk IDs, collect invalid citation IDs
        P-->>C: AskResponse(answer, answer_type, citations, retrieved_chunks, meta)
    end
```

## Evaluation flow

```mermaid
flowchart LR
    CF[(companyfacts)] --> GEN[generators<br/>numeric, comparison,<br/>temporal, false premise]
    MAN[manifest] --> GEN
    HW[handwritten/*.jsonl] --> BUILD[builder<br/>pin sources to accessions]
    GEN --> BUILD
    BUILD --> GOLD[golden_vN.jsonl<br/>+ sha256 meta]
    GOLD --> DQA{dataset QA}
    CHUNKS[chunks.jsonl] --> DQA
    GOLD --> RUN[EvalRunner]
    RUN -->|InjectingBackend for<br/>prompt-injection items| BOT[RAGPipeline]
    BOT --> M1[deterministic metrics]
    BOT --> M2[retrieval metrics]
    BOT --> M3[DeepEval judge<br/>Sonnet via JudgeAdapter]
    M1 --> AGG[per-category means]
    M2 --> AGG
    M3 --> AGG
    AGG --> TH{thresholds.yaml<br/>full / smoke}
    TH --> OUT[results.json<br/>report.md<br/>exit code]
    OUT --> JA[judge agreement<br/>human labels, kappa]
```

## CI

```mermaid
flowchart LR
    PR[pull request] --> LT[lint-and-test<br/>ruff, actionlint, mypy,<br/>unit, dataset schema]
    PR --> IT[integration<br/>Qdrant service]
    PR --> CHG{changes in src/, dataset,<br/>thresholds, deps, workflows?}
    CHG -- yes --> SM[eval-smoke<br/>snapshot index, ~20 items,<br/>real Haiku, deterministic gate]
    CHG -- no --> SKIP[skipped = success]
    CRON[nightly / manual] --> FULL[eval-full<br/>ingest, dataset QA,<br/>full eval + judge, artifact]
```

## Module boundaries

| Module | Depends on | Must not depend on |
|---|---|---|
| `edgar`, `ingest` | stdlib, httpx, bs4 | Qdrant, LLM |
| `index` | `ingest.models`, qdrant-client, fastembed | LLM |
| `llm` | anthropic SDK | everything else |
| `rag` | `index`, `llm` | `evaluation` |
| `evaluation` | everything above, DeepEval | - (no app package imports it; only the CLI wires it in) |

These rules are enforced by `tests/unit/test_architecture.py`.
