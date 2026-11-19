"""Retrieval metrics computed from section labels (no LLM). See ADR-022.

A retrieved chunk is *relevant* if its (accession, section_key) matches one of the item's
expected sources. Each expected source is one thing to find (e.g. a comparison has two).
"""

from filingsqa.evaluation.dataset import GoldenItem
from filingsqa.evaluation.metrics.base import MetricResult
from filingsqa.evaluation.numbers import number_variants
from filingsqa.rag.schemas import AskResponse, RetrievedChunk

INJECTED_ACCESSION = "INJECTED"


def _relevant_source(item: GoldenItem, chunk: RetrievedChunk) -> int | None:
    for idx, src in enumerate(item.expected_sources):
        if src.accession == chunk.accession and chunk.section_key in src.section_keys:
            return idx
    return None


def retrieval_metrics(item: GoldenItem, response: AskResponse) -> list[MetricResult]:
    if not item.expected_sources or item.expected_behavior != "answer":
        return []
    chunks = [c for c in response.retrieved_chunks if c.accession != INJECTED_ACCESSION]
    labels = [_relevant_source(item, c) for c in chunks]

    found_sources = {lbl for lbl in labels if lbl is not None}
    recall = len(found_sources) / len(item.expected_sources)
    first = next((rank for rank, lbl in enumerate(labels, start=1) if lbl is not None), None)
    mrr = 1.0 / first if first else 0.0

    # Contextual precision (rank-aware): mean of precision@k over the ranks k of relevant chunks.
    hits, precisions = 0, []
    for rank, lbl in enumerate(labels, start=1):
        if lbl is not None:
            hits += 1
            precisions.append(hits / rank)
    ctx_precision = sum(precisions) / len(precisions) if precisions else 0.0

    results = [
        MetricResult(
            "recall_at_k", recall, recall == 1.0, {"k": len(chunks), "found": sorted(found_sources)}
        ),
        MetricResult("mrr", mrr, first is not None, {"first_relevant_rank": first}),
        MetricResult("contextual_precision", ctx_precision, ctx_precision >= 0.5),
    ]
    if item.expected_numbers:
        # Contextual recall for numbers: can the answer be built from what was retrieved at all?
        text = "\n".join(c.text for c in chunks)
        present = [
            any(v in text for v in number_variants(n.value, n.unit)) for n in item.expected_numbers
        ]
        score = sum(present) / len(present)
        results.append(
            MetricResult("context_has_numbers", score, score == 1.0, {"per_number": present})
        )
    return results
