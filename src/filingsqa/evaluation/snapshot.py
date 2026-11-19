"""Chunk snapshot for the CI smoke gate (ADR-005, ADR-026).

CI must not download filings from SEC on every PR, so a small, committed, gzip-compressed
subset of the chunk corpus is indexed instead. It contains:
- every chunk of every expected section of the smoke items (the "right answers");
- up to `distractors_per_filing` other chunks of the same filings, chosen deterministically, so
  retrieval still has to pick the right chunk among plausible wrong ones;
- for smoke items without sources (refusals, unknown companies), nothing extra: they are answered
  against whatever the snapshot contains, which is exactly what they test.
Retrieval on the snapshot is easier than on the full corpus; smoke is a regression gate, the
nightly full run measures real retrieval quality.
"""

import hashlib

from filingsqa.evaluation.dataset import GoldenItem
from filingsqa.ingest.models import Chunk


def _order(chunk_id: str) -> str:
    return hashlib.sha1(chunk_id.encode()).hexdigest()


def build_snapshot(
    items: list[GoldenItem], chunks: list[Chunk], distractors_per_filing: int = 60
) -> list[Chunk]:
    wanted_sections = {
        (src.accession, key)
        for item in items
        for src in item.expected_sources
        for key in src.section_keys
    }
    filings = {acc for acc, _ in wanted_sections}

    selected = {
        c.chunk_id: c for c in chunks if (c.filing.accession, c.section_key) in wanted_sections
    }
    for accession in sorted(a for a in filings if a):
        others = sorted(
            (c for c in chunks if c.filing.accession == accession and c.chunk_id not in selected),
            key=lambda c: _order(c.chunk_id),
        )
        for c in others[:distractors_per_filing]:
            selected[c.chunk_id] = c
    # Keep corpus order (filing, then position), so the file diffs nicely between versions.
    position = {c.chunk_id: i for i, c in enumerate(chunks)}
    return sorted(selected.values(), key=lambda c: position[c.chunk_id])
