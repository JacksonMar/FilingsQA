"""Hybrid section detection: our parser by default, edgartools as arbiter for core sections.

Our Item-heading parser (sections.py) handles the usual layout. Some filers (JPMorgan) put the
formal Form 10-K/10-Q items first and the real MD&A and financial statements later in an
annual-report layout, linked from the table of contents; heading-based detection cannot see
that. edgartools resolves those links. Policy (ADR-027):

1. Split with our parser.
2. Ask edgartools for its sections and compare the *core* sections (10-K: Items 7 and 8;
   10-Q: Part I Items 1 and 2) with the same size test the differential test uses.
3. Only if they disagree, take section *boundaries* from edgartools: find where each of its
   sections starts among our blocks. Blocks, table rendering and chunking stay ours.
"""

import logging
import re
import warnings
from dataclasses import dataclass

from filingsqa.ingest.html_to_text import Block
from filingsqa.ingest.sections import (
    FRONT_MATTER,
    FULL_DOCUMENT,
    SIGNATURES,
    TEN_K_ITEMS,
    TEN_Q_ITEMS,
    Section,
    SectionSplit,
)

log = logging.getLogger(__name__)

CORE_SECTIONS = {"10-K": ("item_7", "item_8"), "10-Q": ("part1_item_1", "part1_item_2")}
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO, MIN_ABS_DIFF = 0.5, 2.0, 500
_PART_NUMBER = {"I": 1, "II": 2}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def signature(text: str) -> str:
    """Lowercase alphanumerics only: compares text across different renderings of tables."""
    return _NON_ALNUM.sub("", text.lower())


@dataclass(frozen=True)
class OracleSection:
    key: str  # our naming: item_7, part1_item_2, ...
    size: int  # alphanumeric characters
    lines: tuple[str, ...]  # first meaningful lines of its text, used as a start anchor


def our_key(form: str, item: str, part: str) -> str | None:
    item = item.upper()
    if form.startswith("10-Q"):
        if part.upper() not in _PART_NUMBER or (part.upper(), item) not in TEN_Q_ITEMS:
            return None
        return f"part{_PART_NUMBER[part.upper()]}_item_{item.lower()}"
    return f"item_{item.lower()}" if item in TEN_K_ITEMS else None


def oracle_sections(html: str, form: str) -> list[OracleSection]:
    from edgar.documents import ParserConfig, parse_html  # heavy import, only when needed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = parse_html(html, ParserConfig(form=form))
    out: dict[str, OracleSection] = {}
    for section in doc.sections.values():
        key = our_key(form, str(section.item or ""), str(section.part or ""))
        if key is None or key in out:
            continue
        text = section.text()
        lines = tuple(
            ln for ln in (line.strip() for line in text.split("\n")) if len(signature(ln)) >= 10
        )[:3]
        out[key] = OracleSection(key, len(signature(text)), lines)
    return list(out.values())


def section_sizes(split: SectionSplit) -> dict[str, int]:
    return {
        s.key: sum(len(signature(b.text)) for b in s.blocks)
        for s in split.sections
        if s.key not in (FRONT_MATTER, FULL_DOCUMENT, SIGNATURES)
    }


def sizes_disagree(ours: int | None, oracle: int | None) -> bool:
    if ours is None or oracle is None:
        return ours != oracle
    if abs(ours - oracle) <= MIN_ABS_DIFF:
        return False
    return not MIN_LENGTH_RATIO <= ours / max(oracle, 1) <= MAX_LENGTH_RATIO


def core_disagreement(own: SectionSplit, oracle: list[OracleSection], form: str) -> list[str]:
    ours, theirs = section_sizes(own), {o.key: o.size for o in oracle}
    return [
        key
        for key in CORE_SECTIONS["10-Q" if form.startswith("10-Q") else "10-K"]
        if key in theirs and sizes_disagree(ours.get(key), theirs[key])
    ]


def _canonical_order(form: str) -> dict[str, int]:
    if form.startswith("10-Q"):
        return {our_key(form, i, p) or "": n for n, (p, i) in enumerate(TEN_Q_ITEMS)}
    return {f"item_{i.lower()}": n for n, i in enumerate(TEN_K_ITEMS)}


def _find_start(sigs: list[str], section: OracleSection, after: int) -> int | None:
    """Block index where `section` starts. Prefer a hit whose following blocks also contain
    the section's second line; otherwise the first hit after the previous section's start."""
    if not section.lines:
        return None
    first = signature(section.lines[0])[:40]
    second = signature(section.lines[1])[:25] if len(section.lines) > 1 else ""
    hits = [i for i, s in enumerate(sigs) if s.startswith(first)]
    # the second line must follow right away (same block or the next two)
    confirmed = [i for i in hits if second and second in "".join(sigs[i : i + 3])]
    if len(confirmed) == 1:
        return confirmed[0]
    pool = confirmed or hits
    later = [i for i in pool if i > after]
    candidates = later or pool
    return candidates[0] if candidates else None


def split_with_oracle(
    blocks: list[Block], oracle: list[OracleSection], form: str
) -> SectionSplit | None:
    """Sections from oracle start positions; None if a core section cannot be anchored."""
    sigs = [signature(b.text) for b in blocks]
    order = _canonical_order(form)
    titles = _titles(form)
    starts: dict[str, int] = {}
    previous = -1
    for section in sorted(oracle, key=lambda o: order.get(o.key, 999)):
        idx = _find_start(sigs, section, previous)
        if idx is None:
            continue
        starts[section.key] = idx
        previous = idx
    core = CORE_SECTIONS["10-Q" if form.startswith("10-Q") else "10-K"]
    if any(k in {o.key for o in oracle} and k not in starts for k in core):
        return None

    bounds = sorted((idx, key) for key, idx in starts.items())
    # two sections anchored at the same block: keep the one that comes first canonically
    dedup: list[tuple[int, str]] = []
    for idx, key in bounds:
        if dedup and dedup[-1][0] == idx:
            continue
        dedup.append((idx, key))
    sections: list[Section] = []
    if dedup and dedup[0][0] > 0:
        sections.append(Section(FRONT_MATTER, "Front matter", blocks[: dedup[0][0]]))
    for n, (idx, key) in enumerate(dedup):
        end = dedup[n + 1][0] if n + 1 < len(dedup) else len(blocks)
        sections.append(Section(key, titles.get(key, key), blocks[idx:end]))
    return SectionSplit(sections, ["section boundaries taken from edgartools (core mismatch)"])


def _titles(form: str) -> dict[str, str]:
    if form.startswith("10-Q"):
        return {our_key(form, i, p) or "": t for (p, i), t in TEN_Q_ITEMS.items()}
    return {f"item_{i.lower()}": t for i, t in TEN_K_ITEMS.items()}


def resolve_sections(
    html: str, blocks: list[Block], own: SectionSplit, form: str
) -> tuple[SectionSplit, str]:
    """Return the split to use and its source: "own" or "edgartools"."""
    try:
        oracle = oracle_sections(html, form)
    except Exception as exc:  # the library failing must not stop ingestion
        log.warning("edgartools failed (%s); keeping own sections", exc)
        return own, "own"
    mismatched = core_disagreement(own, oracle, form)
    if not mismatched:
        return own, "own"
    hybrid = split_with_oracle(blocks, oracle, form)
    if hybrid is None:
        own.warnings.append(
            f"core sections {mismatched} disagree with edgartools; anchoring failed"
        )
        return own, "own"
    hybrid.warnings.append(f"core sections {mismatched} disagreed with own parser")
    return hybrid, "edgartools"
