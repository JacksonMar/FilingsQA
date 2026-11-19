"""Quality checks for the golden dataset itself ("test the test data"). See ADR-021.

A wrong golden item produces a false bug report, so the dataset gets its own checks:
- structural: unique ids/questions, category coverage, sources pinned to manifest filings;
- traceability: every expected number points to a filing in the manifest;
- answerability (needs chunks.jsonl): the expected number / key facts actually occur in the
  expected section of the corpus. If not, a "wrong answer" would really be a dataset or
  ingestion problem, and we want to know that before blaming the bot.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from filingsqa.edgar.manifest import Manifest
from filingsqa.evaluation.dataset import GoldenItem
from filingsqa.evaluation.numbers import number_variants
from filingsqa.ingest.models import Chunk

MIN_PER_CATEGORY = {
    "numeric": 20,
    "qualitative": 20,
    "comparison": 5,
    "temporal": 5,
    "adversarial": 15,
}
_WORD = re.compile(r"[a-z0-9&]+")
_STOP = frozenset({"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "its", "is"})


@dataclass
class QAReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _structure(items: list[GoldenItem], report: QAReport, min_per_category: dict[str, int]) -> None:
    for kind, values in (("id", [i.id for i in items]), ("question", [i.question for i in items])):
        for value, n in Counter(values).items():
            if n > 1:
                report.errors.append(f"duplicate {kind}: {value!r} x{n}")
    counts: Counter[str] = Counter(i.category for i in items)
    for category, minimum in min_per_category.items():
        if counts[category] < minimum:
            report.warnings.append(f"category {category}: {counts[category]} items (< {minimum})")
    for i in items:
        if i.source == "handwritten" and not i.verified and i.expected_behavior == "answer":
            report.warnings.append(f"{i.id}: hand-written and not yet verified against filings")


def _traceability(items: list[GoldenItem], manifest: Manifest, report: QAReport) -> None:
    accessions = {f.accession for f in manifest.filings}
    for item in items:
        for num in item.expected_numbers:
            if num.xbrl.accession not in accessions:
                report.errors.append(f"{item.id}: number from filing outside manifest")
        for src in item.expected_sources:
            if src.accession is None:
                report.errors.append(f"{item.id}: unresolved expected source {src.ticker}")
            elif src.accession not in accessions:
                report.errors.append(f"{item.id}: source {src.accession} not in manifest")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _answerability(items: list[GoldenItem], chunks: list[Chunk], report: QAReport) -> None:
    by_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_filing: dict[str, list[str]] = defaultdict(list)
    fiscal: dict[str, tuple[int, str]] = {}
    for c in chunks:
        by_section[(c.filing.accession, c.section_key)].append(c.text)
        by_filing[c.filing.accession].append(c.text)
        fiscal[c.filing.accession] = (c.filing.fiscal_year, c.filing.fiscal_period)

    for item in items:
        sources = [s for s in item.expected_sources if s.accession]
        section_text = "\n".join(
            t
            for s in sources
            for key in s.section_keys
            for t in by_section[(s.accession or "", key)]
        )
        filing_text = "\n".join(t for s in sources for t in by_filing[s.accession or ""])

        for src in sources:
            # DEI (from the HTML) and XBRL (companyfacts) must agree on the fiscal period.
            dei = fiscal.get(src.accession or "")
            if dei and src.fiscal_year is not None and dei != (src.fiscal_year, src.fiscal_period):
                report.errors.append(
                    f"{item.id}: fiscal period mismatch DEI={dei} XBRL="
                    f"{(src.fiscal_year, src.fiscal_period)} for {src.accession}"
                )

        for num in item.expected_numbers:
            variants = number_variants(num.value, num.unit)
            if not any(v in section_text for v in variants):
                where = "filing" if any(v in filing_text for v in variants) else "corpus"
                msg = f"{item.id}: {num.label}={num.value:g} not found in expected sections"
                if where == "corpus":
                    report.errors.append(msg + " nor anywhere in the filing")
                else:
                    report.warnings.append(msg + " (found elsewhere in the filing)")

        if item.key_facts and item.source == "handwritten" and sources:
            words = _words(section_text)
            for fact in item.key_facts:
                missing = _words(fact) - words
                if missing:
                    report.warnings.append(
                        f"{item.id}: key fact {fact!r} - words not in expected sections: "
                        f"{sorted(missing)}"
                    )


def check_dataset(
    items: list[GoldenItem],
    manifest: Manifest,
    chunks: list[Chunk] | None = None,
    min_per_category: dict[str, int] | None = None,
) -> QAReport:
    report = QAReport()
    _structure(items, report, MIN_PER_CATEGORY if min_per_category is None else min_per_category)
    _traceability(items, manifest, report)
    if chunks is not None:
        _answerability(items, chunks, report)
    return report
