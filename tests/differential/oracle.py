"""Differential testing: compare our Item parser with edgartools (an independent implementation).

edgartools is a dev-only dependency used purely as a test oracle (ADR-006). A disagreement is
not automatically our bug: it is a *candidate* to be inspected and, if real, logged in
docs/BUGS_FOUND.md with a regression fixture.
"""

from dataclasses import dataclass, field

from edgar.documents import ParserConfig, parse_html

from filingsqa.ingest.html_to_text import parse_filing_html
from filingsqa.ingest.sections import FRONT_MATTER, FULL_DOCUMENT, split_sections

# Section size is measured in alphanumeric characters only: edgartools renders tables with
# box-drawing characters and padding, so raw lengths are not comparable. Only gross
# disagreements count.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0
_PART_NUMBER = {"I": 1, "II": 2}


def _alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


@dataclass
class Comparison:
    only_ours: list[str] = field(default_factory=list)
    only_oracle: list[str] = field(default_factory=list)
    length_mismatch: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        return not (self.only_ours or self.only_oracle or self.length_mismatch)


def oracle_sections(html: str, form: str) -> dict[str, int]:
    """Section key (in our naming) -> text length, according to edgartools."""
    doc = parse_html(html, ParserConfig(form=form))
    out: dict[str, int] = {}
    for section in doc.sections.values():
        item = str(section.item or "").upper()
        if not item:
            continue
        if form.startswith("10-Q"):
            part = str(section.part or "").upper()
            if part not in _PART_NUMBER:
                continue
            key = f"part{_PART_NUMBER[part]}_item_{item.lower()}"
        else:
            key = f"item_{item.lower()}"
        out[key] = _alnum(section.text())
    return out


def our_sections(html: str, form: str) -> dict[str, int]:
    split = split_sections(parse_filing_html(html).blocks, form)
    return {
        s.key: sum(_alnum(b.text) for b in s.blocks)
        for s in split.sections
        if s.key not in (FRONT_MATTER, FULL_DOCUMENT)
    }


def compare(html: str, form: str) -> Comparison:
    ours, oracle = our_sections(html, form), oracle_sections(html, form)
    result = Comparison(
        only_ours=sorted(set(ours) - set(oracle)),
        only_oracle=sorted(set(oracle) - set(ours)),
    )
    for key in sorted(set(ours) & set(oracle)):
        ratio = ours[key] / max(oracle[key], 1)
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            result.length_mismatch[key] = (ours[key], oracle[key])
    return result
