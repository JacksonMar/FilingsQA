"""Differential testing: compare our Item parser with edgartools (an independent implementation).

A disagreement is not automatically our bug: it is a *candidate* to be inspected and, if real,
logged in docs/BUGS_FOUND.md with a regression fixture. The size test is the same one the
hybrid section resolver uses for core sections (src/filingsqa/ingest/oracle_sections.py).
"""

from dataclasses import dataclass, field

from filingsqa.ingest.html_to_text import parse_filing_html
from filingsqa.ingest.oracle_sections import oracle_sections, section_sizes, sizes_disagree
from filingsqa.ingest.sections import split_sections


@dataclass
class Comparison:
    only_ours: list[str] = field(default_factory=list)
    only_oracle: list[str] = field(default_factory=list)
    length_mismatch: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        return not (self.only_ours or self.only_oracle or self.length_mismatch)


def compare(html: str, form: str) -> Comparison:
    """Our *own* parser (no hybrid) vs edgartools, section by section."""
    ours = section_sizes(split_sections(parse_filing_html(html).blocks, form))
    oracle = {o.key: o.size for o in oracle_sections(html, form)}
    result = Comparison(
        only_ours=sorted(set(ours) - set(oracle)),
        only_oracle=sorted(set(oracle) - set(ours)),
    )
    for key in sorted(set(ours) & set(oracle)):
        if sizes_disagree(ours[key], oracle[key]):
            result.length_mismatch[key] = (ours[key], oracle[key])
    return result
