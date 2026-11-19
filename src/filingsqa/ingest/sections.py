"""Section detection for 10-K and 10-Q filings, using Item headings as boundaries.

Algorithm (see ADR-008):
1. Scan text blocks (never data tables) for short lines that look like `Item 1A. Risk Factors`
   or `PART II`. Each valid Item heading becomes a *candidate*.
2. Item headings also appear in the table of contents and in cross-references, so the same Item
   can have several candidates. For every Item keep the candidate followed by the *largest*
   amount of text before the next candidate: a TOC entry is followed by almost nothing.
3. The chosen headings split the document into sections. Text before the first heading is
   `front_matter`. If no heading is found at all, the whole document is one `full_document`
   section. No block is ever dropped: every block belongs to exactly one section.
"""

import re
from dataclasses import dataclass, field

from filingsqa.ingest.html_to_text import Block

TEN_K_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
    "Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related "
    "Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

TEN_Q_ITEMS: dict[tuple[str, str], str] = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis of Financial Condition and Results of "
    "Operations",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}

FRONT_MATTER = "front_matter"
FULL_DOCUMENT = "full_document"

MAX_HEADING_CHARS = 150
_ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-]?\s*(.*)$", re.IGNORECASE)
_PART_NUMBER = {"I": 1, "II": 2}
_PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b\s*[.:\-]?\s*(.*)$", re.IGNORECASE)


@dataclass
class Section:
    key: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class SectionSplit:
    sections: list[Section]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Candidate:
    block_index: int
    key: str
    title: str
    order: int  # canonical position, used for the order sanity check


def _match_heading(text: str) -> tuple[str, str] | None:
    """Return (kind, number) for 'item'/'part' headings, else None."""
    if len(text) > MAX_HEADING_CHARS:
        return None
    if m := _PART_RE.match(text):
        rest = m.group(2)
        if not rest or rest[0].isupper():
            return "part", m.group(1).upper()
    if m := _ITEM_RE.match(text):
        rest = m.group(2)
        # "Item 7 of this report ..." is a reference, not a heading
        if not rest or rest[0].isupper() or rest[0] in "[(\"'":
            return "item", m.group(1).upper()
    return None


def _candidates(blocks: list[Block], form: str) -> list[_Candidate]:
    is_10q = form.startswith("10-Q")
    k_order = {num: i for i, num in enumerate(TEN_K_ITEMS)}
    q_order = {key: i for i, key in enumerate(TEN_Q_ITEMS)}
    part = "I"
    found: list[_Candidate] = []
    for idx, block in enumerate(blocks):
        if block.kind != "text":
            continue
        heading = _match_heading(block.text)
        if heading is None:
            continue
        kind, number = heading
        if kind == "part":
            part = number
        elif is_10q and (part, number) in TEN_Q_ITEMS:
            key = f"part{_PART_NUMBER[part]}_item_{number.lower()}"
            found.append(_Candidate(idx, key, TEN_Q_ITEMS[(part, number)], q_order[part, number]))
        elif not is_10q and number in TEN_K_ITEMS:
            key = f"item_{number.lower()}"
            found.append(_Candidate(idx, key, TEN_K_ITEMS[number], k_order[number]))
    return found


def split_sections(blocks: list[Block], form: str) -> SectionSplit:
    candidates = _candidates(blocks, form)
    if not candidates:
        return SectionSplit(
            [Section(FULL_DOCUMENT, "Full document", list(blocks))],
            ["no Item headings found; using a single full_document section"],
        )

    # Amount of text that follows each candidate until the next candidate.
    bounds = [c.block_index for c in candidates] + [len(blocks)]
    span = {
        c.block_index: sum(len(b.text) for b in blocks[c.block_index + 1 : bounds[i + 1]])
        for i, c in enumerate(candidates)
    }
    best: dict[str, _Candidate] = {}
    for c in candidates:
        if c.key not in best or span[c.block_index] > span[best[c.key].block_index]:
            best[c.key] = c
    chosen = sorted(best.values(), key=lambda c: c.block_index)

    warnings: list[str] = []
    orders = [c.order for c in chosen]
    if orders != sorted(orders):
        warnings.append(f"Item headings out of canonical order: {[c.key for c in chosen]}")

    sections: list[Section] = []
    if chosen[0].block_index > 0:
        sections.append(Section(FRONT_MATTER, "Front matter", blocks[: chosen[0].block_index]))
    for i, c in enumerate(chosen):
        end = chosen[i + 1].block_index if i + 1 < len(chosen) else len(blocks)
        sections.append(Section(c.key, c.title, blocks[c.block_index : end]))
    return SectionSplit(sections, warnings)
