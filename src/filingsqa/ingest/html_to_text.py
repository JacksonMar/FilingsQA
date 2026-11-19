"""Convert SEC filing HTML (usually inline XBRL) into an ordered list of text and table blocks.

Design notes (see ADR-007):
- Block-level HTML elements become separate text blocks; inline elements are concatenated.
- Data tables are kept as tables and rendered row-by-row with ` | ` separators, after
  normalizing SEC's typical cell splitting (`$` and `)` in their own cells).
- Tiny single-row tables are layout tables (e.g. "Item 1A. | Risk Factors"), so they are
  flattened to text. This lets section headings inside layout tables still be detected.
- The inline XBRL header (`ix:header`) and anything with `display:none` is hidden content and
  is dropped from text, but `dei:*` facts are read from it first (fiscal year/period).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from bs4 import BeautifulSoup, Tag
from bs4.element import Comment, NavigableString

BlockKind = Literal["text", "table"]

BLOCK_TAGS = frozenset(
    {
        "address", "article", "blockquote", "body", "br", "center", "dd", "div", "dl", "dt",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
        "pre", "section", "tr", "ul",
    }
)  # fmt: skip
DROP_TAGS = frozenset({"script", "style", "head", "title", "meta", "link", "ix:header"})

# Unicode punctuation/spaces that SEC HTML uses, mapped to ASCII (helps regexes and BM25).
_CHAR_MAP = {
    0x00A0: " ",  # no-break space
    0x2002: " ",  # en space
    0x2003: " ",  # em space
    0x2009: " ",  # thin space
    0x200B: None,  # zero-width space
    0xFEFF: None,  # byte order mark
    0x2018: "'",  # left single quote
    0x2019: "'",  # right single quote / apostrophe
    0x201C: '"',  # left double quote
    0x201D: '"',  # right double quote
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
}
_WS = re.compile(r"\s+")
_DEI_NAME = re.compile(r"^dei:")

# Lines that are page furniture rather than content.
_ARTIFACT = re.compile(
    r"^(table of contents|\d{1,3}|page \d{1,3}|.{0,80}form 10-[kq]\s*\|?\s*\d{1,3})$",
    re.IGNORECASE,
)
_ATTACH_LEFT = frozenset({")", "%", ")%", "%)"})  # belongs to the previous cell
_ATTACH_RIGHT = frozenset({"$", "("})  # belongs to the next cell


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str


@dataclass
class ParsedDocument:
    blocks: list[Block]
    dei: dict[str, str] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.translate(_CHAR_MAP)).strip()


def _is_hidden(tag: Tag) -> bool:
    style = str(tag.get("style", "")).replace(" ", "").lower()
    return "display:none" in style


def _rendered_text(tag: Tag) -> str:
    """Text of an element as a browser shows it: inline children are concatenated as-is,
    block children and <br> are separated by a space.

    `tag.get_text(" ")` would put a space between *every* pair of elements, which turns
    <span>10-</span><span>Q</span> into "10- Q" (BUG-1 in docs/BUGS_FOUND.md).
    """
    parts: list[str] = []
    for node in tag.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag) and node.name in BLOCK_TAGS:
            parts.append(" ")
    return "".join(parts)


def _extract_dei(soup: BeautifulSoup) -> dict[str, str]:
    facts: dict[str, str] = {}
    for tag in soup.find_all(True, attrs={"name": _DEI_NAME}):
        name = str(tag["name"]).removeprefix("dei:")
        value = normalize_text(_rendered_text(tag))
        if value and name not in facts:
            facts[name] = value
    return facts


def normalize_row(cells: list[str]) -> list[str]:
    """Drop empty cells and glue split currency / parenthesis / percent fragments."""
    out: list[str] = []
    pending_prefix = ""
    for raw in cells:
        cell = normalize_text(raw)
        if not cell:
            continue
        if cell in _ATTACH_RIGHT:
            pending_prefix += cell
            continue
        if cell in _ATTACH_LEFT and out:
            out[-1] += cell
            continue
        out.append(pending_prefix + cell)
        pending_prefix = ""
    if pending_prefix:
        out.append(pending_prefix)
    return out


def _table_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # rows of nested tables are handled by their own table
        cells = [_rendered_text(td) for td in tr.find_all(["td", "th"]) if not _is_hidden(td)]
        row = normalize_row(cells)
        if row:
            rows.append(row)
    return rows


class _Builder:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._buf: list[str] = []

    def add_text(self, text: str) -> None:
        self._buf.append(text)

    def flush(self) -> None:
        text = normalize_text("".join(self._buf))
        self._buf.clear()
        if text and not _ARTIFACT.match(text):
            self.blocks.append(Block("text", text))

    def add_table(self, rows: list[list[str]]) -> None:
        self.flush()
        n_cells = sum(len(r) for r in rows)
        if len(rows) == 1 and n_cells <= 3:
            # layout table: flatten to a text line
            self.add_text(" ".join(rows[0]))
            self.flush()
        else:
            self.blocks.append(Block("table", "\n".join(" | ".join(r) for r in rows)))


def _walk(node: Tag, out: _Builder) -> None:
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            out.add_text(str(child))
            continue
        if not isinstance(child, Tag) or child.name in DROP_TAGS or _is_hidden(child):
            continue
        if child.name == "table":
            rows = _table_rows(child)
            if rows:
                out.add_table(rows)
            continue
        if child.name in BLOCK_TAGS:
            out.flush()
            _walk(child, out)
            out.flush()
        else:
            _walk(child, out)


def parse_filing_html(html: str) -> ParsedDocument:
    soup = BeautifulSoup(html, "lxml")
    dei = _extract_dei(soup)
    root = soup.body or soup
    builder = _Builder()
    _walk(root, builder)
    builder.flush()
    return ParsedDocument(blocks=builder.blocks, dei=dei)
