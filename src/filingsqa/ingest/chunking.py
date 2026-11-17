"""Section-aware chunking. See ADR-010.

Rules:
- Chunks never cross section boundaries.
- Text: paragraphs are packed up to `max_words`; a paragraph longer than that is split on
  sentence boundaries. Consecutive text chunks overlap by up to `overlap_words` (whole trailing
  paragraphs, so overlap never cuts a sentence).
- Tables: a table is its own chunk, never mixed with prose. The short paragraph right before it
  (typically the caption with units, e.g. "(In millions)") is repeated in the table chunk, because
  a number without its unit is useless. Big tables are split by rows, repeating the first row.
- Every chunk carries a `context` header (company, form, fiscal period, section) used for
  embedding, so a chunk is self-describing even out of context.
"""

import re
from dataclasses import dataclass

from filingsqa.ingest.html_to_text import Block
from filingsqa.ingest.models import Chunk, FilingMeta
from filingsqa.ingest.sections import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 350
    overlap_words: int = 60
    max_table_words: int = 800
    max_caption_words: int = 40


def word_count(text: str) -> int:
    return len(text.split())


def split_long_text(text: str, max_words: int) -> list[str]:
    """Split one paragraph into pieces of at most ~max_words, on sentence boundaries."""
    if word_count(text) <= max_words:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(text):
        n = word_count(sentence)
        if current and count + n > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        if n > max_words:  # a single giant "sentence": hard split on words
            words = sentence.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_table(table_text: str, max_words: int) -> list[str]:
    """Split a rendered table by rows; every piece after the first repeats the first row."""
    if word_count(table_text) <= max_words:
        return [table_text]
    rows = table_text.split("\n")
    header, body = rows[0], rows[1:]
    pieces: list[str] = []
    current = [header]
    count = word_count(header)
    for row in body:
        n = word_count(row)
        if len(current) > 1 and count + n > max_words:
            pieces.append("\n".join(current))
            current, count = [header], word_count(header)
        current.append(row)
        count += n
    pieces.append("\n".join(current))
    return pieces


class _SectionChunker:
    def __init__(self, section: Section, filing: FilingMeta, config: ChunkingConfig) -> None:
        self.section = section
        self.filing = filing
        self.config = config
        self.chunks: list[Chunk] = []
        self._paragraphs: list[str] = []
        self._words = 0
        self._has_new_content = False
        self.context = (
            f"{filing.company} ({filing.ticker}) | Form {filing.form} | "
            f"{filing.period_label} (period ended {filing.period_end_date.isoformat()}) | "
            f"{section.title}"
        )

    def _emit(self, text: str, content_type: str) -> None:
        idx = len(self.chunks)
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.filing.ticker}-{self.filing.accession}-{self.section.key}-{idx:03d}",
                text=text,
                context=self.context,
                content_type=content_type,
                section_key=self.section.key,
                section_title=self.section.title,
                chunk_index=idx,
                word_count=word_count(text),
                filing=self.filing,
            )
        )

    def _flush_text(self, keep_overlap: bool) -> None:
        if self._has_new_content:
            self._emit("\n\n".join(self._paragraphs), "text")
        if keep_overlap:
            carried: list[str] = []
            words = 0
            for paragraph in reversed(self._paragraphs):
                n = word_count(paragraph)
                if words + n > self.config.overlap_words:
                    break
                carried.insert(0, paragraph)
                words += n
            self._paragraphs, self._words = carried, words
        else:
            self._paragraphs, self._words = [], 0
        self._has_new_content = False

    def _add_paragraph(self, text: str) -> None:
        n = word_count(text)
        if self._has_new_content and self._words + n > self.config.max_words:
            self._flush_text(keep_overlap=True)
            # drop the overlap if it would push this paragraph over the limit
            if self._words + n > self.config.max_words:
                self._paragraphs, self._words = [], 0
        self._paragraphs.append(text)
        self._words += n
        self._has_new_content = True

    def run(self) -> list[Chunk]:
        previous: Block | None = None
        for block in self.section.blocks:
            if block.kind == "table":
                self._flush_text(keep_overlap=False)
                caption = ""
                if (
                    previous is not None
                    and previous.kind == "text"
                    and word_count(previous.text) <= self.config.max_caption_words
                ):
                    caption = previous.text + "\n"
                for piece in split_table(block.text, self.config.max_table_words):
                    self._emit(caption + piece, "table")
            else:
                for piece in split_long_text(block.text, self.config.max_words):
                    self._add_paragraph(piece)
            previous = block
        self._flush_text(keep_overlap=False)
        return self.chunks


def chunk_sections(
    sections: list[Section], filing: FilingMeta, config: ChunkingConfig | None = None
) -> list[Chunk]:
    cfg = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_SectionChunker(section, filing, cfg).run())
    return chunks
