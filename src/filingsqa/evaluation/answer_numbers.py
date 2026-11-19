"""Extract numbers (with their scale) from a free-text answer. See ADR-022.

"$130.5 billion" -> 130.5e9, "$130,497 million" -> 130_497e6, "$2.94" -> 2.94,
"(1,234)" -> 1234 (negative), "33%" -> percent. A number without a scale word ("$130,497")
is kept as `scaled=False`: the matcher then also tries the "in millions" interpretation, because
answers often copy table values whose unit is only in the table caption.
"""

import re
from dataclasses import dataclass

SCALES = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}  # fmt: skip

_NUMBER = re.compile(
    r"(?P<open>\()?(?P<minus>-)?\s*(?P<cur>\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<close>\))?"
    r"(?:\s*(?P<scale>trillion|billion|million|thousand|tn|bn|mn|mm|[bmk])\b)?"
    r"\s*(?P<pct>%|percent\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerNumber:
    raw: str
    value: float  # already multiplied by the scale word, if any
    scaled: bool  # a scale word (million, billion, ...) was present
    currency: bool
    percent: bool
    negative: bool


def extract_numbers(text: str) -> list[AnswerNumber]:
    out: list[AnswerNumber] = []
    for m in _NUMBER.finditer(text):
        num = float(m.group("num").replace(",", ""))
        scale_word = (m.group("scale") or "").lower()
        # A lone "b"/"m"/"k" only counts as a scale right after a currency amount ("$1.2B"),
        # otherwise "5 m" in prose would turn into five million.
        if len(scale_word) == 1 and not m.group("cur"):
            scale_word = ""
        scale = SCALES.get(scale_word, 1.0)
        negative = bool(m.group("minus")) or bool(m.group("open") and m.group("close"))
        out.append(
            AnswerNumber(
                raw=m.group(0).strip(),
                value=num * scale,
                scaled=bool(scale_word),
                currency=bool(m.group("cur")),
                percent=bool(m.group("pct")),
                negative=negative,
            )
        )
    return out
