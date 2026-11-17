"""Deterministic query analysis: which known companies does a question mention?

Only tickers are inferred automatically. Fiscal year/period are NOT turned into filters: a
FY2025 10-K also reports FY2024 and FY2023 figures as comparatives, so filtering on the
question's year would hide the right chunk. See ADR-015.
"""

import re

from filingsqa.companies import COMPANIES

ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "NVDA": ("nvidia",),
    "JPM": ("jpmorgan", "jp morgan", "j.p. morgan", "jpmorgan chase"),
    "TSLA": ("tesla",),
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"(?<![A-Za-z])(\$?"
        + re.escape(ticker)
        + r"|"
        + "|".join(map(re.escape, names))
        + r")(?![A-Za-z])",
        re.IGNORECASE,
    )
    for ticker, names in ALIASES.items()
}
assert set(_PATTERNS) == set(COMPANIES), "every company needs aliases"


def detect_tickers(question: str) -> list[str]:
    """Known tickers mentioned in the question, in order of first mention."""
    found: list[tuple[int, str]] = []
    for ticker, pattern in _PATTERNS.items():
        if m := pattern.search(question):
            found.append((m.start(), ticker))
    return [t for _, t in sorted(found)]
