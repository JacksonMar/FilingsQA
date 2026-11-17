"""Read SEC XBRL "companyfacts" JSON and pick the fact that answers a question. See ADR-020.

companyfacts layout: facts[taxonomy][tag]["units"][unit] -> list of
  {start?, end, val, accn, fy, fp, form, filed, frame?}
Each entry is one value *as reported in one filing* (accn). A 10-K reports the current year
and comparatives, so one filing has several entries for the same tag with different periods.

Important: `fy`/`fp` on an entry describe the *filing*, not the period of the value. We select
values by their own `start`/`end` dates and use the filing's fy/fp only for the label of the
current period.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

PeriodType = Literal["annual", "quarter", "instant"]

ANNUAL_DAYS = (340, 380)  # 52/53-week fiscal years
QUARTER_DAYS = (80, 100)  # 13/14-week quarters; excludes 6- and 9-month YTD values


@dataclass(frozen=True)
class Fact:
    taxonomy: str
    tag: str
    unit: str
    value: float
    accession: str
    start: date | None
    end: date
    fy: int | None
    fp: str | None
    form: str
    filed: date

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days + 1

    @property
    def period_type(self) -> PeriodType | None:
        days = self.days
        if days is None:
            return "instant"
        if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
            return "annual"
        if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
            return "quarter"
        return None  # YTD or odd period


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def iter_facts(companyfacts: dict[str, Any], taxonomy: str, tag: str, unit: str) -> list[Fact]:
    entries = companyfacts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {})
    out: list[Fact] = []
    for e in entries.get(unit, []):
        out.append(
            Fact(
                taxonomy=taxonomy,
                tag=tag,
                unit=unit,
                value=float(e["val"]),
                accession=e["accn"],
                start=_d(e.get("start")),
                end=date.fromisoformat(e["end"]),
                fy=e.get("fy"),
                fp=e.get("fp"),
                form=e.get("form", ""),
                filed=date.fromisoformat(e["filed"]),
            )
        )
    return out


def find_fact(
    companyfacts: dict[str, Any],
    tags: tuple[str, ...],
    unit: str,
    accession: str,
    period_end: date,
    period_type: PeriodType,
    taxonomy: str = "us-gaap",
) -> Fact | None:
    """First tag (in priority order) with a value reported in `accession` for that exact period.

    Only facts from the filing itself are used: if the filing did not report the value, we do
    not borrow a later restated number from another filing.
    """
    for tag in tags:
        candidates = {
            f.value: f
            for f in iter_facts(companyfacts, taxonomy, tag, unit)
            if f.accession == accession and f.end == period_end and f.period_type == period_type
        }
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        if len(candidates) > 1:
            # Same tag, same period, different values in one filing (e.g. segment members leaking
            # into companyfacts): ambiguous ground truth -> skip rather than guess.
            return None
    return None
