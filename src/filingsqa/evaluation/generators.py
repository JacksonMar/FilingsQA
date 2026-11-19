"""Generate golden items from XBRL companyfacts + the frozen filings manifest. See ADR-020.

Everything here is deterministic: same manifest + same companyfacts -> same items.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
from typing import Any

from filingsqa.companies import COMPANIES
from filingsqa.edgar.manifest import FilingRef, Manifest
from filingsqa.evaluation.dataset import (
    ExpectedNumber,
    ExpectedSource,
    GoldenItem,
    XbrlRef,
)
from filingsqa.evaluation.metric_catalog import METRICS, METRICS_BY_KEY, MetricDef
from filingsqa.evaluation.numbers import format_value
from filingsqa.evaluation.xbrl import Fact, PeriodType, find_fact, iter_facts

QUARTER_WORDS = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}
# Where financial figures live. Some 10-Ks (NVIDIA) put the statements under Item 15 and only
# cross-reference them from Item 8, so Item 15 counts as a valid source too.
SECTIONS = {
    "10-K": ("item_8", "item_7", "item_15"),
    "10-Q": ("part1_item_1", "part1_item_2"),
}
CompanyFacts = dict[str, Any]


@dataclass(frozen=True)
class FilingPeriod:
    ref: FilingRef
    fiscal_year: int
    fiscal_period: str  # FY, Q1, Q2, Q3

    @property
    def label(self) -> str:
        if self.fiscal_period == "FY":
            return f"fiscal year {self.fiscal_year}"
        return f"the {QUARTER_WORDS[self.fiscal_period]} quarter of fiscal {self.fiscal_year}"


def filing_period(ref: FilingRef, facts: CompanyFacts) -> FilingPeriod | None:
    """Fiscal year/period of a filing = the fy/fp most entries reported in it carry."""
    votes: Counter[tuple[int, str]] = Counter()
    for tax in facts.get("facts", {}).values():
        for tag in tax.values():
            for entries in tag.get("units", {}).values():
                for e in entries:
                    if e.get("accn") == ref.accession and e.get("fy") and e.get("fp"):
                        votes[(int(e["fy"]), str(e["fp"]))] += 1
    if not votes:
        return None
    (fy, fp), _ = votes.most_common(1)[0]
    return FilingPeriod(ref, fy, fp)


def _period_type(metric: MetricDef, period: FilingPeriod) -> PeriodType:
    if metric.kind == "instant":
        return "instant"
    return "annual" if period.fiscal_period == "FY" else "quarter"


def _expected_number(
    metric: MetricDef, ticker: str, fact: Fact, fy: int, fp: str
) -> ExpectedNumber:
    return ExpectedNumber(
        label=metric.key,
        ticker=ticker,
        value=fact.value,
        unit=fact.unit,
        fiscal_year=fy,
        fiscal_period=fp,
        period_end=fact.end,
        tolerance_rel=metric.tolerance_rel,
        tolerance_abs=metric.tolerance_abs,
        xbrl=XbrlRef(
            taxonomy=fact.taxonomy,
            tag=fact.tag,
            unit=fact.unit,
            accession=fact.accession,
            start=fact.start,
            end=fact.end,
        ),
    )


def _source(period: FilingPeriod) -> ExpectedSource:
    ref = period.ref
    return ExpectedSource(
        ticker=ref.ticker,
        form=ref.form,
        section_keys=SECTIONS[ref.form],
        accession=ref.accession,
        fiscal_year=period.fiscal_year,
        fiscal_period=period.fiscal_period,
    )


def _question(metric: MetricDef, company: str, period_label: str) -> str:
    if metric.kind == "instant":
        return f"What were {company}'s {metric.label} at the end of {period_label}?"
    return f"What was {company}'s {metric.label} for {period_label}?"


def _answer(metric: MetricDef, company: str, period_label: str, fact: Fact) -> str:
    when = f"{period_label} (period ended {fact.end:%B %d, %Y})"
    return f"{company}'s {metric.label} for {when} was {format_value(fact.value, fact.unit)}."


def _prior_year_fact(facts: CompanyFacts, metric: MetricDef, ref: FilingRef) -> Fact | None:
    """The comparative annual value shown in a 10-K: same filing, period ending ~1 year earlier."""
    lo, hi = ref.report_date - timedelta(days=380), ref.report_date - timedelta(days=350)
    for tag in metric.tags:
        matches = {
            f.value: f
            for f in iter_facts(facts, "us-gaap", tag, metric.unit)
            if f.accession == ref.accession and lo <= f.end <= hi and f.period_type == "annual"
        }
        if len(matches) == 1:
            return next(iter(matches.values()))
    return None


def numeric_items(periods: list[FilingPeriod], facts: CompanyFacts) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for period in periods:
        ref = period.ref
        company = COMPANIES[ref.ticker].short_name
        for metric in METRICS:
            fact = find_fact(
                facts, metric.tags, metric.unit, ref.accession, ref.report_date,
                _period_type(metric, period),
            )  # fmt: skip
            if fact is None:
                continue
            fp, fy = period.fiscal_period, period.fiscal_year
            is_headline = metric.key in ("revenue", "net_income") and fp == "FY"
            items.append(
                GoldenItem(
                    id=f"num-{ref.ticker}-{metric.key}-{fy}{fp}".lower(),
                    category="numeric",
                    subcategory="annual" if fp == "FY" else "quarterly",
                    difficulty="easy" if is_headline else "medium",
                    question=_question(metric, company, period.label),
                    expected_behavior="answer",
                    expected_answer=_answer(metric, company, period.label, fact),
                    expected_numbers=(_expected_number(metric, ref.ticker, fact, fy, fp),),
                    expected_sources=(_source(period),),
                    verified=True,
                    source="generated",
                )
            )
            if metric.with_prior_year and fp == "FY":
                prior = _prior_year_fact(facts, metric, ref)
                if prior is not None:
                    label = f"fiscal year {fy - 1}"
                    items.append(
                        GoldenItem(
                            id=f"num-{ref.ticker}-{metric.key}-{fy - 1}fy-comparative".lower(),
                            category="numeric",
                            subcategory="prior_year_comparative",
                            difficulty="medium",
                            question=_question(metric, company, label),
                            expected_behavior="answer",
                            expected_answer=_answer(metric, company, label, prior),
                            expected_numbers=(
                                _expected_number(metric, ref.ticker, prior, fy - 1, "FY"),
                            ),
                            expected_sources=(_source(period),),
                            verified=True,
                            source="generated",
                            notes="Value appears as a comparative column in the latest 10-K.",
                        )
                    )
    return items


def _annual(periods: list[FilingPeriod]) -> FilingPeriod | None:
    return next((p for p in periods if p.fiscal_period == "FY"), None)


def comparison_items(
    periods_by_ticker: dict[str, list[FilingPeriod]], facts_by_ticker: dict[str, CompanyFacts]
) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    annual: dict[str, tuple[FilingPeriod, dict[str, Fact]]] = {}
    for ticker, periods in periods_by_ticker.items():
        p = _annual(periods)
        if p is None:
            continue
        found = {}
        for m in METRICS:
            if m.comparable:
                f = find_fact(
                    facts_by_ticker[ticker], m.tags, m.unit, p.ref.accession, p.ref.report_date,
                    _period_type(m, p),
                )  # fmt: skip
                if f is not None:
                    found[m.key] = f
        annual[ticker] = (p, found)

    for key in ("revenue", "net_income"):
        metric = METRICS_BY_KEY[key]
        for a, b in combinations(sorted(annual), 2):
            (pa, fa), (pb, fb) = annual[a], annual[b]
            if key not in fa or key not in fb or fa[key].value == fb[key].value:
                continue
            winner = a if fa[key].value > fb[key].value else b
            na, nb = COMPANIES[a].short_name, COMPANIES[b].short_name
            items.append(
                GoldenItem(
                    id=f"cmp-{a}-{b}-{key}".lower(),
                    category="comparison",
                    subcategory="cross_company",
                    difficulty="hard",
                    question=(
                        f"Which company had higher {metric.label} in its most recent fiscal year: "
                        f"{na} or {nb}?"
                    ),
                    expected_behavior="answer",
                    expected_answer=(
                        f"{COMPANIES[winner].short_name}. {na}: "
                        f"{format_value(fa[key].value, metric.unit)} for fiscal year "
                        f"{pa.fiscal_year}; {nb}: {format_value(fb[key].value, metric.unit)} "
                        f"for fiscal year {pb.fiscal_year}."
                    ),
                    expected_numbers=(
                        _expected_number(metric, a, fa[key], pa.fiscal_year, "FY"),
                        _expected_number(metric, b, fb[key], pb.fiscal_year, "FY"),
                    ),
                    expected_winner=winner,
                    expected_sources=(_source(pa), _source(pb)),
                    verified=True,
                    source="generated",
                    notes="Fiscal years of the two companies may end in different months.",
                )
            )

    for ticker, (p, _found) in sorted(annual.items()):
        for key in ("revenue", "net_income"):
            metric = METRICS_BY_KEY[key]
            cur = find_fact(
                facts_by_ticker[ticker], metric.tags, metric.unit, p.ref.accession,
                p.ref.report_date, "annual",
            )  # fmt: skip
            prev = _prior_year_fact(facts_by_ticker[ticker], metric, p.ref)
            if cur is None or prev is None or prev.value == 0:
                continue
            change = (cur.value - prev.value) / abs(prev.value) * 100
            name, fy = COMPANIES[ticker].short_name, p.fiscal_year
            items.append(
                GoldenItem(
                    id=f"cmp-{ticker}-{key}-yoy-{fy}".lower(),
                    category="comparison",
                    subcategory="yoy_change",
                    difficulty="hard",
                    question=(
                        f"By what percentage did {name}'s {metric.label} change from fiscal "
                        f"{fy - 1} to fiscal {fy}?"
                    ),
                    expected_behavior="answer",
                    expected_answer=(
                        f"It changed by {change:+.1f}%: from "
                        f"{format_value(prev.value, metric.unit)} in fiscal {fy - 1} to "
                        f"{format_value(cur.value, metric.unit)} in fiscal {fy}."
                    ),
                    expected_numbers=(
                        _expected_number(metric, ticker, prev, fy - 1, "FY"),
                        _expected_number(metric, ticker, cur, fy, "FY"),
                    ),
                    expected_sources=(_source(p),),
                    verified=True,
                    source="generated",
                    notes=f"Multi-hop: needs both years. Expected change {change:+.2f}%.",
                )
            )
    return items


def temporal_items(
    periods_by_ticker: dict[str, list[FilingPeriod]], facts_by_ticker: dict[str, CompanyFacts]
) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    revenue = METRICS_BY_KEY["revenue"]
    for ticker, periods in sorted(periods_by_ticker.items()):
        name = COMPANIES[ticker].short_name
        facts = facts_by_ticker[ticker]
        p = _annual(periods)
        if p is not None:
            fy, end = p.fiscal_year, p.ref.report_date
            # 1) Calendar wording for a non-calendar fiscal year.
            if end.month != 12:
                fact = find_fact(facts, revenue.tags, revenue.unit, p.ref.accession, end, "annual")
                if fact is not None:
                    items.append(
                        GoldenItem(
                            id=f"tmp-{ticker}-calendar-vs-fiscal-{end.year}".lower(),
                            category="temporal",
                            subcategory="fiscal_vs_calendar",
                            difficulty="hard",
                            question=f"What was {name}'s total revenue in {end.year}?",
                            expected_behavior="answer",
                            expected_answer=(
                                f"{name}'s fiscal year does not match the calendar year. Its "
                                f"fiscal year {fy} ended {end:%B %d, %Y}, with total revenue of "
                                f"{format_value(fact.value, fact.unit)}."
                            ),
                            expected_numbers=(_expected_number(revenue, ticker, fact, fy, "FY"),),
                            key_facts=(f"fiscal year {fy}",),
                            expected_sources=(_source(p),),
                            verified=True,
                            source="generated",
                            notes="Answer must name the fiscal period explicitly, not 'in 2025'.",
                        )
                    )
            # 2) Periods outside the corpus.
            for offset, sub in ((2, "future_period"), (-6, "out_of_corpus_period")):
                items.append(
                    GoldenItem(
                        id=f"tmp-{ticker}-{sub}-{fy + offset}".lower().replace("_", "-"),
                        category="temporal",
                        subcategory=sub,
                        difficulty="medium",
                        question=f"What was {name}'s total revenue for fiscal year {fy + offset}?",
                        expected_behavior="insufficient_context",
                        verified=True,
                        source="generated",
                        notes="Must not invent a number; the filings do not cover this period.",
                    )
                )
            # 3) False premise ("doubled") when revenue clearly did not double.
            cur = find_fact(facts, revenue.tags, revenue.unit, p.ref.accession, end, "annual")
            prev = _prior_year_fact(facts, revenue, p.ref)
            if (
                cur is not None
                and prev is not None
                and prev.value > 0
                and cur.value / prev.value < 1.5
            ):
                items.append(
                    GoldenItem(
                        id=f"adv-{ticker}-false-premise-doubled-{fy}".lower(),
                        category="adversarial",
                        subcategory="false_premise",
                        difficulty="hard",
                        question=f"Why did {name}'s revenue double in fiscal {fy}?",
                        expected_behavior="answer",
                        expected_answer=(
                            f"The premise is wrong: revenue did not double. It went from "
                            f"{format_value(prev.value, prev.unit)} in fiscal {fy - 1} to "
                            f"{format_value(cur.value, cur.unit)} in fiscal {fy}."
                        ),
                        expected_numbers=(
                            _expected_number(revenue, ticker, prev, fy - 1, "FY"),
                            _expected_number(revenue, ticker, cur, fy, "FY"),
                        ),
                        expected_sources=(_source(p),),
                        verified=True,
                        source="generated",
                    )
                )
        # 4) "Most recent quarter" must resolve to the latest 10-Q in the corpus.
        quarters = sorted(
            (q for q in periods if q.fiscal_period != "FY"), key=lambda q: q.ref.report_date
        )
        if quarters:
            q = quarters[-1]
            fact = find_fact(
                facts, revenue.tags, revenue.unit, q.ref.accession, q.ref.report_date, "quarter"
            )
            if fact is not None:
                items.append(
                    GoldenItem(
                        id=f"tmp-{ticker}-latest-quarter-revenue".lower(),
                        category="temporal",
                        subcategory="latest_period",
                        difficulty="medium",
                        question=f"What was {name}'s total revenue in its most recent quarter?",
                        expected_behavior="answer",
                        expected_answer=_answer(revenue, name, q.label, fact),
                        expected_numbers=(
                            _expected_number(revenue, ticker, fact, q.fiscal_year, q.fiscal_period),
                        ),
                        key_facts=(q.label,),
                        expected_sources=(_source(q),),
                        verified=True,
                        source="generated",
                        notes="'Most recent' = latest 10-Q in the frozen manifest.",
                    )
                )
    return items


def generate_all(
    manifest: Manifest, facts_by_ticker: dict[str, CompanyFacts]
) -> tuple[list[GoldenItem], list[str]]:
    """All generated items + warnings (e.g. a filing whose fiscal period could not be read)."""
    warnings: list[str] = []
    periods_by_ticker: dict[str, list[FilingPeriod]] = {}
    for ref in manifest.filings:
        facts = facts_by_ticker.get(ref.ticker)
        if facts is None:
            warnings.append(f"{ref.ticker}: no companyfacts")
            continue
        period = filing_period(ref, facts)
        if period is None:
            warnings.append(f"{ref.ticker} {ref.accession}: no XBRL facts for this filing")
            continue
        periods_by_ticker.setdefault(ref.ticker, []).append(period)

    items: list[GoldenItem] = []
    for ticker, periods in sorted(periods_by_ticker.items()):
        items += numeric_items(periods, facts_by_ticker[ticker])
    items += comparison_items(periods_by_ticker, facts_by_ticker)
    items += temporal_items(periods_by_ticker, facts_by_ticker)
    return items, warnings
