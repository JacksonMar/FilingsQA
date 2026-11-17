"""Which financial metrics become numeric questions, and how to find them in XBRL.

Tags are tried in order: companies use different (equally valid) us-gaap tags for revenue.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str  # wording used in questions
    tags: tuple[str, ...]
    unit: str
    kind: Literal["duration", "instant"]
    tolerance_rel: float = 0.01
    tolerance_abs: float = 0.0
    with_prior_year: bool = False  # also ask about the comparative year shown in the 10-K
    comparable: bool = False  # used in cross-company comparison questions


METRICS: tuple[MetricDef, ...] = (
    MetricDef(
        "revenue",
        "total revenue",
        ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
        "USD",
        "duration",
        with_prior_year=True,
        comparable=True,
    ),
    MetricDef(
        "net_income",
        "net income",
        ("NetIncomeLoss",),
        "USD",
        "duration",
        with_prior_year=True,
        comparable=True,
    ),
    MetricDef(
        "eps_diluted",
        "diluted earnings per share",
        ("EarningsPerShareDiluted",),
        "USD/shares",
        "duration",
        tolerance_rel=0.0,
        tolerance_abs=0.01,
    ),
    MetricDef("operating_income", "operating income", ("OperatingIncomeLoss",), "USD", "duration"),
    MetricDef(
        "rnd",
        "research and development expense",
        ("ResearchAndDevelopmentExpense",),
        "USD",
        "duration",
    ),
    MetricDef("total_assets", "total assets", ("Assets",), "USD", "instant", comparable=True),
    MetricDef(
        "cash",
        "cash and cash equivalents",
        ("CashAndCashEquivalentsAtCarryingValue",),
        "USD",
        "instant",
    ),
)

METRICS_BY_KEY = {m.key: m for m in METRICS}
