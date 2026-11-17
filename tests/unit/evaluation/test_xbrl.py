from datetime import date

from filingsqa.evaluation.xbrl import find_fact, iter_facts
from tests.unit.evaluation.conftest import AAPL_K, AAPL_K_OLD, AAPL_Q3, MSFT_K

REV = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")


def test_period_type_from_dates_not_from_fy_fp(aapl_facts):
    by_value = {
        f.value: f.period_type for f in iter_facts(aapl_facts, "us-gaap", "Revenues", "USD")
    }

    assert by_value[400e9] == "annual"
    assert by_value[90e9] == "quarter"
    assert by_value[300e9] is None  # 9-month YTD


def test_quarter_value_ignores_ytd(aapl_facts):
    fact = find_fact(aapl_facts, REV, "USD", AAPL_Q3, date(2025, 6, 28), "quarter")

    assert fact is not None
    assert fact.value == 90e9


def test_only_values_reported_in_that_filing(aapl_facts):
    fact = find_fact(aapl_facts, REV, "USD", AAPL_K_OLD, date(2025, 9, 27), "annual")

    assert fact is None


def test_tag_fallback_order(msft_facts):
    fact = find_fact(msft_facts, REV, "USD", MSFT_K, date(2025, 6, 30), "annual")

    assert fact is not None
    assert fact.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_ambiguous_values_are_skipped(aapl_facts):
    assert (
        find_fact(aapl_facts, ("OperatingIncomeLoss",), "USD", AAPL_K, date(2025, 9, 27), "annual")
        is None
    )


def test_instant_fact(aapl_facts):
    fact = find_fact(aapl_facts, ("Assets",), "USD", AAPL_K, date(2025, 9, 27), "instant")

    assert fact is not None
    assert fact.value == 360e9
    assert fact.start is None
