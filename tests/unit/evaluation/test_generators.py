import pytest

from filingsqa.evaluation.generators import filing_period, generate_all


@pytest.fixture
def items(manifest, facts_by_ticker):
    items, warnings = generate_all(manifest, facts_by_ticker)
    assert warnings == []
    return {i.id: i for i in items}


def test_filing_period_uses_filing_fy_fp(manifest, aapl_facts):
    q1 = next(f for f in manifest.filings if f.report_date.month == 12)

    period = filing_period(q1, aapl_facts)

    assert (period.fiscal_year, period.fiscal_period) == (2026, "Q1")
    assert period.label == "the first quarter of fiscal 2026"


def test_annual_revenue_item(items):
    item = items["num-aapl-revenue-2025fy"]

    assert item.question == "What was Apple's total revenue for fiscal year 2025?"
    assert item.difficulty == "easy"
    num = item.expected_numbers[0]
    assert (num.value, num.unit, num.fiscal_year, num.fiscal_period) == (400e9, "USD", 2025, "FY")
    assert num.xbrl.tag == "Revenues"
    assert "$400,000 million" in item.expected_answer
    assert item.expected_sources[0].section_keys == ("item_8", "item_7")


def test_prior_year_comparative_comes_from_latest_10k(items):
    item = items["num-aapl-revenue-2024fy-comparative"]

    assert item.expected_numbers[0].value == 390e9
    assert item.expected_numbers[0].xbrl.accession == item.expected_sources[0].accession


def test_quarterly_item_uses_quarter_not_ytd(items):
    assert items["num-aapl-revenue-2025q3"].expected_numbers[0].value == 90e9
    assert items["num-aapl-revenue-2026q1"].question.endswith(
        "for the first quarter of fiscal 2026?"
    )


def test_eps_uses_absolute_tolerance(items):
    num = items["num-aapl-eps_diluted-2025fy"].expected_numbers[0]

    assert (num.tolerance_rel, num.tolerance_abs) == (0.0, 0.01)


def test_ambiguous_metric_produces_no_item(items):
    assert not any("operating_income" in i for i in items)


def test_instant_metric_question_wording(items):
    assert (
        items["num-aapl-total_assets-2025fy"].question
        == "What were Apple's total assets at the end of fiscal year 2025?"
    )


def test_cross_company_comparison(items):
    rev = items["cmp-aapl-msft-revenue"]
    ni = items["cmp-aapl-msft-net_income"]

    assert rev.expected_winner == "AAPL"
    assert ni.expected_winner == "MSFT"
    assert {n.ticker for n in rev.expected_numbers} == {"AAPL", "MSFT"}
    assert rev.difficulty == "hard"


def test_yoy_change_is_multi_hop(items):
    item = items["cmp-aapl-revenue-yoy-2025"]

    assert [n.value for n in item.expected_numbers] == [390e9, 400e9]
    assert "+2.56%" in item.notes


def test_temporal_traps(items):
    assert "tmp-aapl-calendar-vs-fiscal-2025" in items  # FY ends September
    assert "tmp-msft-calendar-vs-fiscal-2025" in items  # FY ends June
    assert items["tmp-aapl-future-period-2027"].expected_behavior == "insufficient_context"
    assert items["tmp-aapl-out-of-corpus-period-2019"].expected_behavior == "insufficient_context"


def test_latest_quarter_is_latest_10q(items):
    item = items["tmp-aapl-latest-quarter-revenue"]

    assert item.expected_numbers[0].value == 120e9
    assert item.key_facts == ("the first quarter of fiscal 2026",)


def test_false_premise_only_when_premise_is_false(items):
    item = items["adv-aapl-false-premise-doubled-2025"]

    assert item.category == "adversarial"
    assert [n.value for n in item.expected_numbers] == [390e9, 400e9]


def test_generation_is_deterministic(manifest, facts_by_ticker):
    assert generate_all(manifest, facts_by_ticker) == generate_all(manifest, facts_by_ticker)


def test_missing_companyfacts_is_a_warning_not_a_crash(manifest, aapl_facts):
    items, warnings = generate_all(manifest, {"AAPL": aapl_facts})

    assert any("MSFT" in w for w in warnings)
    assert all("msft" not in i.id for i in items)
