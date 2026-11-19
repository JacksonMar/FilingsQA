import pytest

from filingsqa.evaluation.answer_numbers import extract_numbers


@pytest.mark.parametrize(
    ("text", "value", "scaled", "currency"),
    [
        ("Revenue was $130.5 billion.", 130.5e9, True, True),
        ("Revenue was $130,497 million.", 130_497e6, True, True),
        ("It reached $1.2B", 1.2e9, True, True),
        ("about 2.1 trillion", 2.1e12, True, False),
        ("Revenue | $130,497", 130_497, False, True),
        ("diluted EPS of $2.94", 2.94, False, True),
    ],
)
def test_values_and_scales(text, value, scaled, currency):
    n = extract_numbers(text)[0]
    assert n.value == pytest.approx(value)
    assert (n.scaled, n.currency) == (scaled, currency)


def test_parenthesized_negative():
    assert extract_numbers("Other income (expense) was (1,234) million")[0].negative


def test_percent():
    n = extract_numbers("Revenue grew 114%.")[0]
    assert n.percent
    assert n.value == 114


def test_single_letter_scale_needs_currency():
    """'5 m' in prose is not five million; '$5M' is."""
    assert extract_numbers("a 5 m wide road")[0].value == 5
    assert extract_numbers("$5M")[0].value == 5e6


def test_multiple_numbers_in_order():
    values = [
        n.value
        for n in extract_numbers(
            "From $60.9 billion in fiscal 2024 to $130.5 billion in fiscal 2025"
        )
    ]
    assert values == pytest.approx([60.9e9, 2024, 130.5e9, 2025])
