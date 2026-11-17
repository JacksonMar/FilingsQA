import pytest

from filingsqa.evaluation.numbers import format_value, number_variants


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (130_497_000_000, "USD", {"130,497", "130,497,000", "130,497,000,000"}),
        (-1_234_000_000, "USD", {"1,234", "1,234,000", "1,234,000,000"}),
        (2.94, "USD/shares", {"2.94"}),
        (123_456_789, "USD", {"123,456,789"}),
    ],
)
def test_number_variants(value, unit, expected):
    assert number_variants(value, unit) == expected


def test_format_value():
    assert format_value(130_497_000_000, "USD") == "$130,497 million"
    assert format_value(-5e8, "USD") == "-$500 million"
    assert format_value(2.944, "USD/shares") == "$2.94"
