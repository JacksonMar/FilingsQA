"""Runs in the default unit suite: the oracle must agree with us on our own fixtures."""

import pytest

from tests.differential.oracle import compare


@pytest.mark.parametrize(
    ("fixture", "form"), [("sample_10k_html", "10-K"), ("sample_10q_html", "10-Q")]
)
def test_oracle_agrees_on_fixtures(fixture, form, request):
    result = compare(request.getfixturevalue(fixture), form)

    assert result.agrees, result
