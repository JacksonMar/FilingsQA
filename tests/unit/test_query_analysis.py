import pytest

from filingsqa.rag.query_analysis import detect_tickers


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What was NVIDIA's revenue in fiscal 2025?", ["NVDA"]),
        ("What was nvda revenue?", ["NVDA"]),
        ("How is $TSLA doing?", ["TSLA"]),
        ("Which had higher net income, MSFT or Apple?", ["MSFT", "AAPL"]),
        ("Compare J.P. Morgan and Tesla", ["JPM", "TSLA"]),
        ("JPMorgan Chase net interest income", ["JPM"]),
        ("What was Amazon's revenue?", []),
    ],
)
def test_detects_known_companies(question, expected):
    assert detect_tickers(question) == expected


@pytest.mark.parametrize(
    "question",
    ["Do pineapple prices matter?", "What is an applet?", "TSLAX fund", "teslas in general"],
)
def test_no_false_positives_inside_other_words(question):
    assert detect_tickers(question) == []
