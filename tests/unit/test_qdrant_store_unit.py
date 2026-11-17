from qdrant_client import models

from filingsqa.index.qdrant_store import SearchFilter, chunk_payload, point_id
from tests.fakes import make_chunk


def test_point_id_is_stable_uuid():
    assert point_id("AAPL-x-item_7-000") == point_id("AAPL-x-item_7-000")
    assert point_id("AAPL-x-item_7-000") != point_id("AAPL-x-item_7-001")


def test_empty_filter_means_no_filter():
    assert SearchFilter().to_qdrant() is None


def test_filter_maps_each_field_to_match_any():
    flt = SearchFilter(tickers=("AAPL", "MSFT"), fiscal_years=(2025,)).to_qdrant()

    assert flt is not None
    assert flt.must == [
        models.FieldCondition(key="ticker", match=models.MatchAny(any=["AAPL", "MSFT"])),
        models.FieldCondition(key="fiscal_year", match=models.MatchAny(any=[2025])),
    ]


def test_payload_has_everything_needed_for_citations_and_filters():
    payload = chunk_payload(make_chunk("NVDA", "Revenue was $130,497 million."))

    for key in ("chunk_id", "text", "ticker", "fiscal_year", "fiscal_period", "period_label",
                "section_key", "section_title", "source_url", "accession"):  # fmt: skip
        assert key in payload
    assert payload["period_label"] == "FY2025"
