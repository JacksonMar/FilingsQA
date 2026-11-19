from filingsqa.evaluation.dataset import ExpectedSource
from filingsqa.evaluation.snapshot import build_snapshot
from filingsqa.ingest.pipeline import load_chunks, save_chunks
from tests.fakes import make_chunk
from tests.unit.evaluation.helpers import item

NVDA = [make_chunk("NVDA", f"risk {i}", section_key="item_1a", index=i) for i in range(10)]
NVDA += [make_chunk("NVDA", f"Revenue | {i}", section_key="item_8", index=i) for i in range(3)]
AAPL = [make_chunk("AAPL", f"apple {i}", section_key="item_8", index=i) for i in range(5)]
ACC = NVDA[0].filing.accession


def _item():
    return item(
        expected_sources=(
            ExpectedSource(ticker="NVDA", form="10-K", section_keys=("item_8",), accession=ACC),
        )
    )


def test_includes_expected_sections_and_capped_distractors():
    snap = build_snapshot([_item()], NVDA + AAPL, distractors_per_filing=4)

    sections = [c.section_key for c in snap]
    assert sections.count("item_8") == 3  # every expected chunk
    assert sections.count("item_1a") == 4  # distractors from the same filing, capped
    assert all(c.filing.ticker == "NVDA" for c in snap)  # other filings not included


def test_snapshot_is_deterministic_and_in_corpus_order():
    a = build_snapshot([_item()], NVDA + AAPL, 4)
    b = build_snapshot([_item()], list(reversed(NVDA + AAPL)), 4)

    assert {c.chunk_id for c in a} == {c.chunk_id for c in b}
    order = [c.chunk_id for c in NVDA]
    assert [c.chunk_id for c in a] == sorted((c.chunk_id for c in a), key=order.index)


def test_gzip_roundtrip_is_byte_stable(tmp_path):
    p1, p2 = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"

    save_chunks(NVDA, p1)
    save_chunks(NVDA, p2)

    assert p1.read_bytes() == p2.read_bytes()  # no timestamp in the gzip header
    assert load_chunks(p1) == NVDA


def test_plain_jsonl_still_supported(tmp_path):
    path = tmp_path / "c.jsonl"
    save_chunks(AAPL, path)

    assert load_chunks(path) == AAPL
