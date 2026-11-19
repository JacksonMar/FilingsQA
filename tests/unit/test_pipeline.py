from filingsqa.edgar.manifest import Manifest
from filingsqa.ingest.pipeline import build_chunks, load_chunks, process_filing


def test_process_10k_fixture_end_to_end(ref_10k, sample_10k_html):
    chunks, report = process_filing(ref_10k, sample_10k_html)

    assert report.period == "FY2025"
    assert set(report.sections) == {"front_matter", "item_1", "item_1a", "item_7", "item_8"}
    assert report.warnings == []
    table = next(c for c in chunks if c.content_type == "table" and c.section_key == "item_8")
    assert "(In millions" in table.text
    assert "Net sales | $410,235 | $391,035" in table.text
    assert all(c.filing.accession == ref_10k.accession for c in chunks)


def test_process_10q_fixture_end_to_end(ref_10q, sample_10q_html):
    chunks, report = process_filing(ref_10q, sample_10q_html)

    assert report.period == "Q1 FY2026"
    risk = [c for c in chunks if c.section_key == "part2_item_1a"]
    assert risk
    assert "no material changes" in risk[0].text


def test_build_chunks_writes_jsonl_and_report(
    tmp_path, ref_10k, ref_10q, sample_10k_html, sample_10q_html
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / ref_10k.raw_filename).write_text(sample_10k_html)
    (raw / ref_10q.raw_filename).write_text(sample_10q_html)
    manifest = Manifest(as_of=ref_10q.filing_date, filings=[ref_10k, ref_10q])
    out = tmp_path / "processed" / "chunks.jsonl"

    reports = build_chunks(manifest, raw, out)

    loaded = load_chunks(out)
    assert len(loaded) == sum(r.n_chunks for r in reports)
    assert len({c.chunk_id for c in loaded}) == len(loaded)
    assert (out.parent / "ingest_report.json").exists()
