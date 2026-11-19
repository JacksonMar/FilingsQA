from pathlib import Path

import pytest

from filingsqa.evaluation.thresholds import Threshold, check_thresholds, load_thresholds

REPO_THRESHOLDS = Path(__file__).resolve().parents[3] / "eval/thresholds.yaml"


@pytest.mark.parametrize("profile", ["full", "smoke"])
def test_committed_thresholds_load(profile):
    thresholds = load_thresholds(REPO_THRESHOLDS, profile)

    assert thresholds
    assert all(0 < t.minimum <= 1 for t in thresholds)


def test_smoke_is_never_stricter_than_full():
    full = {(t.scope, t.metric): t.minimum for t in load_thresholds(REPO_THRESHOLDS, "full")}
    smoke = load_thresholds(REPO_THRESHOLDS, "smoke")

    for t in smoke:
        assert t.minimum <= full.get((t.scope, t.metric), 1.0), t


def test_unknown_profile(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("full: {}\n")

    with pytest.raises(ValueError, match="nightly"):
        load_thresholds(path, "nightly")


def test_check_pass_fail_skip():
    summary = {"overall": {"a": {"mean": 0.9, "n": 10}, "b": {"mean": 0.5, "n": 10}}}
    ts = [
        Threshold("overall", "a", 0.8),
        Threshold("overall", "b", 0.8),
        Threshold("numeric", "c", 0.8),
    ]

    assert [r.status for r in check_thresholds(ts, summary)] == ["pass", "fail", "skipped"]
