"""Quality gates: minimum mean score per (scope, metric), loaded from eval/thresholds.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Status = Literal["pass", "fail", "skipped"]


@dataclass(frozen=True)
class Threshold:
    scope: str  # "overall" or a category name
    metric: str
    minimum: float


@dataclass(frozen=True)
class ThresholdResult:
    threshold: Threshold
    actual: float | None
    n: int
    status: Status


def load_thresholds(path: Path, profile: str) -> list[Threshold]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile not in data:
        raise ValueError(f"profile {profile!r} not in {path}; have {sorted(data)}")
    return [
        Threshold(scope, metric, float(minimum))
        for scope, metrics in data[profile].items()
        for metric, minimum in metrics.items()
    ]


def check_thresholds(
    thresholds: list[Threshold], summary: dict[str, dict[str, dict[str, float]]]
) -> list[ThresholdResult]:
    """summary[scope][metric] = {"mean": ..., "n": ...}. No data for a metric -> skipped."""
    results = []
    for t in thresholds:
        stats = summary.get(t.scope, {}).get(t.metric)
        if not stats or not stats["n"]:
            results.append(ThresholdResult(t, None, 0, "skipped"))
            continue
        ok = stats["mean"] >= t.minimum - 1e-9
        results.append(ThresholdResult(t, stats["mean"], int(stats["n"]), "pass" if ok else "fail"))
    return results
