from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """One metric on one item. score in [0, 1]; passed is the per-item verdict."""

    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def binary(cls, name: str, ok: bool, **details: Any) -> "MetricResult":
        return cls(name, 1.0 if ok else 0.0, ok, details)
