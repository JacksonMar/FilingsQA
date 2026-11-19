"""Module boundaries from docs/ARCHITECTURE.md, enforced (an architecture test).

The bot must not import the evaluation suite, and the lower layers must not import higher
ones; otherwise the system under test and the test harness slowly merge.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "filingsqa"

# package -> filingsqa packages it may import (besides itself)
ALLOWED = {
    "edgar": {"companies", "config"},
    "ingest": {"companies", "config", "edgar"},
    "index": {"config", "ingest"},
    "llm": {"config"},
    "rag": {"companies", "config", "index", "ingest", "llm"},
    "api": {"config", "llm", "rag"},
}


def _imported_packages(package: str) -> set[str]:
    found = set()
    for path in (SRC / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                parts = name.split(".")
                if parts[0] == "filingsqa" and len(parts) > 1:
                    found.add(parts[1])
    return found - {package}


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_package_imports_only_allowed_layers(package):
    forbidden = _imported_packages(package) - ALLOWED[package]

    assert not forbidden, f"{package} must not import {sorted(forbidden)}"


def test_app_never_imports_the_evaluation_suite():
    for package in ALLOWED:
        assert "evaluation" not in _imported_packages(package)
