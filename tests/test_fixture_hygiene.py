"""Fixtures are copies of public pages; make sure none carries a credential."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Patterns GitHub secret scanning also flags; pages embed site keys like these.
SECRET_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "OpenAI key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
}


@pytest.mark.parametrize("path", sorted(p for p in FIXTURES.rglob("*") if p.is_file()), ids=lambda p: p.name)
def test_fixture_contains_no_credentials(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    assert not found, f"{path.name} contains what looks like a {', '.join(found)}; strip it before committing"
