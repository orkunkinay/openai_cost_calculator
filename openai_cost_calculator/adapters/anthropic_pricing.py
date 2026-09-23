"""Seed Claude prices into the legacy three-bucket pricing store.

Used by the Claude Code transcript adapter when a transcript carries token
usage but no per-message cost.  Prices come from the provider catalog (the
same data as :func:`openai_cost_calculator.calculate_cost`), so there is no
second hand-maintained table to drift.  Cache writes have no bucket in the
legacy schema; prefer transcript-supplied costs when available.
"""

from __future__ import annotations

from openai_cost_calculator.legacy import anthropic_legacy_entries
from openai_cost_calculator.pricing import add_pricing_entries


def seed_anthropic_pricing() -> None:
    add_pricing_entries(anthropic_legacy_entries(), replace=True)
