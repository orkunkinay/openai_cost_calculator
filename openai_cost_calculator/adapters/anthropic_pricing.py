from __future__ import annotations

from openai_cost_calculator.pricing import add_pricing_entries


# Prices are USD per 1M tokens. Prompt cache writes are intentionally not
# represented: the core calculator has uncached input, cached input/read, and
# output buckets. Prefer Claude transcript-supplied per-message costs when
# available; these rows are only a fallback for usage-only transcripts.
_ANTHROPIC_PRICES = [
    ("claude-fable-5", "2026-06-20", 10.0, 50.0, 1.0),
    ("claude-mythos-5", "2026-06-20", 10.0, 50.0, 1.0),
    ("claude-opus-4-8", "2026-06-20", 5.0, 25.0, 0.5),
    ("claude-opus-4-7", "2026-06-20", 5.0, 25.0, 0.5),
    ("claude-opus-4-6", "2026-06-20", 5.0, 25.0, 0.5),
    ("claude-opus-4-5", "2026-06-20", 5.0, 25.0, 0.5),
    ("claude-opus-4-1", "2026-06-20", 15.0, 75.0, 1.5),
    ("claude-opus-4", "2026-06-20", 15.0, 75.0, 1.5),
    ("claude-sonnet-5", "2026-06-20", 3.0, 15.0, 0.3),
    ("claude-sonnet-4-6", "2026-06-20", 3.0, 15.0, 0.3),
    ("claude-sonnet-4-5", "2026-06-20", 3.0, 15.0, 0.3),
    ("claude-sonnet-4", "2026-06-20", 3.0, 15.0, 0.3),
    ("claude-haiku-4-5", "2026-06-20", 1.0, 5.0, 0.1),
]


def seed_anthropic_pricing() -> None:
    add_pricing_entries(_ANTHROPIC_PRICES, replace=True)

