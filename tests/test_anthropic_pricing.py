from __future__ import annotations

from decimal import Decimal

import pytest

from openai_cost_calculator.anthropic.pricing import (
    AnthropicPricingError,
    resolve_anthropic_rate,
    split_anthropic_model,
    validate_anthropic_pricing,
)


def test_pricing_table_validates_and_reports_tier_count():
    assert validate_anthropic_pricing() >= 12


def test_split_model_handles_alias_and_dated_ids():
    name, _date = split_anthropic_model("claude-opus-4-8")
    assert name == "claude-opus-4-8"
    assert split_anthropic_model("claude-sonnet-4-5-20250929") == (
        "claude-sonnet-4-5",
        "2025-09-29",
    )
    # A non-date numeric suffix is treated as part of the model name.
    name, _date = split_anthropic_model("claude-haiku-4-5")
    assert name == "claude-haiku-4-5"


def test_cache_multipliers_are_derived_from_base_input():
    rate = resolve_anthropic_rate("claude-opus-4-8", 0)
    assert rate.input == Decimal("5")
    assert rate.cache_read == Decimal("0.5")
    assert rate.cache_write_5m == Decimal("6.25")  # 1.25x
    assert rate.cache_write_1h == Decimal("10")  # 2x


def test_long_context_tier_selection_around_threshold():
    below = resolve_anthropic_rate("claude-sonnet-4-5", 200_000)
    at = resolve_anthropic_rate("claude-sonnet-4-5", 200_001)
    above = resolve_anthropic_rate("claude-sonnet-4-5", 500_000)
    assert below.input == Decimal("3")
    assert at.input == Decimal("6")
    assert above.input == Decimal("6")


def test_unknown_model_and_negative_tokens_raise():
    with pytest.raises(AnthropicPricingError):
        resolve_anthropic_rate("gpt-4o", 10)
    with pytest.raises(AnthropicPricingError):
        resolve_anthropic_rate("claude-opus-4-8", -1)


def test_dated_request_before_effective_date_has_no_pricing():
    with pytest.raises(AnthropicPricingError):
        resolve_anthropic_rate("claude-opus-4-8-20200101", 10)
