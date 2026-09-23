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


def test_cache_prices_for_opus_4_8():
    rate = resolve_anthropic_rate("claude-opus-4-8", 0)
    assert rate.input == Decimal("5")
    assert rate.cache_read == Decimal("0.5")
    assert rate.cache_write_5m == Decimal("6.25")  # 1.25x
    assert rate.cache_write_1h == Decimal("10")  # 2x


def test_first_party_prices_are_flat_across_the_context_window():
    # platform.claude.com pricing (2026-09-23): Claude 4.6+ bill the full 1M
    # window at standard rates; Sonnet 4.5 has a 200k window.
    assert resolve_anthropic_rate("claude-sonnet-4-6", 900_000).input == Decimal("3")
    assert resolve_anthropic_rate("claude-sonnet-4-5", 150_000).input == Decimal("3")


def test_long_context_tier_selection_around_threshold():
    from openai_cost_calculator.catalog import ModelPricing, PriceSet, PricingCatalog, ProviderPricing, Source

    def rates(value):
        return {"input": Decimal(value), "output": Decimal(value)}

    catalog = PricingCatalog([
        ProviderPricing(
            provider="anthropic",
            sources=(Source(id="s", kind="manual", url="https://example.com"),),
            models=(ModelPricing(id="claude-x", source="s", prices=(
                PriceSet(rates=rates("3")),
                PriceSet(rates=rates("6"), min_input_tokens=200_001),
            )),),
        )
    ])
    assert resolve_anthropic_rate("claude-x", 200_000, catalog=catalog).input == Decimal("3")
    assert resolve_anthropic_rate("claude-x", 200_001, catalog=catalog).input == Decimal("6")
    assert resolve_anthropic_rate("claude-x-20260101", 500_000, catalog=catalog).min_input_tokens == 200_001


def test_unknown_model_and_negative_tokens_raise():
    with pytest.raises(AnthropicPricingError):
        resolve_anthropic_rate("gpt-4o", 10)
    with pytest.raises(AnthropicPricingError):
        resolve_anthropic_rate("claude-opus-4-8", -1)


def test_dated_model_id_resolves_at_current_rate():
    # A model id's release-date suffix must not gate pricing: a dated id is
    # billed at the current rate for that model (regression for a live bug where
    # claude-sonnet-5-<date> failed to price).
    dated = resolve_anthropic_rate("claude-sonnet-4-5-20250929", 10)
    undated = resolve_anthropic_rate("claude-sonnet-4-5", 10)
    assert dated.input == undated.input == Decimal("3")


def test_sonnet_5_is_priced_at_the_current_published_rate():
    # Anthropic made the $2/$10 launch price permanent (pricing docs, 2026-09-23).
    rate = resolve_anthropic_rate("claude-sonnet-5", 10)
    assert rate.input == Decimal("2")
    assert rate.output == Decimal("10")


def test_cache_read_multiplier_is_published_per_model():
    # Opus 5.5 cache reads are 0.05x input; Fable 5.1 0.025x - not the 0.1x default.
    assert resolve_anthropic_rate("claude-opus-5-5", 10).cache_read == Decimal("0.2")
    assert resolve_anthropic_rate("claude-fable-5-1", 10).cache_read == Decimal("0.25")
