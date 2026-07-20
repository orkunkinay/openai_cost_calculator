from __future__ import annotations

from decimal import Decimal

import pytest

from openai_cost_calculator.anthropic.usage import (
    AnthropicUsageError,
    extract_anthropic_usage,
    price_anthropic_usage,
    to_cost_breakdown,
    to_ledger_tokens,
    unpriced_usage_fields,
    usage_from_dict,
)


def test_extract_returns_none_without_usage_object():
    assert extract_anthropic_usage({"model": "claude-opus-4-8"}) is None
    assert extract_anthropic_usage({"usage": "nope"}) is None


def test_input_tokens_are_not_reduced_by_cache_fields():
    usage = usage_from_dict(
        {
            "input_tokens": 1_000,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
            "output_tokens": 300,
        }
    )
    assert usage.input_tokens == 1_000  # uncached input is reported verbatim
    assert usage.total_input_tokens == 1_700
    assert to_ledger_tokens(usage) == {
        "prompt_tokens": 1_200,  # input + cache creation
        "completion_tokens": 300,
        "cached_tokens": 500,  # cache reads reported separately
    }


def test_cache_ttl_breakdown_is_priced_separately():
    usage = usage_from_dict(
        {
            "input_tokens": 0,
            "cache_creation_input_tokens": 300,
            "output_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 200,
            },
        }
    )
    cost = price_anthropic_usage("claude-opus-4-8", usage)
    # 100 @ 6.25/M + 200 @ 10/M
    expected = Decimal(100) * Decimal("6.25") / Decimal(1_000_000) + Decimal(200) * Decimal("10") / Decimal(1_000_000)
    assert cost.cache_write_cost == expected
    assert cost.total_cost == expected


def test_cache_creation_remainder_charged_at_five_minute_rate():
    # Aggregate exceeds the TTL breakdown; the remainder is billed at 5m.
    usage = usage_from_dict(
        {
            "input_tokens": 0,
            "cache_creation_input_tokens": 300,
            "output_tokens": 0,
            "cache_creation": {"ephemeral_1h_input_tokens": 100},
        }
    )
    cost = price_anthropic_usage("claude-opus-4-8", usage)
    # 200 remainder @ 6.25/M (5m) + 100 @ 10/M (1h)
    expected = Decimal(200) * Decimal("6.25") / Decimal(1_000_000) + Decimal(100) * Decimal("10") / Decimal(1_000_000)
    assert cost.cache_write_cost == expected


def test_full_cost_maps_onto_three_bucket_breakdown():
    usage = usage_from_dict(
        {
            "input_tokens": 1_000,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
            "output_tokens": 300,
            "cache_creation": {"ephemeral_5m_input_tokens": 200},
        }
    )
    cost = price_anthropic_usage("claude-opus-4-8", usage)
    breakdown = to_cost_breakdown(cost)
    assert breakdown.total_cost == cost.total_cost
    assert (
        breakdown.prompt_cost_uncached
        + breakdown.prompt_cost_cached
        + breakdown.completion_cost
        == breakdown.total_cost
    )
    assert breakdown.prompt_cost_uncached == cost.input_cost + cost.cache_write_cost
    assert breakdown.prompt_cost_cached == cost.cache_read_cost


def test_zero_token_response_costs_zero():
    usage = usage_from_dict({"input_tokens": 0, "output_tokens": 0})
    assert usage.is_empty
    assert price_anthropic_usage("claude-opus-4-8", usage).total_cost == Decimal("0")


def test_malformed_usage_values_raise():
    with pytest.raises(AnthropicUsageError):
        usage_from_dict({"input_tokens": -1})
    with pytest.raises(AnthropicUsageError):
        usage_from_dict({"output_tokens": 1.5})
    with pytest.raises(AnthropicUsageError):
        usage_from_dict({"input_tokens": True})


def test_unpriced_server_tool_fields_are_surfaced():
    assert unpriced_usage_fields(
        {"input_tokens": 1, "server_tool_use": {"web_search_requests": 3}}
    ) == ["server_tool_use"]
    assert unpriced_usage_fields({"server_tool_use": {"web_search_requests": 0}}) == []
    assert unpriced_usage_fields({"input_tokens": 1}) == []


def test_extremely_large_counts_stay_exact():
    usage = usage_from_dict({"input_tokens": 10**12, "output_tokens": 10**12})
    cost = price_anthropic_usage("claude-opus-4-8", usage)
    assert cost.input_cost == Decimal(10**12) * Decimal("5") / Decimal(1_000_000)
    assert cost.output_cost == Decimal(10**12) * Decimal("25") / Decimal(1_000_000)
