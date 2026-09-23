"""Tests for provider specs and the provider-agnostic public API."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from openai_cost_calculator.api import calculate_cost, compare_costs, get_model_pricing, list_models
from openai_cost_calculator.catalog import (
    MissingRateError,
    ModelPricing,
    PriceSet,
    PricingCatalog,
    PricingError,
    PricingUnavailableError,
    ProviderPricing,
    Source,
    UnknownModelError,
    UnknownProviderError,
    Usage,
)
from openai_cost_calculator.providers import generic_candidates, get_provider
from openai_cost_calculator.providers.registry import deepseek_period, parse_bedrock_model

D = Decimal


def _ps(conditions=None, min_input_tokens=0, **rates):
    return PriceSet(rates={k: D(v) for k, v in rates.items()}, conditions=conditions or {}, min_input_tokens=min_input_tokens)


def _provider(pid, *models):
    return ProviderPricing(
        provider=pid,
        sources=(Source(id="src", kind="official_api", url=f"https://{pid}.example/pricing"),),
        models=tuple(models),
        verified_at=date(2026, 9, 1),
    )


def _m(model_id, *prices, **kw):
    return ModelPricing(id=model_id, prices=prices, source="src", **kw)


CATALOG = PricingCatalog(
    [
        _provider(
            "bedrock",
            _m(
                "anthropic.claude-sonnet-4-5",
                _ps({"region": "global"}, input="3", output="15"),
                _ps({"region": "us-east-1"}, input="3.3", output="16.5"),
                _ps({"region": "us-east-1", "service_tier": "batch"}, input="1.65", output="8.25"),
                canonical_id="anthropic/claude-sonnet-4-5",
                vendor="anthropic",
            ),
            _m("meta.llama3-3-70b-instruct", _ps({"region": "us-east-1"}, input="0.72", output="0.72"), canonical_id="meta/llama-3.3-70b-instruct"),
        ),
        _provider(
            "anthropic",
            _m(
                "claude-sonnet-4-5",
                _ps(input="3", cache_write="3.75", cache_write_1h="6", cached_input="0.3", output="15"),
                _ps({"region": "us"}, input="3.3", cache_write="4.125", cache_write_1h="6.6", cached_input="0.33", output="16.5"),
                aliases=("claude-sonnet-4-5-20250929",),
                canonical_id="anthropic/claude-sonnet-4-5",
            ),
        ),
        _provider(
            "deepseek",
            _m(
                "deepseek-flash",
                _ps({"period": "peak"}, input="0.3", cached_input="0.006", output="1.2"),
                _ps({"period": "off_peak"}, input="0.15", cached_input="0.003", output="0.6"),
                aliases=("deepseek-v4-flash",),
            ),
        ),
        _provider(
            "openai",
            _m(
                "gpt-5.5",
                _ps(input="5", cached_input="0.5", output="30"),
                _ps(min_input_tokens=272_001, input="10", cached_input="1", output="45"),
                _ps({"service_tier": "batch"}, input="2.5", cached_input="0.25", output="15"),
            ),
            _m("gpt-4o", _ps(input="2.5", cached_input="1.25", output="10"), aliases=("gpt-4o-2024-08-06",)),
            _m("gpt-4o-2024-05-13", _ps(input="5", output="15")),
        ),
    ]
)


# --------------------------------------------------------------------------- provider specs


def test_provider_aliases_resolve_case_and_separator_insensitively():
    assert get_provider("AWS_Bedrock").id == "bedrock"
    assert get_provider("google-ai-studio").id == "gemini"
    assert get_provider("vertex-ai").id == "vertex"
    with pytest.raises(UnknownProviderError, match="supported providers: anthropic"):
        get_provider("bedrok")


def test_generic_candidates_strip_prefixes_namespaces_and_snapshots():
    assert generic_candidates("models/gemini-2.0-flash-001", ("models/",)) == (
        "models/gemini-2.0-flash-001",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash",
    )
    assert generic_candidates("claude-sonnet-4-5@20250929")[-1] == "claude-sonnet-4-5"
    assert generic_candidates("gpt-4o-2024-08-06") == ("gpt-4o-2024-08-06", "gpt-4o")


@pytest.mark.parametrize(
    "model,region",
    [
        ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", ("global",)),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ("us-east-1",)),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", ("us-east-1", "global")),
        ("arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0", ("us-west-2",)),
    ],
)
def test_bedrock_model_strings_imply_endpoint_pricing(model, region):
    hints = parse_bedrock_model(model)
    assert hints.preferences["region"] == region
    assert "anthropic.claude-sonnet-4-5" in hints.candidates
    assert "anthropic/claude-sonnet-4-5" in hints.candidates


@pytest.mark.parametrize(
    "moment,period",
    [
        (datetime(2026, 9, 21, 2, 30, tzinfo=timezone.utc), "peak"),  # Monday 02:30
        (datetime(2026, 9, 21, 5, 0, tzinfo=timezone.utc), "off_peak"),  # gap between windows
        (datetime(2026, 9, 21, 9, 59, tzinfo=timezone.utc), "peak"),
        (datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc), "off_peak"),
        (datetime(2026, 9, 26, 7, 0, tzinfo=timezone.utc), "off_peak"),  # Saturday
    ],
)
def test_deepseek_peak_hours(moment, period):
    preferences, notes = deepseek_period(moment)
    assert preferences == {"period": (period,)}
    assert "holidays" in notes["period"]


# --------------------------------------------------------------------------- calculate_cost


def test_bedrock_canonical_model_prefers_global_endpoint():
    cost = calculate_cost("aws-bedrock", "anthropic/claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=100_000, catalog=CATALOG)
    assert cost.resolved_model == "anthropic.claude-sonnet-4-5"
    assert cost.total == D("4.5")
    assert cost.conditions["region"] == "global"
    assert cost.source_url == "https://bedrock.example/pricing"
    assert cost.verified_at == date(2026, 9, 1)


def test_bedrock_geo_profile_uses_regional_rate_and_explains_it():
    cost = calculate_cost("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", input_tokens=1_000_000, catalog=CATALOG)
    assert cost.total == D("3.3")
    assert any("us-east-1 regional rate" in a for a in cost.assumptions)
    batch = calculate_cost("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", input_tokens=1_000_000, service_tier="batch", catalog=CATALOG)
    assert batch.total == D("1.65")


def test_bedrock_model_without_global_price_falls_back_to_default_region():
    cost = calculate_cost("bedrock", "meta/llama-3.3-70b-instruct", input_tokens=1_000_000, catalog=CATALOG)
    assert cost.conditions["region"] == "us-east-1"
    assert cost.total == D("0.72")


def test_explicit_unavailable_region_is_an_error_listing_alternatives():
    with pytest.raises(PricingUnavailableError, match="region=global; region=us-east-1"):
        calculate_cost("bedrock", "anthropic.claude-sonnet-4-5", input_tokens=1, region="ap-south-1", catalog=CATALOG)


def test_anthropic_cache_dimensions_and_data_residency():
    cost = calculate_cost(
        "anthropic",
        "claude-sonnet-4-5-20250929",
        input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
        output_tokens=1_000_000,
        catalog=CATALOG,
    )
    assert cost.total == D("3") + D("0.3") + D("3.75") + D("6") + D("15")
    assert cost.by_dimension()["cache_write_1h"] == D("6")
    us = calculate_cost("anthropic", "claude-sonnet-4-5", input_tokens=1_000_000, region="us", catalog=CATALOG)
    assert us.total == D("3.3")


def test_deepseek_period_from_request_time_and_conservative_default():
    peak = calculate_cost("deepseek", "deepseek-v4-flash", input_tokens=1_000_000, at=datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc), catalog=CATALOG)
    off = calculate_cost("deepseek", "deepseek-flash", input_tokens=1_000_000, at=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc), catalog=CATALOG)
    assert (peak.total, off.total) == (D("0.3"), D("0.15"))
    assert any("derived from request time" in a for a in off.assumptions)
    unknown_time = calculate_cost("deepseek", "deepseek-flash", input_tokens=1_000_000, at=date(2026, 9, 21), catalog=CATALOG)
    assert unknown_time.total == D("0.3")
    assert any("assumed period=peak" in a for a in unknown_time.assumptions)
    explicit = calculate_cost("deepseek", "deepseek-flash", input_tokens=1_000_000, period="off_peak", at=date(2026, 9, 21), catalog=CATALOG)
    assert explicit.total == D("0.15") and explicit.assumptions == ()


def test_openai_long_context_batch_and_snapshots():
    short = calculate_cost("openai", "gpt-5.5", input_tokens=272_000, catalog=CATALOG)
    long = calculate_cost("openai", "gpt-5.5", input_tokens=272_001, catalog=CATALOG)
    assert short.total == D("272000") * D("5") / D(1_000_000)
    assert long.total == D("272001") * D("10") / D(1_000_000)
    assert long.min_input_tokens == 272_001
    batch = calculate_cost("openai", "gpt-5.5", input_tokens=1_000_000, service_tier="batch", catalog=CATALOG)
    assert batch.total == D("2.5")
    assert calculate_cost("openai", "gpt-4o-2024-05-13", input_tokens=1_000_000, catalog=CATALOG).total == D("5")
    assert calculate_cost("openai", "gpt-4o-2024-11-20", input_tokens=1_000_000, catalog=CATALOG).total == D("2.5")
    assert calculate_cost("openai", "openai/gpt-4o", input_tokens=1_000_000, catalog=CATALOG).resolved_model == "gpt-4o"


def test_cached_tokens_count_toward_long_context_threshold():
    cost = calculate_cost("openai", "gpt-5.5", input_tokens=200_000, cached_input_tokens=72_001, catalog=CATALOG)
    assert cost.min_input_tokens == 272_001


def test_error_messages_are_actionable():
    with pytest.raises(UnknownModelError, match="did you mean: gpt-5.5"):
        calculate_cost("openai", "gpt-5.55", input_tokens=1, catalog=CATALOG)
    with pytest.raises(MissingRateError, match="input_audio"):
        calculate_cost("openai", "gpt-4o", usage=Usage(input_audio_tokens=5), catalog=CATALOG)
    with pytest.raises(PricingError, match="either usage"):
        calculate_cost("openai", "gpt-4o", usage=Usage(), input_tokens=5, catalog=CATALOG)
    with pytest.raises(PricingUnavailableError, match="service_tier=flex"):
        calculate_cost("openai", "gpt-4o", input_tokens=5, service_tier="flex", catalog=CATALOG)
    with pytest.raises(UnknownProviderError):
        calculate_cost("groq", "x", input_tokens=5, catalog=CATALOG)  # provider absent from this catalog


def test_listing_and_cross_provider_comparison():
    assert {m.id for _, m in list_models("anthropic", catalog=CATALOG)} == {"claude-sonnet-4-5"}
    assert get_model_pricing("bedrock", "global.anthropic.claude-sonnet-4-5-v1:0", catalog=CATALOG).vendor == "anthropic"
    costs = compare_costs("anthropic/claude-sonnet-4-5", usage=Usage(input_tokens=1_000_000), catalog=CATALOG)
    assert [(c.provider, c.total) for c in costs] == [("anthropic", D("3")), ("bedrock", D("3"))]
