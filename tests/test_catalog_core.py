"""Unit tests for the provider-agnostic pricing domain."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from openai_cost_calculator.catalog import (
    AmbiguousModelError,
    CatalogValidationError,
    MissingRateError,
    ModelIndex,
    ModelPricing,
    PriceSet,
    PricingUnavailableError,
    ProviderPricing,
    RequestConditions,
    Source,
    UnknownModelError,
    Usage,
    UsageError,
    price_usage,
    select_price_set,
)
from openai_cost_calculator.catalog.costing import USAGE_FIELDS, Cost
from openai_cost_calculator.catalog.dimensions import DIMENSIONS
from openai_cost_calculator.catalog.io import dump_provider, format_decimal, provider_from_dict
from openai_cost_calculator.catalog.validation import sanity_warnings, validate_provider

D = Decimal
SOURCE = Source(id="docs", kind="official_docs", url="https://example.com/pricing")


def _set(conditions=None, min_input_tokens=0, start=None, end=None, **rates):
    return PriceSet(
        rates={k: D(v) for k, v in rates.items()},
        conditions=conditions or {},
        min_input_tokens=min_input_tokens,
        effective_from=start,
        effective_until=end,
    )


def _model(model_id="m", *prices, **kw):
    return ModelPricing(id=model_id, prices=tuple(prices) or (_set(input="1", output="2"),), source="docs", **kw)


def _provider(*models):
    return ProviderPricing(provider="acme", sources=(SOURCE,), models=tuple(models))


def _request(explicit=(), **prefs):
    preferences = {"service_tier": ("standard",), "region": ("global",)}
    preferences.update({k: tuple(v) if isinstance(v, (list, tuple)) else (v,) for k, v in prefs.items()})
    return RequestConditions(preferences=preferences, explicit=frozenset(explicit))


# --------------------------------------------------------------------------- dimensions / usage


def test_every_dimension_has_exactly_one_usage_field():
    assert set(USAGE_FIELDS) == set(DIMENSIONS)
    assert len(set(USAGE_FIELDS.values())) == len(USAGE_FIELDS)
    assert set(USAGE_FIELDS.values()) == set(Usage.__dataclass_fields__)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_usage_rejects_non_integer_or_negative_counts(bad):
    with pytest.raises(UsageError):
        Usage(input_tokens=bad)


def test_usage_from_totals_splits_cached_and_written_tokens():
    usage = Usage.from_totals(total_input_tokens=1000, cached_input_tokens=300, cache_write_tokens=200, output_tokens=5)
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (500, 300, 200)
    assert usage.total_input_tokens == 1000
    with pytest.raises(UsageError, match="cannot exceed"):
        Usage.from_totals(total_input_tokens=10, cached_input_tokens=11, output_tokens=0)


def test_usage_addition_is_fieldwise():
    total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, web_search_calls=1)
    assert total == Usage(input_tokens=4, output_tokens=2, web_search_calls=1)


# --------------------------------------------------------------------------- costing


def test_price_usage_is_exact_and_itemized():
    price_set = _set(input="3", cached_input="0.3", cache_write="3.75", output="15", web_search="0.01")
    usage = Usage(
        input_tokens=1_000, cached_input_tokens=2_000, cache_write_tokens=400, output_tokens=500, web_search_calls=2
    )
    total, items = price_usage(usage, price_set, where="t")
    expected = D("0.003") + D("0.0006") + D("0.0015") + D("0.0075") + D("0.02")
    assert total == expected
    assert {i.dimension: i.cost for i in items}["web_search"] == D("0.02")
    assert all(i.rate_dimension == i.dimension for i in items)


def test_fallbacks_charge_cache_reads_and_writes_at_input_rate():
    total, items = price_usage(
        Usage(cached_input_tokens=1_000_000, cache_write_tokens=1_000_000), _set(input="2", output="4"), where="t"
    )
    assert total == D("4")
    assert [i.rate_dimension for i in items] == ["input", "input"]


def test_reasoning_is_charged_at_output_rate_unless_priced_separately():
    usage = Usage(output_tokens=1_000_000, reasoning_tokens=1_000_000)
    total, items = price_usage(usage, _set(input="1", output="4"), where="t")
    assert total == D("8") and items[1].rate_dimension == "output"
    total, _ = price_usage(usage, _set(input="1", output="4", reasoning="2"), where="t")
    assert total == D("6")


def test_missing_rate_without_safe_fallback_is_an_explicit_error():
    with pytest.raises(MissingRateError, match="input_audio"):
        price_usage(Usage(input_audio_tokens=10), _set(input="1", output="1"), where="acme/m")
    with pytest.raises(MissingRateError, match="cache_write_1h"):
        price_usage(Usage(cache_write_1h_tokens=10), _set(input="1", output="1", cache_write="2"), where="acme/m")


def test_cost_projects_onto_legacy_breakdown():
    price_set = _set(input="1", cached_input="0.5", cache_write="2", output="4")
    usage = Usage(
        input_tokens=1_000_000, cached_input_tokens=1_000_000, cache_write_tokens=1_000_000, output_tokens=1_000_000
    )
    total, items = price_usage(usage, price_set, where="t")
    breakdown = Cost(provider="acme", model="m", resolved_model="m", total=total, items=items).to_breakdown()
    assert breakdown.prompt_cost_uncached == D("3")
    assert breakdown.prompt_cost_cached == D("0.5")
    assert breakdown.completion_cost == D("4")
    assert breakdown.total_cost == D("7.5")


# --------------------------------------------------------------------------- selection


def test_long_context_tier_threshold_is_inclusive():
    model = _model("m", _set(input="1", output="2"), _set(min_input_tokens=200_001, input="2", output="4"))

    def pick(n):
        return select_price_set(model, _request(), total_input_tokens=n, on=date(2026, 1, 1), provider="acme").price_set

    assert pick(200_000).rates["input"] == D("1")
    assert pick(200_001).rates["input"] == D("2")


def test_service_tier_and_region_preferences_choose_matching_group():
    model = _model(
        "m",
        _set(input="1", output="2"),
        _set({"service_tier": "batch"}, input="0.5", output="1"),
        _set({"region": "us-east-1"}, input="1.1", output="2.2"),
    )
    on = date(2026, 1, 1)
    batch = select_price_set(
        model, _request(explicit={"service_tier"}, service_tier="batch"), total_input_tokens=0, on=on, provider="acme"
    )
    assert batch.price_set.rates["input"] == D("0.5")
    regional = select_price_set(model, _request(region="us-east-1"), total_input_tokens=0, on=on, provider="acme")
    assert regional.price_set.rates["input"] == D("1.1")


def test_region_fallback_order_and_assumption_is_reported():
    model = _model(
        "m", _set({"region": "global"}, input="1", output="2"), _set({"region": "us-east-1"}, input="1.1", output="2.2")
    )
    selection = select_price_set(
        model, _request(region=("global", "us-east-1")), total_input_tokens=0, on=date(2026, 1, 1), provider="acme"
    )
    assert selection.conditions == {"service_tier": "standard", "region": "global"}
    assert any("region=global" in a for a in selection.assumptions)

    only_regional = _model("m", _set({"region": "us-east-1"}, input="1.1", output="2.2"))
    fallback = select_price_set(
        only_regional,
        _request(region=("global", "us-east-1")),
        total_input_tokens=0,
        on=date(2026, 1, 1),
        provider="acme",
    )
    assert fallback.conditions == {"service_tier": "standard", "region": "us-east-1"}


def test_unavailable_conditions_list_the_alternatives():
    model = _model(
        "m", _set({"region": "global"}, input="1", output="2"), _set({"region": "us-east-1"}, input="1.1", output="2.2")
    )
    with pytest.raises(PricingUnavailableError, match="available: region=global; region=us-east-1"):
        select_price_set(
            model,
            _request(explicit={"region"}, region="eu-west-1"),
            total_input_tokens=0,
            on=date(2026, 1, 1),
            provider="acme",
        )
    with pytest.raises(PricingUnavailableError, match="service_tier=flex"):
        select_price_set(
            model,
            _request(explicit={"service_tier"}, service_tier="flex"),
            total_input_tokens=0,
            on=date(2026, 1, 1),
            provider="acme",
        )


def test_effective_dates_select_scheduled_price_changes():
    model = _model(
        "m",
        _set(end=date(2027, 1, 1), input="0.75", output="3.75"),
        _set(start=date(2027, 1, 1), input="1.5", output="7.5"),
    )

    def pick(on):
        return select_price_set(model, _request(), total_input_tokens=0, on=on, provider="acme").price_set.rates[
            "input"
        ]

    assert pick(date(2026, 12, 31)) == D("0.75")
    assert pick(date(2027, 1, 1)) == D("1.5")
    expired = _model("m", _set(end=date(2020, 1, 1), input="1", output="1"))
    with pytest.raises(PricingUnavailableError, match="no price in effect"):
        select_price_set(expired, _request(), total_input_tokens=0, on=date(2026, 1, 1), provider="acme")


def test_explicit_conditions_are_not_reported_as_assumptions():
    model = _model("m", _set(input="1", output="2"), _set({"service_tier": "batch"}, input="0.5", output="1"))
    assumed = select_price_set(model, _request(), total_input_tokens=0, on=date(2026, 1, 1), provider="acme")
    assert assumed.conditions == {"service_tier": "standard"}
    assert assumed.assumptions == ()  # the standard tier is not worth reporting
    explicit = select_price_set(
        model,
        _request(explicit={"service_tier"}, service_tier="batch"),
        total_input_tokens=0,
        on=date(2026, 1, 1),
        provider="acme",
    )
    assert explicit.assumptions == ()


# --------------------------------------------------------------------------- validation / io


def test_validation_rejects_structural_errors():
    bad_models = [
        _model("m", _set(input="-1", output="1")),
        _model("m", _set(bogus="1")),
        _model("m", _set({"colour": "red"}, input="1")),
        _model("m", _set(min_input_tokens=10, input="1")),  # no base tier
        _model("m", _set(input="1"), _set(input="2")),  # overlapping duplicates
        _model("m", _set(start=date(2027, 1, 1), end=date(2026, 1, 1), input="1")),
    ]
    for model in bad_models:
        with pytest.raises(CatalogValidationError):
            validate_provider(_provider(model))
    with pytest.raises(CatalogValidationError, match="used by both"):
        validate_provider(_provider(_model("a", aliases=("x",)), _model("b", aliases=("X",))))
    with pytest.raises(CatalogValidationError, match="unknown source"):
        validate_provider(ProviderPricing(provider="acme", sources=(), models=(_model(),)))


def test_non_overlapping_date_windows_are_valid():
    validate_provider(
        _provider(_model("m", _set(end=date(2027, 1, 1), input="1"), _set(start=date(2027, 1, 1), input="2")))
    )


def test_sanity_warnings_flag_implausible_prices():
    warnings = sanity_warnings(_model("m", _set(input="1", cached_input="2", output="5000")), provider="acme")
    assert any("implausibly high" in w for w in warnings)
    assert any("exceeds input" in w for w in warnings)


def test_serialization_is_deterministic_and_round_trips():
    data = _provider(
        _model(
            "zeta",
            _set({"service_tier": "batch"}, output="1", input="0.5"),
            _set(output="2.50", input="1.0"),
            aliases=("z2", "z1"),
        ),
        _model("alpha", _set(input="3", output="15"), canonical_id="acme/alpha", vendor="acme"),
    )
    text = dump_provider(data)
    reordered = _provider(*reversed(data.models))
    assert dump_provider(reordered) == text
    raw = json.loads(text)
    assert [m["id"] for m in raw["models"]] == ["alpha", "zeta"]
    assert raw["models"][1]["aliases"] == ["z1", "z2"]
    assert raw["models"][1]["prices"][0]["rates"] == {"input": "1", "output": "2.5"}
    assert list(raw["models"][1]["prices"][0]["rates"]) == ["input", "output"]
    assert dump_provider(provider_from_dict(raw)) == text


def test_loader_rejects_floats_and_wrong_schema_version():
    raw = json.loads(dump_provider(_provider(_model())))
    raw["models"][0]["prices"][0]["rates"]["input"] = 0.1
    with pytest.raises(CatalogValidationError, match="decimal string"):
        provider_from_dict(raw)
    raw["schema_version"] = 99
    with pytest.raises(CatalogValidationError, match="schema_version"):
        provider_from_dict(raw)


@pytest.mark.parametrize(
    "value,text", [("2.50", "2.5"), ("10", "10"), ("1E+1", "10"), ("0.000", "0"), ("0.0000001", "0.0000001")]
)
def test_format_decimal_is_plain_and_minimal(value, text):
    assert format_decimal(D(value)) == text


# --------------------------------------------------------------------------- lookup


def test_model_index_lookup_order_and_suggestions():
    index = ModelIndex(
        _provider(
            _model(
                "claude-sonnet-4-5", aliases=("claude-sonnet-4-5-20250929",), canonical_id="anthropic/claude-sonnet-4-5"
            ),
            _model("claude-haiku-4-5", canonical_id="anthropic/claude-haiku-4-5"),
        )
    )
    assert index.find(["CLAUDE-SONNET-4-5-20250929"]).id == "claude-sonnet-4-5"
    assert index.find(["anthropic/claude-haiku-4-5"]).id == "claude-haiku-4-5"
    assert index.find(["anthropic/claude-sonnet-4.5"]).id == "claude-sonnet-4-5"
    assert index.find(["claude-sonnet-4.5"]).id == "claude-sonnet-4-5"
    with pytest.raises(UnknownModelError, match="did you mean: claude-sonnet-4-5"):
        index.resolve("claude-sonet-4-5", ["claude-sonet-4-5"])


def test_model_index_refuses_to_guess_between_models():
    index = ModelIndex(_provider(_model("gpt-4-1", canonical_id="x/a"), _model("gpt-41", canonical_id="x/b")))
    with pytest.raises(AmbiguousModelError):
        index.find(["gpt4.1"])
