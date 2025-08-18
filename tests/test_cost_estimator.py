from datetime import datetime, timezone
from decimal import Decimal
from typing import List
import pytest

import openai_cost_calculator as occ

from openai_cost_calculator.core import calculate_cost, calculate_cost_typed
from openai_cost_calculator.estimate import estimate_cost, estimate_cost_typed, CostEstimateError
from openai_cost_calculator.parser import extract_model_details, extract_usage
from openai_cost_calculator.types import CostBreakdown
from openai_cost_calculator.telemetry import (
set_global_tags,
get_global_tags,
register_cost_sink,
unregister_cost_sink,
clear_cost_sinks,
)

class _Struct:
    """Tiny helper to build ad-hoc objects with attributes."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _classic_response(prompt_t, completion_t, cached_t, model="gpt-4o-mini-2024-07-18"):
    usage = _Struct(
        prompt_tokens        = prompt_t,
        completion_tokens    = completion_t,
        prompt_tokens_details= _Struct(cached_tokens=cached_t),
    )
    return _Struct(model=model, usage=usage)


def _new_response(input_t, output_t, cached_t, model="gpt-4o-mini-2024-07-18"):
    usage = _Struct(
        input_tokens         = input_t,
        output_tokens        = output_t,
        input_tokens_details = _Struct(cached_tokens=cached_t),
    )
    return _Struct(model=model, usage=usage)


# Static pricing used in every test (USD / 1M tokens)
_PRICING = {("gpt-4o-mini", "2024-07-18"): {
    "input_price"       : 0.50,
    "cached_input_price": 0.25,
    "output_price"      : 1.00,
}}

@pytest.fixture(autouse=True)
def monkeypatch_pricing(monkeypatch):
    """Force `load_pricing()` to return our static dict."""
    monkeypatch.setattr(occ.pricing, "load_pricing", lambda: _PRICING)


# --------------------------------------------------------------------------- #
# Unit tests                                                                  #
# --------------------------------------------------------------------------- #
def test_calculate_cost_basic_rounding():
    usage  = {"prompt_tokens": 1_000, "completion_tokens": 2_000, "cached_tokens": 200}
    rates  = {"input_price": 1.0, "cached_input_price": 0.5, "output_price": 2.0}
    costs  = calculate_cost(usage, rates)

    assert costs == {
        "prompt_cost_uncached": "0.00080000",   # 800 / 1M * $1
        "prompt_cost_cached"  : "0.00010000",   # 200 / 1M * $0.5
        "completion_cost"     : "0.00400000",   # 2 000 / 1M * $2
        "total_cost"          : "0.00490000",
    }


def test_calculate_cost_typed_basic():
    """Test the new typed cost calculation function."""
    usage  = {"prompt_tokens": 1_000, "completion_tokens": 2_000, "cached_tokens": 200}
    rates  = {"input_price": 1.0, "cached_input_price": 0.5, "output_price": 2.0}
    cost_breakdown = calculate_cost_typed(usage, rates)

    # Verify the result is a CostBreakdown instance
    assert isinstance(cost_breakdown, CostBreakdown)
    
    # Verify all fields are Decimal objects
    assert isinstance(cost_breakdown.prompt_cost_uncached, Decimal)
    assert isinstance(cost_breakdown.prompt_cost_cached, Decimal)
    assert isinstance(cost_breakdown.completion_cost, Decimal)
    assert isinstance(cost_breakdown.total_cost, Decimal)
    
    # Verify correct values
    assert cost_breakdown.prompt_cost_uncached == Decimal("0.0008")   # 800 / 1M * $1
    assert cost_breakdown.prompt_cost_cached == Decimal("0.0001")     # 200 / 1M * $0.5
    assert cost_breakdown.completion_cost == Decimal("0.004")         # 2000 / 1M * $2
    assert cost_breakdown.total_cost == Decimal("0.0049")


def test_calculate_cost_compatibility():
    """Test that old and new calculate_cost functions return equivalent results."""
    usage  = {"prompt_tokens": 1_000, "completion_tokens": 2_000, "cached_tokens": 200}
    rates  = {"input_price": 1.0, "cached_input_price": 0.5, "output_price": 2.0}
    
    # Get results from both functions
    old_result = calculate_cost(usage, rates)
    new_result = calculate_cost_typed(usage, rates)
    
    # Convert typed result to dict with strings
    typed_as_dict = new_result.as_dict(stringify=True)
    
    # They should be identical
    assert old_result == typed_as_dict


def test_cost_breakdown_as_dict():
    """Test the CostBreakdown.as_dict() method with both stringify options."""
    usage  = {"prompt_tokens": 1_000, "completion_tokens": 2_000, "cached_tokens": 200}
    rates  = {"input_price": 1.0, "cached_input_price": 0.5, "output_price": 2.0}
    cost_breakdown = calculate_cost_typed(usage, rates)
    
    # Test stringify=True (default)
    string_dict = cost_breakdown.as_dict(stringify=True)
    assert all(isinstance(v, str) for v in string_dict.values())
    assert string_dict["total_cost"] == "0.00490000"
    
    # Test stringify=False  
    decimal_dict = cost_breakdown.as_dict(stringify=False)
    assert all(isinstance(v, Decimal) for v in decimal_dict.values())
    assert decimal_dict["total_cost"] == Decimal("0.0049")


@pytest.mark.parametrize(
    "model, exp_date",
    [("gpt-4o-mini-2024-07-18", "2024-07-18"),
     ("gpt-4o-mini",            datetime.now(timezone.utc).strftime("%Y-%m-%d"))]
)
def test_extract_model_details(model, exp_date):
    details = extract_model_details(model)
    assert details == {"model_name": "gpt-4o-mini", "model_date": exp_date}


def test_extract_usage_classic_and_new():
    classic = _classic_response(100, 50, 30)
    new     = _new_response(100, 50, 30)
    for obj in (classic, new):
        assert extract_usage(obj) == {
            "prompt_tokens"   : 100,
            "completion_tokens": 50,
            "cached_tokens"   : 30,
        }


# --------------------------------------------------------------------------- #
# Integration tests: estimate_cost                                            #
# --------------------------------------------------------------------------- #
def test_estimate_cost_single_response():
    resp  = _classic_response(1_000, 500, 100)
    cost  = estimate_cost(resp)
    # Quick sanity: strings, not floats & total sum matches parts
    assert all(isinstance(v, str) for v in cost.values())
    total = sum(map(float, (cost["prompt_cost_uncached"],
                            cost["prompt_cost_cached"],
                            cost["completion_cost"])))
    assert float(cost["total_cost"]) == pytest.approx(total)


def test_estimate_cost_typed_single_response():
    """Test the new typed estimate function."""
    resp = _classic_response(1_000, 500, 100)
    cost = estimate_cost_typed(resp)
    
    # Verify the result is a CostBreakdown instance
    assert isinstance(cost, CostBreakdown)
    
    # Verify all fields are Decimal objects
    assert isinstance(cost.prompt_cost_uncached, Decimal)
    assert isinstance(cost.prompt_cost_cached, Decimal)
    assert isinstance(cost.completion_cost, Decimal)
    assert isinstance(cost.total_cost, Decimal)
    
    # Verify total is sum of parts (with Decimal precision)
    expected_total = cost.prompt_cost_uncached + cost.prompt_cost_cached + cost.completion_cost
    assert cost.total_cost == expected_total


def test_estimate_cost_compatibility():
    """Test that old and new estimate functions return equivalent results."""
    resp = _classic_response(1_000, 500, 100)
    
    # Get results from both functions
    old_result = estimate_cost(resp)
    new_result = estimate_cost_typed(resp)
    
    # Convert typed result to dict with strings
    typed_as_dict = new_result.as_dict(stringify=True)
    
    # They should be identical
    assert old_result == typed_as_dict


def test_estimate_cost_stream(monkeypatch):
    # two chunks: first w/o usage, last with usage
    dummy_chunks = (
        _Struct(model="ignored", foo="bar"),
        _classic_response(2_000, 0, 0),
    )
    cost = estimate_cost(iter(dummy_chunks))
    assert float(cost["completion_cost"]) == pytest.approx(0.0)
    assert float(cost["total_cost"]) != pytest.approx(0.0)


def test_estimate_cost_typed_stream():
    """Test the new typed estimate function with streaming."""
    # two chunks: first w/o usage, last with usage
    dummy_chunks = (
        _Struct(model="ignored", foo="bar"),
        _classic_response(2_000, 0, 0),
    )
    cost = estimate_cost_typed(iter(dummy_chunks))
    
    assert isinstance(cost, CostBreakdown)
    assert cost.completion_cost == Decimal("0")
    assert cost.total_cost > Decimal("0")


def test_missing_pricing_raises(monkeypatch):
    resp = _classic_response(10, 10, 0, model="non-existent-2099-01-01")
    with pytest.raises(CostEstimateError):
        estimate_cost(resp)


def test_missing_pricing_raises_typed():
    """Test that the typed version also raises CostEstimateError for missing pricing."""
    resp = _classic_response(10, 10, 0, model="non-existent-2099-01-01")
    with pytest.raises(CostEstimateError):
        estimate_cost_typed(resp)


def test_public_api_imports():
    """Test that all new functions are properly exported."""
    # Test that new functions are available in the public API
    assert hasattr(occ, 'estimate_cost_typed')
    assert hasattr(occ, 'calculate_cost_typed')
    assert hasattr(occ, 'CostBreakdown')
    
    # Test that legacy functions are still available
    assert hasattr(occ, 'estimate_cost')
    assert hasattr(occ, 'refresh_pricing')
    assert hasattr(occ, 'CostEstimateError')

def _resp_small():
    # a tiny, deterministic response for tests
    return _classic_response(1_000, 200, 100)  # uses _PRICING fixture


@pytest.fixture()
def reset_telemetry():
    # Ensure global state doesn't leak across tests
    occ.clear_cost_sinks()
    occ.set_global_tags({})
    yield
    occ.clear_cost_sinks()
    occ.set_global_tags({})


def test_metadata_included_in_dict_when_tags_present(reset_telemetry):
    resp = _resp_small()

    # 1) per-call tags only
    d = estimate_cost(resp, tags={"project": "MarketingBot"})
    assert "metadata" in d
    assert d["metadata"] == {"project": "MarketingBot"}
    # numeric fields should remain strings
    for k, v in d.items():
        if k != "metadata":
            assert isinstance(v, str)

    # 2) global tags only
    occ.set_global_tags({"env": "prod"})
    d2 = estimate_cost(resp)  # no per-call tags
    assert "metadata" in d2 and d2["metadata"] == {"env": "prod"}

    # 3) merged (per-call overrides global)
    d3 = estimate_cost(resp, tags={"env": "staging", "feature": "assist"})
    assert d3["metadata"] == {"env": "staging", "feature": "assist"}


def test_metadata_on_typed_api_and_merge_order(reset_telemetry):
    resp = _resp_small()

    occ.set_global_tags({"team": "growth", "feature": "old"})
    cost = estimate_cost_typed(resp, tags={"feature": "new", "user_id": "42"})

    # Typed API always exposes metadata as dict on the object
    assert isinstance(cost, CostBreakdown)
    assert cost.metadata == {"team": "growth", "feature": "new", "user_id": "42"}


def test_tag_length_validation_on_globals(reset_telemetry):
    long_val = "x" * 129
    with pytest.raises(ValueError):
        occ.set_global_tags({"ok": long_val})
    with pytest.raises(ValueError):
        occ.set_global_tags({long_val: "ok"})

    # After failed sets, globals should still be empty
    assert occ.get_global_tags() == {}


def test_tag_length_validation_on_per_call_tags(reset_telemetry):
    resp = _resp_small()
    long_val = "x" * 129

    # Per-call validation errors bubble as CostEstimateError (wrapped)
    with pytest.raises(CostEstimateError):
        estimate_cost_typed(resp, tags={"bad": long_val})
    with pytest.raises(CostEstimateError):
        estimate_cost_typed(resp, tags={long_val: "bad"})


def test_no_metadata_key_when_no_tags(reset_telemetry):
    # With no global tags and no per-call tags, dict API should not include "metadata"
    resp = _resp_small()
    d = estimate_cost(resp)
    assert "metadata" not in d
    # And all values should remain strings (legacy behavior)
    assert all(isinstance(v, str) for v in d.values())


def test_sink_hooks_receive_breakdowns(reset_telemetry):
    resp = _resp_small()
    received: List[CostBreakdown] = []

    def sink(cb: CostBreakdown):
        received.append(cb)

    occ.register_cost_sink(sink)
    cost = estimate_cost_typed(resp, tags={"t": "1"})

    assert len(received) == 1
    assert isinstance(received[0], CostBreakdown)
    # Same object content as returned
    assert received[0].total_cost == cost.total_cost
    assert received[0].metadata.get("t") == "1"

    # Clear sinks and ensure no more emissions
    occ.clear_cost_sinks()
    estimate_cost_typed(resp, tags={"t": "2"})
    assert len(received) == 1  # unchanged


def test_get_global_tags_returns_copy(reset_telemetry):
    occ.set_global_tags({"a": "1"})
    g = occ.get_global_tags()
    assert g == {"a": "1"}
    g["a"] = "mutated"
    # internal state should not change
    assert occ.get_global_tags() == {"a": "1"}