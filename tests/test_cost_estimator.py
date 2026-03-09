from datetime import datetime, timezone
from decimal import Decimal
import importlib
import pytest

import openai_cost_calculator as occ

from openai_cost_calculator.core import calculate_cost, calculate_cost_typed
from openai_cost_calculator.estimate import estimate_cost, estimate_cost_typed, CostEstimateError
from openai_cost_calculator.parser import extract_model_details, extract_usage
from openai_cost_calculator.types import CostBreakdown

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
_PRICING_TIERED = {("gpt-4o-mini", "2024-07-18"): [{
    "input_price"       : 0.50,
    "cached_input_price": 0.25,
    "output_price"      : 1.00,
    "minimum_tokens"    : 0,
}]}

@pytest.fixture(autouse=True)
def monkeypatch_pricing(monkeypatch):
    """Force pricing loaders to return our static dicts."""
    monkeypatch.setattr(occ.pricing, "load_pricing", lambda: _PRICING)
    monkeypatch.setattr(occ.pricing, "load_pricing_tiered", lambda: _PRICING_TIERED)


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


def test_estimate_cost_tier_boundary_selection(monkeypatch):
    tiered = {
        ("gpt-5.4", "2026-01-01"): [
            {"input_price": 2.50, "cached_input_price": 0.25, "output_price": 15.00, "minimum_tokens": 0},
            {"input_price": 5.00, "cached_input_price": 0.50, "output_price": 22.50, "minimum_tokens": 272001},
        ]
    }
    monkeypatch.setattr(occ.pricing, "load_pricing_tiered", lambda: tiered)

    at_boundary = _classic_response(272_000, 0, 0, model="gpt-5.4-2026-01-01")
    over_boundary = _classic_response(272_001, 0, 0, model="gpt-5.4-2026-01-01")

    cost_at_boundary = estimate_cost_typed(at_boundary)
    cost_over_boundary = estimate_cost_typed(over_boundary)

    assert cost_at_boundary.prompt_cost_uncached == Decimal("0.68")
    assert cost_over_boundary.prompt_cost_uncached == Decimal("1.360005")


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

# --------------------------------------------------------------------------- #
# Pricing source (remote CSV) + 24h cache + overrides tests                   #
# --------------------------------------------------------------------------- #

def _csv_text(rows):
    header = "Model Name,Model Date,Input Price,Cached Input Price,Output Price\n"
    return header + "\n".join(rows) + "\n"


def _csv_text_with_minimum(rows):
    header = "Model Name,Model Date,Input Price,Cached Input Price,Output Price,Minimum Tokens\n"
    return header + "\n".join(rows) + "\n"


def test_load_pricing_fetch_cache_and_ttl_refresh(monkeypatch):
    # Reload to undo the autouse fixture's monkeypatch on load_pricing
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    # Reset globals
    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False
    pricing._TTL = 10  # small TTL for test

    # Two CSV versions (same key, different numbers)
    csv_v1 = _csv_text([
        "gpt-4o-mini,2024-07-18,0.50,0.25,1.00",
    ])
    csv_v2 = _csv_text([
        "gpt-4o-mini,2024-07-18,0.60,0.30,1.20",
    ])

    call_count = {"n": 0}
    current_csv = {"text": csv_v1}

    class _Resp:
        def __init__(self, text): self.text = text
        def raise_for_status(self): return None

    def fake_get(url, timeout):
        call_count["n"] += 1
        return _Resp(current_csv["text"])

    # Simulate time progression
    times = [1000, 1005, 1012]  # second call within TTL, third beyond TTL
    def fake_time():
        return times[0] if len(times) == 1 else times.pop(0)

    monkeypatch.setattr(pricing, "_PRICING_CSV_URL", "http://example.com/test.csv")
    monkeypatch.setattr(pricing.requests, "get", fake_get)
    monkeypatch.setattr(pricing.time, "time", fake_time)

    # 1) First fetch
    data1 = pricing.load_pricing()
    assert call_count["n"] == 1
    assert data1[("gpt-4o-mini", "2024-07-18")] == {
        "input_price": 0.50, "cached_input_price": 0.25, "output_price": 1.00
    }

    # 2) Within TTL → no refetch
    data2 = pricing.load_pricing()
    assert call_count["n"] == 1  # unchanged
    assert data2 == data1

    # 3) After TTL → refetch to v2
    current_csv["text"] = csv_v2
    data3 = pricing.load_pricing()
    assert call_count["n"] == 2
    assert data3[("gpt-4o-mini", "2024-07-18")] == {
        "input_price": 0.60, "cached_input_price": 0.30, "output_price": 1.20
    }


def test_offline_mode_uses_only_local_overrides(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    # Reset
    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False

    # Fail if network is touched
    def boom(*a, **k): raise AssertionError("Network should not be used in offline mode")
    monkeypatch.setattr(pricing.requests, "get", boom)

    pricing.set_offline_mode(True)
    pricing.add_pricing_entry(
        "gpt-4o-mini", "2025-08-01",
        input_price=0.20, output_price=0.60, cached_input_price=0.04
    )
    data = pricing.load_pricing()
    assert ("gpt-4o-mini", "2025-08-01") in data
    assert data[("gpt-4o-mini", "2025-08-01")] == {
        "input_price": 0.20, "cached_input_price": 0.04, "output_price": 0.60
    }

    # No-op refresh in offline mode (and still no network)
    pricing.refresh_pricing()
    data2 = pricing.load_pricing()
    assert data2 == data


def test_local_overrides_take_precedence(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False

    csv_remote = _csv_text([
        "gpt-4o-mini,2024-07-18,0.50,0.25,1.00",
    ])

    class _Resp:
        def __init__(self, text): self.text = text
        def raise_for_status(self): return None

    monkeypatch.setattr(pricing.requests, "get", lambda url, timeout: _Resp(csv_remote))

    # Add override with the same key but different values
    pricing.add_pricing_entry(
        "gpt-4o-mini", "2024-07-18",
        input_price=0.99, output_price=2.22, cached_input_price=0.11
    )

    data = pricing.load_pricing()
    assert data[("gpt-4o-mini", "2024-07-18")] == {
        "input_price": 0.99, "cached_input_price": 0.11, "output_price": 2.22
    }


def test_clear_local_pricing(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._LOCAL_OVERRIDES.clear()
    pricing.set_offline_mode(True)

    pricing.add_pricing_entry(
        "foo", "2025-01-01", input_price=1.0, output_price=2.0, cached_input_price=0.5
    )
    assert ("foo", "2025-01-01") in pricing.load_pricing()

    pricing.clear_local_pricing()
    assert pricing.load_pricing() == {}  # offline mode → only local overrides considered


def test_add_pricing_entry_validation():
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    with pytest.raises(ValueError):
        pricing.add_pricing_entry("", "2025-01-01", input_price=0.1, output_price=0.2)

    with pytest.raises(ValueError):
        pricing.add_pricing_entry("m", "2025-1-1", input_price=0.1, output_price=0.2)

    with pytest.raises(ValueError):
        pricing.add_pricing_entry("m", "2025-01-01", input_price=-0.1, output_price=0.2)

    with pytest.raises(ValueError):
        pricing.add_pricing_entry("m", "2025-01-01", input_price=0.1, output_price=-0.2)


def test_add_pricing_entries_bulk_and_replace_flag():
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._LOCAL_OVERRIDES.clear()
    pricing.set_offline_mode(True)

    pricing.add_pricing_entries([
        ("m1", "2025-01-01", 0.1, 0.2, 0.05),
        ("m2", "2025-01-01", 0.3, 0.4, None),
    ])

    d = pricing.load_pricing()
    assert ("m1", "2025-01-01") in d and ("m2", "2025-01-01") in d

    # Attempt to add existing without replace → KeyError
    with pytest.raises(KeyError):
        pricing.add_pricing_entries([
            ("m1", "2025-01-01", 9.9, 9.9, 9.9),
        ], replace=False)

    # Replace = True should update
    pricing.add_pricing_entries([
        ("m1", "2025-01-01", 1.1, 2.2, 0.0),  # 0.0 → None
        ("m1", "2025-01-01", 1.9, 2.9, 0.9, 200000),  # tiered tuple format
    ], replace=True)

    d2 = pricing.load_pricing()
    assert d2[("m1", "2025-01-01")] == {
        "input_price": 1.1, "cached_input_price": None, "output_price": 2.2
    }
    assert pricing.load_pricing_tiered()[("m1", "2025-01-01")][1]["minimum_tokens"] == 200000


def test_refresh_pricing_immediate_fetch(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False

    current = {"text": _csv_text(["m,2025-01-01,0.1,,0.2"])}

    class _Resp:
        def __init__(self, text): self.text = text
        def raise_for_status(self): return None

    def fake_get(url, timeout):
        return _Resp(current["text"])

    monkeypatch.setattr(pricing.requests, "get", fake_get)

    # First refresh -> load v1
    pricing.refresh_pricing()
    d1 = pricing.load_pricing()
    assert d1[("m", "2025-01-01")] == {
        "input_price": 0.1, "cached_input_price": None, "output_price": 0.2
    }

    # Change remote to v2 and refresh again
    current["text"] = _csv_text(["m,2025-01-01,9.9,0.7,8.8"])
    pricing.refresh_pricing()
    d2 = pricing.load_pricing()
    assert d2[("m", "2025-01-01")] == {
        "input_price": 9.9, "cached_input_price": 0.7, "output_price": 8.8
    }


def test_fetch_csv_parses_blank_cached_price_as_none(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False

    csv_blank_cached = _csv_text([
        "x,2025-02-02,0.50,,1.00",  # blank cached_input_price
    ])

    class _Resp:
        def __init__(self, text): self.text = text
        def raise_for_status(self): return None

    monkeypatch.setattr(pricing.requests, "get", lambda url, timeout: _Resp(csv_blank_cached))

    d = pricing.load_pricing()
    assert d[("x", "2025-02-02")]["cached_input_price"] is None


def test_fetch_csv_parses_minimum_tokens_and_keeps_tiers(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False

    csv_tiered = _csv_text_with_minimum([
        "gpt-5.4,2026-01-01,2.50,0.25,15.00,0",
        "gpt-5.4,2026-01-01,5.00,0.50,22.50,272001",
    ])

    class _Resp:
        def __init__(self, text): self.text = text
        def raise_for_status(self): return None

    monkeypatch.setattr(pricing.requests, "get", lambda url, timeout: _Resp(csv_tiered))

    tiered = pricing.load_pricing_tiered()
    assert tiered[("gpt-5.4", "2026-01-01")] == [
        {"input_price": 2.5, "cached_input_price": 0.25, "output_price": 15.0, "minimum_tokens": 0},
        {"input_price": 5.0, "cached_input_price": 0.5, "output_price": 22.5, "minimum_tokens": 272001},
    ]

    # Backward-compatible flat view should still expose the base tier.
    flat = pricing.load_pricing()
    assert flat[("gpt-5.4", "2026-01-01")] == {
        "input_price": 2.5, "cached_input_price": 0.25, "output_price": 15.0
    }


def test_add_pricing_entry_with_minimum_tokens(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing.set_offline_mode(True)

    pricing.add_pricing_entry(
        "gpt-5.4", "2026-01-01",
        input_price=2.5, output_price=15.0, cached_input_price=0.25, minimum_tokens=0
    )
    pricing.add_pricing_entry(
        "gpt-5.4", "2026-01-01",
        input_price=5.0, output_price=22.5, cached_input_price=0.5, minimum_tokens=272001
    )

    tiered = pricing.load_pricing_tiered()
    assert tiered[("gpt-5.4", "2026-01-01")][1]["minimum_tokens"] == 272001

    with pytest.raises(KeyError):
        pricing.add_pricing_entry(
            "gpt-5.4", "2026-01-01",
            input_price=9.9, output_price=9.9, cached_input_price=9.9, minimum_tokens=272001, replace=False
        )


def test_set_offline_mode_refresh_noop(monkeypatch):
    importlib.reload(occ.pricing)
    pricing = occ.pricing

    pricing._CACHE = None
    pricing._CACHE_TS = 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing.set_offline_mode(True)

    # Any network usage would fail the test
    def boom(*a, **k): raise AssertionError("Should not call network in offline mode")
    monkeypatch.setattr(pricing.requests, "get", boom)

    # No error should occur, and cache remains empty until overrides added
    pricing.refresh_pricing()
    assert pricing.load_pricing() == {}

    # Add an entry with cached_input_price=0 (treated as None)
    pricing.add_pricing_entry("y", "2025-03-03", input_price=0.2, output_price=0.4, cached_input_price=0.0)
    d = pricing.load_pricing()
    assert d[("y", "2025-03-03")] == {
        "input_price": 0.2, "cached_input_price": None, "output_price": 0.4
    }
