import builtins
import types
from datetime import datetime
import pytest

import openai_cost_calculator as occ

from openai_cost_calculator.core import calculate_cost
from openai_cost_calculator.estimate import estimate_cost, CostEstimateError
from openai_cost_calculator.parser import extract_model_details, extract_usage
import openai_cost_calculator.pricing as pricing_module

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


@pytest.mark.parametrize(
    "model, exp_date",
    [("gpt-4o-mini-2024-07-18", "2024-07-18"),
     ("gpt-4o-mini",            datetime.utcnow().strftime("%Y-%m-%d"))]
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


def test_estimate_cost_stream(monkeypatch):
    # two chunks: first w/o usage, last with usage
    dummy_chunks = (
        _Struct(model="ignored", foo="bar"),
        _classic_response(2_000, 0, 0),
    )
    cost = estimate_cost(iter(dummy_chunks))
    assert float(cost["completion_cost"]) == pytest.approx(0.0)
    assert float(cost["total_cost"]) != pytest.approx(0.0)


def test_missing_pricing_raises(monkeypatch):
    resp = _classic_response(10, 10, 0, model="non-existent-2099-01-01")
    with pytest.raises(CostEstimateError):
        estimate_cost(resp)
