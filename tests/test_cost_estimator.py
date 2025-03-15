import pytest
from openai_cost_calculator.usage_parser import parse_completion_usage, extract_model_details
from openai_cost_calculator.cost_calculator import calculate_cost
from openai_cost_calculator.cost_estimator import CostEstimator
from openai_cost_calculator.pricing import load_pricing

# --- Tests for usage_parser functions ---

def test_parse_completion_usage():
    usage_str = (
        "prompt_tokens=21, completion_tokens=500, total_tokens=521, accepted_prediction_tokens=0, "
        "completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), "
        "prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)"
    )
    parsed = parse_completion_usage(usage_str)
    expected = {
        "completion_tokens": 500,
        "prompt_tokens": 21,
        "total_tokens": 521,
        "accepted_prediction_tokens": 0,
        "audio_tokens_completion": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
        "audio_tokens_prompt": 0,
        "cached_tokens": 0
    }
    assert parsed == expected

def test_extract_model_details():
    model_str = "gpt-4o-mini-2024-07-18"
    details = extract_model_details(model_str)
    expected = {"model_name": "gpt-4o-mini", "model_date": "2024-07-18"}
    assert details == expected

# --- Tests for cost_calculator function ---

def test_calculate_cost_both():
    usage_data = {
        "prompt_tokens": 1000,
        "completion_tokens": 2000,
        "audio_tokens_prompt": 500,
        "audio_tokens_completion": 1000,
        "accepted_prediction_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "rejected_prediction_tokens": 0,
        "cached_tokens": 0
    }
    pricing_text = {"input_price": 2.5, "output_price": 10.0}
    pricing_audio = {"input_price": 40.0, "output_price": 80.0}
    
    cost = calculate_cost(usage_data, pricing_text, pricing_audio)
    
    # Expected cost calculations:
    # Text cost: (1000/1e6)*2.5 = 0.0025, (2000/1e6)*10.0 = 0.02, total text = 0.0225
    # Audio cost: (500/1e6)*40.0 = 0.02, (1000/1e6)*80.0 = 0.08, total audio = 0.1
    # Combined total = 0.0225 + 0.1 = 0.1225
    expected = {
        "text_input_cost": "0.00250000",
        "text_output_cost": "0.02000000",
        "text_total_cost": "0.02250000",
        "audio_input_cost": "0.02000000",
        "audio_output_cost": "0.08000000",
        "audio_total_cost": "0.10000000",
        "total_cost": "0.12250000"
    }
    assert cost == expected

# --- Tests for CostEstimator ---

# Dummy class to simulate ChatCompletionChunk
class DummyChunk:
    def __init__(self, model, usage):
        self.model = model
        self.usage = usage  # usage can be a string representing usage details

def test_cost_estimator_single():
    # Simulate a ChatCompletion object (non-streaming) with text tokens only.
    usage_str = (
        "prompt_tokens=1000, completion_tokens=2000, total_tokens=3000, accepted_prediction_tokens=0, "
        "completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), "
        "prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)"
    )
    dummy = DummyChunk("gpt-4o-mini-2024-07-18", usage_str)
    estimator = CostEstimator()
    
    cost = estimator.estimate_cost(dummy)
    # Check that total_cost is computed and convertible to float.
    total_cost = cost.get("total_cost")
    assert total_cost is not None
    float(total_cost)  # Will raise ValueError if conversion fails

def test_cost_estimator_iterable():
    # Simulate a streaming response (iterable) with both text and audio tokens.
    usage_str = (
        "prompt_tokens=500, completion_tokens=1000, total_tokens=1500, accepted_prediction_tokens=0, "
        "completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=200, reasoning_tokens=0, rejected_prediction_tokens=0), "
        "prompt_tokens_details=PromptTokensDetails(audio_tokens=300, cached_tokens=0)"
    )
    dummy = DummyChunk("gpt-4o-mini-2024-07-18", usage_str)
    estimator = CostEstimator()
    
    # Provide the dummy as an iterable (e.g., list with one chunk)
    response = [dummy]
    cost = estimator.estimate_cost(response)
    total_cost = cost.get("total_cost")
    assert total_cost is not None
    float(total_cost)

def test_estimate_cost_with_invalid_response():
    estimator = CostEstimator()
    # Passing an integer should raise a TypeError
    with pytest.raises(TypeError):
        estimator.estimate_cost(2)

def test_estimate_cost_with_empty_iterable():
    estimator = CostEstimator()
    # Passing an empty iterable should raise a ValueError
    with pytest.raises(ValueError):
        estimator.estimate_cost([])
