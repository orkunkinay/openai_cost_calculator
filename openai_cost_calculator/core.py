"""
Pure cost arithmetic – no OpenAI–specific code lives here.
All numbers are BIGINT-safe (ints in Python are arbitrary precision).
"""

from decimal import Decimal, ROUND_HALF_UP


def _usd(value: float) -> str:
    """Format to 8-decimal-place USD string."""
    return str(Decimal(value).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def calculate_cost(usage: dict, rates: dict) -> dict:
    """
    Parameters
    ----------
    usage
        {
            "prompt_tokens"    : int,
            "completion_tokens": int,
            "cached_tokens"    : int,
        }
        *All keys are required. 0 is fine.*

    rates
        {
            "input_price"        : float   (USD / 1M tokens – uncached)
            "cached_input_price" : float   or None
            "output_price"       : float   (USD / 1M tokens)
        }

    Returns
    -------
    dict
        {
            "prompt_cost_uncached": "...",
            "prompt_cost_cached"  : "...",
            "completion_cost"     : "...",
            "total_cost"          : "..."
        }
    """
    if not isinstance(usage, dict):
        raise TypeError("`usage` must be a dict")

    required = {"prompt_tokens", "completion_tokens", "cached_tokens"}
    if not required.issubset(usage):
        missing = required.difference(usage)
        raise ValueError(f"usage missing keys: {missing}")

    million = 1_000_000
    uncached_prompt = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    cached_prompt = usage["cached_tokens"]

    prompt_uncached_cost = (uncached_prompt / million) * rates["input_price"] # prompt (uncached)

    cached_rate = rates.get("cached_input_price") or rates["input_price"] # prompt (cached)
    prompt_cached_cost = (cached_prompt / million) * cached_rate

    completion_cost = (usage["completion_tokens"] / million) * rates["output_price"] # completion

    total = prompt_uncached_cost + prompt_cached_cost + completion_cost

    return {
        "prompt_cost_uncached": _usd(prompt_uncached_cost),
        "prompt_cost_cached"  : _usd(prompt_cached_cost),
        "completion_cost"     : _usd(completion_cost),
        "total_cost"          : _usd(total),
    }
