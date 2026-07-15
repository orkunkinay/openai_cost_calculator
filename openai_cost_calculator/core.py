"""
Pure cost arithmetic – no OpenAI–specific code lives here.
All numbers are BIGINT-safe (ints in Python are arbitrary precision).
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from .types import CostBreakdown


def _usd(value: float) -> str:
    """Format to 8-decimal-place USD string."""
    return str(Decimal(value).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def _calculate_cost_typed(usage: dict, rates: dict) -> CostBreakdown:
    """
    Internal function that performs cost calculation using Decimal arithmetic
    and returns a strongly-typed CostBreakdown dataclass.
    
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
    CostBreakdown
        Strongly-typed dataclass with Decimal fields for precise arithmetic.
    """
    if not isinstance(usage, dict):
        raise TypeError("`usage` must be a dict")

    required = {"prompt_tokens", "completion_tokens", "cached_tokens"}
    if not required.issubset(usage):
        missing = required.difference(usage)
        raise ValueError(f"usage missing keys: {missing}")

    normalized_usage = {}
    for key in required:
        value = usage[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"usage {key!r} must be a non-negative integer")
        normalized_usage[key] = value

    if normalized_usage["cached_tokens"] > normalized_usage["prompt_tokens"]:
        raise ValueError("usage cached_tokens cannot exceed prompt_tokens")

    million = Decimal("1000000")
    
    uncached_prompt = (
        normalized_usage["prompt_tokens"] - normalized_usage["cached_tokens"]
    )
    cached_prompt = normalized_usage["cached_tokens"]

    # Convert rates to Decimal for precise arithmetic
    try:
        input_price = Decimal(str(rates["input_price"]))
        cached_rate = rates.get("cached_input_price")
        if cached_rate is None:
            cached_rate = rates["input_price"]
        cached_input_price = Decimal(str(cached_rate))
        output_price = Decimal(str(rates["output_price"]))
    except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("rates must contain valid decimal prices") from exc
    for name, price in (
        ("input_price", input_price),
        ("cached_input_price", cached_input_price),
        ("output_price", output_price),
    ):
        if not price.is_finite() or price < 0:
            raise ValueError(f"rate {name!r} must be a finite non-negative number")

    # Calculate costs using Decimal arithmetic
    prompt_uncached_cost = (Decimal(str(uncached_prompt)) / million) * input_price
    prompt_cached_cost = (Decimal(str(cached_prompt)) / million) * cached_input_price
    completion_cost = (
        Decimal(normalized_usage["completion_tokens"]) / million
    ) * output_price
    
    total = prompt_uncached_cost + prompt_cached_cost + completion_cost

    return CostBreakdown(
        prompt_cost_uncached=prompt_uncached_cost,
        prompt_cost_cached=prompt_cached_cost,
        completion_cost=completion_cost,
        total_cost=total
    )


def calculate_cost_typed(usage: dict, rates: dict) -> CostBreakdown:
    """
    Calculate costs and return a strongly-typed CostBreakdown dataclass.
    
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
    CostBreakdown
        Strongly-typed dataclass with Decimal fields containing:
        - prompt_cost_uncached: Decimal
        - prompt_cost_cached: Decimal  
        - completion_cost: Decimal
        - total_cost: Decimal
        
    Examples
    --------
    >>> usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 100}
    >>> rates = {"input_price": 0.5, "cached_input_price": 0.25, "output_price": 1.0}
    >>> cost = calculate_cost_typed(usage, rates)
    >>> cost.total_cost  # Decimal('0.00095000')
    """
    return _calculate_cost_typed(usage, rates)


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
        
    Note
    ----
    This function is maintained for backward compatibility. For new code,
    consider using calculate_cost_typed() which returns a strongly-typed
    CostBreakdown dataclass.
    """
    cost_breakdown = _calculate_cost_typed(usage, rates)
    return cost_breakdown.as_dict(stringify=True)
