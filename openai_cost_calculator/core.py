"""
Pure cost arithmetic – no OpenAI–specific code lives here.
All numbers are BIGINT-safe (ints in Python are arbitrary precision).
"""

from decimal import Decimal, ROUND_HALF_UP
from .types import CostBreakdown


def _usd(value: float) -> str:
    """Format to 8-decimal-place USD string."""
    return str(Decimal(value).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def _calculate_cost_typed(usage: dict, rates: dict, tool_usage: dict = None, tool_pricing: dict = None) -> CostBreakdown:
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
    
    tool_usage
        {
            "WebSearchTool": 2,
            "FileSearchTool": 1,
            ...
        }
        Optional. Dict mapping tool names to call counts. Defaults to empty dict.
    
    tool_pricing
        {
            "WebSearchTool": 0.01,
            "FileSearchTool": 0.05,
            ...
        }
        Optional. Dict mapping tool names to price per call (USD). Defaults to empty dict.

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

    million = Decimal("1000000")
    
    uncached_prompt = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    cached_prompt = usage["cached_tokens"]

    # Convert rates to Decimal for precise arithmetic
    input_price = Decimal(str(rates["input_price"]))
    cached_rate = rates.get("cached_input_price") or rates["input_price"]
    cached_input_price = Decimal(str(cached_rate))
    output_price = Decimal(str(rates["output_price"]))

    # Calculate costs using Decimal arithmetic
    prompt_uncached_cost = (Decimal(str(uncached_prompt)) / million) * input_price
    prompt_cached_cost = (Decimal(str(cached_prompt)) / million) * cached_input_price
    completion_cost = (Decimal(str(usage["completion_tokens"])) / million) * output_price
    
    # Calculate tool costs
    tool_cost = Decimal("0")
    if tool_usage and tool_pricing:
        for tool_name, call_count in tool_usage.items():
            if tool_name in tool_pricing and call_count > 0:
                price_per_call = Decimal(str(tool_pricing[tool_name]))
                tool_cost += Decimal(str(call_count)) * price_per_call
    
    total = prompt_uncached_cost + prompt_cached_cost + completion_cost + tool_cost

    return CostBreakdown(
        prompt_cost_uncached=prompt_uncached_cost,
        prompt_cost_cached=prompt_cached_cost,
        completion_cost=completion_cost,
        tool_cost=tool_cost,
        total_cost=total
    )


def calculate_cost_typed(usage: dict, rates: dict, tool_usage: dict = None, tool_pricing: dict = None) -> CostBreakdown:
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
    
    tool_usage
        {
            "WebSearchTool": 2,
            "FileSearchTool": 1,
            ...
        }
        Optional. Dict mapping tool names to call counts. Defaults to empty dict.
    
    tool_pricing
        {
            "WebSearchTool": 0.01,
            "FileSearchTool": 0.05,
            ...
        }
        Optional. Dict mapping tool names to price per call (USD). Defaults to empty dict.

    Returns
    -------
    CostBreakdown
        Strongly-typed dataclass with Decimal fields containing:
        - prompt_cost_uncached: Decimal
        - prompt_cost_cached: Decimal  
        - completion_cost: Decimal
        - tool_cost: Decimal
        - total_cost: Decimal
        
    Examples
    --------
    >>> usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 100}
    >>> rates = {"input_price": 0.5, "cached_input_price": 0.25, "output_price": 1.0}
    >>> tool_usage = {"WebSearchTool": 2}
    >>> tool_pricing = {"WebSearchTool": 0.01}
    >>> cost = calculate_cost_typed(usage, rates, tool_usage, tool_pricing)
    >>> cost.total_cost  # Decimal('0.02095000')
    """
    return _calculate_cost_typed(usage, rates, tool_usage, tool_pricing)


def calculate_cost(usage: dict, rates: dict, tool_usage: dict = None, tool_pricing: dict = None) -> dict:
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
    
    tool_usage
        Optional. Dict mapping tool names to call counts. Defaults to empty dict.
    
    tool_pricing
        Optional. Dict mapping tool names to price per call (USD). Defaults to empty dict.

    Returns
    -------
    dict
        {
            "prompt_cost_uncached": "...",
            "prompt_cost_cached"  : "...",
            "completion_cost"     : "...",
            "tool_cost"           : "...",
            "total_cost"          : "..."
        }
        
    Note
    ----
    This function is maintained for backward compatibility. For new code,
    consider using calculate_cost_typed() which returns a strongly-typed
    CostBreakdown dataclass.
    """
    cost_breakdown = _calculate_cost_typed(usage, rates, tool_usage, tool_pricing)
    return cost_breakdown.as_dict(stringify=True)
