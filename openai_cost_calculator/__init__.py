"""
OpenAI Cost Calculator
~~~~~~~~~~~~~~~~~~~~~~

A lightweight, user friendly library to estimate USD costs for OpenAI API responses.

Example usage:

```python
from openai_cost_calculator import estimate_cost

cost = estimate_cost(response)
print(cost["total_cost"])
```

For strongly-typed usage with Decimal precision:

```python
from openai_cost_calculator import estimate_cost_typed

cost = estimate_cost_typed(response)
print(cost.total_cost)  # Decimal object
```

You can also manually refresh the pricing cache:

```python
from openai_cost_calculator import refresh_pricing

refresh_pricing()
```
"""
from .estimate import estimate_cost, estimate_cost_typed, CostEstimateError
from .core import calculate_cost_typed
from .pricing import (
    add_pricing_entry, add_pricing_entries, clear_local_pricing,
    set_offline_mode, refresh_pricing
)
from .types import CostBreakdown
from .tracker import CostTracker, Turn, CallRecord
from .adapters.anthropic_pricing import seed_anthropic_pricing

__all__ = [
    "estimate_cost", 
    "estimate_cost_typed", 
    "calculate_cost_typed",
    "refresh_pricing", 
    "add_pricing_entry",
    "add_pricing_entries",
    "clear_local_pricing",
    "set_offline_mode",
    "CostEstimateError",
    "CostBreakdown",
    "CostTracker",
    "Turn",
    "CallRecord",
    "seed_anthropic_pricing",
]
