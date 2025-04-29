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
You can also manually refresh the pricing cache:

```python
from openai_cost_calculator import refresh_pricing

refresh_pricing()
```
"""

__all__ = ["estimate_cost", "refresh_pricing", "CostEstimateError"]