# openai_cost_calculator

[![PyPI version](https://img.shields.io/pypi/v/openai-cost-calculator)](https://pypi.org/project/openai-cost-calculator/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A lightweight, user-friendly library to estimate USD costs for OpenAI and Azure OpenAI API responses.

## Features

- **Query-level cost estimation**: calculates **cost per user query** individually, based on the actual tokens used — no guesswork, no aggregate billing needed.
- **Dual-API support**: works with `chat.completions.create()` and the new `responses.create()`.
- **Zero boilerplate**: one import & one call: `estimate_cost(response)`.
- **Strongly-typed API**: new typed functions return `CostBreakdown` dataclass with `Decimal` precision for accurate financial calculations.
- **Backward compatibility**: legacy string-based API remains unchanged.
- **Pricing auto-refresh**: daily CSV pull with a helper `refresh_pricing()`.
- **Edge-case aware**: cached tokens, undated models, streaming generators, Azure deployments handled.
- **Predictable output**: legacy API returns strings formatted to 8 decimal places; typed API uses `Decimal` for precise arithmetic.

> **Note:**  
> `openai_cost_calculator` computes the **exact USD cost for each individual user query**,  
> based on **token counts** directly returned by OpenAI or Azure OpenAI.  
> It does **not estimate based on model type or guessing** — it uses precise usage data attached to each response.

## Installation

```bash
pip install openai-cost-calculator
```

> **Note:** Package name on PyPI uses dashes; import name is `openai_cost_calculator`.

## Quickstart

### Basic Usage (Legacy String API)

```python
from openai import OpenAI
from openai_cost_calculator import estimate_cost

client = OpenAI(api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Hi there!"}],
)

print(estimate_cost(resp))
# {'prompt_cost_uncached': '0.00000150',
#  'prompt_cost_cached'  : '0.00000000',
#  'completion_cost'     : '0.00000600',
#  'total_cost'          : '0.00000750'}
```

### Strongly-Typed Usage (Recommended for New Code)

```python
from openai import OpenAI
from openai_cost_calculator import estimate_cost_typed

client = OpenAI(api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Hi there!"}],
)

cost = estimate_cost_typed(resp)
print(f"Total cost: ${cost.total_cost}")  # Decimal('0.00000750')

# Access individual components
print(f"Prompt (uncached): ${cost.prompt_cost_uncached}")
print(f"Prompt (cached): ${cost.prompt_cost_cached}")
print(f"Completion: ${cost.completion_cost}")

# Convert to dict with precise Decimal values
decimal_dict = cost.as_dict(stringify=False)

# Convert to legacy string format
string_dict = cost.as_dict(stringify=True)
```

### Chat Completion API

```python
from openai import OpenAI
from openai_cost_calculator import estimate_cost

client = OpenAI(api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Hi there!"}],
)

print(estimate_cost(resp))
# {'prompt_cost_uncached': '0.00000150',
#  'prompt_cost_cached'  : '0.00000000',
#  'completion_cost'     : '0.00000600',
#  'total_cost'          : '0.00000750'}
```

### Responses API

```python
resp = client.responses.create(
    model="gpt-4.1-mini",
    input=[{"role":"user","content":"Hi there!"}],
)
print(estimate_cost(resp))
```

## API Reference

### Strongly-Typed API (Recommended)

```python
from openai_cost_calculator import estimate_cost_typed, calculate_cost_typed, CostBreakdown
```

#### `estimate_cost_typed(response) → CostBreakdown`

- Accepts a ChatCompletion, streamed chunks, or Response object.
- Returns a `CostBreakdown` dataclass with `Decimal` fields:
  - `prompt_cost_uncached: Decimal`
  - `prompt_cost_cached: Decimal`
  - `completion_cost: Decimal`
  - `total_cost: Decimal`

#### `calculate_cost_typed(usage, rates) → CostBreakdown`

- Lower-level function for direct cost calculation.
- `usage`: dict with `prompt_tokens`, `completion_tokens`, `cached_tokens`
- `rates`: dict with `input_price`, `cached_input_price`, `output_price`
- Returns the same `CostBreakdown` dataclass.

#### `CostBreakdown` dataclass

```python
@dataclass(frozen=True, slots=True)
class CostBreakdown:
    prompt_cost_uncached: Decimal
    prompt_cost_cached: Decimal
    completion_cost: Decimal
    total_cost: Decimal
    
    def as_dict(self, stringify: bool = True) -> Dict[str, str | Decimal]:
        """
        Convert to dictionary.
        - If stringify=True: return 8-decimal-place strings (legacy format)
        - If stringify=False: return raw Decimal objects
        """
```

### Legacy String API (Backward Compatibility)

```python
from openai_cost_calculator import estimate_cost, refresh_pricing, CostEstimateError
```

#### `estimate_cost(response) → dict[str, str]`

- Accepts a ChatCompletion, streamed chunks, or Response object.
- Returns a dict with:
  - `prompt_cost_uncached`: str
  - `prompt_cost_cached`  : str
  - `completion_cost`     : str
  - `total_cost`          : str

### Common Functions

#### `refresh_pricing() → None`

- Force-reload the remote pricing CSV (cache TTL is 24h).

#### `CostEstimateError`

- Unified exception for recoverable input, parsing, or pricing errors.

## When to Use Which API

### Use the **Typed API** (`estimate_cost_typed`) when:

- Building new applications
- Performing financial calculations requiring precision
- Working with accounting or billing systems
- Need to aggregate costs across multiple requests
- Want type safety and IDE support

### Use the **Legacy API** (`estimate_cost`) when:

- Maintaining existing codebases
- Need string output for JSON serialization
- Working with systems expecting the original format
- Quick debugging or logging

## Example: Financial Calculations

```python
from openai_cost_calculator import estimate_cost_typed
from decimal import Decimal

# Calculate costs for multiple requests
total_cost = Decimal('0')
for response in responses:
    cost = estimate_cost_typed(response)
    total_cost += cost.total_cost
    
print(f"Total cost: ${total_cost}")  # Precise Decimal arithmetic
```

## Pricing Data

Pricing is loaded from a remote CSV at:

```
https://raw.githubusercontent.com/orkunkinay/openai_cost_calculator/refs/heads/main/data/gpt_pricing_data.csv
```

Cached for 24 hours by default; use `refresh_pricing()` to force an update immediately.

## Troubleshooting

- **New model raises "pricing not found"**
  1. Verify the model/date in the [pricing CSV on GitHub](https://github.com/orkunkinay/openai_cost_calculator/blob/main/data/gpt_pricing_data.csv).
  2. If missing, open an issue or email the maintainer.
  3. If present, call `refresh_pricing()`.

- **`cached_tokens = 0` even though some were cached**
  - Ensure you request `include_usage_details=True` (classic) or `stream_options={"include_usage": True}` (streaming).

## Contributing

PRs for additional edge-cases, new pricing formats, or SDK changes are welcome!

## License

MIT License © 2025 Orkun Kınay, Murat Barkın Kınay
