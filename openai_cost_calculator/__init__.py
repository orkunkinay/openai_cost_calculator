"""
LLM cost calculator
~~~~~~~~~~~~~~~~~~~

Provider-agnostic, auditable USD cost calculation for LLM API usage across
OpenAI, Anthropic, Google (Gemini API and Vertex AI), Azure, Amazon Bedrock,
OpenRouter, DeepSeek, Together, Groq, Fireworks, DeepInfra and Mistral.

```python
from openai_cost_calculator import calculate_cost

cost = calculate_cost(
    provider="aws-bedrock",
    model="anthropic/claude-sonnet-4-5",
    input_tokens=12_000,
    output_tokens=800,
)
cost.total        # Decimal, exact
cost.items        # itemized by dimension
cost.assumptions  # e.g. which endpoint region was assumed
cost.source_url   # where the price came from
```

Price a raw provider response in any supported usage format:

```python
from openai_cost_calculator import estimate_response_cost

cost = estimate_response_cost(anthropic_message, provider="anthropic")
```

The original OpenAI-response API is unchanged:

```python
from openai_cost_calculator import estimate_cost, estimate_cost_typed

estimate_cost_typed(openai_response).total_cost
```
"""
from .adapters.anthropic_pricing import seed_anthropic_pricing
from .api import (
    calculate_cost,
    compare_costs,
    estimate_response_cost,
    get_model_pricing,
    list_models,
    list_providers,
)
from .catalog import (
    AmbiguousModelError,
    Cost,
    LineItem,
    MissingRateError,
    PricingCatalog,
    PricingError,
    PricingUnavailableError,
    UnknownModelError,
    UnknownProviderError,
    Usage,
    UsageError,
)
from .core import calculate_cost_typed
from .estimate import CostEstimateError, estimate_cost, estimate_cost_typed
from .pricing import (
    add_pricing_entries,
    add_pricing_entry,
    clear_local_pricing,
    refresh_pricing,
    set_offline_mode,
)
from .tracker import CallRecord, CostTracker, Turn
from .types import CostBreakdown
from .usage import extract_usage

__all__ = [
    # provider-agnostic API
    "calculate_cost",
    "compare_costs",
    "estimate_response_cost",
    "extract_usage",
    "get_model_pricing",
    "list_models",
    "list_providers",
    "Cost",
    "LineItem",
    "PricingCatalog",
    "Usage",
    "PricingError",
    "UnknownProviderError",
    "UnknownModelError",
    "AmbiguousModelError",
    "PricingUnavailableError",
    "MissingRateError",
    "UsageError",
    # original OpenAI-response API (unchanged)
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
