"""Anthropic Messages accounting primitives for the Claude Code integration.

This package is deliberately independent of the OpenAI Responses/Chat pricing
model.  Anthropic distinguishes four separately priced token categories
(uncached input, cache-creation writes with a 5-minute or 1-hour TTL, cache
reads, and output) which the OpenAI three-bucket schema cannot represent.
Keeping the two pricing schemas apart avoids forcing Anthropic semantics into
abstractions designed for the Responses API.

The modules here are pure: they perform no I/O and never touch credentials.
They convert observed Anthropic usage into a :class:`CostBreakdown` that the
existing proxy ledger, turn/session aggregation, and checkpoint machinery can
store unchanged.
"""

from openai_cost_calculator.anthropic.pricing import (
    AnthropicPricingError,
    AnthropicRate,
    resolve_anthropic_rate,
    validate_anthropic_pricing,
)
from openai_cost_calculator.anthropic.usage import (
    AnthropicUsage,
    extract_anthropic_usage,
    price_anthropic_usage,
    to_cost_breakdown,
)

__all__ = [
    "AnthropicPricingError",
    "AnthropicRate",
    "AnthropicUsage",
    "extract_anthropic_usage",
    "price_anthropic_usage",
    "resolve_anthropic_rate",
    "to_cost_breakdown",
    "validate_anthropic_pricing",
]
