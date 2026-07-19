"""Extraction and pricing of Anthropic Messages token usage.

The functions here turn a raw Anthropic ``usage`` object (from a non-streaming
response body or an assembled stream) into an :class:`AnthropicUsage`, price it
with :mod:`openai_cost_calculator.anthropic.pricing`, and map the result onto the
existing :class:`~openai_cost_calculator.types.CostBreakdown` so the proxy ledger
can store it without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from openai_cost_calculator.anthropic.pricing import (
    MILLION,
    AnthropicPricingError,
    AnthropicRate,
    resolve_anthropic_rate,
)
from openai_cost_calculator.types import CostBreakdown


class AnthropicUsageError(ValueError):
    """Raised when an Anthropic usage object is present but malformed."""


# Usage keys that can carry a separate charge which this integration does not
# price.  They are surfaced so the caller can record a diagnostic rather than
# silently reporting a request as fully costed.
_SERVER_TOOL_KEYS = ("server_tool_use",)


@dataclass(frozen=True)
class AnthropicUsage:
    """Normalized Anthropic token usage for a single request."""

    input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int
    output_tokens: int

    @property
    def total_input_tokens(self) -> int:
        """Input tokens counted toward long-context tier selection."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.input_tokens == 0
            and self.cache_read_input_tokens == 0
            and self.cache_creation_input_tokens == 0
            and self.output_tokens == 0
        )


@dataclass(frozen=True)
class AnthropicCost:
    """Priced Anthropic request with an itemized breakdown."""

    input_cost: Decimal
    cache_write_cost: Decimal
    cache_read_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    rate: AnthropicRate


def _coerce_int(value: Any, field: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnthropicUsageError(f"usage field {field!r} must be an integer")
    if value < 0:
        raise AnthropicUsageError(f"usage field {field!r} must be non-negative")
    return value


def _nested_int(usage: dict[str, Any], parent: str, field: str) -> int:
    container = usage.get(parent)
    if container is None:
        return 0
    if not isinstance(container, dict):
        raise AnthropicUsageError(f"usage field {parent!r} must be an object")
    return _coerce_int(container.get(field), f"{parent}.{field}")


def extract_anthropic_usage(payload: dict[str, Any]) -> Optional[AnthropicUsage]:
    """Extract usage from an Anthropic Messages response payload.

    Returns ``None`` when the payload carries no ``usage`` object.  Raises
    :class:`AnthropicUsageError` when a usage object is present but malformed.
    """
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage_from_dict(usage)


def usage_from_dict(usage: dict[str, Any]) -> AnthropicUsage:
    """Build an :class:`AnthropicUsage` from a raw Anthropic usage object."""
    if not isinstance(usage, dict):
        raise AnthropicUsageError("usage must be an object")

    input_tokens = _coerce_int(usage.get("input_tokens"), "input_tokens")
    cache_read = _coerce_int(
        usage.get("cache_read_input_tokens"), "cache_read_input_tokens"
    )
    cache_creation = _coerce_int(
        usage.get("cache_creation_input_tokens"), "cache_creation_input_tokens"
    )
    output_tokens = _coerce_int(usage.get("output_tokens"), "output_tokens")

    write_5m = _nested_int(usage, "cache_creation", "ephemeral_5m_input_tokens")
    write_1h = _nested_int(usage, "cache_creation", "ephemeral_1h_input_tokens")

    # If the aggregate is absent but a TTL breakdown is present, derive it.
    if cache_creation == 0 and (write_5m or write_1h):
        cache_creation = write_5m + write_1h

    return AnthropicUsage(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        cache_creation_5m_input_tokens=write_5m,
        cache_creation_1h_input_tokens=write_1h,
        output_tokens=output_tokens,
    )


def unpriced_usage_fields(usage: dict[str, Any]) -> list[str]:
    """Return billable-looking usage keys this integration does not price."""
    if not isinstance(usage, dict):
        return []
    found: list[str] = []
    for key in _SERVER_TOOL_KEYS:
        value = usage.get(key)
        if isinstance(value, dict) and any(value.values()):
            found.append(key)
    return found


def price_anthropic_usage(model: str, usage: AnthropicUsage) -> AnthropicCost:
    """Price normalized Anthropic usage using the Anthropic pricing table."""
    if not isinstance(usage, AnthropicUsage):
        raise AnthropicUsageError("usage must be an AnthropicUsage")
    rate = resolve_anthropic_rate(model, usage.total_input_tokens)

    input_cost = Decimal(usage.input_tokens) * rate.input / MILLION
    cache_read_cost = Decimal(usage.cache_read_input_tokens) * rate.cache_read / MILLION

    # Attribute cache-creation tokens by TTL.  Any tokens not covered by an
    # explicit 5m/1h breakdown are charged at the 5-minute rate; this keeps the
    # aggregate exact without double counting when a breakdown is missing or
    # inconsistent with the total.
    write_1h = usage.cache_creation_1h_input_tokens
    write_5m = usage.cache_creation_5m_input_tokens
    remainder = usage.cache_creation_input_tokens - write_5m - write_1h
    if remainder > 0:
        write_5m += remainder
    elif remainder < 0:
        # The breakdown exceeds the aggregate; trust the breakdown total.
        pass
    cache_write_cost = (
        Decimal(write_5m) * rate.cache_write_5m / MILLION
        + Decimal(write_1h) * rate.cache_write_1h / MILLION
    )

    output_cost = Decimal(usage.output_tokens) * rate.output / MILLION
    total = input_cost + cache_write_cost + cache_read_cost + output_cost
    return AnthropicCost(
        input_cost=input_cost,
        cache_write_cost=cache_write_cost,
        cache_read_cost=cache_read_cost,
        output_cost=output_cost,
        total_cost=total,
        rate=rate,
    )


def to_cost_breakdown(cost: AnthropicCost) -> CostBreakdown:
    """Map an :class:`AnthropicCost` onto the ledger's three-bucket breakdown.

    Cache-creation (write) cost is folded into the uncached-prompt bucket and
    cache-read cost into the cached-prompt bucket.  The itemized Anthropic
    breakdown is preserved separately by callers that need it; the ledger only
    requires that the three buckets sum to the total.
    """
    return CostBreakdown(
        prompt_cost_uncached=cost.input_cost + cost.cache_write_cost,
        prompt_cost_cached=cost.cache_read_cost,
        completion_cost=cost.output_cost,
        total_cost=cost.total_cost,
    )


def to_ledger_tokens(usage: AnthropicUsage) -> dict[str, int]:
    """Map Anthropic usage onto the ledger's ``prompt/completion/cached`` triple.

    Cache-creation (write) tokens are counted as prompt tokens; cache-read
    tokens are reported separately as cached tokens.
    """
    return {
        "prompt_tokens": usage.input_tokens + usage.cache_creation_input_tokens,
        "completion_tokens": usage.output_tokens,
        "cached_tokens": usage.cache_read_input_tokens,
    }


__all__ = [
    "AnthropicCost",
    "AnthropicUsage",
    "AnthropicUsageError",
    "extract_anthropic_usage",
    "price_anthropic_usage",
    "to_cost_breakdown",
    "to_ledger_tokens",
    "unpriced_usage_fields",
    "usage_from_dict",
]
