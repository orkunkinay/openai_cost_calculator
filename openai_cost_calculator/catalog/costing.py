"""Normalized usage, itemized costs, and the (provider-free) cost arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterator, Mapping, Optional, Tuple

from ..types import CostBreakdown
from .dimensions import DIMENSIONS, get_dimension
from .errors import MissingRateError, UsageError
from .model import PriceSet

#: Usage field name for every pricing dimension.
USAGE_FIELDS: Mapping[str, str] = {
    "input": "input_tokens",
    "cached_input": "cached_input_tokens",
    "cache_write": "cache_write_tokens",
    "cache_write_1h": "cache_write_1h_tokens",
    "input_audio": "input_audio_tokens",
    "cached_input_audio": "cached_input_audio_tokens",
    "input_image": "input_image_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
    "output_audio": "output_audio_tokens",
    "output_image": "output_image_tokens",
    "web_search": "web_search_calls",
    "request": "requests",
}


@dataclass(frozen=True)
class Usage:
    """Billable quantities for one request, in *disjoint* buckets.

    ``input_tokens`` counts only uncached text input: cache reads belong in
    ``cached_input_tokens`` and cache writes in ``cache_write_tokens``.  (The
    OpenAI API's ``prompt_tokens`` *includes* cached tokens; use
    :meth:`from_totals` to convert.)  ``reasoning_tokens`` may be reported
    separately from ``output_tokens``; they are charged at the output rate
    unless the model publishes a distinct reasoning rate.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    input_audio_tokens: int = 0
    cached_input_audio_tokens: int = 0
    input_image_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    output_audio_tokens: int = 0
    output_image_tokens: int = 0
    web_search_calls: int = 0
    requests: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise UsageError(f"{f.name} must be an integer, got {value!r}")
            if value < 0:
                raise UsageError(f"{f.name} must be non-negative, got {value}")

    @classmethod
    def from_totals(
        cls,
        *,
        total_input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> "Usage":
        """Build usage from a total that *includes* cached and cache-write tokens."""
        uncached = total_input_tokens - cached_input_tokens - cache_write_tokens
        if uncached < 0:
            raise UsageError(
                "cached_input_tokens + cache_write_tokens cannot exceed total_input_tokens "
                f"({cached_input_tokens} + {cache_write_tokens} > {total_input_tokens})"
            )
        return cls(
            input_tokens=uncached,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            output_tokens=output_tokens,
        )

    def quantities(self) -> Iterator[Tuple[str, int]]:
        """Yield ``(dimension, quantity)`` for every non-zero bucket."""
        for dimension, field_name in USAGE_FIELDS.items():
            quantity = getattr(self, field_name)
            if quantity:
                yield dimension, quantity

    @property
    def total_input_tokens(self) -> int:
        """Input-side tokens used to select long-context tiers."""
        return sum(q for d, q in self.quantities() if DIMENSIONS[d].side == "input")

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})


@dataclass(frozen=True)
class LineItem:
    dimension: str
    quantity: int
    #: Dimension whose rate was charged (differs from ``dimension`` on fallback).
    rate_dimension: str
    unit_price: Decimal
    unit: str
    cost: Decimal


@dataclass(frozen=True)
class Cost:
    """An itemized, exactly computed cost with its pricing provenance."""

    provider: str
    model: str
    resolved_model: str
    total: Decimal
    items: Tuple[LineItem, ...]
    currency: str = "USD"
    canonical_model: Optional[str] = None
    conditions: Mapping[str, str] = field(default_factory=dict)
    min_input_tokens: int = 0
    priced_on: Optional[date] = None
    source_url: Optional[str] = None
    verified_at: Optional[date] = None
    assumptions: Tuple[str, ...] = ()

    def by_dimension(self) -> Dict[str, Decimal]:
        return {item.dimension: item.cost for item in self.items}

    def rounded(self, places: int = 8) -> Decimal:
        return self.total.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)

    def to_breakdown(self) -> CostBreakdown:
        """Project onto the legacy three-bucket :class:`CostBreakdown`.

        Cache reads map to the cached-prompt bucket, every other input-side
        dimension (including cache writes) to the uncached-prompt bucket, and
        everything else to the completion bucket.
        """
        uncached = cached = completion = Decimal(0)
        for item in self.items:
            dimension = DIMENSIONS[item.dimension]
            if item.dimension in ("cached_input", "cached_input_audio"):
                cached += item.cost
            elif dimension.side == "input":
                uncached += item.cost
            else:
                completion += item.cost
        return CostBreakdown(
            prompt_cost_uncached=uncached,
            prompt_cost_cached=cached,
            completion_cost=completion,
            total_cost=self.total,
        )


def _rate_for(dimension: str, rates: Mapping[str, Decimal]) -> Optional[Tuple[str, Decimal]]:
    current: Optional[str] = dimension
    while current is not None:
        if current in rates:
            return current, rates[current]
        current = get_dimension(current).fallback
    return None


def price_usage(usage: Usage, price_set: PriceSet, *, where: str) -> Tuple[Decimal, Tuple[LineItem, ...]]:
    items = []
    total = Decimal(0)
    for dimension, quantity in usage.quantities():
        found = _rate_for(dimension, price_set.rates)
        if found is None:
            priced = ", ".join(sorted(price_set.rates))
            raise MissingRateError(
                f"{where}: usage has {quantity} {dimension} units but the price set only prices: {priced}"
            )
        rate_dimension, unit_price = found
        spec = DIMENSIONS[dimension]
        cost = Decimal(quantity) * unit_price / spec.unit_size
        total += cost
        items.append(
            LineItem(
                dimension=dimension,
                quantity=quantity,
                rate_dimension=rate_dimension,
                unit_price=unit_price,
                unit=spec.unit_label,
                cost=cost,
            )
        )
    return total, tuple(items)
