"""Helpers shared by source parsers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ...catalog.model import PriceSet

PER_MILLION = Decimal(1_000_000)


def price_set(
    rates: Mapping[str, Optional[Decimal]],
    *,
    service_tier: str = "standard",
    region: Optional[str] = None,
    period: Optional[str] = None,
    min_input_tokens: int = 0,
    effective_from: Optional[date] = None,
    effective_until: Optional[date] = None,
) -> Optional[PriceSet]:
    """Build a price set, dropping unpriced dimensions; ``None`` if nothing is priced.

    ``standard`` is the implicit service tier and is therefore not stored.
    """
    kept: Dict[str, Decimal] = {k: v for k, v in rates.items() if v is not None}
    if not kept:
        return None
    conditions: Dict[str, str] = {}
    if service_tier != "standard":
        conditions["service_tier"] = service_tier
    if region is not None:
        conditions["region"] = region
    if period is not None:
        conditions["period"] = period
    return PriceSet(
        rates=kept,
        conditions=conditions,
        min_input_tokens=min_input_tokens,
        effective_from=effective_from,
        effective_until=effective_until,
    )


def present(sets: Iterable[Optional[PriceSet]]) -> Tuple[PriceSet, ...]:
    """The price sets that were actually built (``price_set`` returns None when nothing is priced)."""
    return tuple(s for s in sets if s is not None)


def scale(rates: Mapping[str, Decimal], factor: Decimal) -> Dict[str, Decimal]:
    return {k: v * factor for k, v in rates.items()}


def per_token_to_per_million(value: object) -> Optional[Decimal]:
    """Convert a per-token price (string/number) to USD per 1M tokens."""
    if value is None or value == "":
        return None
    amount = Decimal(str(value))
    return amount * PER_MILLION
