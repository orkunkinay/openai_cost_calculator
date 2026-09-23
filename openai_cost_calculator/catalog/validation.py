"""Structural and sanity validation of pricing data.

Validation runs both when bundled data is loaded and before the automated
updater accepts fetched prices, so a malformed upstream parse can never reach
the checked-in catalog.  Structural errors raise
:class:`~.errors.CatalogValidationError`; :func:`sanity_warnings` reports
values that are *valid* but implausible, for human review.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Set, Tuple

from .dimensions import DIMENSIONS
from .errors import CatalogValidationError
from .model import CONDITION_KEYS, MODEL_STATUSES, SOURCE_KINDS, ModelPricing, PriceSet, ProviderPricing

#: No per-1M-token price in any researched catalog exceeds this; a larger value
#: almost certainly means a unit-conversion error (per-token vs per-1M).
MAX_PLAUSIBLE_TOKEN_PRICE = Decimal("2000")


def validate_price_set(price_set: PriceSet, *, where: str) -> None:
    if not price_set.rates:
        raise CatalogValidationError(f"{where}: price set has no rates")
    for dimension, rate in price_set.rates.items():
        if dimension not in DIMENSIONS:
            raise CatalogValidationError(f"{where}: unknown dimension {dimension!r}")
        if not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0:
            raise CatalogValidationError(
                f"{where}: rate {dimension!r} must be a finite non-negative Decimal, got {rate!r}"
            )
    for key, value in price_set.conditions.items():
        if key not in CONDITION_KEYS:
            raise CatalogValidationError(
                f"{where}: unknown condition {key!r}; allowed: {', '.join(CONDITION_KEYS)}"
            )
        if not isinstance(value, str) or not value:
            raise CatalogValidationError(f"{where}: condition {key!r} must be a non-empty string")
    if (
        not isinstance(price_set.min_input_tokens, int)
        or isinstance(price_set.min_input_tokens, bool)
        or price_set.min_input_tokens < 0
    ):
        raise CatalogValidationError(f"{where}: min_input_tokens must be a non-negative integer")
    start, end = price_set.effective_from, price_set.effective_until
    if start is not None and end is not None and end <= start:
        raise CatalogValidationError(f"{where}: effective_until must be after effective_from")


def _windows_overlap(a: PriceSet, b: PriceSet) -> bool:
    a_start, a_end = a.effective_from, a.effective_until
    b_start, b_end = b.effective_from, b.effective_until
    starts_before_b_ends = b_end is None or a_start is None or a_start < b_end
    b_starts_before_a_ends = a_end is None or b_start is None or b_start < a_end
    return starts_before_b_ends and b_starts_before_a_ends


def validate_model(model: ModelPricing, *, provider: str) -> None:
    where = f"{provider}/{model.id}"
    if not model.id or model.id != model.id.strip():
        raise CatalogValidationError(f"{provider}: model id {model.id!r} is empty or padded")
    if model.status not in MODEL_STATUSES:
        raise CatalogValidationError(f"{where}: status must be one of {MODEL_STATUSES}")
    if not model.prices:
        raise CatalogValidationError(f"{where}: model has no price sets")
    by_key: Dict[Tuple[Tuple[str, str], ...], List[PriceSet]] = defaultdict(list)
    for index, price_set in enumerate(model.prices):
        validate_price_set(price_set, where=f"{where} price[{index}]")
        by_key[price_set.condition_key].append(price_set)
    for condition_key, sets in by_key.items():
        label = dict(condition_key) or "default conditions"
        if not any(s.min_input_tokens == 0 for s in sets):
            raise CatalogValidationError(f"{where}: {label} has no base tier (min_input_tokens=0)")
        for i, first in enumerate(sets):
            for second in sets[i + 1 :]:
                if first.min_input_tokens == second.min_input_tokens and _windows_overlap(first, second):
                    raise CatalogValidationError(
                        f"{where}: {label} has overlapping price sets at "
                        f"min_input_tokens={first.min_input_tokens}"
                    )


def validate_provider(data: ProviderPricing) -> None:
    source_ids: Set[str] = set()
    for source in data.sources:
        if source.kind not in SOURCE_KINDS:
            raise CatalogValidationError(
                f"{data.provider}: source {source.id!r} kind must be one of {SOURCE_KINDS}"
            )
        if source.id in source_ids:
            raise CatalogValidationError(f"{data.provider}: duplicate source id {source.id!r}")
        source_ids.add(source.id)

    seen: Dict[str, str] = {}
    for model in data.models:
        validate_model(model, provider=data.provider)
        if model.source not in source_ids:
            raise CatalogValidationError(
                f"{data.provider}/{model.id}: unknown source {model.source!r}"
            )
        keys = [identifier.lower() for identifier in model.identifiers]
        if len(set(keys)) != len(keys):
            raise CatalogValidationError(f"{data.provider}/{model.id}: duplicate alias")
        for identifier, key in zip(model.identifiers, keys):
            owner = seen.get(key)
            if owner is not None:
                raise CatalogValidationError(
                    f"{data.provider}: identifier {identifier!r} is used by both "
                    f"{owner!r} and {model.id!r}"
                )
            seen[key] = model.id


def sanity_warnings(model: ModelPricing, *, provider: str) -> List[str]:
    """Return human-readable reasons a valid model's prices look implausible."""
    warnings: List[str] = []
    where = f"{provider}/{model.id}"
    for price_set in model.prices:
        rates = price_set.rates
        for dimension, rate in rates.items():
            if DIMENSIONS[dimension].unit_label == "1M tokens" and rate > MAX_PLAUSIBLE_TOKEN_PRICE:
                warnings.append(f"{where}: {dimension} rate {rate} per 1M tokens is implausibly high")
        base = rates.get("input")
        cached = rates.get("cached_input")
        if base is not None and cached is not None and cached > base:
            warnings.append(f"{where}: cached_input rate {cached} exceeds input rate {base}")
    return warnings
