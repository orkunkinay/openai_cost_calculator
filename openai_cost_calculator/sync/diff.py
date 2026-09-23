"""Compare fetched pricing with the checked-in catalog."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple

from ..catalog.io import format_decimal
from ..catalog.model import ModelPricing, PriceSet

PriceKey = Tuple[Tuple[Tuple[str, str], ...], int, Optional[date], Optional[date]]


def price_key(price_set: PriceSet) -> PriceKey:
    return (
        price_set.condition_key,
        price_set.min_input_tokens,
        price_set.effective_from,
        price_set.effective_until,
    )


def describe_price_key(key: PriceKey) -> str:
    conditions, minimum, start, end = key
    parts = [f"{k}={v}" for k, v in conditions] or ["standard"]
    if minimum:
        parts.append(f">= {minimum:,} input tokens")
    if start:
        parts.append(f"from {start.isoformat()}")
    if end:
        parts.append(f"until {end.isoformat()}")
    return ", ".join(parts)


@dataclass(frozen=True)
class RateChange:
    price: str
    dimension: str
    old: Optional[Decimal]
    new: Optional[Decimal]

    @property
    def ratio(self) -> Optional[Decimal]:
        if self.old is None or self.new is None or self.old == 0:
            return None
        return self.new / self.old

    def __str__(self) -> str:
        render = lambda v: "(none)" if v is None else format_decimal(v)  # noqa: E731
        return f"{self.price}: {self.dimension} {render(self.old)} -> {render(self.new)}"


@dataclass(frozen=True)
class ModelDiff:
    model_id: str
    kind: str  # "added" | "removed" | "changed" | "unchanged"
    old: Optional[ModelPricing]
    new: Optional[ModelPricing]
    rate_changes: Tuple[RateChange, ...] = ()
    metadata_changes: Tuple[str, ...] = ()


def _index(prices: Iterable[PriceSet]) -> Dict[PriceKey, PriceSet]:
    return {price_key(p): p for p in prices}


def compare_models(old: ModelPricing, new: ModelPricing) -> ModelDiff:
    rate_changes: List[RateChange] = []
    old_sets, new_sets = _index(old.prices), _index(new.prices)
    for key in sorted(set(old_sets) | set(new_sets), key=lambda k: describe_price_key(k)):
        label = describe_price_key(key)
        old_rates = old_sets[key].rates if key in old_sets else {}
        new_rates = new_sets[key].rates if key in new_sets else {}
        for dimension in sorted(set(old_rates) | set(new_rates)):
            before, after = old_rates.get(dimension), new_rates.get(dimension)
            if before != after:
                rate_changes.append(RateChange(label, dimension, before, after))
    metadata = []
    for attribute in ("vendor", "canonical_id", "display_name"):
        before, after = getattr(old, attribute), getattr(new, attribute)
        if before != after:
            metadata.append(f"{attribute}: {before!r} -> {after!r}")
    if set(old.aliases) != set(new.aliases):
        metadata.append(f"aliases: {sorted(old.aliases)} -> {sorted(new.aliases)}")
    kind = "changed" if rate_changes or metadata else "unchanged"
    return ModelDiff(old.id, kind, old, new, tuple(rate_changes), tuple(metadata))


def diff_models(current: Iterable[ModelPricing], fetched: Iterable[ModelPricing]) -> List[ModelDiff]:
    old = {m.id: m for m in current}
    new = {m.id: m for m in fetched}
    diffs: List[ModelDiff] = []
    for model_id in sorted(set(old) | set(new)):
        if model_id not in old:
            diffs.append(ModelDiff(model_id, "added", None, new[model_id]))
        elif model_id not in new:
            diffs.append(ModelDiff(model_id, "removed", old[model_id], None))
        else:
            diffs.append(compare_models(old[model_id], new[model_id]))
    return diffs
