"""Choose the price set that applies to one request.

Selection is provider-agnostic.  A request supplies, for each condition key,
an ordered tuple of acceptable values (the caller's explicit choice, or the
provider's defaults with fallbacks).  A price set matches when each of its
conditions is acceptable; a set that does not mention a key applies to every
value of it (except ``service_tier``, whose absence means ``standard``).  Among matching condition groups the one whose values rank
earliest wins; within that group the long-context tier with the greatest
``min_input_tokens`` not exceeding the request's input size applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from .errors import PricingUnavailableError
from .model import CONDITION_KEYS, IMPLICIT_CONDITIONS, ModelPricing, PriceSet

ConditionKey = Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class RequestConditions:
    #: key -> acceptable values, most preferred first.
    preferences: Mapping[str, Tuple[str, ...]]
    #: keys whose preferences the caller chose explicitly (not defaults).
    explicit: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class Selection:
    price_set: PriceSet
    #: Condition values in effect (explicit, defaulted, or wildcard).
    conditions: Mapping[str, str]
    assumptions: Tuple[str, ...]


def _rank(condition_key: ConditionKey, request: RequestConditions) -> Optional[Tuple[int, ...]]:
    conditions = {**IMPLICIT_CONDITIONS, **dict(condition_key)}
    ranks: List[int] = []
    for key in CONDITION_KEYS:
        if key not in conditions:
            ranks.append(0)  # wildcard: applies to every value
            continue
        acceptable = request.preferences.get(key, ())
        if conditions[key] not in acceptable:
            return None
        ranks.append(acceptable.index(conditions[key]))
    return tuple(ranks)


def _describe(condition_keys: List[ConditionKey]) -> str:
    rendered = sorted(
        ", ".join(f"{k}={v}" for k, v in key) or "default conditions" for key in condition_keys
    )
    return "; ".join(rendered)


def select_price_set(
    model: ModelPricing,
    request: RequestConditions,
    *,
    total_input_tokens: int,
    on: date,
    provider: str,
) -> Selection:
    effective = [s for s in model.prices if s.is_effective(on)]
    if not effective:
        raise PricingUnavailableError(
            f"{provider}/{model.id} has no price in effect on {on.isoformat()}"
        )

    groups: Dict[ConditionKey, List[PriceSet]] = {}
    for price_set in effective:
        groups.setdefault(price_set.condition_key, []).append(price_set)

    ranked = []
    for condition_key in groups:
        rank = _rank(condition_key, request)
        if rank is not None:
            # More specific groups win ties against wildcard groups.
            ranked.append((rank, -len(condition_key), condition_key))
    if not ranked:
        wanted = ", ".join(
            f"{k}={'|'.join(v)}" for k, v in sorted(request.preferences.items()) if v
        )
        raise PricingUnavailableError(
            f"{provider}/{model.id} has no price for {wanted}; "
            f"available: {_describe(list(groups))}"
        )
    ranked.sort()
    best_rank, best_specificity, best_key = ranked[0]
    ties = [key for rank, spec, key in ranked if rank == best_rank and spec == best_specificity]
    if len(ties) > 1:
        raise PricingUnavailableError(
            f"{provider}/{model.id} has ambiguous prices for the request: {_describe(ties)}"
        )

    tiers = sorted(groups[best_key], key=lambda s: s.min_input_tokens)
    chosen = tiers[0]
    for tier in tiers:
        if tier.min_input_tokens <= total_input_tokens:
            chosen = tier

    in_effect = {**IMPLICIT_CONDITIONS, **dict(best_key)}
    assumptions = []
    varying = _varying_keys(effective)
    for key in sorted(varying):
        if key in request.explicit:
            continue
        value = in_effect.get(key)
        if value is not None:
            assumptions.append(f"assumed {key}={value} (pass {key}=... to price another option)")
    return Selection(price_set=chosen, conditions=in_effect, assumptions=tuple(assumptions))


def _varying_keys(price_sets: List[PriceSet]) -> FrozenSet[str]:
    values: Dict[str, set] = {}
    for price_set in price_sets:
        for key in CONDITION_KEYS:
            values.setdefault(key, set()).add(price_set.condition(key))
    return frozenset(key for key, seen in values.items() if len(seen) > 1)
