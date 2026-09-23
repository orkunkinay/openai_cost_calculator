"""DeepInfra: official public model list API (``GET /models/list``).

DeepInfra quotes list prices in *cents per token* and everything else as
multipliers of the input price: cache reads, explicit 5m/1h cache writes, and
the flex/priority service tiers.  An active promotion appears as
``discount`` (a fraction off every price) with an optional
``discount_ends_at``; the effective (billed) price is list x (1 - discount),
which DeepInfra's model pages confirm.  Context-tiered models describe their
tiers only in the ``full`` prose field ("$1.2 in $6 out $0.24 cached <= 32K,
..."), which is parsed strictly or flagged.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ...catalog.model import ModelPricing, PriceSet, Source
from ..base import Fetcher, SourceResult, get_json
from .common import price_set, scale

URL = "https://api.deepinfra.com/models/list"
SOURCE = Source(
    id="deepinfra-models-api",
    kind="official_api",
    url=URL,
    description="DeepInfra public model list API",
)

#: cents per token -> USD per 1M tokens
_CENTS_PER_TOKEN = Decimal(1_000_000) / Decimal(100)
_TIER_SEGMENT = re.compile(
    r"^\$(?P<input>[\d.]+) in \$(?P<output>[\d.]+) out(?: \$(?P<cached>[\d.]+) cached)?"
    r"(?:\s*(?P<op><=|>)\s*(?P<k>\d+)K)?$"
)


_QUANTUM = Decimal("0.000001")


def _quantize(rates: Dict[str, Decimal]) -> Dict[str, Decimal]:
    """Round multiplier-derived prices to 1e-6 USD/1M (DeepInfra multipliers are floats)."""
    return {k: v.quantize(_QUANTUM).normalize() for k, v in rates.items()}


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    return Decimal(str(value))


def _context_tiers(full: str) -> Optional[List[Tuple[int, Dict[str, Decimal]]]]:
    """Parse "$a in $b out $c cached <= 32K, ..." into (min_input_tokens, rates)."""
    tiers: List[Tuple[int, Dict[str, Decimal]]] = []
    minimum = 0
    for segment in (s.strip() for s in full.split(",")):
        match = _TIER_SEGMENT.match(segment)
        if not match:
            return None
        rates = {"input": Decimal(match.group("input")), "output": Decimal(match.group("output"))}
        if match.group("cached"):
            rates["cached_input"] = Decimal(match.group("cached"))
        tiers.append((minimum, rates))
        if match.group("op") == "<=":
            minimum = int(match.group("k")) * 1000 + 1
        elif match.group("op") == ">":
            minimum = int(match.group("k")) * 1000 + 1
            tiers[-1] = (minimum, rates)
    return tiers


def _sets_for(model: str, pricing: Dict[str, Any], result: SourceResult) -> List[Optional[PriceSet]]:
    kind = pricing.get("type")
    input_cents = _decimal(pricing.get("cents_per_input_token"))
    if input_cents is None:
        return []
    output_cents = _decimal(pricing.get("cents_per_output_token")) if kind == "tokens" else None
    base: Dict[str, Decimal] = {"input": input_cents * _CENTS_PER_TOKEN}
    if output_cents is not None:
        base["output"] = output_cents * _CENTS_PER_TOKEN

    tiers: List[Tuple[int, Dict[str, Decimal]]] = [(0, base)]
    full = pricing.get("full")
    if isinstance(full, str) and ("<=" in full or ">" in full):
        parsed = _context_tiers(full)
        if parsed is None:
            result.issue(f"unrecognized tier description {full!r}", model)
            return []
        tiers = parsed  # the prose already states effective (discounted) prices
        discount = Decimal(0)
    else:
        discount = _decimal(pricing.get("discount")) or Decimal(0)

    cached = _decimal(pricing.get("rate_per_input_token_cached"))
    explicit = pricing.get("rate_per_explicit_cache_write_token") or {}
    tier_multipliers = {
        "flex": _decimal(pricing.get("rate_per_service_tier_flex")),
        "priority": _decimal(pricing.get("rate_per_service_tier_priority")),
    }
    ends = pricing.get("discount_ends_at")
    until = datetime.fromisoformat(str(ends).replace("Z", "+00:00")).date() if ends else None

    sets: List[Optional[PriceSet]] = []
    for minimum, rates in tiers:
        effective = scale(rates, 1 - discount)
        if cached is not None and "cached_input" not in effective:
            effective["cached_input"] = effective["input"] * cached
        if isinstance(explicit, dict):
            for key, dimension in (("5m", "cache_write"), ("1h", "cache_write_1h")):
                if key in explicit:
                    effective[dimension] = effective["input"] * Decimal(str(explicit[key]))
        variants = [("standard", effective)] + [
            (tier, scale(effective, multiplier)) for tier, multiplier in tier_multipliers.items() if multiplier
        ]
        for tier, tier_rates in variants:
            sets.append(
                price_set(_quantize(tier_rates), service_tier=tier, min_input_tokens=minimum, effective_until=until)
            )
            if until is not None and discount:
                list_rates = _quantize(scale(tier_rates, 1 / (1 - discount)))
                sets.append(price_set(list_rates, service_tier=tier, min_input_tokens=minimum, effective_from=until))
    return sets


def parse(payload: Any) -> SourceResult:
    result = SourceResult()
    if not isinstance(payload, list):
        result.issue("response is not a list of models; the API format may have changed")
        return result
    for item in payload:
        if not isinstance(item, dict) or item.get("private"):
            continue
        model = item.get("model_name")
        pricing = item.get("pricing")
        if not isinstance(model, str) or not isinstance(pricing, dict):
            continue
        if pricing.get("type") not in ("tokens", "input_tokens"):
            continue  # images, audio seconds, characters: outside the token catalog
        try:
            sets = [s for s in _sets_for(model, pricing, result) if s is not None]
        except (ArithmeticError, ValueError, TypeError) as exc:
            result.issue(f"unparseable pricing: {exc}", model)
            continue
        if not sets:
            continue
        vendor = model.split("/", 1)[0].lower() if "/" in model else None
        result.add(
            ModelPricing(
                id=model,
                prices=tuple(sets),
                source=SOURCE.id,
                vendor=vendor,
                canonical_id=model.lower(),
                status="deprecated" if item.get("deprecated") else "active",
            )
        )
    return result


class DeepInfraSource:
    provider = "deepinfra"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(get_json(fetcher, URL))
