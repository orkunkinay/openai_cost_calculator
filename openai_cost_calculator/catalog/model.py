"""Domain model for provider pricing data.

The model deliberately separates four concepts that the original OpenAI-only
schema conflated:

* the **billing provider** (:class:`ProviderPricing`) - who invoices you;
* the **offering** (:class:`ModelPricing`) - a model as sold by that
  provider, under the provider's own identifier, with an optional
  cross-provider ``canonical_id`` and the ``vendor`` who built it;
* the **price set** (:class:`PriceSet`) - rates that apply under specific
  conditions (service tier, region, time-of-day period), above an input-size
  threshold, and within an effective-date window;
* the **provenance** (:class:`Source`) - where the numbers came from.

Rates are ``Decimal`` values quoted per :data:`~.dimensions.Dimension.unit_size`
(per 1M tokens for token dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, Mapping, Optional, Tuple

#: Condition keys a price set may be qualified by.  Each is a real variation
#: observed in provider pricing: service tiers (batch/flex/priority), deployment
#: regions or scopes, and DeepSeek-style peak/off-peak periods.
CONDITION_KEYS: Tuple[str, ...] = ("service_tier", "region", "period")

#: Conditions whose *absence* from a price set means a specific value rather
#: than "any value".  A set without a service tier is the standard tier: it must
#: not silently price a batch or flex request.  A set without a region, by
#: contrast, genuinely applies in every region.
IMPLICIT_CONDITIONS: Mapping[str, str] = {"service_tier": "standard"}

SOURCE_KINDS: Tuple[str, ...] = ("official_api", "official_docs", "aggregator", "manual")

MODEL_STATUSES: Tuple[str, ...] = ("active", "deprecated", "unverified")


@dataclass(frozen=True)
class Source:
    """Provenance of pricing data."""

    id: str
    kind: str
    url: str
    description: str = ""


@dataclass(frozen=True)
class PriceSet:
    rates: Mapping[str, Decimal]
    conditions: Mapping[str, str] = field(default_factory=dict)
    #: The set applies when the request's total input tokens are at least this
    #: many (long-context tiers).  ``0`` is the base tier.
    min_input_tokens: int = 0
    effective_from: Optional[date] = None
    #: Exclusive end date.
    effective_until: Optional[date] = None

    def is_effective(self, on: date) -> bool:
        if self.effective_from is not None and on < self.effective_from:
            return False
        if self.effective_until is not None and on >= self.effective_until:
            return False
        return True

    @property
    def condition_key(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self.conditions.items()))

    def condition(self, key: str) -> Optional[str]:
        """The set's value for ``key``; ``None`` means it applies to any value."""
        return self.conditions.get(key, IMPLICIT_CONDITIONS.get(key))


@dataclass(frozen=True)
class ModelPricing:
    id: str
    prices: Tuple[PriceSet, ...]
    source: str
    vendor: Optional[str] = None
    canonical_id: Optional[str] = None
    display_name: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    status: str = "active"

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return (self.id, *self.aliases)


@dataclass(frozen=True)
class ProviderPricing:
    """All pricing for one billing provider (one generated data file)."""

    provider: str
    sources: Tuple[Source, ...]
    models: Tuple[ModelPricing, ...]
    currency: str = "USD"
    #: Date the data was last confirmed against its sources.
    verified_at: Optional[date] = None

    def source(self, source_id: str) -> Optional[Source]:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def models_by_id(self) -> Dict[str, ModelPricing]:
        return {model.id: model for model in self.models}
