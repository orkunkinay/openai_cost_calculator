"""Anthropic first-party token pricing for the Messages API proxy.

Prices come from the provider catalog (``data/pricing/anthropic.json``), which
the automated sync keeps in line with Anthropic's pricing documentation.  This
module adapts catalog entries to the :class:`AnthropicRate` shape the proxy
accounting uses.  Rates are USD per one million tokens as :class:`Decimal`.

Anthropic separates four billable token categories:

``input``
    Uncached input tokens (Anthropic's ``input_tokens`` field already excludes
    cache reads and cache writes, so no subtraction is required).
``cache_write_5m`` / ``cache_write_1h``
    Cache-creation writes, priced by their time-to-live.
``cache_read``
    Cache-read (hit) input tokens.  The multiplier differs by model, so the
    published price is used rather than a derived one.
``output``
    Generated output tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from ..catalog import PricingCatalog, PricingError, RequestConditions, select_price_set

MILLION = Decimal(1_000_000)

#: Anthropic's documented default for 1-hour cache writes (2x base input),
#: used only if a catalog entry lacks an explicit 1-hour price.
_CACHE_WRITE_1H_MULTIPLIER = Decimal("2")

_DATED_MODEL_RE = re.compile(r"^(?P<name>.+)-(?P<date>\d{8})$")


class AnthropicPricingError(ValueError):
    """Raised when Anthropic pricing data is invalid or cannot be resolved."""


@dataclass(frozen=True)
class AnthropicRate:
    """Per-million-token prices for one Anthropic pricing tier."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    min_input_tokens: int = 0

    def validate(self, *, model: str, date: str) -> None:
        for name in ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise AnthropicPricingError(f"{model} ({date}) rate {name!r} must be a Decimal")
            if not value.is_finite() or value < 0:
                raise AnthropicPricingError(f"{model} ({date}) rate {name!r} must be finite and non-negative")
        if (
            not isinstance(self.min_input_tokens, int)
            or isinstance(self.min_input_tokens, bool)
            or self.min_input_tokens < 0
        ):
            raise AnthropicPricingError(f"{model} ({date}) min_input_tokens must be a non-negative integer")


def _catalog(catalog: Optional[PricingCatalog]) -> PricingCatalog:
    from ..catalog import bundled_catalog

    return catalog or bundled_catalog()


def _to_rate(rates, min_input_tokens: int) -> AnthropicRate:
    base = rates["input"]
    return AnthropicRate(
        input=base,
        output=rates["output"],
        cache_read=rates.get("cached_input", base),
        cache_write_5m=rates.get("cache_write", base),
        cache_write_1h=rates.get("cache_write_1h", base * _CACHE_WRITE_1H_MULTIPLIER),
        min_input_tokens=min_input_tokens,
    )


def validate_anthropic_pricing(catalog: Optional[PricingCatalog] = None) -> int:
    """Validate the Anthropic catalog entries; returns the number of price tiers."""
    try:
        data = _catalog(catalog).get("anthropic")
    except PricingError as exc:
        raise AnthropicPricingError(str(exc)) from exc
    if data is None or not data.models:
        raise AnthropicPricingError("no Anthropic pricing data in the catalog")
    tiers = 0
    for model in data.models:
        for price in model.prices:
            if "input" not in price.rates or "output" not in price.rates:
                raise AnthropicPricingError(f"{model.id} has a price without input/output rates")
            _to_rate(price.rates, price.min_input_tokens).validate(model=model.id, date=str(data.verified_at))
            tiers += 1
    return tiers


def split_anthropic_model(model: str) -> tuple[str, str]:
    """Split an Anthropic model id into ``(name, request_date)``.

    Accepts undated aliases (``claude-opus-4-8``) and dated ids using
    Anthropic's ``-YYYYMMDD`` suffix (``claude-sonnet-4-5-20250929``).  When no
    date is embedded, today's UTC date is used.
    """
    if not isinstance(model, str) or not model:
        raise AnthropicPricingError("model must be a non-empty string")
    match = _DATED_MODEL_RE.match(model)
    if match:
        digits = match.group("date")
        try:
            parsed = datetime.strptime(digits, "%Y%m%d")
        except ValueError:
            parsed = None
        if parsed is not None:
            return match.group("name"), parsed.strftime("%Y-%m-%d")
    return model, datetime.now(timezone.utc).strftime("%Y-%m-%d")


def resolve_anthropic_rate(
    model: str, total_input_tokens: int, *, catalog: Optional[PricingCatalog] = None
) -> AnthropicRate:
    """Resolve the standard, globally routed rate for a model and request size.

    ``total_input_tokens`` should be the sum of uncached input, cache-read, and
    cache-creation tokens, matching how long-context tiers are evaluated.  A
    dated model id is billed at the model's current rate.
    """
    from ..api import resolve_model

    if not isinstance(total_input_tokens, int) or isinstance(total_input_tokens, bool):
        raise AnthropicPricingError("total_input_tokens must be an integer")
    if total_input_tokens < 0:
        raise AnthropicPricingError("total_input_tokens must be non-negative")
    try:
        entry = resolve_model("anthropic", model, catalog=_catalog(catalog)).model
        selection = select_price_set(
            entry,
            RequestConditions(preferences={"service_tier": ("standard",), "region": ("global",)}),
            total_input_tokens=total_input_tokens,
            on=datetime.now(timezone.utc).date(),
            provider="anthropic",
        )
        return _to_rate(selection.price_set.rates, selection.price_set.min_input_tokens)
    except (PricingError, KeyError) as exc:
        raise AnthropicPricingError(f"no Anthropic pricing for model {model!r}: {exc}") from exc
