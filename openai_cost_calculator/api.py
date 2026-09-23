"""Public, provider-agnostic pricing API.

    >>> from openai_cost_calculator import calculate_cost
    >>> cost = calculate_cost(
    ...     provider="aws-bedrock",
    ...     model="anthropic/claude-sonnet-4-5",
    ...     input_tokens=12_000,
    ...     output_tokens=800,
    ... )
    >>> cost.total, cost.conditions, cost.assumptions

Application code names a provider and a model; everything provider-specific
(identifier formats, regional/global endpoints, batch tiers, peak hours,
long-context thresholds) is resolved from the provider spec and the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from .catalog import (
    Cost,
    ModelPricing,
    PricingCatalog,
    PricingError,
    ProviderPricing,
    RequestConditions,
    Usage,
    bundled_catalog,
    price_usage,
    select_price_set,
)
from .catalog.errors import UnknownProviderError
from .providers import ProviderSpec, get_provider, iter_providers
from .usage import extract_usage

When = Union[datetime, date, None]


@dataclass(frozen=True)
class ResolvedModel:
    provider: ProviderSpec
    data: ProviderPricing
    model: ModelPricing
    hints_preferences: dict
    hints_notes: dict


def _catalog(catalog: Optional[PricingCatalog]) -> PricingCatalog:
    return catalog if catalog is not None else bundled_catalog()


def _provider_data(spec: ProviderSpec, catalog: PricingCatalog) -> ProviderPricing:
    data = catalog.get(spec.id)
    if data is None:
        raise UnknownProviderError(spec.id, catalog.provider_ids())
    return data


def resolve_model(provider: str, model: str, *, catalog: Optional[PricingCatalog] = None) -> ResolvedModel:
    """Resolve ``(provider, model)`` to the catalog offering that prices it."""
    if not isinstance(model, str) or not model.strip():
        raise PricingError("model must be a non-empty string")
    spec = get_provider(provider)
    cat = _catalog(catalog)
    data = _provider_data(spec, cat)
    hints = spec.hints(model)
    found = cat.index(spec.id).resolve(model, hints.candidates)
    return ResolvedModel(spec, data, found, dict(hints.preferences), dict(hints.notes))


def get_model_pricing(provider: str, model: str, *, catalog: Optional[PricingCatalog] = None) -> ModelPricing:
    """Return the full pricing entry (all tiers, regions, dates) for a model."""
    return resolve_model(provider, model, catalog=catalog).model


def _split_when(at: When) -> Tuple[date, Optional[datetime]]:
    if at is None:
        now = datetime.now(timezone.utc)
        return now.date(), now
    if isinstance(at, datetime):
        return (at.astimezone(timezone.utc) if at.tzinfo else at).date(), at
    if isinstance(at, date):
        return at, None
    raise PricingError(f"at must be a datetime or date, not {type(at).__name__}")


def calculate_cost(
    provider: str,
    model: str,
    *,
    usage: Optional[Usage] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    service_tier: Optional[str] = None,
    region: Optional[str] = None,
    period: Optional[str] = None,
    at: When = None,
    catalog: Optional[PricingCatalog] = None,
) -> Cost:
    """Price one request.

    Token counts are **disjoint**: ``input_tokens`` excludes cache reads
    (``cached_input_tokens``) and cache writes (``cache_write_tokens``, or
    ``cache_write_1h_tokens`` for Anthropic's 1-hour cache).  Pass a
    :class:`Usage` for audio/image tokens or tool calls.

    ``service_tier`` (``standard``/``batch``/``flex``/``priority``...),
    ``region`` and ``period`` select conditional prices; when omitted the
    provider's defaults apply and are reported in :attr:`Cost.assumptions`.
    ``at`` prices the request as of a date/time (scheduled price changes,
    DeepSeek peak hours); it defaults to now.
    """
    token_kwargs = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_write_1h_tokens": cache_write_1h_tokens,
    }
    if usage is None:
        usage = Usage(**token_kwargs)
    elif any(token_kwargs.values()):
        raise PricingError("pass either usage=... or token counts, not both")

    resolved = resolve_model(provider, model, catalog=catalog)
    spec, data, entry = resolved.provider, resolved.data, resolved.model
    priced_on, moment = _split_when(at)

    preferences = {key: tuple(values) for key, values in spec.default_preferences.items()}
    preferences["service_tier"] = ("standard",)
    preferences["region"] = resolved.hints_preferences.get("region", spec.default_regions)
    notes = dict(resolved.hints_notes)
    if spec.time_conditions is not None and moment is not None:
        derived, time_notes = spec.time_conditions(moment)
        preferences.update(derived)
        notes.update(time_notes)

    if service_tier is not None:
        service_tier = spec.service_tier_aliases.get(service_tier, service_tier)
    explicit = set()
    for key, value in (("service_tier", service_tier), ("region", region), ("period", period)):
        if value is not None:
            preferences[key] = spec.regions_for(value) if key == "region" else (value,)
            explicit.add(key)
            notes.pop(key, None)

    selection = select_price_set(
        entry,
        RequestConditions(preferences=preferences, explicit=frozenset(explicit)),
        total_input_tokens=usage.total_input_tokens,
        on=priced_on,
        provider=spec.id,
    )
    total, items = price_usage(usage, selection.price_set, where=f"{spec.id}/{entry.id}")

    # A provider note ("us. profile billed at the us-east-1 rate") replaces the
    # generic "assumed region=..." message whenever that condition set the price.
    used_notes = {key: note for key, note in notes.items() if key in selection.price_set.conditions}
    assumptions = [a for a in selection.assumptions if not any(a.startswith(f"assumed {k}=") for k in used_notes)]
    assumptions += used_notes.values()
    source = data.source(entry.source)
    return Cost(
        provider=spec.id,
        model=model,
        resolved_model=entry.id,
        total=total,
        items=items,
        currency=data.currency,
        canonical_model=entry.canonical_id,
        conditions=selection.conditions,
        min_input_tokens=selection.price_set.min_input_tokens,
        priced_on=priced_on,
        source_url=source.url if source else None,
        verified_at=data.verified_at,
        assumptions=tuple(assumptions),
    )


def list_providers() -> Tuple[ProviderSpec, ...]:
    return tuple(iter_providers())


def list_models(
    provider: Optional[str] = None, *, catalog: Optional[PricingCatalog] = None
) -> List[Tuple[str, ModelPricing]]:
    """``(provider_id, model)`` pairs, optionally for one provider."""
    cat = _catalog(catalog)
    ids: Iterable[str] = [get_provider(provider).id] if provider else cat.provider_ids()
    pairs: List[Tuple[str, ModelPricing]] = []
    for pid in ids:
        data = cat.get(pid)
        if data is not None:
            pairs.extend((pid, model) for model in data.models)
    return pairs


def compare_costs(
    model: str,
    *,
    usage: Usage,
    providers: Optional[Sequence[str]] = None,
    catalog: Optional[PricingCatalog] = None,
    at: When = None,
) -> List[Cost]:
    """Price the same usage for ``model`` on every provider that offers it.

    ``model`` is typically a canonical id such as ``"anthropic/claude-sonnet-4-5"``
    or ``"meta-llama/llama-3.3-70b-instruct"``.  Providers that do not offer the
    model (or cannot price this usage) are skipped.  Results are cheapest first.
    """
    cat = _catalog(catalog)
    names = providers if providers is not None else cat.provider_ids()
    results: List[Cost] = []
    for name in names:
        try:
            results.append(calculate_cost(name, model, usage=usage, catalog=cat, at=at))
        except PricingError:
            continue
    return sorted(results, key=lambda cost: cost.total)


def estimate_response_cost(
    response: Any,
    *,
    provider: str,
    model: Optional[str] = None,
    service_tier: Optional[str] = None,
    region: Optional[str] = None,
    period: Optional[str] = None,
    at: When = None,
    catalog: Optional[PricingCatalog] = None,
) -> Cost:
    """Price a provider response (SDK object or JSON dict) in any supported usage format.

    ``provider`` is the *billing* provider: the same Anthropic-format response
    costs differently on ``anthropic``, ``bedrock`` and ``vertex``.  ``model``
    overrides the response's model field (useful when an Azure deployment name
    or a proxy hides the underlying model).
    """
    extracted = extract_usage(response)
    model_id = model or extracted.model
    if not model_id:
        raise PricingError("response does not name its model; pass model=...")
    return calculate_cost(
        provider,
        model_id,
        usage=extracted.usage,
        service_tier=service_tier,
        region=region,
        period=period,
        at=at,
        catalog=catalog,
    )
