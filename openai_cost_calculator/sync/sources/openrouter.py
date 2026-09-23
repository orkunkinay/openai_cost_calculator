"""OpenRouter: official public models API (``GET /api/v1/models``).

Prices are USD per token (strings).  ``overrides`` with ``min_prompt_tokens``
are long-context tiers; OpenRouter's thresholds are inclusive (OpenAI's 272K
tier is published as ``min_prompt_tokens: 272000``), matching the catalog's
``min_input_tokens`` semantics.

Deliberately not mapped: ``image``/``image_output``/``audio_output`` (units
are not documented precisely enough to price safely), overrides bounded by
``utc_start``/``utc_end`` (time-of-day promotions; the base price is kept and
the override is noted), and negative prices, which OpenRouter uses for
routers whose price depends on the model they pick.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Mapping, Optional

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult, get_json
from .common import per_token_to_per_million, price_set

URL = "https://openrouter.ai/api/v1/models"
SOURCE = Source(
    id="openrouter-models-api",
    kind="official_api",
    url=URL,
    description="OpenRouter public models API",
)

_PER_TOKEN = {
    "prompt": "input",
    "completion": "output",
    "internal_reasoning": "reasoning",
    "input_cache_read": "cached_input",
    "input_cache_write": "cache_write",
    "input_cache_write_1h": "cache_write_1h",
    "audio": "input_audio",
    "input_audio_cache": "cached_input_audio",
}
_PER_UNIT = {"web_search": "web_search", "request": "request"}


def _rates(pricing: Mapping[str, Any]) -> Dict[str, Optional[Decimal]]:
    rates: Dict[str, Optional[Decimal]] = {}
    for key, dimension in _PER_TOKEN.items():
        rates[dimension] = per_token_to_per_million(pricing.get(key))
    for key, dimension in _PER_UNIT.items():
        value = pricing.get(key)
        rates[dimension] = Decimal(str(value)) if value not in (None, "") else None
    # Zero per-request / per-search fees carry no information; keep token zeros
    # (free models) but drop zero unit fees to keep the catalog readable.
    for dimension in _PER_UNIT.values():
        if rates.get(dimension) == 0:
            rates[dimension] = None
    if rates.get("reasoning") is not None and rates["reasoning"] == rates.get("output"):
        rates["reasoning"] = None  # identical to output: nothing to record
    return rates


def parse(payload: Any) -> SourceResult:
    result = SourceResult()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        result.issue("response has no 'data' list; the API format may have changed")
        return result
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        pricing = item.get("pricing") if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not isinstance(pricing, dict):
            result.issue(f"malformed model entry: {str(item)[:120]}")
            continue
        try:
            base = _rates(pricing)
        except (ArithmeticError, ValueError) as exc:
            result.issue(f"unparseable price: {exc}", model_id)
            continue
        if any(v is not None and v < 0 for v in base.values()):
            continue  # variable-price router (e.g. openrouter/auto)
        sets = [price_set(base)]
        for override in pricing.get("overrides") or []:
            if "utc_start" in override or "utc_end" in override:
                result.note("time-of-day price override not modelled; base price kept", model_id)
                continue
            minimum = override.get("min_prompt_tokens")
            if not isinstance(minimum, int) or minimum <= 0:
                result.issue(f"unrecognized price override {override}", model_id)
                continue
            merged = {**{k: v for k, v in pricing.items() if k != "overrides"}, **override}
            sets.append(price_set(_rates(merged), min_input_tokens=minimum))
        vendor = model_id.split("/", 1)[0] if "/" in model_id else None
        slug = item.get("canonical_slug")
        # Variants (":batch", ":free") share their base model's slug; only the
        # base id may claim it, or two entries would own one identifier.
        aliases = (slug,) if isinstance(slug, str) and slug and slug != model_id and ":" not in model_id else ()
        result.add(
            ModelPricing(
                id=model_id,
                prices=tuple(s for s in sets if s is not None),
                source=SOURCE.id,
                vendor=vendor,
                canonical_id=model_id.split(":", 1)[0],
                display_name=item.get("name"),
                aliases=aliases,
            )
        )
    return result


class OpenRouterSource:
    provider = "openrouter"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(get_json(fetcher, URL))
