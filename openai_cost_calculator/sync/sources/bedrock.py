"""Amazon Bedrock: official AWS Price List API (bulk offer files).

Two offers are needed:

* ``AmazonBedrockFoundationModels`` - Anthropic models, sold through AWS
  Marketplace.  Products are named "Claude Sonnet 4.5 (Amazon Bedrock
  Edition)"; usage types encode the dimension, batch and global-endpoint
  pricing (``InputTokenCount_Global_Batch``, ``cache_write_tokens_1h_global_standard``).
* ``AmazonBedrock`` - Amazon, Meta, Mistral, DeepSeek, OpenAI open-weight and
  other models.  Usage types look like ``USE1-deepseek.v3.2-mantle-input-tokens-flex``;
  the code before the dimension is the Bedrock model id when it is
  vendor-qualified (``deepseek.v3.2``) and is qualified with the provider
  otherwise (``NovaPro`` -> ``amazon.novapro``, matching ``amazon.nova-pro-v1:0``
  once punctuation is ignored).

Prices are per region; each configured region contributes regional prices and
global-endpoint prices are recorded as ``region="global"``.  Customization,
provisioned throughput, reserved capacity, latency-optimized and non-token
(image/video/second) meters are outside the catalog and skipped.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult, get_json
from ..text import slugify
from .common import price_set

OFFER_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{offer}/current/{region}/index.json"
REGIONS: Tuple[str, ...] = ("us-east-1", "us-west-2", "eu-central-1")
SOURCE = Source(
    id="aws-price-list-api",
    kind="official_api",
    url="https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json",
    description="AWS Price List API: AmazonBedrock and AmazonBedrockFoundationModels offers "
    f"({', '.join(REGIONS)} and global endpoints)",
)

_VENDORS = {
    "amazon": "amazon", "meta": "meta", "mistral ai": "mistral", "mistral": "mistral", "deepseek": "deepseek",
    "openai": "openai", "google": "google", "moonshot ai": "moonshotai", "qwen": "qwen", "minimax": "minimax",
    "nvidia": "nvidia", "writer": "writer", "z ai": "zai", "xai": "xai", "cohere": "cohere", "ai21 labs": "ai21",
}
#: Usage-type codes without a vendor or ``provider`` attribute.
_CODE_VENDORS = (("nova", "amazon"), ("titan", "amazon"))
_QUALIFIED_CODE = re.compile(r"^[a-z][a-z0-9-]*\.[a-z]")
_SKIP_WORDS = {"custom", "customization", "provisionedthroughput", "provisioned", "reserved", "latencyoptimized",
               "video", "second", "image", "created", "storage", "training", "search", "units", "embed"}


def _tier_and_scope(words: Iterable[str]) -> Tuple[str, bool]:
    tier, is_global = "standard", False
    for word in words:
        if word == "global":
            is_global = True
        elif word in {"batch", "flex", "priority"}:
            tier = word
    return tier, is_global


def _dimension(words: List[str]) -> Optional[str]:
    if "cache" in words and "read" in words:
        return "cached_input"
    if "cache" in words and "write" in words:
        return "cache_write_1h" if "1h" in words else "cache_write"
    if "audio" in words and "input" in words:
        return "input_audio"
    if "input" in words and not any(w in words for w in ("image", "video")):
        return "input"
    if "output" in words and not any(w in words for w in ("image", "video")):
        return "output"
    return None


# --------------------------------------------------------------------------- Anthropic offer

_FM_USAGE = re.compile(r"^[A-Z0-9]+-MP:[A-Z0-9]+_(?P<code>.+)-Units$")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _fm_words(code: str) -> List[str]:
    # InputTokenCount_Global_Batch -> input token count global batch; CacheWrite1h... -> cache write 1h
    code = code.replace("1h", "_1h_")
    return [w for w in re.split(r"[_\s]+", _CAMEL.sub(" ", code).lower().replace(" ", "_")) if w]


def parse_foundation_models(offer: Dict[str, Any], region: str, result: SourceResult) -> Dict[Tuple, Decimal]:
    prices: Dict[Tuple, Decimal] = {}
    terms = offer.get("terms", {}).get("OnDemand", {})
    for sku, product in offer.get("products", {}).items():
        attributes = product.get("attributes", {})
        name = str(attributes.get("servicename", ""))
        match = re.match(r"^(?P<name>Claude .+?) \(Amazon Bedrock Edition\)$", name)
        usage = _FM_USAGE.match(str(attributes.get("usagetype", "")))
        if not match or not usage:
            continue
        words = _fm_words(usage.group("code"))
        if _SKIP_WORDS.intersection(words) or "million" in words:
            continue
        dimension = _dimension(words)
        if dimension is None:
            continue
        tier, is_global = _tier_and_scope(words)
        for term in terms.get(sku, {}).values():
            for dimension_price in term.get("priceDimensions", {}).values():
                if dimension_price.get("unit") != "1M tokens":
                    continue
                model = "anthropic." + slugify(match.group("name"))
                key = (model, "global" if is_global else region, tier, dimension, match.group("name"))
                prices[key] = Decimal(dimension_price["pricePerUnit"]["USD"])
    return prices


# --------------------------------------------------------------------------- general offer

_TAIL = re.compile(r"-(?:mantle-)?(?=(?:input|output|cache|text|prompt)-)")


def _model_code(usagetype: str) -> Optional[str]:
    body = usagetype.split("-", 1)[1] if "-" in usagetype else usagetype
    parts = _TAIL.split(body, maxsplit=1)
    return parts[0] if len(parts) == 2 else None


def parse_general(offer: Dict[str, Any], region: str, result: SourceResult) -> Dict[Tuple, Decimal]:
    prices: Dict[Tuple, Decimal] = {}
    terms = offer.get("terms", {}).get("OnDemand", {})
    for sku, product in offer.get("products", {}).items():
        attributes = product.get("attributes", {})
        usagetype = str(attributes.get("usagetype", ""))
        feature = str(attributes.get("feature") or "").lower()
        if "provisioned" in feature or "custom" in feature:
            continue
        code = _model_code(usagetype)
        if code is None:
            continue
        tail = usagetype[usagetype.index(code) + len(code):].lower()
        words = [w for w in re.split(r"[-\s]+", tail) if w]
        words += re.split(r"[\s-]+", str(attributes.get("inferenceType") or "").lower())
        words += re.split(r"[\s-]+", str(attributes.get("service_tier") or "").lower())
        if "batch" in feature:
            words.append("batch")
        if _SKIP_WORDS.intersection(words) or "model" in words:
            continue
        dimension = _dimension(words)
        if dimension is None:
            continue
        tier, is_global = _tier_and_scope(words)
        vendor = _VENDORS.get(str(attributes.get("provider") or "").lower())
        if vendor is None:
            vendor = next((v for prefix, v in _CODE_VENDORS if code.lower().startswith(prefix)), None)
        if _QUALIFIED_CODE.match(code.lower()):
            model = code.lower()
        elif vendor:
            model = f"{vendor}.{code.lower()}"
        else:
            continue  # cannot qualify the model id safely
        display = str(attributes.get("model") or code)
        for term in terms.get(sku, {}).values():
            for dimension_price in term.get("priceDimensions", {}).values():
                unit = str(dimension_price.get("unit", "")).lower()
                usd = Decimal(dimension_price["pricePerUnit"]["USD"])
                if unit in {"1m tokens", "million tokens"}:
                    per_million = usd
                elif unit in {"1k tokens", "thousand tokens"}:
                    per_million = usd * 1000
                else:
                    continue
                prices[(model, "global" if is_global else region, tier, dimension, display)] = per_million
    return prices


def build(prices: Dict[Tuple, Decimal], result: SourceResult) -> None:
    grouped: Dict[str, Dict[Tuple[str, str], Dict[str, Decimal]]] = {}
    displays: Dict[str, str] = {}
    for (model, region, tier, dimension, display), amount in prices.items():
        grouped.setdefault(model, {}).setdefault((region, tier), {})[dimension] = amount
        displays.setdefault(model, display)
    for model, groups in sorted(grouped.items()):
        sets = [
            price_set(rates, service_tier=tier, region=region)
            for (region, tier), rates in sorted(groups.items())
            if "input" in rates and "output" in rates
        ]
        sets = [s for s in sets if s is not None]
        if not sets:
            continue
        vendor, _, rest = model.partition(".")
        result.add(
            ModelPricing(
                id=model,
                prices=tuple(sets),
                source=SOURCE.id,
                vendor=vendor,
                canonical_id=f"{vendor}/{rest}",
                display_name=displays[model],
            )
        )


class BedrockSource:
    provider = "bedrock"
    source = SOURCE

    def __init__(self, regions: Tuple[str, ...] = REGIONS) -> None:
        self.regions = regions

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        result = SourceResult()
        prices: Dict[Tuple, Decimal] = {}
        for region in self.regions:
            general = get_json(fetcher, OFFER_URL.format(offer="AmazonBedrock", region=region))
            prices.update(parse_general(general, region, result))
            anthropic = get_json(fetcher, OFFER_URL.format(offer="AmazonBedrockFoundationModels", region=region))
            prices.update(parse_foundation_models(anthropic, region, result))
        build(prices, result)
        return result
