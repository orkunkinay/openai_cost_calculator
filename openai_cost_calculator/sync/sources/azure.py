"""Azure OpenAI (Microsoft Foundry): official Azure Retail Prices API.

``prices.azure.com/api/retail/prices`` is public and structured, but each
price is identified only by a free-form *meter name* built from abbreviations,
e.g. ``5.4 longco batch cd inp Dz 1M Tokens`` or ``56luna LoCo Cd Wr Fl Gl``.
Meter names are tokenized and every token must be either a known modifier
(deployment type, direction, cache read/write, service tier, context length)
or part of a model name that matches OpenAI's naming scheme.  Meters that do
not fit - fine-tuning, audio, image, realtime, grader and tool meters, or
novel abbreviations - are skipped and counted in a report note rather than
guessed at.

Deployment types map to the ``region`` condition: ``global``, ``data-zone``,
or the Azure region for regional deployments (:data:`REGION`).  Snapshot
suffixes are Azure's ``MMDD`` form (``gpt-4o-0806``); the Azure provider spec
maps API model names such as ``gpt-4o-2024-08-06`` onto them.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult, get_json
from .common import present, price_set

URL = "https://prices.azure.com/api/retail/prices"
REGION = "eastus2"
PRODUCTS = ("Azure OpenAI", "Azure OpenAI GPT5", "Azure OpenAI GPT6", "Azure OpenAI Reasoning")
SOURCE = Source(
    id="azure-retail-prices-api",
    kind="official_api",
    url=URL,
    description=f"Azure Retail Prices API, Foundry Models meters ({REGION} and global/data-zone deployments)",
)
#: Long-context meters ("LongCo") follow OpenAI's 272K threshold.
LONG_CONTEXT_THRESHOLD = 272_000

_DEPLOYMENT = {
    "gl": "global",
    "glb": "global",
    "glbl": "global",
    "global": "global",
    "dz": "data-zone",
    "dzn": "data-zone",
    "dzone": "data-zone",
    "datazone": "data-zone",
    "regn": "regional",
    "regnl": "regional",
    "rgnl": "regional",
    "regional": "regional",
}
_DIRECTION = {
    "inp": "input",
    "inpt": "input",
    "input": "input",
    "in": "input",
    "outp": "output",
    "opt": "output",
    "out": "output",
    "output": "output",
    "outpt": "output",
}
_CACHED = {"cd", "cached", "cchd", "ccchd", "cched"}
_WRITE = {"wr"}
_TIER = {
    "batch": "batch",
    "pp": "priority",
    "priority": "priority",
    "fl": "flex",
    "flex": "flex",
    "std": "standard",
    "standard": "standard",
}
_CONTEXT = {"longco": "long", "loco": "long", "lngco": "long", "shortco": "short", "shco": "short"}
_NOISE = {"tokens", "token", "1m", "1k"}
#: Tokens that identify meters outside the text-token catalog.
_UNSUPPORTED = {
    "ft",
    "dev",
    "training",
    "trng",
    "grader",
    "grdr",
    "mdl",
    "mdel",
    "model",
    "rft",
    "aud",
    "audio",
    "rt",
    "realtime",
    "rtime",
    "realtimeprvw",
    "prvw",
    "preview",
    "txt",
    "text",
    "transcribe",
    "tts",
    "img",
    "image",
    "embedding",
    "embed",
    "computer",
    "use",
    "search",
    "file",
    "tool",
    "calls",
    "deep",
    "research",
    "tkn",
    "hosting",
    "provisioned",
    "data",
    "zone",
}
_NAME_WORDS = {"mini", "nano", "pro", "chat", "codex", "max", "luna", "sol", "terra", "astra", "cyber", "latest"}
_GPT_VERSION = re.compile(r"^(?P<major>\d)(?:\.(?P<minor>\d))?$|^(?P<glued>[56])(?P<gminor>\d)(?P<word>[a-z]*)$")
_O_SERIES = re.compile(r"^o\d$")
_SNAPSHOT = re.compile(r"^\d{4}$")
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}$")


def _model_id(tokens: List[str]) -> Optional[str]:
    """Build an OpenAI-style id from meter name tokens, or None if it does not fit."""
    words = [t for t in tokens if t not in {"gpt", "az", "azure"}]
    if not words:
        return None
    head, rest = words[0], words[1:]
    if _O_SERIES.match(head):
        parts = [head]
    else:
        match = _GPT_VERSION.match(head)
        if match is None:
            if head in {"4o", "4.1", "4.5"}:
                parts = ["gpt", head]
            else:
                return None
        elif match.group("glued"):
            # "56luna" / "54" -> 5.6-luna / 5.4 (dot dropped in some GPT-5 meters)
            parts = ["gpt", f"{match.group('glued')}.{match.group('gminor')}"]
            if match.group("word"):
                rest = [match.group("word"), *rest]
        else:
            version = match.group("major") + (f".{match.group('minor')}" if match.group("minor") else "")
            parts = ["gpt", version]
    snapshot = None
    for word in rest:
        if _SNAPSHOT.match(word) and snapshot is None:
            snapshot = word
        elif word in _NAME_WORDS and snapshot is None:
            parts.append(word)
        else:
            return None
    if snapshot:
        parts.append(snapshot)
    return "-".join(parts)


def parse_meter(meter: str) -> Optional[Tuple[str, str, str, str, str]]:
    """``(model, region, tier, context, dimension)`` for a text-token meter, else None."""
    tokens = [t for t in re.split(r"[\s\-]+", meter.lower()) if t]
    if any(t in _UNSUPPORTED for t in tokens):
        return None
    name: List[str] = []
    deployment = direction = None
    tier, context = "standard", "short"
    cached = write = False
    seen_modifier = False
    for token in tokens:
        if token in _NOISE:
            continue
        if token in _DEPLOYMENT:
            deployment = _DEPLOYMENT[token]
        elif token in _DIRECTION:
            direction = _DIRECTION[token]
        elif token in _CACHED:
            cached = True
        elif token in _WRITE:
            write = True
        elif token in _TIER:
            tier = _TIER[token]
        elif token in _CONTEXT:
            context = _CONTEXT[token]
        elif seen_modifier:
            return None  # unknown abbreviation after the model name
        else:
            name.append(token)
            continue
        seen_modifier = True
    if deployment is None or (direction is None and not (cached and write)):
        return None
    if write:
        dimension = "cache_write"
    elif cached:
        if direction != "input":
            return None
        dimension = "cached_input"
    else:
        dimension = direction or ""
    model = _model_id(name)
    if model is None:
        return None
    region = REGION if deployment == "regional" else deployment
    return model, region, tier, context, dimension


def _per_million(price: Any, unit: str) -> Optional[Decimal]:
    amount = Decimal(str(price))
    if unit == "1M":
        return amount
    if unit == "1K":
        return amount * 1000
    return None


def parse(items: List[Dict[str, Any]]) -> SourceResult:
    result = SourceResult()
    latest: Dict[Tuple[str, str, str, str, str], Tuple[str, Decimal]] = {}
    skipped = 0
    for item in items:
        if item.get("productName") not in PRODUCTS or item.get("type") != "Consumption":
            continue
        meter = str(item.get("meterName", ""))
        if not meter.lower().endswith("tokens"):
            continue
        parsed = parse_meter(meter)
        amount = _per_million(item.get("retailPrice"), str(item.get("unitOfMeasure")))
        if parsed is None or amount is None:
            skipped += 1
            continue
        started = str(item.get("effectiveStartDate", ""))
        previous = latest.get(parsed)
        if previous is None or started > previous[0]:
            latest[parsed] = (started, amount)

    grouped: Dict[str, Dict[Tuple[str, str, str], Dict[str, Decimal]]] = {}
    for (model, region, tier, context, dimension), (_, amount) in latest.items():
        grouped.setdefault(model, {}).setdefault((region, tier, context), {})[dimension] = amount
    for model, groups in sorted(grouped.items()):
        built = []
        for (region, tier, context), rates in sorted(groups.items()):
            if "input" not in rates or "output" not in rates:
                continue  # incomplete meter family: not enough to price a request
            minimum = LONG_CONTEXT_THRESHOLD if context == "long" else 0
            built.append(price_set(rates, service_tier=tier, region=region, min_input_tokens=minimum))
        sets = list(present(built))
        # A long-context tier needs its short-context base for the same conditions.
        bases = {s.condition_key for s in sets if s.min_input_tokens == 0}
        sets = [s for s in sets if s.min_input_tokens == 0 or s.condition_key in bases]
        if sets:
            result.add(
                ModelPricing(
                    id=model,
                    prices=tuple(sets),
                    source=SOURCE.id,
                    vendor="openai",
                    canonical_id="openai/" + _SNAPSHOT_SUFFIX.sub("", model),
                )
            )
    if skipped:
        result.note(f"{skipped} token meters were outside the text-token catalog or unrecognized and were skipped")
    return result


def fetch_items(fetcher: Fetcher, region: str = REGION) -> List[Dict[str, Any]]:
    params = {"$filter": f"serviceName eq 'Foundry Models' and armRegionName eq '{region}' and type eq 'Consumption'"}
    payload = get_json(fetcher, URL, params=params)
    items = list(payload.get("Items", []))
    next_link = payload.get("NextPageLink")
    pages = 1
    while next_link and pages < 50:
        payload = get_json(fetcher, next_link)
        items += payload.get("Items", [])
        next_link = payload.get("NextPageLink")
        pages += 1
    return items


class AzureSource:
    provider = "azure"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetch_items(fetcher))
