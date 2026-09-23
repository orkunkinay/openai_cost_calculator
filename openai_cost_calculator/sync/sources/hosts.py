"""Open-model hosts publishing Markdown price tables: Together AI, Groq, Fireworks AI.

These three share a shape - a documentation page (served as Markdown) with a
per-model table - but not a column layout, so each has a small row parser.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import List, Optional

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult
from ..text import clean_cell, markdown_tables, parse_money
from .common import price_set, scale

# --------------------------------------------------------------------------- Together AI

TOGETHER_URL = "https://docs.together.ai/docs/serverless/models.md"
TOGETHER_SOURCE = Source(
    id="together-models-docs",
    kind="official_docs",
    url="https://docs.together.ai/docs/serverless/models",
    description="Together AI serverless model catalog with per-model pricing",
)


def parse_together(text: str) -> SourceResult:
    result = SourceResult()
    for table in markdown_tables(text):
        model_col = table.column("API model string", "Model string for API")
        input_col = table.column("Input pricing (per 1M tokens)")
        output_col = table.column("Output pricing (per 1M tokens)")
        cached_col = table.column("Cached input pricing (per 1M tokens)")
        if model_col is None or input_col is None:
            continue
        organization_col = table.column("Organization")
        for row in table.rows:
            model = clean_cell(row[model_col])
            try:
                rates = {
                    "input": parse_money(row[input_col]),
                    "output": parse_money(row[output_col]) if output_col is not None else None,
                    "cached_input": parse_money(row[cached_col]) if cached_col is not None else None,
                }
            except ValueError as exc:
                result.issue(f"unparseable price: {exc}", model)
                continue
            built = price_set(rates)
            if built is None or not model:
                continue
            vendor = clean_cell(row[organization_col]).lower() if organization_col is not None else None
            result.add(
                ModelPricing(
                    id=model,
                    prices=(built,),
                    source=TOGETHER_SOURCE.id,
                    vendor=vendor,
                    canonical_id=model.lower(),
                    display_name=clean_cell(row[table.column("Model name") or 0]),
                )
            )
    if not result.models:
        result.issue("no priced models found; the page layout may have changed")
    return result


class TogetherSource:
    provider = "together"
    source = TOGETHER_SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse_together(fetcher.get_text(TOGETHER_URL))


# --------------------------------------------------------------------------- Groq

GROQ_URL = "https://console.groq.com/docs/models.md"
GROQ_SOURCE = Source(
    id="groq-models-docs",
    kind="official_docs",
    url="https://console.groq.com/docs/models",
    description="GroqCloud supported models with per-model pricing",
)
_GROQ_LINK = re.compile(r"\]\(/docs/model/(?P<id>[^)]+)\)")
_GROQ_PRICE = re.compile(r"^\$(?P<input>[\d.]+) input\s*\$(?P<output>[\d.]+) output$")


def parse_groq(text: str) -> SourceResult:
    result = SourceResult()
    for table in markdown_tables(text):
        price_col = table.column("PRICE PER 1M TOKENS")
        if price_col is None:
            continue
        for row in table.rows:
            link = _GROQ_LINK.search(row[0])
            if not link:
                result.issue(f"model cell without a model link: {clean_cell(row[0])!r}")
                continue
            model = link.group("id")
            price = clean_cell(row[price_col])
            match = _GROQ_PRICE.match(price)
            if match is None:
                # Enterprise-only ("ContactSales") or non-token units (per hour/character).
                if price and "$" in price and not re.search(r"per (hour|1M characters|minute)", price):
                    result.issue(f"unrecognized price {price!r}", model)
                continue
            rates = {"input": Decimal(match.group("input")), "output": Decimal(match.group("output"))}
            result.add(
                ModelPricing(
                    id=model,
                    prices=(price_set(rates),),  # type: ignore[arg-type]
                    source=GROQ_SOURCE.id,
                    vendor=model.split("/", 1)[0] if "/" in model else None,
                    canonical_id=model.lower(),
                )
            )
    if not result.models:
        result.issue("no priced models found; the page layout may have changed")
    return result


class GroqSource:
    provider = "groq"
    source = GROQ_SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse_groq(fetcher.get_text(GROQ_URL))


# --------------------------------------------------------------------------- Fireworks AI

FIREWORKS_URL = "https://docs.fireworks.ai/serverless/pricing.md"
FIREWORKS_SOURCE = Source(
    id="fireworks-pricing-docs",
    kind="official_docs",
    url="https://docs.fireworks.ai/serverless/pricing",
    description="Fireworks AI serverless pricing documentation",
)
_FW_LINK = re.compile(r"\((?:https://app\.fireworks\.ai)?/models/(?P<account>[\w-]+)/(?P<slug>[\w.\-]+)\)")
_FW_TRIPLE = re.compile(r"^\$(?P<input>[\d.]+) / \$(?P<cached_input>[\d.]+) / \$(?P<output>[\d.]+)$")
_FW_BATCH = re.compile(r"Batch inference\W+is billed at\W+(?P<pct>\d+)% of serverless pricing", re.I)
_FW_VARIANT = re.compile(r"\s+(?:\((?P<paren>US)\)|(?P<fast>Fast))(?:\s+(?P<us>US))?\s*$")


def _fireworks_triple(cell: str) -> Optional[dict]:
    text = clean_cell(cell)
    if text in {"—", "-", ""}:
        return None
    match = _FW_TRIPLE.match(text)
    if not match:
        raise ValueError(f"not an input/cached/output triple: {text!r}")
    return {k: Decimal(v) for k, v in match.groupdict().items()}


def parse_fireworks(text: str) -> SourceResult:
    result = SourceResult()
    batch = _FW_BATCH.search(clean_cell(text))
    batch_factor = Decimal(batch.group("pct")) / 100 if batch else None
    if batch is None:
        result.issue("batch pricing rule not found; batch prices were not generated")
    for table in markdown_tables(text):
        standard_col, priority_col = table.column("Standard"), table.column("Priority")
        if standard_col is None:
            continue
        for row in table.rows:
            link = _FW_LINK.search(row[0])
            name = clean_cell(row[0])
            if not link:
                result.issue(f"model cell without a model link: {name!r}")
                continue
            model = f"accounts/{link.group('account')}/models/{link.group('slug')}"
            variant = _FW_VARIANT.search(name)
            service_tier = "fast" if variant and variant.group("fast") else "standard"
            region = "us" if variant and (variant.group("paren") or variant.group("us")) else None
            try:
                standard = _fireworks_triple(row[standard_col])
                priority = _fireworks_triple(row[priority_col]) if priority_col is not None else None
            except ValueError as exc:
                result.issue(str(exc), model)
                continue
            if standard is None:
                continue
            sets: List = [price_set(standard, service_tier=service_tier, region=region)]
            if priority is not None:
                sets.append(price_set(priority, service_tier="priority", region=region))
            if batch_factor is not None and service_tier == "standard":
                io = {k: v for k, v in standard.items() if k in ("input", "output")}
                sets.append(price_set(scale(io, batch_factor), service_tier="batch", region=region))
            base_name = _FW_VARIANT.sub("", name) if variant else name
            result.add(
                ModelPricing(
                    id=model,
                    prices=tuple(s for s in sets if s is not None),
                    source=FIREWORKS_SOURCE.id,
                    canonical_id=f"{link.group('account')}/{link.group('slug')}",
                    display_name=base_name,
                )
            )
    if not result.models:
        result.issue("no priced models found; the page layout may have changed")
    return result


class FireworksSource:
    provider = "fireworks"
    source = FIREWORKS_SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse_fireworks(fetcher.get_text(FIREWORKS_URL))
