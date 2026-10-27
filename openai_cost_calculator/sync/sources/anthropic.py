"""Anthropic: official pricing documentation (Markdown rendition).

Parsed from ``platform.claude.com/docs/en/about-claude/pricing.md``:

* the model table gives base input, 5-minute and 1-hour cache writes, cache
  reads and output per model - cache prices are taken as published rather than
  derived, because the cache-read multiplier differs by model (0.025x-0.1x);
* the batch and fast-mode tables give input/output; the documentation states
  that cache multipliers stack on top of both, so cache prices are scaled by
  the same factor as base input;
* a model priced by prompt length has one row per tier ("Claude Haiku 5.5
  (for prompts over 100,000 tokens)"); the higher tier becomes a
  ``min_input_tokens`` price set, in both the model and batch tables;
* the data-residency sentence ("For Claude 4.6 and later models ... 1.1x")
  yields ``region="us"`` prices.  If that sentence stops matching, regional
  prices are not emitted and the run is flagged for review.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from ...catalog.model import ModelPricing, PriceSet, Source
from ..base import Fetcher, SourceResult
from ..text import Table, clean_cell, markdown_tables, parse_money, slugify
from .common import present, price_set, scale

URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"
SOURCE = Source(
    id="anthropic-pricing-docs",
    kind="official_docs",
    url="https://platform.claude.com/docs/en/about-claude/pricing",
    description="Claude API pricing documentation (Markdown rendition)",
)

#: Display names whose API id does not follow ``claude-<family>-<version>``.
ID_OVERRIDES = {"claude-haiku-3-5": "claude-3-5-haiku"}

_MODEL_COLUMNS = {
    "input": "base input tokens",
    "cache_write": "5m cache writes",
    "cache_write_1h": "1h cache writes",
    "cached_input": "cache hits and refreshes",
    "output": "output tokens",
}
_RESIDENCY = re.compile(
    r"For Claude (?P<version>\d+(?:\.\d+)?) and later models, specifying US-only inference .*?"
    r"(?P<multiplier>\d+(?:\.\d+)?)x multiplier on all token pricing categories",
    re.S,
)
_WEB_SEARCH = re.compile(r"\$(?P<price>\d+(?:\.\d+)?)(?: USD)? per (?P<count>[\d,]+) searches")
_PROMPT_TIER = re.compile(r"\(for prompts (?P<bound>up to|over) (?P<tokens>[\d,]+) tokens\)", re.I)
_VERSION = re.compile(r"(\d+)(?:\.(\d+))?$")


def _names(cell: str) -> List[str]:
    """Model display names in a cell ("Claude Opus 5 / Claude Opus 4.8" -> two names)."""
    text = clean_cell(cell)
    removed = 1
    while removed:  # innermost first: notes can nest "(retired, except on ... (Google managed))"
        text, removed = re.subn(r"\([^()]*\)", "", text)
    return [name.strip() for name in text.split("/") if name.strip()]


def _min_input_tokens(cell: str) -> int:
    """``(for prompts over 100,000 tokens)`` -> 100001; no tier note -> 0."""
    match = _PROMPT_TIER.search(clean_cell(cell))
    if match is None or match.group("bound").lower() == "up to":
        return 0
    return int(match.group("tokens").replace(",", "")) + 1


def model_id(display_name: str) -> str:
    slug = slugify(display_name)
    return ID_OVERRIDES.get(slug, slug)


def _version(display_name: str) -> Tuple[int, int]:
    match = _VERSION.search(display_name.strip())
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2) or 0)


def _find(tables: List[Table], *columns: str, heading: str = "") -> Optional[Table]:
    for table in tables:
        lowered = [h.lower() for h in table.header]
        in_section = not heading or any(heading in h.lower() for h in table.headings)
        if in_section and all(c in lowered for c in columns):
            return table
    return None


def _two_column_rates(
    table: Optional[Table], result: SourceResult, what: str
) -> Dict[Tuple[str, int], Tuple[Decimal, Decimal]]:
    rates: Dict[Tuple[str, int], Tuple[Decimal, Decimal]] = {}
    if table is None:
        result.issue(f"{what} table not found")
        return rates
    for row in table.rows:
        try:
            values = parse_money(row[1]), parse_money(row[2])
        except ValueError as exc:
            result.issue(f"unparseable {what} price: {exc}")
            continue
        if values[0] is None or values[1] is None:
            continue
        for name in _names(row[0]):
            rates[(model_id(name), _min_input_tokens(row[0]))] = (values[0], values[1])
    return rates


def _derived(base: Dict[str, Decimal], new_input: Decimal, new_output: Decimal) -> Dict[str, Decimal]:
    """Scale cache prices by the input factor; take input/output as published."""
    factor = new_input / base["input"]
    derived = scale({k: v for k, v in base.items() if k.startswith("cache")}, factor)
    derived.update(input=new_input, output=new_output)
    return derived


def parse(text: str) -> SourceResult:
    result = SourceResult()
    tables = markdown_tables(text)
    model_table = _find(tables, "model", "base input tokens", "output tokens")
    if model_table is None:
        result.issue("model pricing table not found; the page layout may have changed")
        return result
    batch = _two_column_rates(_find(tables, "model", "batch input", "batch output", heading="batch"), result, "batch")
    fast = _two_column_rates(_find(tables, "model", "input", "output", heading="fast mode"), result, "fast mode")

    residency = _RESIDENCY.search(text)
    if residency is None:
        result.issue("data-residency pricing sentence not found; region='us' prices were not generated")
    web_search = _WEB_SEARCH.search(text)
    web_search_price = (
        Decimal(web_search.group("price")) / Decimal(web_search.group("count").replace(",", "")) if web_search else None
    )

    columns = {dim: model_table.column(header) for dim, header in _MODEL_COLUMNS.items()}
    sets_by_model: Dict[str, List[Optional[PriceSet]]] = {}
    names_by_model: Dict[str, str] = {}
    for row in model_table.rows:
        names = _names(row[0])
        if len(names) != 1:
            result.issue(f"unrecognized model cell {row[0]!r}")
            continue
        name = names[0]
        mid = model_id(name)
        minimum = _min_input_tokens(row[0])
        try:
            base = {dim: parse_money(row[col]) for dim, col in columns.items() if col is not None}
        except ValueError as exc:
            result.issue(f"unparseable price: {exc}", mid)
            continue
        if base.get("input") is None or base.get("output") is None:
            result.issue("missing base input/output price", mid)
            continue
        standard = {k: v for k, v in base.items() if v is not None}
        extras = {"web_search": web_search_price} if web_search_price is not None else {}

        sets: List[Optional[PriceSet]] = [price_set({**standard, **extras}, min_input_tokens=minimum)]
        if (mid, minimum) in batch:
            batch_rates = _derived(standard, *batch[(mid, minimum)])
            sets.append(price_set({**batch_rates, **extras}, service_tier="batch", min_input_tokens=minimum))
        if (mid, minimum) in fast:
            fast_rates = _derived(standard, *fast[(mid, minimum)])
            sets.append(price_set({**fast_rates, **extras}, service_tier="fast", min_input_tokens=minimum))
        if residency and _version(name) >= _version(residency.group("version")):
            multiplier = Decimal(residency.group("multiplier"))
            for price in [s for s in sets if s is not None]:
                tier = price.conditions.get("service_tier", "standard")
                token_rates = {k: v for k, v in price.rates.items() if k != "web_search"}
                sets.append(
                    price_set(
                        {**scale(token_rates, multiplier), **extras},
                        service_tier=tier,
                        region="us",
                        min_input_tokens=minimum,
                    )
                )

        sets_by_model.setdefault(mid, []).extend(sets)
        names_by_model.setdefault(mid, name)

    for mid, sets in sets_by_model.items():
        prices = present(sets)
        if not any(p.min_input_tokens == 0 for p in prices):
            result.issue("prompt-length tier without a base tier", mid)
            continue
        result.add(
            ModelPricing(
                id=mid,
                prices=prices,
                source=SOURCE.id,
                vendor="anthropic",
                canonical_id=f"anthropic/{mid}",
                display_name=names_by_model[mid],
            )
        )
    return result


class AnthropicSource:
    provider = "anthropic"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
