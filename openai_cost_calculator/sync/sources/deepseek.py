"""DeepSeek: official API documentation pricing page (HTML).

The pricing table has one column per model and PEAK/OFF-PEAK rows for cache
hits, cache misses and output.  The table's footnotes are parsed too:

* the legacy-name footnote supplies aliases (``deepseek-v4-flash`` ->
  ``deepseek-flash``);
* the peak-hours footnote is checked against the schedule hard-coded in the
  DeepSeek provider spec (:func:`~openai_cost_calculator.providers.registry.deepseek_period`).
  If DeepSeek changes its peak hours the run is flagged for review, so a
  code-level assumption cannot silently drift from the published rule.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult
from ..text import html_tables, html_text, parse_money
from .common import present, price_set

URL = "https://api-docs.deepseek.com/quick_start/pricing"
SOURCE = Source(
    id="deepseek-pricing-docs", kind="official_docs", url=URL, description="DeepSeek API pricing documentation"
)

#: The schedule implemented by ``providers.registry.deepseek_period``.
EXPECTED_PEAK_HOURS = "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday"

_ROW_DIMENSIONS = (("CACHE HIT", "cached_input"), ("CACHE MISS", "input"), ("OUTPUT TOKENS", "output"))
_LEGACY = re.compile(
    r"Use (?P<model>[a-z0-9.\-]+) as the model name\. The legacy names (?P<names>.+?) are still accepted"
)


def parse(document: str) -> SourceResult:
    result = SourceResult()
    text = html_text(document)
    if EXPECTED_PEAK_HOURS not in re.sub(r"\s+", " ", text):
        result.issue(
            "peak-hours footnote no longer matches the schedule implemented in "
            "providers.registry.deepseek_period; review and update both"
        )
    table = next((t for t in html_tables(document) if any(r and r[0] == "MODEL" for r in t.rows)), None)
    if table is None:
        result.issue("pricing table not found; the page layout may have changed")
        return result

    models: List[str] = []
    rates: Dict[str, Dict[str, Dict[str, Decimal]]] = {}
    dimension = None
    for row in table.rows:
        if row and row[0] == "MODEL":
            models = [re.sub(r"\(\d+\)$", "", cell).strip() for cell in row[1:]]
            continue
        label = " ".join(row).upper()
        for marker, dim in _ROW_DIMENSIONS:
            if marker in label:
                dimension = dim
        period = "off_peak" if "OFF-PEAK" in label else "peak" if "PEAK" in label else None
        if period is None or dimension is None or not models:
            continue
        prices = row[-len(models) :]
        for model, cell in zip(models, prices):
            try:
                value = parse_money(cell)
            except ValueError:
                result.issue(f"unparseable price {cell!r}", model)
                continue
            if value is not None:
                rates.setdefault(model, {}).setdefault(period, {})[dimension] = value

    legacy = _LEGACY.search(re.sub(r"\s+", " ", text))
    aliases = {}
    if legacy:
        aliases[legacy.group("model")] = tuple(
            n.strip() for n in re.split(r",| and ", legacy.group("names")) if n.strip()
        )
    for model in models:
        periods = rates.get(model, {})
        sets = present(price_set(r, period=p) for p, r in sorted(periods.items()))
        if len(sets) != 2:
            result.issue("expected peak and off-peak prices", model)
            continue
        result.add(
            ModelPricing(
                id=model,
                prices=sets,
                source=SOURCE.id,
                vendor="deepseek",
                canonical_id=f"deepseek/{model}",
                aliases=aliases.get(model, ()),
            )
        )
    return result


class DeepSeekSource:
    provider = "deepseek"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
