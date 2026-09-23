"""The legacy three-bucket pricing view, derived from the catalog.

Before the catalog existed, prices lived in ``data/gpt_pricing_data.csv``
(OpenAI models plus a few ``google/`` Gemini rows) and every installed copy of
this package downloads that file from the repository's ``main`` branch at run
time.  The file is therefore a public interface: it must keep its exact
columns, keep every row old clients may look up, and stay parseable by the
old parser.  It is now *generated* from the catalog by this module
(``openai-cost-calculator pricing export-legacy``), so it is always current
with the official sources and never edited by hand.

The same projection backs :func:`openai_cost_calculator.pricing.load_pricing_tiered`
when the remote CSV cannot be fetched, and seeds Claude prices for the Claude
Code adapter.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .catalog import PricingCatalog, bundled_catalog
from .catalog.io import format_decimal
from .catalog.model import ModelPricing, PriceSet

#: (catalog provider, prefix used for its model names in the legacy CSV)
LEGACY_PROVIDERS: Sequence[Tuple[str, str]] = (("openai", ""), ("gemini", "google/"))
HEADER = ["Model Name", "Model Date", "Input Price", "Cached Input Price", "Output Price", "Minimum Tokens"]

_DATED = re.compile(r"^(?P<name>.+?)-(?P<date>\d{4}-\d{2}-\d{2})$")

LegacyKey = Tuple[str, str]


def split_legacy_name(identifier: str) -> LegacyKey:
    match = _DATED.match(identifier)
    return (match.group("name"), match.group("date")) if match else (identifier, "")


def _standard_sets(model: ModelPricing, on: date) -> List[PriceSet]:
    """Standard-tier, global (or region-less) prices in effect on ``on``, by tier."""
    sets = [
        p
        for p in model.prices
        if p.is_effective(on)
        and p.conditions.get("service_tier", "standard") == "standard"
        and p.conditions.get("region") in (None, "global")
        and not p.conditions.get("period")
    ]
    by_minimum: Dict[int, PriceSet] = {}
    for price in sets:
        # Prefer an explicit "global" set over a region-less one if both exist.
        if price.min_input_tokens not in by_minimum or price.conditions.get("region") == "global":
            by_minimum[price.min_input_tokens] = price
    return [by_minimum[k] for k in sorted(by_minimum)]


def _row(price: PriceSet) -> Optional[dict]:
    rates = price.rates
    if "input" not in rates or "output" not in rates:
        return None  # the legacy schema requires both
    return {
        "input_price": float(rates["input"]),
        "cached_input_price": float(rates["cached_input"]) if "cached_input" in rates else None,
        "output_price": float(rates["output"]),
        "minimum_tokens": price.min_input_tokens,
    }


def legacy_tiered_pricing(
    catalog: Optional[PricingCatalog] = None, *, on: Optional[date] = None
) -> Dict[LegacyKey, List[dict]]:
    """``{(model_name, model_date): [tier rows]}`` in the legacy loader's shape."""
    catalog = catalog or bundled_catalog()
    on = on or date.today()
    result: Dict[LegacyKey, List[dict]] = {}
    for provider, prefix in LEGACY_PROVIDERS:
        data = catalog.get(provider)
        if data is None:
            continue
        for model in data.models:
            rows = [r for r in (_row(p) for p in _standard_sets(model, on)) if r is not None]
            if not rows or rows[0]["minimum_tokens"] != 0:
                continue
            for identifier in model.identifiers:
                key = split_legacy_name(prefix + identifier)
                result.setdefault(key, [dict(r) for r in rows])
    return result


def _format(value: Optional[float]) -> str:
    return "" if value is None else format_decimal(Decimal(str(value)))


def render_legacy_csv(catalog: Optional[PricingCatalog] = None, *, on: Optional[date] = None) -> str:
    """The legacy CSV, deterministically ordered by model name, date and tier."""
    tiered = legacy_tiered_pricing(catalog, on=on)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for (name, model_date) in sorted(tiered):
        for row in tiered[(name, model_date)]:
            writer.writerow(
                [
                    name,
                    model_date,
                    _format(row["input_price"]),
                    _format(row["cached_input_price"]),
                    _format(row["output_price"]),
                    str(row["minimum_tokens"]),
                ]
            )
    return buffer.getvalue()


def anthropic_legacy_entries(
    catalog: Optional[PricingCatalog] = None, *, on: Optional[date] = None
) -> Iterable[tuple]:
    """``(model, date, input, output, cached)`` tuples for ``add_pricing_entries``."""
    catalog = catalog or bundled_catalog()
    on = on or date.today()
    data = catalog.get("anthropic")
    if data is None:
        return []
    entries = []
    for model in data.models:
        sets = _standard_sets(model, on)
        rows = [r for r in (_row(p) for p in sets) if r is not None]
        for row in rows:
            for identifier in model.identifiers:
                entries.append(
                    (identifier, on.isoformat(), row["input_price"], row["output_price"], row["cached_input_price"], row["minimum_tokens"])
                )
    return entries
