#!/usr/bin/env python3
"""One-time migration: carry legacy CSV rows no official source covers into the catalog.

Kept for provenance.  When the catalog was introduced, some rows of
``data/gpt_pricing_data.csv`` described models the providers' current pricing
pages no longer list (preview snapshots, retired variants).  Installed
clients may still look them up, so they were imported as hand-maintained
entries (source ``legacy-csv``, status ``unverified``).  The sync never edits
or removes such entries; an official source listing the same model
supersedes them automatically.

Usage: python scripts/migrate_legacy_csv.py data/gpt_pricing_data.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

from openai_cost_calculator.api import resolve_model
from openai_cost_calculator.catalog import BUNDLED_DATA_DIR, PricingCatalog, PricingError
from openai_cost_calculator.catalog.io import load_provider_file, write_provider_file
from openai_cost_calculator.catalog.model import ModelPricing, PriceSet, ProviderPricing, Source

LEGACY_SOURCE = Source(
    id="legacy-csv",
    kind="manual",
    url="https://github.com/orkunkinay/openai_cost_calculator/blob/main/data/gpt_pricing_data.csv",
    description="Rows carried over from the pre-catalog pricing CSV; not listed by any official source at migration time",
)


def _price_set(row: Dict[str, str]) -> PriceSet:
    rates = {"input": Decimal(row["Input Price"]), "output": Decimal(row["Output Price"])}
    if row["Cached Input Price"].strip():
        rates["cached_input"] = Decimal(row["Cached Input Price"])
    return PriceSet(rates=rates, min_input_tokens=int(row["Minimum Tokens"] or 0))


def main(csv_path: str) -> int:
    catalog = PricingCatalog.from_directory(BUNDLED_DATA_DIR)
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name, model_date = row["Model Name"].strip(), row["Model Date"].strip()
            provider, model = ("gemini", name[len("google/"):]) if name.startswith("google/") else ("openai", name)
            identifier = f"{model}-{model_date}" if model_date else model
            try:
                resolve_model(provider, identifier, catalog=catalog)
                continue  # an official source covers it
            except PricingError:
                groups[(provider, identifier)].append(row)

    additions: Dict[str, List[ModelPricing]] = defaultdict(list)
    for (provider, identifier), rows in sorted(groups.items()):
        additions[provider].append(
            ModelPricing(
                id=identifier,
                prices=tuple(_price_set(r) for r in rows),
                source=LEGACY_SOURCE.id,
                vendor="openai" if provider == "openai" else "google",
                status="unverified",
            )
        )
    for provider, models in additions.items():
        path = Path(BUNDLED_DATA_DIR) / f"{provider}.json"
        data = load_provider_file(path)
        write_provider_file(
            path,
            ProviderPricing(
                provider=data.provider,
                sources=tuple(s for s in data.sources if s.id != LEGACY_SOURCE.id) + (LEGACY_SOURCE,),
                models=tuple(m for m in data.models if m.source != LEGACY_SOURCE.id) + tuple(models),
                currency=data.currency,
                verified_at=data.verified_at,
            ),
        )
        print(f"{provider}: carried over {len(models)} legacy entries: {', '.join(m.id for m in models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/gpt_pricing_data.csv"))
