"""Backwards compatibility of the legacy OpenAI pricing path.

``data/gpt_pricing_data.csv`` is downloaded at run time by every installed
release, so these tests pin its contract against the catalog it is now
generated from.
"""

from __future__ import annotations

import csv
import importlib
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import requests

import openai_cost_calculator as occ
from openai_cost_calculator import estimate, legacy
from openai_cost_calculator.catalog import bundled_catalog

ROOT = Path(__file__).resolve().parents[1]
PRE_CATALOG_CSV = Path(__file__).parent / "fixtures" / "legacy_pricing_2026-09-23_pre_catalog.csv"
TODAY = datetime.now(timezone.utc).date()


def test_committed_csv_is_the_generated_projection():
    assert (ROOT / "data" / "gpt_pricing_data.csv").read_text(encoding="utf-8") == legacy.render_legacy_csv(on=TODAY), (
        "run `openai-cost-calculator pricing export-legacy`"
    )


def test_generated_csv_is_accepted_by_the_parser_old_releases_ship():
    parsed = occ.pricing._parse_csv(legacy.render_legacy_csv(on=date(2026, 9, 23)))
    assert ("gpt-4o", "2024-08-06") in parsed
    assert all(tiers[0]["minimum_tokens"] == 0 for tiers in parsed.values())


def test_every_pre_catalog_row_still_resolves_for_old_clients(monkeypatch):
    projection = legacy.legacy_tiered_pricing(on=date(2026, 9, 23))
    monkeypatch.setattr(occ.pricing, "load_pricing_tiered", lambda: projection)
    with PRE_CATALOG_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        name, model_date = row["Model Name"], row["Model Date"] or "2026-09-23"
        tokens = int(row["Minimum Tokens"])
        rates = estimate._find_rates(name, model_date, tokens)  # the old lookup algorithm
        assert rates["input_price"] >= 0 and rates["output_price"] >= 0


def test_dated_snapshots_keep_their_own_prices():
    projection = legacy.legacy_tiered_pricing(on=date(2026, 9, 23))
    assert projection[("gpt-4o", "2024-05-13")][0]["input_price"] == 5.0
    assert projection[("gpt-4o", "2024-08-06")][0]["input_price"] == 2.5


def test_legacy_and_new_api_agree_on_standard_prices():
    projection = legacy.legacy_tiered_pricing(on=date(2026, 9, 23))
    for name in ("gpt-5.5", "gpt-4o-mini", "o3"):
        row = projection[(name, "")][0]
        # 100K tokens each: below every long-context threshold.
        cost = occ.calculate_cost("openai", name, input_tokens=100_000, output_tokens=100_000, at=date(2026, 9, 23))
        assert (Decimal(str(row["input_price"])) + Decimal(str(row["output_price"]))) / 10 == cost.total


@pytest.fixture
def fresh_pricing():
    pricing = importlib.reload(occ.pricing)
    pricing._CACHE, pricing._CACHE_TS = None, 0
    pricing._LOCAL_OVERRIDES.clear()
    pricing._OFFLINE_ONLY = False
    yield pricing
    importlib.reload(occ.pricing)


def test_unreachable_csv_falls_back_to_the_bundled_catalog(fresh_pricing, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(fresh_pricing.requests, "get", offline)
    tiered = fresh_pricing.load_pricing_tiered()
    assert tiered[("gpt-4o-mini", "")][0]["input_price"] == 0.15


def test_failed_refresh_keeps_the_last_good_copy(fresh_pricing, monkeypatch):
    class _Resp:
        text = "Model Name,Model Date,Input Price,Cached Input Price,Output Price,Minimum Tokens\nm,,1,,2,0\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(fresh_pricing.requests, "get", lambda url, timeout: _Resp())
    assert ("m", "") in fresh_pricing.load_pricing_tiered()
    fresh_pricing._CACHE_TS = 0  # expire the cache

    def offline(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(fresh_pricing.requests, "get", offline)
    assert ("m", "") in fresh_pricing.load_pricing_tiered()


def test_seed_anthropic_pricing_uses_catalog_prices(fresh_pricing):
    from openai_cost_calculator.adapters.anthropic_pricing import seed_anthropic_pricing

    fresh_pricing.set_offline_mode(True)
    seed_anthropic_pricing()
    flat = fresh_pricing.load_pricing()
    key = next(k for k in flat if k[0] == "claude-sonnet-5")
    assert flat[key]["input_price"] == 2.0 and flat[key]["output_price"] == 10.0
    assert any(k[0] == "claude-haiku-4-5" for k in flat)


def test_bundled_catalog_covers_every_supported_provider():
    from openai_cost_calculator.providers import provider_ids

    catalog = bundled_catalog()
    assert set(catalog.provider_ids()) == set(provider_ids())
    for provider in provider_ids():
        data = catalog.get(provider)
        assert data.models and data.verified_at is not None and data.sources
