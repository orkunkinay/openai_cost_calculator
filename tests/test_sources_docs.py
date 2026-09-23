"""Contract tests for documentation-page sources (OpenAI, Anthropic, Gemini).

Fixtures are the official pages as retrieved on 2026-09-23; expected values
were checked by hand against the same pages.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from openai_cost_calculator.catalog.validation import validate_model
from openai_cost_calculator.sync.base import FixtureFetcher, merge_duplicate_models
from openai_cost_calculator.sync.sources import anthropic, gemini, openai

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
D = Decimal


def _load(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def _models(result):
    merged = merge_duplicate_models(result)
    return {m.id: m for m in merged.models}, merged.issues


def _rates(model, **conditions):
    minimum = conditions.pop("min_input_tokens", 0)
    on = conditions.pop("on", None)
    for price in model.prices:
        if dict(price.conditions) == conditions and price.min_input_tokens == minimum and (on is None or price.is_effective(on)):
            return {k: v for k, v in price.rates.items()}
    raise AssertionError(f"no price set {conditions} >= {minimum} for {model.id}")


# --------------------------------------------------------------------------- OpenAI


@pytest.fixture(scope="module")
def openai_models():
    models, issues = _models(openai.parse(_load("openai_pricing.md")))
    assert [i for i in issues if i.blocking] == []
    return models


def test_openai_flagship_rates_with_cache_writes_and_long_context(openai_models):
    sol = openai_models["gpt-6-sol"]
    assert _rates(sol) == {"input": D("2"), "cached_input": D("0.2"), "cache_write": D("2.5"), "output": D("10")}
    assert _rates(sol, min_input_tokens=272_000) == {"input": D("4"), "cached_input": D("0.4"), "cache_write": D("5"), "output": D("15")}
    assert sol.canonical_id == "openai/gpt-6-sol" and sol.vendor == "openai"


def test_openai_service_tiers(openai_models):
    assert _rates(openai_models["gpt-5.4"], service_tier="batch")["input"] == D("1.25")
    assert _rates(openai_models["gpt-5-mini"], service_tier="flex")["output"] == D("1")
    assert _rates(openai_models["gpt-4o"], service_tier="priority") == {"input": D("4.25"), "cached_input": D("2.125"), "output": D("17")}


def test_openai_snapshots_and_legacy_models(openai_models):
    assert _rates(openai_models["gpt-4o-2024-05-13"]) == {"input": D("5"), "output": D("15")}
    assert _rates(openai_models["gpt-4o-mini"]) == {"input": D("0.15"), "cached_input": D("0.075"), "output": D("0.6")}
    assert _rates(openai_models["o3"])["cached_input"] == D("0.5")


def test_openai_modality_tables(openai_models):
    realtime = _rates(openai_models["gpt-realtime"])
    assert realtime["input_audio"] == D("32") and realtime["output_audio"] == D("64")
    assert realtime["input"] == D("4") and realtime["input_image"] == D("5")
    assert _rates(openai_models["gpt-image-1"], service_tier="batch")["output_image"] == D("20")
    assert "tts-1" not in openai_models  # priced per character: out of scope


def test_openai_all_entries_are_valid(openai_models):
    assert len(openai_models) >= 50
    for model in openai_models.values():
        validate_model(model, provider="openai")


def test_openai_reports_unparseable_rows():
    page = "Standard\n### Standard pricing data\n| Model | Input | Output |\n| --- | --- | --- |\n| gpt-x | $1.00 | about $2 |\n| Weird Model! | $1 | $2 |\n"
    result = openai.parse(page)
    assert [i.model_id for i in result.issues] == ["gpt-x", None, None]
    assert result.models == [] and result.issues[-1].message.startswith("no priced models")


def test_openai_empty_page_is_flagged():
    assert openai.parse("# Pricing\nNothing here").issues[0].message.startswith("no priced models")


def test_openai_source_fetches_the_markdown_rendition():
    fetcher = FixtureFetcher({openai.URL: _load("openai_pricing.md")})
    assert openai.OpenAISource().fetch(fetcher).models
    assert fetcher.requested == [openai.URL]


# --------------------------------------------------------------------------- Anthropic


@pytest.fixture(scope="module")
def anthropic_models():
    models, issues = _models(anthropic.parse(_load("anthropic_pricing.md")))
    assert issues == []
    return models


def test_anthropic_published_cache_prices_are_used_verbatim(anthropic_models):
    # Fable 5.1 cache reads are 0.025x and Opus 5.5 0.05x: not derivable from 0.1x.
    assert _rates(anthropic_models["claude-fable-5-1"])["cached_input"] == D("0.25")
    assert _rates(anthropic_models["claude-opus-5-5"]) == {
        "input": D("4"), "cache_write": D("5"), "cache_write_1h": D("8"), "cached_input": D("0.2"), "output": D("20"), "web_search": D("0.01"),
    }


def test_anthropic_batch_fast_and_data_residency(anthropic_models):
    opus = anthropic_models["claude-opus-4-8"]
    assert _rates(opus, service_tier="batch")["input"] == D("2.5")
    assert _rates(opus, service_tier="batch")["cache_write_1h"] == D("5")
    assert _rates(opus, service_tier="fast")["output"] == D("50")
    assert _rates(opus, region="us")["input"] == D("5.5")
    assert _rates(opus, service_tier="fast", region="us")["input"] == D("11")
    # Data residency pricing applies to Claude 4.6 and later only.
    assert all(p.conditions.get("region") is None for p in anthropic_models["claude-sonnet-4-5"].prices)
    assert all(p.conditions.get("service_tier") != "fast" for p in anthropic_models["claude-opus-4-7"].prices)


def test_anthropic_current_prices(anthropic_models):
    assert _rates(anthropic_models["claude-sonnet-5"])["input"] == D("2")
    assert _rates(anthropic_models["claude-sonnet-5"])["output"] == D("10")
    assert _rates(anthropic_models["claude-3-5-haiku"])["input"] == D("0.8")
    assert set(anthropic_models) >= {"claude-haiku-4-5", "claude-opus-4-1", "claude-mythos-5-1", "claude-sonnet-4"}


def test_anthropic_missing_residency_sentence_is_flagged_not_guessed():
    page = _load("anthropic_pricing.md").replace("specifying US-only inference", "choosing a US region")
    models, issues = _models(anthropic.parse(page))
    assert any("data-residency" in i.message for i in issues)
    assert all(p.conditions.get("region") is None for m in models.values() for p in m.prices)


def test_anthropic_redesigned_page_yields_no_models():
    result = anthropic.parse("# Pricing\n| Model | Price |\n| --- | --- |\n| Claude | $1 |\n")
    assert result.models == [] and "not found" in result.issues[0].message


# --------------------------------------------------------------------------- Gemini


@pytest.fixture(scope="module")
def gemini_models():
    models, issues = _models(gemini.parse(_load("gemini_pricing.md")))
    assert [i for i in issues if i.blocking] == []
    return models


def test_gemini_long_context_tiers(gemini_models):
    pro = gemini_models["gemini-2.5-pro"]
    assert _rates(pro) == {"input": D("1.25"), "output": D("10"), "cached_input": D("0.125")}
    assert _rates(pro, min_input_tokens=200_001) == {"input": D("2.5"), "output": D("15"), "cached_input": D("0.25")}
    assert _rates(pro, service_tier="batch")["input"] == D("0.625")


def test_gemini_modality_prices(gemini_models):
    flash = _rates(gemini_models["gemini-2.5-flash"])
    assert flash["input"] == D("0.3") and flash["input_audio"] == D("1") and flash["cached_input_audio"] == D("0.1")
    assert _rates(gemini_models["gemini-2.5-flash-preview-tts"])["output_audio"] == D("10")


def test_gemini_scheduled_price_changes_become_effective_windows(gemini_models):
    flash = gemini_models["gemini-3.8-flash"]
    before = _rates(flash, on=date(2026, 12, 31))
    after = _rates(flash, on=date(2027, 1, 1))
    assert (before["input"], after["input"]) == (D("0.75"), D("1.5"))
    assert (before["cached_input"], after["cached_input"]) == (D("0.075"), D("0.15"))


def test_gemini_non_token_and_footnoted_prices_are_not_misread(gemini_models):
    batch = _rates(gemini_models["gemini-3-pro-image"], service_tier="batch")
    assert "input_image" not in batch  # "$0.0006 (image)*" is per image, not per 1M tokens
    embedding = _rates(gemini_models["gemini-embedding-2"])
    assert embedding["input_image"] == D("0.45")  # "($0.00012 per image)" is an equivalence note


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("$1.25, prompts \\<= 200k tokens $2.50, prompts \\> 200k tokens", [("input", "1.25", 0), ("input", "2.50", 200_001)]),
        ("$0.30 (text / image / video) $1.00 (audio)", [("input", "0.30", None), ("input_image", "0.30", None), ("input_audio", "1.00", None)]),
        ("$3.50 or $0.005/min^\\*^ (audio)", [("input_audio", "3.50", None)]),
    ],
)
def test_gemini_cell_grammar(cell, expected):
    clauses, problems, _ = gemini.parse_cell(cell, "input", frozenset())
    assert problems == []
    assert sorted((c.dimension, str(c.amount), c.min_input_tokens) for c in clauses) == sorted(expected)


def test_gemini_unknown_phrasing_blocks_the_model():
    page = "## Gemini X\n*[`gemini-x`](u)*\n### Standard\n|   | Free Tier | Paid Tier, per 1M tokens in USD |\n|---|---|---|\n| Input price | Free | $1.00 for early adopters |\n| Output price | Free | $2.00 |\n"
    result = gemini.parse(page)
    assert result.models == []
    assert result.issues[0].model_id == "gemini-x" and "early adopters" in result.issues[0].message
