"""Contract tests for DeepSeek, Together, Groq, Fireworks, Mistral and Vertex AI sources.

HTML fixtures are the official pages retrieved on 2026-09-23 with scripts,
styles and attributes stripped (element structure and text are unchanged).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from openai_cost_calculator.catalog.validation import validate_model
from openai_cost_calculator.sync.base import merge_duplicate_models
from openai_cost_calculator.sync.sources import deepseek, hosts, mistral, vertex

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
D = Decimal


def _text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def _models(result, provider="test"):
    merged = merge_duplicate_models(result)
    for model in merged.models:
        validate_model(model, provider=provider)
    return {m.id: m for m in merged.models}, merged.issues


def _rates(model, min_input_tokens=0, on=None, **conditions):
    for price in model.prices:
        if dict(price.conditions) == conditions and price.min_input_tokens == min_input_tokens and (on is None or price.is_effective(on)):
            return dict(price.rates)
    raise AssertionError(f"no price {conditions} for {model.id}")


# --------------------------------------------------------------------------- DeepSeek


def test_deepseek_peak_and_off_peak_with_legacy_aliases():
    models, issues = _models(deepseek.parse(_text("deepseek_pricing.html")))
    assert issues == []
    flash = models["deepseek-flash"]
    assert _rates(flash, period="peak") == {"cached_input": D("0.006"), "input": D("0.3"), "output": D("1.2")}
    assert _rates(flash, period="off_peak") == {"cached_input": D("0.003"), "input": D("0.15"), "output": D("0.6")}
    assert set(flash.aliases) == {"deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}
    assert _rates(models["deepseek-v4-pro"], period="peak")["output"] == D("3.96")


def test_deepseek_changed_peak_hours_are_flagged():
    page = _text("deepseek_pricing.html").replace("06:00 - 10:00", "07:00 - 11:00")
    result = deepseek.parse(page)
    assert any("peak-hours footnote" in i.message and i.blocking for i in result.issues)


# --------------------------------------------------------------------------- Together / Groq / Fireworks


def test_together_catalog_prices_and_duplicate_tables():
    models, issues = _models(hosts.parse_together(_text("together_models.md")))
    assert issues == []
    assert _rates(models["moonshotai/Kimi-K3"]) == {"input": D("3"), "output": D("15"), "cached_input": D("0.3")}
    # Listed in both the chat and vision tables; the vision row lacks the cached column.
    assert _rates(models["MiniMaxAI/MiniMax-M3"])["cached_input"] == D("0.06")
    assert _rates(models["Prism-ML/Ternary-Bonsai-27B"]) == {"input": D("0"), "output": D("0")}


def test_groq_prices_skip_enterprise_and_non_token_models():
    models, issues = _models(hosts.parse_groq(_text("groq_models.md")))
    assert issues == []
    assert _rates(models["openai/gpt-oss-120b"]) == {"input": D("0.15"), "output": D("0.6")}
    assert "llama-3.3-70b-versatile" not in models  # "Contact sales"
    assert "whisper-large-v3" not in models  # priced per hour


def test_fireworks_standard_priority_fast_us_and_batch():
    models, issues = _models(hosts.parse_fireworks(_text("fireworks_pricing.md")))
    assert issues == []
    kimi = models["accounts/fireworks/models/kimi-k3"]
    assert _rates(kimi) == {"input": D("3"), "cached_input": D("0.3"), "output": D("15")}
    assert _rates(kimi, service_tier="priority")["output"] == D("18.75")
    assert _rates(kimi, service_tier="fast")["input"] == D("4.5")
    assert _rates(kimi, region="us")["input"] == D("4.5")
    assert _rates(kimi, service_tier="batch") == {"input": D("1.5"), "output": D("7.5")}


def test_fireworks_missing_batch_rule_is_flagged():
    page = _text("fireworks_pricing.md").replace("50% of serverless pricing", "a discount")
    result = hosts.parse_fireworks(page)
    assert any("batch pricing rule" in i.message for i in result.issues)
    assert all("batch" not in p.conditions.get("service_tier", "") for m in result.models for p in m.prices)


# --------------------------------------------------------------------------- Mistral


def test_mistral_cards_and_service_wide_modifiers():
    models, issues = _models(mistral.parse(_text("mistral_pricing.html")))
    assert issues == []
    medium = models["mistral-medium-3-5"]
    assert _rates(medium) == {"input": D("1.5"), "output": D("7.5"), "cached_input": D("0.15")}
    assert _rates(medium, service_tier="batch")["output"] == D("3.75")
    assert _rates(medium, region="eu")["input"] == D("1.65")
    assert "mistral-medium-latest" in medium.aliases
    assert _rates(models["mistral-embed"]) == {"input": D("0.1")}
    assert "voxtral-small" not in models  # audio priced per minute
    assert models["glm-5-2"].canonical_id is None  # third-party model hosted by Mistral


def test_mistral_missing_modifier_sentence_is_flagged_not_guessed():
    page = _text("mistral_pricing.html").replace("Regional inference", "Local processing")
    models, issues = _models(mistral.parse(page))
    assert any("regional inference" in i.message for i in issues)
    assert all(p.conditions.get("region") is None for m in models.values() for p in m.prices)


# --------------------------------------------------------------------------- Vertex AI


@pytest.fixture(scope="module")
def vertex_result():
    return _models(vertex.parse(_text("vertex_pricing.html")))


def test_vertex_claude_region_tabs_and_long_context(vertex_result):
    models, _ = vertex_result
    sonnet = models["claude-sonnet-4-5"]
    assert _rates(sonnet, region="global")["input"] == D("3")
    assert _rates(sonnet, min_input_tokens=200_001, region="global")["input"] == D("6")
    assert _rates(sonnet, region="us-east5")["input"] == D("3.3")  # not offered on the US multi-region
    assert _rates(models["claude-opus-5-5"], region="eu")["output"] == D("22")
    assert models["claude-opus-5-5"].canonical_id == "anthropic/claude-opus-5-5"


def test_vertex_contradictory_rows_are_left_out_and_noted(vertex_result):
    models, issues = vertex_result
    assert [i for i in issues if i.blocking] == []
    notes = {(i.model_id, i.message.split(";")[0]) for i in issues}
    assert ("claude-sonnet-4-5", "the Global table lists contradictory '5m Batch Cache Write' prices") in notes
    tiers = {(p.conditions.get("region"), p.conditions.get("service_tier")) for p in models["claude-sonnet-4-5"].prices}
    assert ("global", "batch") not in tiers and ("global", None) in tiers


def test_vertex_gemini_global_regional_and_scheduled_prices(vertex_result):
    models, _ = vertex_result
    flash = models["gemini-3.8-flash"]
    assert _rates(flash, region="global", on=date(2026, 12, 1))["input"] == D("0.75")
    assert _rates(flash, region="global", on=date(2027, 2, 1))["input"] == D("1.5")
    assert _rates(flash, region="regional", on=date(2026, 12, 1))["input"] == D("0.825")
    pro = models["gemini-2.5-pro"]
    assert _rates(pro, region="global", min_input_tokens=200_001)["output"] == D("15")
    assert _rates(pro, region="global", service_tier="priority")["input"] == D("2.25")


def test_vertex_partner_models(vertex_result):
    models, _ = vertex_result
    assert _rates(models["deepseek-v3.1"]) == {"input": D("0.6"), "output": D("1.7")}
    assert "deepseek-r1" in models  # "DeepSeek R1 (0528)" -> parenthetical dropped
    assert _rates(models["grok-4.6"], min_input_tokens=200_001)["input"] == D("4")


def test_vertex_tab_count_mismatch_falls_back_to_global_only():
    page = _text("vertex_pricing.html").replace("EU Multi-Region (EU)", "")
    models, issues = _models(vertex.parse(page))
    assert any("region tabs" in i.message for i in issues)
    assert {p.conditions.get("region") for p in models["claude-opus-5-5"].prices} == {"global"}
