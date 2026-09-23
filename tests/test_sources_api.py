"""Contract tests for official-API sources (OpenRouter, DeepInfra, Azure, Bedrock).

Fixtures are trimmed, structure-preserving copies of the live API responses
retrieved on 2026-09-23.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from openai_cost_calculator.catalog.validation import validate_model
from openai_cost_calculator.sync.base import FixtureFetcher, SourceError
from openai_cost_calculator.sync.sources import azure, bedrock, deepinfra, openrouter

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
D = Decimal


def _json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _by_id(result):
    for model in result.models:
        validate_model(model, provider="test")
    return {m.id: m for m in result.models}


def _rates(model, min_input_tokens=0, **conditions):
    for price in model.prices:
        if dict(price.conditions) == conditions and price.min_input_tokens == min_input_tokens:
            return dict(price.rates)
    raise AssertionError(f"no price {conditions} for {model.id}")


# --------------------------------------------------------------------------- OpenRouter


def test_openrouter_per_token_prices_and_long_context_overrides():
    models = _by_id(openrouter.parse(_json("openrouter_models.json")))
    sonnet = models["anthropic/claude-sonnet-4.5"]
    assert _rates(sonnet) == {
        "input": D("3"),
        "output": D("15"),
        "cached_input": D("0.3"),
        "cache_write": D("3.75"),
        "cache_write_1h": D("6"),
        "web_search": D("0.01"),
    }
    assert _rates(sonnet, min_input_tokens=200_000)["input"] == D("6")
    assert sonnet.aliases == ("anthropic/claude-4.5-sonnet-20250929",)
    assert sonnet.canonical_id == "anthropic/claude-sonnet-4.5" and sonnet.vendor == "anthropic"
    assert _rates(models["openai/gpt-6-luna"], min_input_tokens=272_000)["input"] == D("0.2")


def test_openrouter_separately_priced_reasoning_and_skipped_routers():
    models = _by_id(openrouter.parse(_json("openrouter_models.json")))
    assert _rates(models["perplexity/sonar-deep-research"])["reasoning"] == D("3")
    assert "reasoning" not in _rates(models["openai/gpt-5"])  # identical to output: omitted
    assert "openrouter/auto" not in models  # negative price = variable router


def test_openrouter_time_window_overrides_are_noted_not_applied():
    result = openrouter.parse(_json("openrouter_models.json"))
    notes = [i for i in result.issues if not i.blocking]
    assert notes and all("time-of-day" in n.message for n in notes)
    assert [i for i in result.issues if i.blocking] == []


@pytest.mark.parametrize("payload", [{"models": []}, [], {"data": [{"id": 3}]}])
def test_openrouter_unexpected_payloads_are_flagged(payload):
    result = openrouter.parse(payload)
    assert result.models == [] and result.issues[0].blocking


# --------------------------------------------------------------------------- DeepInfra


def test_deepinfra_discounts_multipliers_and_tiers():
    models = _by_id(deepinfra.parse(_json("deepinfra_models.json")))
    glm = models["zai-org/GLM-5.2"]  # list $0.75/$2.40 with 25% promotional discount
    assert _rates(glm) == {
        "input": D("0.5625"),
        "output": D("1.8"),
        "cached_input": D("0.105"),
        "cache_write": D("0.703125"),
        "cache_write_1h": D("1.125"),
    }
    assert _rates(glm, service_tier="priority")["input"] == D("0.84375")
    assert _rates(glm, service_tier="flex")["output"] == D("1.44")
    seed = models["ByteDance/Seed-1.8"]
    assert _rates(seed, min_input_tokens=128_001) == {"input": D("0.5"), "output": D("4"), "cached_input": D("0.1")}
    assert models["meta-llama/Llama-3.3-70B-Instruct"].status == "deprecated"
    assert _rates(models["sentence-transformers/multi-qa-mpnet-base-dot-v1"]) == {"input": D("0.005")}
    assert "Qwen/Qwen-Image-Edit" not in models  # priced per image


def test_deepinfra_discount_with_end_date_reverts_to_list_price():
    item = _json("deepinfra_models.json")[1]
    item = json.loads(json.dumps(item))
    item["pricing"]["discount_ends_at"] = "2026-10-01T00:00:00Z"
    [model] = deepinfra.parse([item]).models
    windows = sorted(
        (str(p.effective_from), str(p.effective_until), p.rates["input"]) for p in model.prices if not p.conditions
    )
    assert windows == [("2026-10-01", "None", D("0.75")), ("None", "2026-10-01", D("0.5625"))]


def test_deepinfra_unparseable_tier_prose_blocks_the_model():
    item = json.loads(json.dumps(_json("deepinfra_models.json")[4]))
    item["pricing"]["full"] = "$0.25 in $2 out <= 128K, then ask sales"
    result = deepinfra.parse([item])
    assert result.models == [] and result.issues[0].model_id == item["model_name"]


# --------------------------------------------------------------------------- Azure


@pytest.mark.parametrize(
    "meter,expected",
    [
        ("5.4 longco batch cd inp Dz 1M Tokens", ("gpt-5.4", "data-zone", "batch", "long", "cached_input")),
        ("56luna LoCo Cd Wr Fl Gl 1M Tokens", ("gpt-5.6-luna", "global", "flex", "long", "cache_write")),
        ("gpt 4o 0806 Inp regnl Tokens", ("gpt-4o-0806", "eastus2", "standard", "short", "input")),
        ("GPT 5 Nano Batch Inpt cchd Dzone 1M Tokens", ("gpt-5-nano", "data-zone", "batch", "short", "cached_input")),
        ("o4-mini 0416 Outp glbl Tokens", ("o4-mini-0416", "global", "standard", "short", "output")),
    ],
)
def test_azure_meter_grammar(meter, expected):
    assert azure.parse_meter(meter) == expected


@pytest.mark.parametrize(
    "meter",
    [
        "gpt 4.1 dev ft training glbl Tokens",
        "gpt-4o-rt-aud-0603 cchd Inp DZn Tokens",
        "gpt-35-trb16K-Batch-125-Inp-glbl",
        "5.4 mystery inp Gl",
    ],
)
def test_azure_unsupported_or_unknown_meters_are_skipped(meter):
    assert azure.parse_meter(meter) is None


def test_azure_prices_by_deployment_type():
    items = _json("azure_retail_prices.json")["Items"]
    result = azure.parse(items)
    models = _by_id(result)
    gpt54 = models["gpt-5.4"]
    assert _rates(gpt54, region="global") == {"input": D("2.5"), "cached_input": D("0.25"), "output": D("15")}
    assert _rates(gpt54, region="data-zone")["input"] == D("2.75")
    assert _rates(gpt54, min_input_tokens=272_000, region="global")["output"] == D("22.5")
    assert _rates(gpt54, service_tier="priority", region="global")["input"] == D("5")
    assert _rates(models["gpt-4o-0806"], region="eastus2")["input"] == D("2.75")
    assert any(not i.blocking for i in result.issues)  # skipped meters are counted, not blocking


def test_azure_follows_pagination():
    first = {"Items": _json("azure_retail_prices.json")["Items"][:10], "NextPageLink": "https://prices.azure.com/page2"}
    second = {"Items": _json("azure_retail_prices.json")["Items"][10:20], "NextPageLink": None}
    fetcher = FixtureFetcher({azure.URL: json.dumps(first), "https://prices.azure.com/page2": json.dumps(second)})
    assert len(azure.fetch_items(fetcher)) == 20


# --------------------------------------------------------------------------- Bedrock


@pytest.fixture(scope="module")
def bedrock_models():
    fetcher = FixtureFetcher(
        {
            bedrock.OFFER_URL.format(offer="AmazonBedrock", region="us-east-1"): (
                FIXTURES / "bedrock_general_use1.json"
            ).read_text(),
            bedrock.OFFER_URL.format(offer="AmazonBedrockFoundationModels", region="us-east-1"): (
                FIXTURES / "bedrock_fm_use1.json"
            ).read_text(),
        }
    )
    return _by_id(bedrock.BedrockSource(regions=("us-east-1",)).fetch(fetcher))


def test_bedrock_anthropic_global_regional_and_batch(bedrock_models):
    sonnet = bedrock_models["anthropic.claude-sonnet-4-5"]
    assert _rates(sonnet, region="global") == {
        "input": D("3"),
        "output": D("15"),
        "cached_input": D("0.3"),
        "cache_write": D("3.75"),
        "cache_write_1h": D("6"),
    }
    assert _rates(sonnet, region="us-east-1")["input"] == D("3.3")  # regional endpoints: +10%
    assert _rates(sonnet, service_tier="batch", region="us-east-1")["output"] == D("8.25")
    assert sonnet.canonical_id == "anthropic/claude-sonnet-4-5"
    assert "global" not in {p.conditions.get("region") for p in bedrock_models["anthropic.claude-3-haiku"].prices}


def test_bedrock_general_offer_models_and_tiers(bedrock_models):
    deepseek = bedrock_models["deepseek.v3.2"]
    assert _rates(deepseek, region="us-east-1") == {"input": D("0.62"), "output": D("1.85")}
    assert _rates(deepseek, service_tier="priority", region="us-east-1")["input"] == D("1.085")
    assert _rates(bedrock_models["meta.llama3-3-70b"], service_tier="batch", region="us-east-1")["input"] == D("0.36")
    assert "amazon.novapro" in bedrock_models
    assert _rates(bedrock_models["xai.grok-4.6"], region="global")


def test_bedrock_network_failure_surfaces_as_source_error():
    with pytest.raises(SourceError):
        bedrock.BedrockSource(regions=("us-east-1",)).fetch(FixtureFetcher({}))
