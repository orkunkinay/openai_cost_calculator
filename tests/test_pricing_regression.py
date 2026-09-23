"""End-to-end pricing regressions: official page -> parser -> catalog -> cost.

Each expected amount is computed by hand from the provider's published price
(pages retrieved 2026-09-23, stored under ``tests/fixtures/sources``).  The
catalog is built from those fixtures, not from the bundled data, so these
tests pin the calculation pipeline without failing when the weekly sync
legitimately updates bundled prices.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from openai_cost_calculator import Usage, calculate_cost, estimate_response_cost
from openai_cost_calculator.catalog import PricingCatalog, ProviderPricing
from openai_cost_calculator.sync.base import FixtureFetcher, merge_duplicate_models
from openai_cost_calculator.sync.sources import (
    anthropic,
    azure,
    bedrock,
    deepinfra,
    deepseek,
    gemini,
    hosts,
    mistral,
    openai,
    openrouter,
    vertex,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sources"
ON = date(2026, 9, 23)
D = Decimal


def _text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def _provider(source_module, provider, result):
    merged = merge_duplicate_models(result)
    return ProviderPricing(
        provider=provider, sources=(source_module.SOURCE,), models=tuple(merged.models), verified_at=ON
    )


@pytest.fixture(scope="module")
def catalog():
    bedrock_fetcher = FixtureFetcher(
        {
            bedrock.OFFER_URL.format(offer="AmazonBedrock", region="us-east-1"): _text("bedrock_general_use1.json"),
            bedrock.OFFER_URL.format(offer="AmazonBedrockFoundationModels", region="us-east-1"): _text(
                "bedrock_fm_use1.json"
            ),
        }
    )
    fireworks = hosts.parse_fireworks(_text("fireworks_pricing.md"))
    together = hosts.parse_together(_text("together_models.md"))
    groq = hosts.parse_groq(_text("groq_models.md"))

    class _Src:
        def __init__(self, source):
            self.SOURCE = source

    return PricingCatalog(
        [
            _provider(openai, "openai", openai.parse(_text("openai_pricing.md"))),
            _provider(anthropic, "anthropic", anthropic.parse(_text("anthropic_pricing.md"))),
            _provider(gemini, "gemini", gemini.parse(_text("gemini_pricing.md"))),
            _provider(openrouter, "openrouter", openrouter.parse(json.loads(_text("openrouter_models.json")))),
            _provider(azure, "azure", azure.parse(json.loads(_text("azure_retail_prices.json"))["Items"])),
            _provider(bedrock, "bedrock", bedrock.BedrockSource(regions=("us-east-1",)).fetch(bedrock_fetcher)),
            _provider(vertex, "vertex", vertex.parse(_text("vertex_pricing.html"))),
            _provider(deepseek, "deepseek", deepseek.parse(_text("deepseek_pricing.html"))),
            _provider(_Src(hosts.TOGETHER_SOURCE), "together", together),
            _provider(_Src(hosts.GROQ_SOURCE), "groq", groq),
            _provider(_Src(hosts.FIREWORKS_SOURCE), "fireworks", fireworks),
            _provider(deepinfra, "deepinfra", deepinfra.parse(json.loads(_text("deepinfra_models.json")))),
            _provider(mistral, "mistral", mistral.parse(_text("mistral_pricing.html"))),
        ]
    )


@pytest.mark.parametrize(
    "provider,model,kwargs,expected",
    [
        # OpenAI gpt-6-sol: $2 in, $0.20 cached, $2.50 cache write, $10 out per 1M.
        (
            "openai",
            "gpt-6-sol",
            dict(input_tokens=100_000, cached_input_tokens=50_000, cache_write_tokens=20_000, output_tokens=10_000),
            D("0.2") + D("0.01") + D("0.05") + D("0.1"),
        ),
        # gpt-5.5 above 272K input: $10 in / $45 out.
        ("openai", "gpt-5.5-2026-04-01", dict(input_tokens=300_000, output_tokens=1_000), D("3") + D("0.045")),
        # Batch tier: gpt-5-mini $0.125 / $1.
        (
            "openai",
            "gpt-5-mini",
            dict(input_tokens=1_000_000, output_tokens=1_000_000, service_tier="batch"),
            D("1.125"),
        ),
        # Anthropic Opus 4.8: $5 in, $6.25 5m write, $10 1h write, $0.50 read, $25 out.
        (
            "anthropic",
            "claude-opus-4-8",
            dict(
                input_tokens=10_000,
                cache_write_tokens=10_000,
                cache_write_1h_tokens=10_000,
                cached_input_tokens=100_000,
                output_tokens=2_000,
            ),
            D("0.05") + D("0.0625") + D("0.1") + D("0.05") + D("0.05"),
        ),
        # Anthropic US data residency (1.1x) on Sonnet 5 ($2 -> $2.20).
        ("anthropic", "claude-sonnet-5", dict(input_tokens=1_000_000, region="us"), D("2.2")),
        # Gemini 2.5 Pro above 200K: $2.50 in / $15 out.
        ("gemini", "models/gemini-2.5-pro", dict(input_tokens=250_000, output_tokens=10_000), D("0.625") + D("0.15")),
        # OpenRouter passes through Anthropic pricing incl. its >=200K tier.
        ("openrouter", "anthropic/claude-sonnet-4.5", dict(input_tokens=200_000, output_tokens=0), D("1.2")),
        # Azure global deployment of GPT-5.4 = OpenAI list price; Data Zone +10%.
        ("azure", "gpt-5.4", dict(input_tokens=100_000, output_tokens=100_000), D("0.25") + D("1.5")),
        ("azure", "gpt-5.4", dict(input_tokens=100_000, region="data-zone"), D("0.275")),
        # ...and 300K input tokens crosses the 272K long-context tier ($5).
        ("azure", "gpt-5.4", dict(input_tokens=300_000), D("1.5")),
        ("azure", "gpt-4o-2024-08-06", dict(input_tokens=1_000_000, region="eastus2"), D("2.75")),
        # Bedrock: global Claude = first-party price; in-region +10%.
        (
            "aws-bedrock",
            "anthropic/claude-sonnet-4-5",
            dict(input_tokens=1_000_000, output_tokens=100_000),
            D("3") + D("1.5"),
        ),
        ("bedrock", "anthropic.claude-sonnet-4-5-20250929-v1:0", dict(input_tokens=1_000_000), D("3.3")),
        ("bedrock", "deepseek.v3.2", dict(input_tokens=1_000_000, output_tokens=1_000_000), D("0.62") + D("1.85")),
        # Vertex: Claude Sonnet 4.5 keeps a >200K tier ($6); regional Gemini is +10%.
        ("vertex", "claude-sonnet-4-5@20250929", dict(input_tokens=250_000), D("1.5")),
        ("vertex", "gemini-2.5-pro", dict(input_tokens=100_000), D("0.125")),
        (
            "deepseek",
            "deepseek-v4-pro",
            dict(input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=1_000_000, period="peak"),
            D("1.32") + D("0.044") + D("3.96"),
        ),
        (
            "together",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            dict(input_tokens=1_000_000, output_tokens=1_000_000),
            D("2.08"),
        ),
        ("groq", "openai/gpt-oss-20b", dict(input_tokens=2_000_000, output_tokens=1_000_000), D("0.15") + D("0.3")),
        (
            "fireworks",
            "accounts/fireworks/models/kimi-k3",
            dict(input_tokens=1_000_000, service_tier="batch"),
            D("1.5"),
        ),
        ("fireworks", "accounts/fireworks/models/kimi-k3", dict(input_tokens=1_000_000, region="us"), D("4.5")),
        # DeepInfra GLM-5.2: $0.75 list, 25% promotional discount -> $0.5625.
        ("deepinfra", "zai-org/GLM-5.2", dict(input_tokens=1_000_000, output_tokens=1_000_000), D("0.5625") + D("1.8")),
        (
            "mistral",
            "mistral-medium-latest",
            dict(input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=1_000_000),
            D("1.5") + D("7.5") + D("0.15"),
        ),
    ],
)
def test_documented_prices_end_to_end(catalog, provider, model, kwargs, expected):
    cost = calculate_cost(provider, model, catalog=catalog, at=ON, **kwargs)
    assert cost.total == expected


def test_openai_fast_mode_is_the_priority_tier(catalog):
    fast = calculate_cost("openai", "gpt-4o", input_tokens=1_000_000, service_tier="fast", catalog=catalog, at=ON)
    priority = calculate_cost("openai", "gpt-4o", input_tokens=1_000_000, service_tier="priority", catalog=catalog, at=ON)
    assert fast.total == priority.total == D("4.25")


def test_vertex_regional_gemini_uses_non_global_price(catalog):
    cost = calculate_cost(
        "vertex", "gemini-3.8-flash", input_tokens=1_000_000, region="us-central1", catalog=catalog, at=ON
    )
    assert cost.total == D("0.825") and cost.conditions["region"] == "regional"


def test_scheduled_gemini_price_change(catalog):
    before = calculate_cost(
        "gemini", "gemini-3.8-flash", input_tokens=1_000_000, catalog=catalog, at=date(2026, 12, 31)
    )
    after = calculate_cost("gemini", "gemini-3.8-flash", input_tokens=1_000_000, catalog=catalog, at=date(2027, 1, 1))
    assert (before.total, after.total) == (D("0.75"), D("1.5"))


def test_deepseek_off_peak_by_request_time(catalog):
    saturday = datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc)
    cost = calculate_cost("deepseek", "deepseek-flash", input_tokens=1_000_000, catalog=catalog, at=saturday)
    assert cost.total == D("0.15")


def test_same_claude_response_on_three_billing_providers(catalog):
    # The identical Anthropic-format response, billed three ways: Anthropic bills
    # Sonnet 4.5 flat, Bedrock's global endpoint matches it, and Vertex's
    # us-east5 endpoint adds 10% and keeps a >200K long-context tier.
    usage = {"input_tokens": 1_000_000, "cache_read_input_tokens": 0, "output_tokens": 0}
    message = {"type": "message", "model": "claude-sonnet-4-5-20250929", "usage": usage}
    totals = {
        provider: estimate_response_cost(message, provider=provider, catalog=catalog, at=ON, **extra).total
        for provider, extra in (("anthropic", {}), ("bedrock", {}), ("vertex", {"region": "us-east5"}))
    }
    assert totals == {"anthropic": D("3"), "bedrock": D("3"), "vertex": D("6.6")}


def test_audio_and_reasoning_dimensions(catalog):
    usage = Usage(input_tokens=1_000_000, input_audio_tokens=1_000_000, output_audio_tokens=1_000_000)
    assert calculate_cost("openai", "gpt-realtime", usage=usage, catalog=catalog, at=ON).total == D("4") + D("32") + D(
        "64"
    )
    reasoning = Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=1_000_000)
    assert calculate_cost(
        "openrouter", "perplexity/sonar-deep-research", usage=reasoning, catalog=catalog, at=ON
    ).total == D("11")
