# openai_cost_calculator

[![PyPI version](https://img.shields.io/pypi/v/openai-cost-calculator)](https://pypi.org/project/openai-cost-calculator/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Exact, auditable USD cost calculation for LLM API usage across 13 billing providers** —
OpenAI, Anthropic, Google Gemini API, Google Vertex AI, Azure OpenAI, Amazon Bedrock,
OpenRouter, DeepSeek, Together AI, Groq, Fireworks AI, DeepInfra and Mistral — with prices
pulled automatically from each provider's **official** pricing API or documentation.

```python
from openai_cost_calculator import calculate_cost

cost = calculate_cost(
    provider="aws-bedrock",
    model="anthropic/claude-sonnet-4-5",
    input_tokens=12_000,
    output_tokens=800,
)
print(cost.total)        # Decimal('0.048')
print(cost.conditions)   # {'service_tier': 'standard', 'region': 'global'}
print(cost.assumptions)  # ('assumed region=global (pass region=... to price another option)',)
print(cost.source_url)   # the official source the price came from
```

Application code names a provider and a model. Everything provider-specific — model id
formats, global vs regional endpoints, batch/flex/priority tiers, cache-write TTLs,
long-context thresholds, peak hours, scheduled price changes — is resolved by the library.

> The import name is still `openai_cost_calculator`; the original OpenAI-response API is
> unchanged (see [OpenAI response API](#openai-response-api)).

---

## Installation

```bash
pip install openai-cost-calculator
```

The pricing catalog ships inside the package: the provider-agnostic API makes **no network
calls**, so results are deterministic and work offline.

---

## Why this is harder than `tokens × price`

Researching the thirteen providers turned up real differences a single
`input/cached/output` table cannot express:

| Variation | Examples |
| --- | --- |
| Cache reads *and* writes, with TTLs | Anthropic 5-minute and 1-hour writes; OpenAI GPT-5.6+ cache writes; per-model cache-read multipliers (0.025×–0.1×) |
| Long-context tiers | OpenAI ≥272K, Gemini and Vertex Claude >200K — while Anthropic's own API bills Claude 4.6+ flat |
| Service tiers | batch, flex, priority / "fast" — not offered for every model |
| Deployment region | Bedrock and Vertex regional endpoints +10%; Azure Global / Data Zone / Regional; Anthropic `inference_geo="us"` 1.1× |
| Time | DeepSeek peak/off-peak hours; Gemini prices that change on a published date; DeepInfra promotions |
| Modalities | audio and image tokens priced separately; reasoning tokens priced separately on some OpenRouter models |
| Identifiers | `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, `claude-sonnet-4-5@20250929`, `gpt-4o-0806` (Azure), `accounts/fireworks/models/kimi-k3` |

The same Claude response can therefore cost different amounts depending on who bills it:

```python
from openai_cost_calculator import estimate_response_cost

message = {  # an Anthropic Messages API response (dict or SDK object)
    "type": "message",
    "model": "claude-sonnet-4-5-20250929",
    "usage": {"input_tokens": 1_000_000, "cache_read_input_tokens": 0, "output_tokens": 0},
}
estimate_response_cost(message, provider="anthropic").total                   # Decimal('3')
estimate_response_cost(message, provider="bedrock").total                     # Decimal('3')    global endpoint
estimate_response_cost(message, provider="vertex", region="us-east5").total  # Decimal('6.60') regional, >200K tier
```

---

## Usage

### Price a request

Token counts are **disjoint**: `input_tokens` excludes cache reads (`cached_input_tokens`) and
cache writes (`cache_write_tokens`, or `cache_write_1h_tokens` for Anthropic's 1-hour cache).

```python
from openai_cost_calculator import calculate_cost

cost = calculate_cost(
    provider="anthropic",
    model="claude-opus-4-8",
    input_tokens=10_000,
    cached_input_tokens=100_000,
    cache_write_tokens=10_000,
    output_tokens=2_000,
)
for item in cost.items:
    print(item.dimension, item.quantity, item.unit_price, item.cost)
# input        10000   5     0.05
# cached_input 100000  0.5   0.05
# cache_write  10000   6.25  0.0625
# output       2000    25    0.05
```

For audio, image, reasoning tokens or tool calls, pass a `Usage`:

```python
from openai_cost_calculator import Usage, calculate_cost

usage = Usage(input_tokens=2_000, input_audio_tokens=50_000, output_audio_tokens=20_000)
calculate_cost("openai", "gpt-realtime", usage=usage).total
```

If you have an OpenAI-style total that *includes* cached tokens, convert it explicitly:

```python
Usage.from_totals(total_input_tokens=prompt_tokens, cached_input_tokens=cached, output_tokens=completion)
```

### Conditions: tiers, regions, time

```python
calculate_cost("openai", "gpt-5-mini", input_tokens=1_000_000, service_tier="batch")    # $0.125
calculate_cost("anthropic", "claude-sonnet-5", input_tokens=1_000_000, region="us")    # $2.20 (inference_geo=us)
calculate_cost("azure", "gpt-5.4", input_tokens=100_000, region="data-zone")           # Data Zone deployment
calculate_cost("bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", input_tokens=1_000)  # regional rate, explained
calculate_cost("vertex", "gemini-2.5-pro", input_tokens=1_000, region="us-central1")   # non-global Gemini price

from datetime import date, datetime, timezone
calculate_cost("gemini", "gemini-3.8-flash", input_tokens=1_000_000, at=date(2027, 1, 2))  # scheduled price
calculate_cost("deepseek", "deepseek-flash", input_tokens=1_000_000,
               at=datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc))                          # off-peak (Saturday)
```

When you don't specify a condition that affects the price, the provider default is used and
reported in `cost.assumptions`. When a condition is unavailable, you get an error listing the
alternatives rather than a guess:

```
PricingUnavailableError: bedrock/anthropic.claude-sonnet-4-5 has no price for region=ap-south-1, ...;
available: region=global; region=us-east-1; region=us-west-2; ...
```

### Price a provider response

`estimate_response_cost` reads usage from any supported wire format — OpenAI Chat
Completions and Responses (also used by Azure, OpenRouter, Groq, Together, Fireworks,
DeepInfra, Mistral), DeepSeek, Anthropic Messages, Gemini `usageMetadata` and Bedrock
Converse — as SDK objects or JSON dicts. The *billing provider* is passed separately,
because the same format is billed differently by different providers.

```python
estimate_response_cost(openai_response, provider="openai")
estimate_response_cost(converse_response, provider="bedrock", model="anthropic.claude-haiku-4-5")
```

### Compare providers, inspect prices

```python
from openai_cost_calculator import Usage, compare_costs, get_model_pricing, list_models

for cost in compare_costs("anthropic/claude-sonnet-4-5", usage=Usage(input_tokens=50_000, output_tokens=2_000)):
    print(cost.provider, cost.resolved_model, cost.total)

get_model_pricing("bedrock", "global.anthropic.claude-opus-4-8-v1:0").prices  # every tier/region/date
list_models("groq")
```

### Errors

All pricing errors derive from `PricingError` (a `ValueError`) and say what to do next:

| Error | Meaning |
| --- | --- |
| `UnknownProviderError` | lists the supported providers and aliases |
| `UnknownModelError` | includes "did you mean" suggestions |
| `AmbiguousModelError` | an identifier matches several offerings; lists them |
| `PricingUnavailableError` | the model exists but not for the requested tier/region/date; lists what is available |
| `MissingRateError` | usage contains a dimension the model has no price for (e.g. audio tokens on a text model) |
| `UsageError` | negative, non-integer or inconsistent usage |

Missing rates are never silently replaced by another dimension's price, with two deliberate
exceptions: cache reads/writes fall back to the input rate (how providers without separate
cache pricing bill them) and reasoning tokens to the output rate.

---

## Supported providers

| Provider id | Aliases | Pricing source (kind) |
| --- | --- | --- |
| `openai` | | OpenAI pricing docs, Markdown rendition (official docs) |
| `anthropic` | `claude` | Claude API pricing docs, Markdown rendition (official docs) |
| `gemini` | `google-gemini`, `google-ai-studio` | Gemini API pricing docs, Markdown rendition (official docs) |
| `vertex` | `vertex-ai`, `google-vertex` | Vertex AI generative AI pricing page (official docs) |
| `azure` | `azure-openai`, `azure-ai` | Azure Retail Prices API (official API) |
| `bedrock` | `aws-bedrock`, `amazon-bedrock` | AWS Price List API, two offers (official API) |
| `openrouter` | | OpenRouter models API (official API) |
| `deepseek` | | DeepSeek API docs (official docs) |
| `together` | `together-ai` | Together serverless model catalog (official docs) |
| `groq` | | GroqCloud models page (official docs) |
| `fireworks` | `fireworks-ai` | Fireworks serverless pricing docs (official docs) |
| `deepinfra` | | DeepInfra model list API (official API) |
| `mistral` | `mistral-ai` | Mistral API pricing page (official docs) |

`openai-cost-calculator pricing providers` shows model counts and when each provider's data
was last verified. Every `Cost` carries `source_url` and `verified_at`.

---

## Keeping prices current

Prices are data, not code. A weekly GitHub Actions job runs
`openai-cost-calculator pricing sync`, which fetches each provider's official source, parses
it, validates it, diffs it against the checked-in catalog and applies **only** changes that
pass every safety check; anything ambiguous is left unchanged and listed for review in the
pull request it opens. An independent community table (LiteLLM) is used only as a second
opinion that can hold a suspicious change back — never as a source of truth.

See [docs/PRICING_UPDATES.md](docs/PRICING_UPDATES.md) for the full pipeline, what counts as
"confident", and what happens when a page changes format.

---

## Command line

```bash
openai-cost-calculator pricing cost bedrock us.anthropic.claude-sonnet-4-5-20250929-v1:0 --input 12000 --output 800
openai-cost-calculator pricing models anthropic
openai-cost-calculator pricing providers
openai-cost-calculator pricing sync --provider openai --dry-run   # maintainers
openai-cost-calculator pricing stale --max-age-days 45
openai-cost-calculator pricing validate
```

---

## OpenAI response API

The original API is unchanged and remains supported:

```python
from openai import OpenAI
from openai_cost_calculator import estimate_cost, estimate_cost_typed

client = OpenAI()
resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])

estimate_cost_typed(resp).total_cost  # Decimal; also works for the Responses API and streams
estimate_cost(resp)                   # dict of 8-decimal strings (legacy format)
```

For streams, request usage in the final chunk with `stream_options={"include_usage": True}`.

This path keeps its original behaviour: it reads the published
[`data/gpt_pricing_data.csv`](data/gpt_pricing_data.csv) (refreshed at most every 24 hours),
lets local overrides win, and uses only local overrides in offline mode. That CSV is now
**generated from the catalog**, so it is updated by the same official-source pipeline. If it
cannot be downloaded the last good copy, then the bundled catalog, is used instead of failing.

```python
from openai_cost_calculator import add_pricing_entry, refresh_pricing, set_offline_mode

refresh_pricing()          # re-download now
set_offline_mode(True)     # never touch the network
add_pricing_entry("ollama/qwen3:30b", "2025-08-01", input_price=0.20, output_price=0.60, cached_input_price=0.04)
```

See [MIGRATION.md](MIGRATION.md) for how the two APIs relate and what changed.

---

## Integrations

`CostTracker` (per-turn totals for wrapped OpenAI clients), the local accounting proxy, and the
Claude Code and Codex status-line integrations are documented in
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

---

## Limitations

- Prices are list prices. Negotiated discounts, credits, free tiers and invoicing rounding
  are not modelled; subscription (OAuth) usage is an API-equivalent estimate.
- Non-token billing (per image, per second of audio/video, per character, per page,
  provisioned throughput, storage) is out of scope and skipped by the parsers.
- Free-quota tool fees (e.g. Gemini grounding after N free requests) are not modelled.
- Regional pricing covers the regions the sources are configured for (Bedrock: us-east-1,
  us-west-2, eu-central-1; Azure regional deployments: eastus2); other regions raise
  `PricingUnavailableError` rather than guessing.
- DeepSeek's off-peak calendar excludes Chinese public holidays (billed off-peak).
- Cross-provider comparison relies on canonical ids, which are exact for first-party models
  (OpenAI, Anthropic, Google) and best-effort for open-weight models hosted by many providers.

---

## Development

```bash
pip install -r requirements-dev.txt -e ".[proxy]" ruff mypy types-requests
pytest            # unit, contract, regression and compatibility tests (no network)
ruff check .
mypy
```

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — domain model, layering and design decisions
- [docs/ADDING_A_PROVIDER.md](docs/ADDING_A_PROVIDER.md) — adding a provider step by step
- [docs/PRICING_UPDATES.md](docs/PRICING_UPDATES.md) — the automated update pipeline

---

## Links

- **Source:** https://github.com/orkunkinay/openai_cost_calculator
- **Issues:** https://github.com/orkunkinay/openai_cost_calculator/issues

---

## License

MIT © 2025 Orkun Kınay & Murat Barkın Kınay
