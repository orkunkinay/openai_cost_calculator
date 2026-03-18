# openai_cost_calculator

[![PyPI version](https://img.shields.io/pypi/v/openai-cost-calculator)](https://pypi.org/project/openai-cost-calculator/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Instant, accurate **USD cost estimates** for OpenAI & Azure OpenAI API calls. Works with **Chat Completions** and the **Responses API**, streaming or not. Offers a **typed** `Decimal`-based API for finance-safe math and a **legacy** string API for drop-ins.

**Docs:** https://orkunkinay.github.io/openai_cost_calculator/

---

## Installation

```bash
pip install openai-cost-calculator
```

> Import name uses underscores: `import openai_cost_calculator`

---

## Quickstart

**Typed (recommended)**

```python
from openai import OpenAI
from openai_cost_calculator import estimate_cost_typed

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi there!"}],
)

cost = estimate_cost_typed(resp)  # -> CostBreakdown (Decimal fields)
print(cost.total_cost)            # Decimal('0.00000750')
print(cost.as_dict(stringify=True))  # 8-dp strings if you prefer
```

**Legacy (string output)**

```python
from openai_cost_calculator import estimate_cost
print(estimate_cost(resp))  # dict of 8-dp strings
```

**Responses API**

```python
resp = client.responses.create(model="gpt-4.1-mini", input=[{"role":"user","content":"Hi"}])
from openai_cost_calculator import estimate_cost_typed
print(estimate_cost_typed(resp))
```

**Streaming**

```python
stream = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role":"user","content":"Hi"}],
  stream=True,
  stream_options={"include_usage": True},
)
from openai_cost_calculator import estimate_cost_typed
print(estimate_cost_typed(stream))
```

---

## Highlights

- **Typed API:** `CostBreakdown` dataclass with `Decimal` precision  
- **Drop-in legacy API:** 8-decimal strings (backward compatible)  
- **Handles edge cases:** cached tokens, undated model strings, streaming generators, Azure deployment names  
- **Pricing sources:** Remote CSV (24h cache) + **local overrides** and **offline mode**

---

## Comparison: LiteLLM vs openai_cost_calculator

There are multiple ways to estimate LLM usage costs. One alternative is LiteLLM’s `completion_cost` helper, which is part of a broader multi-provider framework.

This library takes a different approach: it focuses on being a **small, framework-independent cost estimation layer** that can be used anywhere.

---

### When to use openai_cost_calculator

This library is designed to be:

* **Lightweight with minimal dependencies** (no heavy frameworks)
* **Simple and transparent** (explicit pricing logic)
* **Framework-independent** (works with any pipeline or stack)
* **Easy to integrate into existing systems**

👉 Best if you want a **minimal, reliable cost estimation utility** without adopting a larger LLM framework.

---

### When to use LiteLLM (`completion_cost`)

LiteLLM provides a more feature-complete solution:

* Supports **multiple providers** (OpenAI, Gemini, Anthropic, etc.)
* Handles **advanced request types** (tool calls, audio/video, service tiers)
* Integrates with a full **LLM routing/orchestration layer**

👉 Best if you are already using LiteLLM or need **broad provider support and advanced features**.

---

### Key Tradeoffs

| Aspect            | openai_cost_calculator | LiteLLM            |
| ----------------- | ---------------------- | ------------------ |
| Dependencies      | Minimal (`requests`)   | Requires LiteLLM   |
| Scope             | Cost estimation only   | Full LLM framework |
| Transparency      | High (explicit logic)  | Abstracted         |
| Setup complexity  | Minimal                | Higher             |
| Advanced features | Limited                | Extensive          |

---

## Pricing utilities

```python
from openai_cost_calculator import (
  refresh_pricing, set_offline_mode,
  add_pricing_entry, add_pricing_entries, clear_local_pricing
)

# Force refresh (bypasses 24h cache)
refresh_pricing()

# Run fully offline (no network calls)
set_offline_mode(True)

# Teach custom prices (per 1M tokens)
add_pricing_entry(
  "ollama/qwen3:30b", "2025-08-01",
  input_price=0.20, output_price=0.60, cached_input_price=0.04,
  minimum_tokens=0,  # optional tier floor; default is 0
)
```

If a model has tiered pricing by prompt/input size, add multiple rows for the same
`(model_name, model_date)` with different `minimum_tokens` values. The calculator picks
the highest matching tier where `minimum_tokens <= prompt_tokens`.

Remote CSV (auto-fetched, cached 24h):  
`https://raw.githubusercontent.com/orkunkinay/openai_cost_calculator/refs/heads/main/data/gpt_pricing_data.csv`

---

## Errors

Recoverable issues raise `CostEstimateError` with a clear message (missing pricing row, unexpected input shape, etc.).

---

## Troubleshooting

- **“Pricing not found”** → confirm row exists in the CSV; call `refresh_pricing()`.  
- **`cached_tokens = 0`** → ensure `include_usage_details=True` (classic) or `stream_options={"include_usage": True}` (streaming).  
- **Model string has no date** → the latest row with `date ≤ today` is used.

---

## Links

- **Docs & examples:** https://orkunkinay.github.io/openai_cost_calculator/  
- **Source:** https://github.com/orkunkinay/openai_cost_calculator  
- **Issues:** https://github.com/orkunkinay/openai_cost_calculator/issues

---

## License

MIT © 2025 Orkun Kınay & Murat Barkın Kınay
