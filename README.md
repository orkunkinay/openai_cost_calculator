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

## Tracking agent turns

Use `CostTracker` when one user-visible turn can make many OpenAI calls and you
want a running total.

```python
from openai import OpenAI
from openai_cost_calculator import CostTracker

tracker = CostTracker()
client = tracker.wrap(OpenAI())

with tracker.turn("Refactor auth") as turn:
    client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "Plan the refactor"}],
    )
    client.responses.create(
        model="gpt-5.4-mini",
        input=[{"role": "user", "content": "Summarize the changes"}],
    )

print(turn.total_cost)
print(turn.cost_by_model)
print(tracker.session_total)
```

For streaming calls, pass `stream_options={"include_usage": True}` so the final
usage chunk is available for cost tracking.

---

## Zero-config tracking with the proxy

Install the optional proxy dependencies:

```bash
pip install openai-cost-calculator[proxy]
```

Run the local OpenAI-compatible proxy:

```bash
openai-cost-calculator proxy --port 8100
```

Point any OpenAI-compatible agent, IDE, or SDK client at:

```text
http://127.0.0.1:8100/v1
```

The proxy forwards requests to `https://api.openai.com/v1` by default, preserving
the request body and `Authorization` header. To use another OpenAI-compatible
upstream:

```bash
openai-cost-calculator proxy --port 8100 --upstream https://example.com/v1
```

Read accumulated costs at:

```text
http://127.0.0.1:8100/_occ/costs
```

For grouping, send `X-OCC-Session` to separate agent sessions and `X-OCC-Turn`
to group calls within a visible turn. Streaming calls should set
`stream_options={"include_usage": true}` so the final SSE usage event can be
costed.

---

## Show cost inside Claude Code & Codex

Install the Claude Code UI adapters:

```bash
openai-cost-calculator install claude-code
```

This adds a Claude Code status line command and a `Stop` hook. The status line
shows the running Claude Code session cost, and the hook prints an inline
per-turn line after each assistant response:

```text
💰 $0.0123 session · last 8.5k->1.2k tok (cache 2.0k) · Sonnet 4.6 · ctx 14%
💰 This turn cost $0.0041 (12.3k in / 1.1k out)
```

Claude Code session totals come from Claude Code's own status-line JSON. The
per-turn hook prefers exact per-message costs from the local transcript; if only
usage is available, it estimates from seeded Claude prices.

Install the Codex adapters:

```bash
openai-cost-calculator install codex --proxy-url http://127.0.0.1:8100 --session default
```

Run the proxy and route Codex's OpenAI-compatible provider through it:

```bash
openai-cost-calculator proxy --port 8100
```

Codex notifications use the proxy checkpoint endpoint, so one
`agent-turn-complete` notification corresponds to one cost checkpoint:

```text
💰 $0.0188 session · last $0.0032
💰 Turn $0.0032 · gpt-5.5 1.2k->500
```

Codex does not pass token or cost data to `notify`; these numbers come from the
local proxy. Current Codex docs expose `notify` as an external program and the
TUI status line as built-in footer items, so `occ-codex-statusline` is provided
for wrappers or future Codex builds that can run an external status command.

Undo either install with:

```bash
openai-cost-calculator uninstall claude-code
openai-cost-calculator uninstall codex
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
