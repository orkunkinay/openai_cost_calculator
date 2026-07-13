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

Run the local OpenAI-compatible proxy for a Platform API-key client:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode api-key
```

Point any OpenAI-compatible agent, IDE, or SDK client at:

```text
http://127.0.0.1:8100/v1
```

The proxy forwards Platform API-key requests to `https://api.openai.com/v1` and ChatGPT-backed Codex requests to `https://chatgpt.com/backend-api/codex`.
`--auth-mode auto` reads only the `auth_mode` field in Codex authentication metadata and otherwise retains the backward-compatible Platform default.
It never silently retries a request against the other authentication domain.
Known-incompatible authentication and upstream combinations fail at startup.
To use a trusted OpenAI-compatible upstream explicitly:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode api-key --upstream https://example.com/v1
```

An explicit custom upstream receives the forwarded `Authorization` header, so only configure an endpoint you trust.
The proxy removes hop-by-hop transport headers and its local `X-OCC-Session` and `X-OCC-Turn` headers while preserving required end-to-end headers.

Read accumulated costs at:

```text
http://127.0.0.1:8100/_occ/costs
```

For grouping, send `X-OCC-Session` to separate agent sessions and `X-OCC-Turn` to group calls within a visible turn.
Calls without a session use `default`, and calls with the same session and turn label aggregate into one turn.
Streaming calls should set `stream_options={"include_usage": true}` so the final SSE usage event can be costed.
The proxy forwards streaming bytes incrementally and performs accounting only as a side effect.
An accounting failure never replaces or edits a successful upstream response.

Cached input is removed from uncached input before pricing, then charged only at the cached-input rate.
Malformed usage where cached input exceeds total input is reported as a diagnostic and is not recorded as a zero-cost success.
Pricing is selected by model alias, model date, and the highest applicable minimum-token tier.

### Durable accounting

Accounting is in memory by default and is lost when the proxy exits.
Enable the optional durable JSON ledger when totals must survive restarts:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode auto --ledger ~/.local/state/openai-cost-calculator/ledger.json
```

The ledger stores model names, token counts, calculated `Decimal` cost components, turn labels, bounded diagnostics, and checkpoint cursors.
It never stores request or response bodies, authorization headers, cookies, or authentication files.
Writes use a same-directory temporary file, `fsync`, and atomic replacement, so an interrupted write leaves the last complete ledger usable.
Historical calls retain their originally calculated prices rather than being repriced after a restart.
Status output separates `historical_total` loaded at startup from `process_total` observed by the current proxy.

One ledger supports exactly one proxy process at a time.
The proxy holds an exclusive file lock and refuses a second writer instead of claiming unsupported cross-process safety.
Use a different ledger for each simultaneously running proxy.

Inspect or reset an offline ledger with:

```bash
openai-cost-calculator ledger inspect ~/.local/state/openai-cost-calculator/ledger.json
openai-cost-calculator ledger reset ~/.local/state/openai-cost-calculator/ledger.json --yes
```

Use the proxy reset command while the proxy owns the ledger:

```bash
openai-cost-calculator reset --yes
```

Reset is destructive and clears cumulative totals, diagnostics, and checkpoint cursors.

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

Run the proxy and let it select the route from Codex's existing login metadata:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode auto
```

The Codex installer adds a managed `~/.codex/config.toml` block that selects an
`openai_cost_calculator` custom provider, points it at
`http://127.0.0.1:8100/v1`, uses `wire_api = "responses"`, and disables
websockets so the HTTP proxy can observe usage.
The provider uses Codex's existing OpenAI authentication, so sign in with ChatGPT or an API key before using the integration.
The configured session is sent as an `X-OCC-Session` header so proxy accounting and notifications use the same isolated session.

Use an explicit mode when automatic detection is unavailable:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode chatgpt
openai-cost-calculator proxy --port 8100 --auth-mode api-key
```

Codex notifications use the proxy checkpoint endpoint, so one `agent-turn-complete` notification consumes all calls since the preceding successful checkpoint:

```text
💰 $0.0188 session · last $0.0032
💰 Turn $0.0032 · gpt-5.5 1.2k->500
```

Checkpoint consumption is incremental and exactly once from the package's perspective.
Repeating a notifier after a successful checkpoint returns zero and does not change cumulative session totals.
If a durable cursor cannot be written, the checkpoint fails and remains unconsumed.
Status reads never consume checkpoints or mutate totals.

Codex does not pass token or cost data to `notify`; these numbers come from the local proxy.
Some Codex surfaces, including `codex exec`, run the notifier but do not display its stdout.
Notifier failures never fail Codex and are written to a bounded, permission-protected diagnostic file in `CODEX_HOME`.
Inspect proxy and notifier diagnostics without reading internal files:

```bash
openai-cost-calculator status --session default --diagnostics
openai-cost-calculator status --session default --json
openai-cost-calculator checkpoint --session default --json
```

`occ-codex-statusline` remains available for wrappers or Codex surfaces that can run an external status command.

The installer changes only managed blocks plus the top-level `notify` and `model_provider` keys it must replace.
It preserves the previous values inside the managed block so uninstall can restore them.
It uses permission-preserving atomic replacement and refuses invalid TOML, malformed managed markers, symlinked configuration, or an existing user-owned provider named `openai_cost_calculator`.
It does not copy the configuration into retained backup files because user configuration may contain secrets.
If safe automatic merging is impossible, installation stops before modifying the file.

Undo either install with:

```bash
openai-cost-calculator uninstall claude-code
openai-cost-calculator uninstall codex
```

### Maintainer Codex self-test

After installing the current checkout in editable mode with proxy and development dependencies, run the real integration self-test:

```bash
python scripts/self_test_codex_integration.py
```

The script uses a free local port and a unique session, creates an isolated temporary `CODEX_HOME`, installs the adapter through the public CLI, copies an existing file-backed Codex login or imports `OPENAI_API_KEY` without printing it, launches one read-only `codex exec` turn, independently recomputes the cost from the working-tree pricing CSV, verifies checkpoint stability, and removes all temporary files.
Use `--model` or `--upstream` only when the detected Codex login needs an explicit override.
The command exits nonzero with a sanitized diagnostic when credentials, model access, routing, usage, pricing, or accounting validation fails.
The live test is opt-in, makes exactly one paid model request when authentication succeeds, and costs whatever that request's selected model and token usage cost.
Ordinary unit tests never contact an OpenAI service.

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

- **401 from `api.openai.com` with a ChatGPT login** → restart with `--auth-mode chatgpt`; ChatGPT-backed Codex credentials are not Platform API keys.
- **401 from `chatgpt.com/backend-api/codex` with an API key** → restart with `--auth-mode api-key`; the proxy never falls back between authentication domains.
- **Unsupported or inaccessible model** → confirm the selected login has access and that the isolated self-test inherited the intended source model, or pass `--model` explicitly.
- **“Pricing not found” or `cost_estimation_failed`** → run `openai-cost-calculator pricing validate`, confirm the model has a pricing row, and inspect `openai-cost-calculator status --diagnostics`.
- **`missing_usage`** → for streaming Chat Completions request final usage with `stream_options={"include_usage": true}`; the proxy cannot infer tokens from response text.
- **`cached_tokens = 0`** → the upstream did not report cached-token details; missing cached fields are treated as zero, not guessed.
- **Model string has no date** → the latest pricing row with `date ≤ today` is used, including intentionally undated aliases.
- **Notifier output is hidden** → use `openai-cost-calculator status --diagnostics`; Codex execution does not depend on notifier stdout.
- **Ledger is already in use** → inspect through the running proxy or stop it before using offline `ledger` commands.
- **Ledger health is `ERROR`** → current in-memory totals may be newer than the last durable snapshot; fix the filesystem error before relying on restart recovery.

The administrative `/_occ` endpoints have no authentication.
Keep the default loopback bind unless a separate trusted access-control layer protects the proxy.

---

## Links

- **Docs & examples:** https://orkunkinay.github.io/openai_cost_calculator/  
- **Source:** https://github.com/orkunkinay/openai_cost_calculator  
- **Issues:** https://github.com/orkunkinay/openai_cost_calculator/issues

---

## License

MIT © 2025 Orkun Kınay & Murat Barkın Kınay
