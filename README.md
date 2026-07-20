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
HTTP streaming and WebSocket messages are relayed incrementally without changing upstream bytes, message types, ordering, queries, or negotiated subprotocols.
Terminal WebSocket Responses events are costed immediately, and duplicate terminal events are ignored.

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
Use the SQLite backend when totals must survive restarts or more than one proxy process must share accounting state:

```bash
openai-cost-calculator proxy --port 8100 --auth-mode auto --database ~/.local/state/openai-cost-calculator/accounting.sqlite3
```

SQLite uses WAL, full synchronous transactions, concurrent readers and writers, incremental call inserts, streamed summaries, and an atomic checkpoint transaction.
It works across platforms supported by Python's standard `sqlite3` module and does not rely on POSIX file locking.
Several proxy processes may share one local SQLite database, and the first successful checkpoint transaction consumes each call exactly once across those processes.
SQLite storage is not supported on filesystems whose locking semantics are incompatible with SQLite, including many network filesystem configurations.

The database stores model names, token counts, calculated `Decimal` cost components, turn labels, bounded diagnostics, and checkpoint cursors.
It never stores request or response bodies, authorization headers, cookies, or authentication files.
Historical calls retain their originally calculated prices rather than being repriced after a restart.
Status output separates `historical_total` present at startup from `process_total` added since startup.
After a reset by any process, all connected processes detect the new storage generation and restart those process-relative totals safely.

Inspect or reset the database while proxies are running or stopped:

```bash
openai-cost-calculator database inspect ~/.local/state/openai-cost-calculator/accounting.sqlite3
openai-cost-calculator database reset ~/.local/state/openai-cost-calculator/accounting.sqlite3 --yes
```

The previous atomic JSON snapshot backend remains available for compatibility:

```bash
openai-cost-calculator proxy --port 8100 --ledger ~/.local/state/openai-cost-calculator/ledger.json
openai-cost-calculator ledger inspect ~/.local/state/openai-cost-calculator/ledger.json
openai-cost-calculator ledger reset ~/.local/state/openai-cost-calculator/ledger.json --yes
```

JSON ledgers remain single-process and POSIX-only; new installations should use SQLite.
JSON writes use a same-directory temporary file, `fsync`, exclusive locking, and atomic replacement.
To keep legacy in-memory snapshots bounded, JSON mode accepts up to 1,024 sessions and 50,000 calls per session.
SQLite removes the per-session call-history limit and aggregates summaries and checkpoints without reconstructing every historical call in memory.
Both backends retain at most 100 diagnostics per session and report capacity or storage failures explicitly.

Use the proxy reset command while the proxy owns the ledger:

```bash
openai-cost-calculator reset --yes
```

Reset is destructive and clears cumulative totals, diagnostics, and checkpoint cursors.

### Remote exposure and administrative authentication

The proxy binds to loopback by default.
A non-loopback bind is refused unless remote exposure is explicit and a strong administrative bearer token is configured.
Create a permission-protected token file and pass only its path:

```bash
mkdir -p ~/.config/openai-cost-calculator
python -c 'import secrets; print(secrets.token_urlsafe(48))' > ~/.config/openai-cost-calculator/admin-token
chmod 600 ~/.config/openai-cost-calculator/admin-token
OCC_ADMIN_TOKEN_FILE=~/.config/openai-cost-calculator/admin-token \
  openai-cost-calculator proxy --host 0.0.0.0 --allow-remote --auth-mode auto
```

When configured, the token protects every `/_occ` endpoint, including status streams and mutation commands.
The CLI, notifier, and status line read `OCC_ADMIN_TOKEN` or `OCC_ADMIN_TOKEN_FILE` and send the token only in the administrative `Authorization` header.
Tokens must contain at least 32 characters, token files must not be accessible by group or others, and token values are never printed or persisted by the package.
Remote deployments must additionally use TLS and trusted network controls, normally through a reverse proxy.

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

## Proxy-backed Claude Code cost observation

The `install claude-code` adapter above reads Claude Code's own reported cost from the transcript.
The proxy-backed integration below is independent: it routes Claude Code's Anthropic Messages traffic through the local OCC proxy, prices every request from this project's Anthropic pricing table, and shows both the current-turn and cumulative-session cost that OCC observed.

Why a proxy: Claude uses the Anthropic Messages API, whose usage schema separates uncached input, cache-creation writes (with 5-minute and 1-hour TTLs), cache reads, and output.
These are priced independently and cannot be expressed by the OpenAI three-bucket schema, so the Anthropic path is kept separate from the Codex/Responses path rather than forced into it.

Install the settings.json integration and start the matching proxy:

```bash
openai-cost-calculator claude install --proxy-url http://127.0.0.1:8100
openai-cost-calculator proxy --port 8100 --protocol anthropic-messages
```

The installer merges `env.ANTHROPIC_BASE_URL` (and `env.OCC_PROXY_URL`), the `occ-claude-statusline` status line, and the `occ-claude-hook` lifecycle hooks (`UserPromptSubmit`, `Stop`, `SessionEnd`) into `settings.json` under `CLAUDE_CONFIG_DIR` (or `~/.claude`).
It writes atomically, records a non-sensitive manifest for exact restoration, never touches authentication tokens, managed settings, MCP, permissions, plugins, or agents, and refuses to overwrite an existing status line unless you pass `--replace-statusline`.
An existing `ANTHROPIC_BASE_URL` gateway is preserved and reported so you can chain to it with `proxy --upstream <url>`, and it is restored on uninstall.

The status line shows both totals directly inside Claude Code:

```text
OCC Turn $0.0124 · Session $0.0837
```

Subscription (Pro/Max/Team/Enterprise) OAuth sessions are labelled `API-eq` because token pricing is an API-equivalent estimate, not an amount actually billed:

```text
OCC Turn API-eq $0.0124 · Session API-eq $0.0837
```

When no turn is active the line shows the last completed turn (`OCC Last turn …`); a just-opened turn shows `OCC Turn $0.0000` until its first request is accounted; and an accounting gap shows `inspect diagnostics` or `OCC cost unavailable` rather than a false zero.

Definitions.
A *session* is one Claude Code session, keyed by its stable session id (`x-claude-code-session-id`); its total is the sum of every accounted request in the session and increases monotonically across turns.
A *turn* is one user prompt: it opens on `UserPromptSubmit`, aggregates every Messages request that belongs to it — including tool-use continuations, compaction, retries, and main-agent and subagent calls — and finalizes on `Stop` (completed) or `SessionEnd` (interrupted).
Subagent cost is included in both the turn and the session total.
Turn boundaries come from the hooks, but request cost is recorded at the proxy as each response is observed, so totals update without waiting for the final hook.
A *checkpoint* is a separate incremental cursor: the first checkpoint returns the cost accumulated since the previous one, a repeated checkpoint returns zero, and status reads never consume a checkpoint or mutate any total.

Inspect and manage from the CLI:

```bash
openai-cost-calculator claude status          # current/last turn and session cost
openai-cost-calculator claude status --json    # stable, decimal-safe JSON
openai-cost-calculator claude check            # effective config and conflicts
openai-cost-calculator claude checkpoint       # consume the next incremental delta
openai-cost-calculator claude pricing validate # validate the Anthropic pricing table
```

Cost semantics and supported providers.
A direct Anthropic API key yields a close *billed estimate* (still not an invoice: credits, tiers, and beta pricing may differ).
Subscription OAuth yields an *API-equivalent* estimate.
The proxy infers which of these applies from the credential *header kind* — an API key is sent as `x-api-key`, subscription OAuth as `authorization: Bearer` — so a subscription login stored in the OS keychain is still labelled `API-eq` even though the resolver cannot read it.
Direct Anthropic API keys, subscription OAuth via a local `ANTHROPIC_BASE_URL`, bearer-token Anthropic-format gateways, and chainable custom Anthropic-format base URLs are supported; a custom gateway's cost is reported as unavailable unless a pricing profile is known.
Amazon Bedrock, Google Vertex, and Microsoft Foundry are detected and refused rather than mispriced, because their payloads are not the Anthropic Messages protocol; unsupported providers never produce a false zero-cost result.

Session attribution uses the `x-claude-code-session-id` request header by default.
If your Claude Code build sends a different header, set `OCC_CLAUDE_SESSION_HEADER` on the proxy to that name — no code change needed.
To confirm which headers Claude Code actually sends when routed through the proxy, start it with `OCC_CLAUDE_DEBUG_HEADERS=1` and run one session; the proxy records the inbound header *names* (never their values) once per session as a `claude_headers_seen` diagnostic visible via `claude status --diagnostics`.

Streaming responses are commonly `gzip`-encoded; the proxy forwards the raw bytes unchanged and decodes a private copy only for accounting.

Persistence and limitations.
Add `--database <path>` to the proxy for a concurrent SQLite ledger; turn and session totals *and the turn lifecycle* (open/finalized turn states) survive proxy restarts, and restored history is reported separately from current-process cost.
The JSON ledger persists the same lifecycle in its snapshot.
The integration has been validated with one real Claude Code (2.1.x) subscription request end to end: the request routed through the proxy, `x-claude-code-session-id` attributed it to the active turn, two Messages requests aggregated into one turn, the gzip stream was priced, and the status line showed matching API-equivalent turn and session totals with an idempotent checkpoint.
A subscription (Keychain) login is used read-only against the real config directory, because it is not visible inside an isolated empty home; file-backed or `ANTHROPIC_API_KEY` credentials run fully isolated.

### Maintainer Claude self-test

`scripts/self_test_claude_integration.py` is an opt-in, paid, end-to-end test; ordinary tests never contact a paid service.

```bash
python scripts/self_test_claude_integration.py
```

It creates a unique temporary working directory and an isolated `CLAUDE_CONFIG_DIR`, copies only minimal file-backed authentication when present (never printing or exporting it, never reading keychain secrets, never modifying the source configuration), installs the OCC hooks and status line, starts the anthropic-messages proxy, launches one tool-free `claude` prompt with no automatic paid retries, independently recomputes the request cost from the Anthropic pricing table, verifies that request, turn, session, and first-checkpoint costs agree while the second checkpoint is zero and status reads do not mutate totals, terminates the proxy, and removes all temporary material.
If no safe authentication prerequisite exists it states that the deterministic paths passed and the live path remains untested rather than fabricating success.

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
`http://127.0.0.1:8100/v1`, and uses `wire_api = "responses"`.
HTTP is the default because it preserves the validated one-request self-test behavior.
The proxy also observes Responses WebSocket traffic; opt in when desired:

```bash
openai-cost-calculator install codex --websockets --proxy-url http://127.0.0.1:8100 --session default
```

Codex may issue more than one Responses request for one CLI turn when its WebSocket transport is enabled, and every completed response is accounted independently.
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

The installer changes only managed blocks plus the top-level `notify` and `model_provider` assignments it must replace.
Those assignments are converted in place to reversible lexical placeholders rather than moved or normalized.
Unrelated comments, whitespace, quoting, Unicode, line endings, section order, inline comments, and the presence or absence of a final newline are preserved byte-for-byte.
Uninstall decodes the original assignment bytes at their original locations, while unrelated user edits made after installation remain intact.
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
Use `--auth-mode chatgpt` to require an isolated copy of an existing ChatGPT login.
Use `--auth-mode api-key` to require `OPENAI_API_KEY` and exercise the Platform route even when the source Codex installation is logged in through ChatGPT:

```bash
OPENAI_API_KEY=... python scripts/self_test_codex_integration.py --auth-mode api-key --model MODEL
```

The API key is passed to `codex login --with-api-key` over stdin and never appears in a command line or report.
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
- **JSON ledger is already in use** → inspect through the running proxy, stop it before using offline `ledger` commands, or migrate future operation to SQLite with `--database`.
- **Ledger health is `ERROR`** → current in-memory totals may be newer than the last durable snapshot; fix the filesystem error before relying on restart recovery.
- **`admin_auth_required`** → set the same `OCC_ADMIN_TOKEN` or `OCC_ADMIN_TOKEN_FILE` for the proxy, CLI, notifier, and status-line processes.
- **Remote bind refused** → add `--allow-remote` and a protected administrative token only after TLS and network access controls are ready.
- **`websocket_missing_terminal`** → the upstream connection closed before a terminal Responses event; inspect diagnostics and upstream connectivity without assuming a zero-cost success.
- **WebSocket handshake failure** → confirm the selected authentication domain, custom upstream WebSocket support, forwarded subprotocol, and proxy optional dependencies.
- **Claude `401` or subscription OAuth routing failure** → confirm Claude Code is logged in and that `env.ANTHROPIC_BASE_URL` points at the running proxy; the proxy forwards Claude's own credentials and never mixes API-key and subscription domains.
- **Claude cost shows `unavailable` or `inspect diagnostics`** → run `openai-cost-calculator claude status --diagnostics`; a custom gateway or unsupported provider (Bedrock/Vertex/Foundry) reports cost as unavailable rather than a false zero.
- **Claude turn cost is unavailable while the session cost is known** → one request in the turn could not be priced (unknown model or pricing gap); the session total remains correct and the unpriced request is listed in diagnostics.
- **Claude status line differs from Claude Code's own estimate** → they are independent; Claude Code's `cost.total_cost_usd` is a reconciliation signal, not the OCC ledger, and is never added to OCC totals.
- **Claude project settings override user settings** → `openai-cost-calculator claude check` reports a project `.claude/settings.json` that overrides `ANTHROPIC_BASE_URL`.
- **Claude turn attribution failure (`turn_unattributed`)** → a request arrived with no open turn (for example before the first `UserPromptSubmit` hook); it is recorded under a named session-only synthetic turn and still counted in the session total.

Keep the default loopback bind unless remote access is operationally necessary.

---

## Links

- **Docs & examples:** https://orkunkinay.github.io/openai_cost_calculator/  
- **Source:** https://github.com/orkunkinay/openai_cost_calculator  
- **Issues:** https://github.com/orkunkinay/openai_cost_calculator/issues

---

## License

MIT © 2025 Orkun Kınay & Murat Barkın Kınay
