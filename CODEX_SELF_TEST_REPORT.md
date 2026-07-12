# Codex Integration Self-Test Report

## Result

The final real self-test passed.
A separate Codex process sent one real streaming Responses API request through this repository's local proxy, returned exactly `OCC_SELF_TEST_OK`, and produced fresh token and cost accounting in a unique session.
The independently recomputed cost matched the proxy, cumulative session total, first checkpoint, and status line.
A repeated checkpoint returned zero and did not mutate cumulative costs.

## Environment

- Date: 2026-07-12.
- Operating system: macOS 15.5, build 24F74, Apple Silicon.
- Python: CPython 3.13.0.
- Package version: 1.2.0.
- Package installation: editable install from the current working tree with `.[proxy]` and `requirements-dev.txt` in `.context/venv`.
- Codex: `codex-cli 0.144.1`.
- Proxy port for the final run: 52500.
- Proxy upstream for the final run: `https://chatgpt.com/backend-api/codex`.
- Codex home: isolated temporary `CODEX_HOME`.
- Authentication prerequisite: satisfied by copying the existing file-backed ChatGPT Codex login into the temporary home with mode `0600`.
- `OPENAI_API_KEY`: not present and not required for the tested ChatGPT-login path.
- Secrets: no credential values were printed, copied into the repository, written to the report, or retained after cleanup.

## Product Contract and Acceptance Criteria

The package exposes the `openai-cost-calculator`, `occ-codex-notify`, and `occ-codex-statusline` entry points.
The proxy forwards `/v1/{path}` requests, supports buffered JSON and streaming SSE Responses payloads, extracts final Responses usage, and stores an in-memory ledger grouped by `X-OCC-Session` and optional `X-OCC-Turn`.
The tracker subtracts cached input from uncached input before applying per-million-token rates with `Decimal` arithmetic.
Pricing resolution selects the newest applicable model date and the highest `Minimum Tokens` tier that does not exceed prompt tokens.
Checkpoints advance a per-session record cursor while cumulative status reads remain non-mutating.
The Codex installer selects a managed Responses provider, disables WebSockets, reuses Codex OpenAI authentication, and sends the configured session header.

The real acceptance criteria were one fresh successful nested request, the fixed response, a unique session, positive input and output usage, one observed model, exactly one recorded call, non-negative cost, exact independent recomputation, equal request/turn/session/checkpoint totals, stable status reads, a zero second checkpoint, installer idempotency, isolated credentials, and complete temporary cleanup.

## Baseline

The first system-Python installation attempt was refused by the macOS externally managed environment guard.
No system environment was modified; the baseline was rerun in `.context/venv`.

The untouched editable baseline collected 49 tests and passed all 49 in 0.89 seconds.
CLI help and package/proxy import smoke tests passed.
The repository defines no formatter, linter, or type-check command, so none was available to run.
The baseline emitted one Starlette/httpx deprecation warning and one `datetime.utcnow()` deprecation warning.

Inspection found that the installer configured `occ_codex_session` for the notifier but did not add `X-OCC-Session` to provider requests.
Traffic would therefore accumulate under `default` while the notifier queried the named session.
Inspection also found that the installer required `OPENAI_API_KEY` even though Codex was already authenticated through ChatGPT and supports `requires_openai_auth` for custom providers.

The first real routing attempt reached the local proxy and `/v1/responses`, but the documented Platform upstream rejected the forwarded ChatGPT token with HTTP 401 because it did not have the `api.responses.write` scope.
The next attempt used the ChatGPT Codex upstream but confirmed that `gpt-5.3-codex` was unavailable for this ChatGPT account.
The harness was corrected to derive the upstream from the login mode and to use the source Codex model unless `--model` is explicitly supplied.

## Final Real Self-Test Evidence

The final unique session was `occ-self-test-1783892942`.

The sanitized proxy command was:

```text
"$REPO/.context/venv/bin/python" "-m" "openai_cost_calculator.cli" "proxy" "--host" "127.0.0.1" "--port" "52500" "--upstream" "https://chatgpt.com/backend-api/codex"
```

The sanitized nested Codex command was:

```text
"/opt/homebrew/bin/codex" "exec" "--ephemeral" "--skip-git-repo-check" "--ignore-rules" "--sandbox" "read-only" "--color" "never" "--model" "gpt-5.6-sol" "--config" "notify=[]" "--config" "model_reasoning_effort=\"low\"" "--output-last-message" "$TMP/last-message.txt" "--cd" "$TMP/empty-workdir" "Reply with exactly `OCC_SELF_TEST_OK`. Do not inspect or modify files."
```

`notify=[]` was applied only to the nested command so the harness could read and verify the checkpoint itself instead of allowing `codex exec` to consume it through a notifier whose stdout Codex does not display.
The installed configuration still contained `notify = ["occ-codex-notify"]`, and notifier behavior remains covered by focused adapter tests.

- Nested Codex exit status: 0.
- Nested Codex final response: `OCC_SELF_TEST_OK`.
- Observed model: `gpt-5.6-sol`.
- Recorded calls: 1.
- Input tokens: 9,007.
- Cached input tokens: 6,912.
- Uncached input tokens: 2,095.
- Output tokens: 9.
- Pricing row: undated `gpt-5.6-sol` alias, `Minimum Tokens = 0`.
- Input rate: $5.00 per million tokens.
- Cached input rate: $0.50 per million tokens.
- Output rate: $30.00 per million tokens.
- Recorded request and turn cost: $0.01420100.
- Session total before: $0.00000000.
- Session total after: $0.01420100.
- First checkpoint cost: $0.01420100.
- Second checkpoint cost: $0.00000000.
- Status line: `💰 $0.0142 session · last $0.0142`.
- Installer idempotency: byte-for-byte pass.
- All values matched: yes.

## Independent Cost Recalculation

The harness parsed the working-tree `data/gpt_pricing_data.csv` directly instead of asking the proxy or tracker to recalculate its own result.
The 9,007-token prompt selected the minimum-zero pricing tier because it is below the 272,001-token tier threshold.

```text
uncached input = 2,095 × $5.00 / 1,000,000  = $0.010475
cached input   = 6,912 × $0.50 / 1,000,000  = $0.003456
output         =     9 × $30.00 / 1,000,000 = $0.000270
total                                             $0.014201
8-decimal proxy representation                    $0.01420100
```

Cached tokens were charged only at the cached rate and were removed from the uncached token count.
The independent total equaled the request, turn, session, first checkpoint, and model total.
The second checkpoint was zero, and repeated status/checkpoint reads left cumulative costs unchanged.

## Defects Found and Corrected

### Session routing mismatch

Symptom: the notifier queried the installed session, but Codex provider requests did not carry that session and would be recorded under `default`.
Root cause: the managed provider omitted Codex's supported static `http_headers` configuration.
Correction: the installer now writes `http_headers = { "X-OCC-Session" = "<session>" }`.
Regression coverage: `test_installers_are_idempotent_and_reversible` verifies the exact managed header and preserved unrelated settings.

### Authentication configuration excluded ChatGPT login

Symptom: a valid existing ChatGPT Codex login could not satisfy the installer's `env_key = "OPENAI_API_KEY"` requirement.
Root cause: the provider selected one credential source instead of Codex's OpenAI authentication abstraction.
Correction: the managed provider now uses `requires_openai_auth = true`, and installation guidance covers both ChatGPT and API-key upstreams.
Regression coverage: the installer test verifies `requires_openai_auth`, rejects an embedded bearer token, and the final real run reused an isolated ChatGPT login successfully.

### Authentication and upstream mismatch in the self-test workflow

Symptom: forwarding a ChatGPT access token to `api.openai.com/v1` produced a real 401, while a fixed test model was not available to the ChatGPT account.
Root cause: the initial harness assumed API-key routing and a globally available model.
Correction: the harness detects `auth_mode`, selects the corresponding OpenAI upstream, and defaults to the model already configured for the source Codex installation.
Regression coverage: the final no-override self-test selected the ChatGPT backend and `gpt-5.6-sol` and passed.

### Silent missing-usage and pricing failures

Symptom: successful responses without usage and unsupported pricing could appear as empty `$0.00` sessions.
Root cause: proxy recording returned early or swallowed tracker errors without exposing them.
Correction: the registry now records sanitized per-session diagnostics with codes such as `missing_usage`, `missing_model`, and `cost_estimation_failed`, and publishes diagnostic changes to stream subscribers.
Regression coverage: `test_missing_usage_is_reported_instead_of_appearing_as_zero_cost_success` and `test_costing_failure_is_swallowed_and_response_is_unchanged` verify explicit diagnostics without changing upstream responses.

### Hop-by-hop request forwarding

Symptom: request hop-by-hop headers and headers named by `Connection` could be forwarded upstream.
Root cause: request filtering removed only `Host` and `Content-Length`.
Correction: request and response forwarding now removes the standard hop-by-hop set plus dynamically nominated `Connection` headers while preserving `Authorization` and end-to-end headers.
Regression coverage: `test_request_hop_by_hop_headers_are_not_forwarded` verifies credential forwarding and hop-by-hop exclusion.

### Unsafe modification of invalid Codex configuration

Symptom: the installer could rewrite an already invalid TOML file, and direct writes could leave a truncated file if interrupted.
Root cause: installation did not parse the existing file and wrote directly to the destination.
Correction: install and uninstall refuse invalid TOML before backup or mutation, and JSON/TOML configuration writes now use a flushed atomic replacement while preserving permissions.
Regression coverage: `test_codex_installer_refuses_invalid_config_without_modifying_it` verifies unchanged invalid input and no backup or temporary-file residue.

## Verification

- Focused proxy and adapter tests: 15 passed before installer hardening, then 7 adapter tests passed after hardening.
- Final complete suite: 52 passed in 0.25 seconds.
- Final CLI help smoke: passed.
- Final package and proxy import smoke: passed.
- Final editable installation: passed.
- Final real nested Codex self-test: passed.
- Final status-line check: passed.
- Installer idempotency: passed byte for byte.
- Uninstall restoration with unrelated settings: passed in the adapter suite.
- Invalid-config preservation: passed.
- Temporary cleanup: passed; the harness terminated the proxy in `finally`, removed its temporary Codex home and work directory, and left no `occ-codex-self-test-*` directory.
- Original user configuration: untouched; the harness only read the source configuration and copied file-backed authentication into its temporary home.
- Formatter/linter: not run because the repository configures none.
- Type checker: not run because the repository configures none.

The final suite still reports two pre-existing warnings: Starlette's deprecated `httpx` test-client integration and the package's use of deprecated `datetime.utcnow()`.

## Remaining Limitations

The proxy ledger is intentionally in memory, so costs are lost when the proxy process restarts and separate proxy processes do not share totals.
The real run covered ChatGPT-login authentication; API-key authentication remains covered by configuration and forwarding tests but was not exercised with a live key because none was available.
Streaming, final-event usage, upstream-error, missing-usage, unknown-model, header, checkpoint, session-isolation, and installer edge cases are deterministic local tests; only the primary Responses path was exercised against the real service.
`codex exec` does not display notifier stdout, so the harness disables notify for its one nested turn, verifies the real checkpoint directly, and separately exercises the installed status line and notifier unit coverage.
The ChatGPT Codex upstream is distinct from the Platform API upstream and may need updating if Codex changes that service route.
The two existing deprecation warnings remain unresolved because they are unrelated to the self-observation defects.
