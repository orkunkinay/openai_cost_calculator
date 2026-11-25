# Codex accounting audit

## Verdict

The baseline missed all observed usage in a successful live GPT-6.2 Sol turn. The audited branch captured a later equivalent single turn. Its 14,313 input and nine output tokens agreed with the independent sanitized rollout, and the standard API-equivalent token subtotal matched exactly at **USD 0.028716**. The result applies to this isolated standard token estimate. Trust wider token and session totals only to the extent that the integration observes and attributes all relevant requests. Project aggregation, subscription credits and actual financial charges remain unverified.

## Production pipeline

1. The installer selects a Responses provider using a local proxy URL and a configured `X-OCC-Session` tag. Authentication chooses the OpenAI Platform or ChatGPT Codex upstream. Calls bypassing that provider are invisible.
2. The proxy extracts usage from JSON or SSE, or observes terminal WebSocket responses. For SSE it now examines a decoded copy and records explicit completion before forwarding the terminal chunk. The response model takes precedence. A stream without a model can use the request model fallback.
3. OpenAI Responses and Chat Completions counters are normalized to input, cached input and output. Cached tokens are subtracted from ordinary input. Reasoning tokens are validated when present and remain inside output. No independent reasoning total is persisted.
4. The tracker looks up the legacy price projection. The remote CSV is refreshed with a 24-hour TTL. Local overrides take precedence and a model missing entirely from the remote CSV can use the bundled catalog. Dates are derived from snapshot suffixes or the current UTC date. Tiers are chosen by prompt tokens. The calculator multiplies Decimal rates by their respective token counts and divides by one million.
5. The registry aggregates by the caller-supplied session and optional turn labels. A notifier checkpoint consumes the records since the prior checkpoint for that label. The process-wide report aggregates observed sessions. The tag is neither a native Codex thread ID nor an OpenAI project ID. Sessions sharing a tag can merge, including concurrent interactions. The JSON and SQLite ledgers persist exact decimal values for new records. Summary and status amounts round for display.
6. The production path does not discover, import or read historical rollout files. Cumulative usage, forks and compaction affect the audit fixture reader only. Ledger history is previously captured proxy traffic and is read at its recorded cost rather than repriced.

## Results

| Experiment | Model | Input | Cached | Output | Reasoning | Reference USD | Reported USD | Absolute error USD | Relative error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ChatGPT model eligibility | GPT-5.3-Codex | — | — | — | — | — | — | — | — | Rejected by upstream |
| Single turn before terminal capture and pricing fixes | GPT-6.2 Sol | 14,310 | 0 | 9 | 0 | 0.028710 | 0.00000000 | 0.028710 | 100% | Missing capture |
| Single turn after fixes | GPT-6.2 Sol | 14,313 | 0 | 9 | 0 | 0.028716 | 0.02871600 | 0 | 0% | Matched rollout and proxy estimate |
| Single turn with pricing diagnostic | GPT-6.2 Sol | 14,307 | 0 | 9 | 0 | 0.028704 | 0.02870400 | 0 | 0% | Standard subtotal matched. Additional dimension flagged. |
| Two turns across CLI resume | GPT-6.2 Sol | 34,313 | 0 | 18 | 0 | 0.068806 | 0.06880600 | 0 | 0% | Both checkpoints and cumulative rollout matched. Additional dimension flagged. |
| Durable restart, before precision fix | GPT-5.3-Codex | 20 | 20 | 0 | 0 | 0.00000350 | 0.00000360 | 0.00000010 | 2.857% | Rounded each persisted call |
| Durable restart, after precision fix | GPT-5.3-Codex | 20 | 20 | 0 | 0 | 0.00000350 | 0.00000350 | 0 | 0% | Both backends matched offline |

The follow-up live request had three more input tokens than the earlier request. This difference appears in the rollout and is reflected in the reference. No evidence was collected about which hidden input changed.

- Token, model and session attribution matched in 5,007 deterministic differential records covering two models and five explicitly tagged sessions per model. Two completed and priced isolated live turns matched after the capture and pricing fixes. The second also flagged an additional pricing dimension and reported incomplete coverage. The failed baseline is retained above. There is no representative population-wide accuracy score.
- Mathematical error before rounding was zero in the differential records. Both ledgers retained exact amounts in the restart test after the precision fix. Ledgers written before the fix retain whatever precision was already stored.
- Repeated checkpoints and duplicate terminal events within one HTTP stream did not add cost in tested cases. WebSocket deduplication is bounded and connection-local. Cross-request duplicates, reconnections and process restarts are not covered by a stable response ID in the ledger.
- Unknown model testing verified an explicit diagnostic for an unpriced model. Numeric summary fields describe the priced subtotal. Use the accounting diagnostic state to detect partial results. A missing observation may produce no diagnostic at all.
- Actual billing reconciliation is unverified. No authorized admin key or isolated Platform API billing experiment was available.

## Confirmed defects and fixes

- Missing or malformed wire totals became integer zeroes or were coerced. Regression `test_invalid_response_usage_is_not_priced` failed for eight shapes before validation. It now requires valid nonnegative integer counters and checks input/output subsets and a supplied total.
- Compressed SSE was passed directly to the accounting parser. Regression `test_openai_compressed_stream_is_accounted_without_rewriting_body` reproduced the zero subtotal. The proxy now uses the existing streaming decoder on its accounting copy.
- Codex consumed the terminal SSE event and could disconnect before EOF. A live investigation observed the completed payload even though the proxy recorded no turn. The proxy now records explicit completion before yielding it and suppresses repeats within that stream. `test_terminal_responses_usage_is_committed_before_next_upstream_chunk` verifies order and duplicate handling.
- Discovery GET requests emitted usage errors even though they return model metadata. Only POST traffic enters buffered usage accounting now. See `test_openai_model_discovery_does_not_create_usage_errors`.
- GPT-6.2 Sol was absent from both the legacy CSV and bundled snapshot. After terminal capture was fixed, the live probe produced a pricing diagnostic. The catalog now provides rates for the supported standard token projection with a 272,001 first long-context token. The bundled fallback test simulates a stale remote CSV.
- Both ledgers serialized individual request costs at eight decimal places. The restart regression reproduced cumulative error. New ledger rows retain exact decimal values and preserve existing read compatibility.
- The Codex status line presented monetary figures without a basis and could show zero for absent or unpriced usage. Status now labels API-equivalent estimates, reports incomplete coverage when diagnostics exist, and identifies sessions without observed usage.

## Coverage still open

Long-context live requests, confirmed cache hits, reasoning-intensive requests, file reads and edits, shell execution, MCP, images, compaction, forks, model switching, subagents, cancelled or incomplete requests, retries, concurrent Codex processes, midnight and reporting-period boundaries, and multiple reasoning efforts are unverified. Platform API mode and billing could not be reconciled here. The offline suite exercises selected stream, model and attribution mechanics but makes no live claim for those scenarios.

The production three-bucket projection still omits tier modifiers, cache-write premiums and separate tool charges. Some explicit extra dimensions now raise a diagnostic while retaining the standard token subtotal. Missing metadata, implicit service tier selection and unknown future schemas remain limitations. There is no period or native project attribution, rollout backfill, stable cross-connection deduplication, released-price history guarantee or exhaustive independent catalog coverage. Before expanding coverage, add sanitized fixtures per Codex version, pin dated official price and schema evidence, and require an independent boundary test for each new rate dimension.
