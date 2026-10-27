# Changelog

## 1.3.1

### Fixed

- Streaming: usage is no longer lost when a stream ends with chunks that carry no usage
  (#40, thanks @GreedyC).
- Pricing sync parsers, updated for upstream page changes:
  - OpenAI: Flex prices nested under the Batch section, the "Fast" (formerly Priority) and
    new Ultrafast tiers, and the renamed "30m write" cache-write columns.
  - Anthropic: prompt-length price tiers and prices written with a "USD" suffix.
  - Amazon Bedrock: long-context meters are priced as a separate tier instead of overwriting
    the base price.
  - DeepSeek: fetch the pricing page that the CDN serves correctly.
  - Vertex AI: dated prices written as "(Starting <date>)".
  - Gemini API: per-10s audio equivalents and caching remarks no longer block a model.
  - Mistral: license badges are no longer read as model names.
  - Compact Markdown table separators are accepted.

### Changed

- Pricing data refreshed from official sources. New models include gpt-6.1-sol,
  gpt-6.1-terra, gemini-4-argon, gemini-er-2.0 and the Gemini 3.8 TTS models; the Ultrafast
  tier is priced as `service_tier="ultrafast"`.
- Deprecated Gemini ids that route to a successor (`gemini-3.5-flash`, `gemini-3.7-flash`)
  are aliases of that successor and priced as it.
- The pricing sync folds a removed model into another when it reappears as that model's
  alias.

## 1.3.0

### Added

- Provider-agnostic API: `calculate_cost`, `estimate_response_cost`, `compare_costs`,
  `get_model_pricing`, `list_models`, `list_providers`, `extract_usage`, with an itemized
  `Cost` that records the conditions applied, any assumptions, and the price's source URL and
  verification date.
- Bundled pricing catalog for 13 billing providers: OpenAI, Anthropic, Gemini API, Vertex AI,
  Azure OpenAI, Amazon Bedrock, OpenRouter, DeepSeek, Together AI, Groq, Fireworks AI,
  DeepInfra and Mistral.
- Pricing dimensions for cache writes (5-minute and 1-hour), cached input, audio, image,
  reasoning, web search and per-request fees; conditions for service tier, region and
  time-of-day period; long-context tiers; effective-date windows.
- Usage extraction for OpenAI Chat/Responses, DeepSeek, Anthropic Messages, Gemini and Bedrock
  Converse payloads.
- `openai-cost-calculator pricing` commands: `cost`, `models`, `providers`, `sync`, `stale`,
  `export-legacy`.
- Automated, fail-closed pricing sync from official sources with Markdown/JSON reports, a
  weekly GitHub Actions workflow, and lint/type-check CI gates.

### Changed

- `data/gpt_pricing_data.csv` is generated from the catalog (format unchanged).
- The legacy loader falls back to the last good copy, then the bundled catalog, when the CSV
  cannot be downloaded.
- Anthropic proxy pricing and `seed_anthropic_pricing` read the catalog; prices now match
  Anthropic's published pricing.
- `pyproject.toml` is the single packaging definition; the package ships `py.typed`.

### Removed

- `scripts/check_pricing.py`, superseded by `openai-cost-calculator pricing sync`.
