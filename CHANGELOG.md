# Changelog

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
