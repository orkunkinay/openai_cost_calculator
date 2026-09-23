# Architecture

This document explains how the library is structured and, more importantly, *why*: every
abstraction below exists because a specific provider priced something in a way a simpler
design could not represent. The evidence comes from each provider's official pricing source
(retrieved 2026-09-23; fixtures in `tests/fixtures/sources/`).

## Goals

1. Application code prices usage with `calculate_cost(provider=..., model=..., ...)` and
   never branches on the provider.
2. Prices are **data**, produced by parsers from official sources, validated, versioned and
   auditable (source URL + verification date on every result).
3. Keeping prices current needs close to zero manual work, and fails *closed*: an ambiguous
   upstream change is never applied silently.
4. Deterministic, exact arithmetic (`Decimal`) with no network I/O on the pricing path.
5. The original OpenAI-only API keeps working unchanged.

## Layers

```
            application code
                   |
   api.py  calculate_cost / estimate_response_cost / compare_costs      (facade)
     |            |                          |
 providers/   usage.py                   catalog/                      (pure, no I/O)
 ProviderSpec  wire format -> Usage      model, validation, io, lookup,
 (quirks)                                selection, costing
                                              ^
                                              | generated JSON (package data)
                                              |
   sync/  sources/<provider>.py -> policy -> runner -> report            (offline tooling)
     ^
     | official APIs and documentation pages
```

Dependencies point downward only. `catalog/` knows nothing about any provider; `providers/`
knows provider identifier formats and defaults but not prices; `sync/` produces catalog
data but is never imported by the pricing path. The legacy modules (`estimate.py`,
`pricing.py`, `core.py`) and the proxy/integrations sit beside the facade and read the same
catalog through `legacy.py` and `anthropic/pricing.py`.

## Domain model (`catalog/model.py`)

| Concept | Type | Example |
| --- | --- | --- |
| Billing provider | `ProviderPricing` (one JSON file) | `bedrock` |
| Offering | `ModelPricing` | `anthropic.claude-sonnet-4-5` as sold by Bedrock |
| Provider model id | `ModelPricing.id` + `aliases` | `anthropic.claude-sonnet-4-5`, dated snapshots |
| Vendor | `ModelPricing.vendor` | `anthropic` (who built it) |
| Canonical identity | `ModelPricing.canonical_id` | `anthropic/claude-sonnet-4-5` (cross-provider) |
| Conditional prices | `PriceSet` | rates + conditions + `min_input_tokens` + effective dates |
| Pricing dimension | `Dimension` (`catalog/dimensions.py`) | `input`, `cache_write_1h`, `web_search` |
| Provenance | `Source`, `verified_at` | official API / official docs / manual |

Separating *billing provider* from *vendor* is what makes "Claude on Bedrock" and "Claude on
Vertex" distinct offerings of one canonical model with different prices, endpoints and
identifier formats.

### Dimensions and disjoint usage

Usage (`catalog/costing.Usage`) is a set of **disjoint** buckets: a token is counted once.
Providers report overlapping counters (OpenAI's `prompt_tokens` includes cached tokens and
cache writes; Gemini's `promptTokenCount` includes cached content; Anthropic's `input_tokens`
excludes both). Converting to disjoint buckets is the job of the per-format extractor in
`usage.py`, so the arithmetic is a plain sum over dimensions with no provider logic.

Each dimension may declare a *fallback* used when a model has no rate for it. Fallbacks exist
only where they reflect how providers bill (a cache write on a provider without write pricing
is ordinary input; reasoning tokens are output). Every other missing rate raises
`MissingRateError` instead of guessing — pricing audio tokens at the text rate, for example,
would silently under-count by ~10×.

The dimension set is small and closed; each member was added because a provider prices it:

| Dimension | Why it exists |
| --- | --- |
| `cache_write`, `cache_write_1h` | Anthropic TTL-specific writes; OpenAI GPT-5.6+ writes; Bedrock/Vertex/DeepInfra/OpenRouter |
| `cached_input_audio` | OpenAI realtime and Gemini price cached audio separately |
| `input_audio`, `output_audio`, `input_image`, `output_image` | realtime/audio/image models |
| `reasoning` | OpenRouter lists a model whose reasoning tokens cost differently from output |
| `web_search`, `request` | Anthropic/OpenRouter per-call fees |

### Conditions (`PriceSet.conditions`)

Three condition keys, each backed by a real variation:

* `service_tier` — batch/flex/priority/fast (OpenAI, Anthropic, Gemini, Azure, Bedrock,
  Fireworks, DeepInfra, Mistral). A set without a tier *is* the standard tier; it must not
  silently price a batch request (this was caught by a test during development).
* `region` — deployment scope. Bedrock and Vertex regional endpoints (+10%), Azure
  Global/Data Zone/Regional, Anthropic `inference_geo="us"` (1.1×), Fireworks US-only,
  Mistral EU. A set without a region applies everywhere.
* `period` — DeepSeek's peak/off-peak pricing.

`min_input_tokens` models long-context tiers (OpenAI ≥272K, Gemini/Vertex >200K), which
apply to *every* token once the request's total input crosses the threshold.
`effective_from`/`effective_until` model scheduled price changes (Gemini publishes prices
that change on 2027-01-01) and time-limited promotions (DeepInfra).

### Selection (`catalog/selection.py`)

A request gives, per condition key, an ordered tuple of acceptable values — the caller's
explicit choice, or the provider's defaults with fallbacks (Bedrock: `global`, then
`us-east-1`). Selection filters sets by effective date, ranks condition groups by preference,
prefers specific groups over wildcards, raises on ties, then picks the highest long-context
tier not above the request's input size. Defaults that affected the price are returned as
human-readable `assumptions`. This single algorithm serves all thirteen providers.

## Provider specs (`providers/`)

`ProviderSpec` holds only what the core cannot know:

| Hook | Used by | Why |
| --- | --- | --- |
| `parse_model` | Bedrock, Vertex, Azure | `us.`/`global.` inference profiles and ARNs imply the endpoint's region; `@version` and `-maas` suffixes; Azure `MMDD` snapshots |
| `default_regions` | Bedrock | prefer the global endpoint, fall back to a default region |
| `expand_region` | Vertex | a concrete region falls back to Gemini's "non-global" price |
| `time_conditions` | DeepSeek | peak/off-peak derived from the request time |
| `default_preferences` | DeepSeek | conservative (peak) when the time is unknown |
| `service_tier_aliases` | OpenAI, Azure | "fast" is the renamed "priority" tier |
| `strip_prefixes` | Gemini, Fireworks | `models/…`, `accounts/fireworks/models/…` |

Everything else (snapshot suffix stripping, `vendor/` namespaces, punctuation-insensitive
matching, "did you mean" suggestions) is generic and lives in `generic_candidates` and
`catalog.ModelIndex`. There are no `if provider == ...` branches outside provider specs and
source parsers.

## Wire formats vs billing providers (`usage.py`)

The usage schema is determined by the API protocol, not by who bills: Azure, OpenRouter,
Groq, Together, Fireworks, DeepInfra and Mistral speak OpenAI's; Claude on Bedrock/Vertex
speaks Anthropic's; Gemini on Vertex speaks Gemini's. `extract_usage` therefore detects the
format from the payload shape, and `estimate_response_cost` takes the billing provider as a
separate argument. The same Anthropic response priced on `anthropic`, `bedrock` and `vertex`
yields three different costs — the regression tests assert exactly that.

## Data as generated artifacts

* `openai_cost_calculator/data/pricing/<provider>.json` — the catalog, written only by
  `pricing sync` (or by a reviewed one-off migration). Serialization is canonical: sorted
  models and price sets, dimension-ordered rates, decimals as plain strings, so an unchanged
  source produces a byte-identical file and diffs show only real changes.
* `data/gpt_pricing_data.csv` — generated by `pricing export-legacy` from the catalog. It is
  downloaded at run time by every installed release, so it is treated as a public interface:
  tests check it against the parser old releases ship and against every row of the previous
  hand-maintained file.
* Hand-maintained knowledge lives in code next to the parser that needs it, is small, and is
  documented where it is used: OpenAI dated snapshot aliases, Anthropic's `claude-3-5-haiku`
  id, Mistral `-latest` aliases. Rows no official source lists (retired previews) were carried
  over once as `unverified` manual entries; an official source supersedes them automatically.

## The update pipeline (`sync/`)

See [PRICING_UPDATES.md](PRICING_UPDATES.md). In short: a `PricingSource` fetches through an
injected `Fetcher` (real HTTP in CI, recorded fixtures in tests) and returns catalog entries
plus parser *issues*; the runner validates, diffs against the checked-in data and asks the
policy which changes are safe. Parsers are pure functions of the fetched text, so every
parser has contract tests against real recorded pages, including malformed variants.

## Dependency inversion where it pays

* `Fetcher` protocol: sources never import `requests`; tests inject `FixtureFetcher`.
* `PricingCatalog` is a parameter of every public function: tests and applications can price
  against a custom or pinned catalog; the Anthropic proxy module is tested with one.
* `Corroborator` is a plain callable: the policy doesn't know LiteLLM exists.

Nothing else is abstracted: sources are plain classes with a `fetch` method, specs are
frozen dataclasses, and the facade is a module of functions.

## Decisions and trade-offs

* **Bundled data, no network on the pricing path.** The original library fetched a CSV from
  GitHub on first use and whenever its 24-hour cache expired, which put network latency and
  a failure mode inside every estimate (including inside the proxy's request path) and let
  anyone who could push to `main` change every user's costs. The new API reads data shipped
  with the package; freshness comes from releases and the weekly sync. The legacy API keeps
  its remote-refresh contract for compatibility, now with a fallback to bundled data.
* **Official sources over aggregators.** The previous checker trusted LiteLLM's community
  JSON. It is now only a corroborator that can hold a suspicious change back.
* **Documentation pages are parsed, not scraped visually.** OpenAI, Anthropic and Gemini
  serve Markdown renditions (`.md`, `.md.txt`); Together, Groq and Fireworks serve Markdown
  docs; DeepSeek, Mistral and Vertex are server-rendered HTML parsed with the standard
  library. No headless browser, no third-party HTML dependency.
* **Structural validation at load time.** Bundled data is validated on first use; a
  malformed file fails loudly rather than pricing silently wrong.
* **Legacy three-bucket view is a projection.** `CostBreakdown` and the legacy CSV are views
  of the richer catalog; nothing is maintained twice. Before this change the Anthropic prices
  existed in two hand-maintained tables that had both drifted from Anthropic's published
  prices.

## Package layout

```
openai_cost_calculator/
  api.py              provider-agnostic facade
  usage.py            wire format -> Usage
  legacy.py           catalog -> legacy CSV / three-bucket view
  pricing_cli.py      `openai-cost-calculator pricing ...`
  catalog/            domain model, validation, (de)serialization, lookup, selection, costing
  providers/          ProviderSpec + registry of the 13 providers
  sync/               base protocols, text helpers, diff, policy, runner, report, corroborate
  sync/sources/       one parser per upstream source
  data/pricing/       generated catalog (package data)
  estimate.py, pricing.py, core.py, parser.py, types.py, tracker.py   original API
  anthropic/, proxy/, adapters/                                       integrations
```
