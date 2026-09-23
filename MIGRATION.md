# Migrating to the provider-agnostic API (1.3)

**Nothing you use today needs to change.** `estimate_cost`, `estimate_cost_typed`,
`calculate_cost_typed`, `CostTracker`, the pricing utilities, the proxy and the Claude Code and
Codex integrations keep their signatures and return types. This guide explains what changed
underneath and how to adopt the new API.

## Two APIs, one source of prices

| | Original API | Provider-agnostic API |
| --- | --- | --- |
| Entry points | `estimate_cost(response)`, `estimate_cost_typed(response)` | `calculate_cost(provider, model, ...)`, `estimate_response_cost(response, provider=...)` |
| Providers | OpenAI-protocol responses (plus `google/` Gemini rows) | 13 billing providers |
| Result | `CostBreakdown` (uncached prompt / cached prompt / completion) | `Cost`: itemized by dimension, conditions, assumptions, provenance |
| Prices | `data/gpt_pricing_data.csv`, fetched from GitHub, 24h cache | catalog bundled in the package, no network |
| Tiers/regions/time | long-context tiers only | tiers, service tiers, regions, periods, effective dates |

Both read the same catalog: the CSV is now *generated* from it.

## Recommended migration

```python
# before
from openai_cost_calculator import estimate_cost_typed
total = estimate_cost_typed(response).total_cost

# after
from openai_cost_calculator import estimate_response_cost
cost = estimate_response_cost(response, provider="openai")   # or "azure", "openrouter", ...
total = cost.total
legacy_shape = cost.to_breakdown()                            # a CostBreakdown, if you need one
```

Differences you may notice when switching:

* **Cache writes are priced.** OpenAI GPT-5.6+ and Anthropic bill cache writes; the new API
  prices them separately (the old three buckets folded them into uncached input at the input
  rate).
* **Audio/image tokens are priced at their own rates** where published, and an explicit
  `MissingRateError` is raised when a model has no such rate, instead of silently using the
  text rate.
* **No network I/O.** Prices update with releases (and the weekly data refresh), not at run
  time. Pin or override prices by passing your own `PricingCatalog`
  (`PricingCatalog.from_directory(path)`).
* **Errors are typed** (`UnknownModelError`, `PricingUnavailableError`, ...), all subclasses of
  `PricingError`/`ValueError`, instead of a single `CostEstimateError`.

## What changed in the original API

* `data/gpt_pricing_data.csv` is generated from official sources by the automated sync. Its
  columns, and every row family older releases look up, are unchanged; tests verify both.
  Long-context rows now use OpenAI's published threshold (`272000`) instead of `272001`.
* If the CSV cannot be downloaded, the last good copy is used; without one, the bundled
  catalog is used. Previously every estimate failed. A CSV that downloads but is malformed
  still raises.
* `seed_anthropic_pricing()` and the Anthropic proxy accounting read the catalog. Their prices
  now match Anthropic's published pricing (for example Claude Sonnet 5 is $2/$10 per 1M tokens,
  and Claude 4.6+ models bill the full 1M context window at the standard rate; the previous
  hand-maintained tables still had $3/$15 and a >200K surcharge).
* `openai-cost-calculator pricing validate` now also validates the catalog.

## Removed

* `scripts/check_pricing.py` (the LiteLLM-driven checker) is replaced by
  `openai-cost-calculator pricing sync`, which reads official sources and uses LiteLLM only as
  a corroborating second opinion.
* `setup.py` no longer duplicates packaging metadata; `pyproject.toml` is the single
  definition (the old `setup.py` had drifted: it lacked two console scripts and pointed at a
  nonexistent data file).

## Why not rename the package?

The distribution and import names stay `openai-cost-calculator` / `openai_cost_calculator` so
existing installs, imports and the published CSV URL keep working. A future rename can ship a
thin alias package; the code is already provider-agnostic.
