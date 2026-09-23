# Adding a provider

Adding a provider touches four places and usually no core code. The worked example below
adds a hypothetical provider `acme` whose pricing is published as JSON at
`https://api.acme.example/v1/models`.

## 0. Research first

Before writing code, read the provider's **current official** pricing documentation and
answer:

* Is there a structured source (public API, JSON, Markdown rendition via `.md`/`llms.txt`)?
  Prefer it over HTML.
* Which dimensions does it bill? Map each to an existing dimension in
  `catalog/dimensions.py` (input, cached_input, cache_write, cache_write_1h, output,
  reasoning, audio/image variants, web_search, request).
* Which conditions? Service tiers (`service_tier`), regional/global endpoints (`region`),
  time-of-day periods (`period`), long-context thresholds (`min_input_tokens`), scheduled
  changes (`effective_from`/`effective_until`).
* What do model identifiers look like in API requests and responses, and how do they differ
  from the pricing page?
* Which usage wire format do responses use (OpenAI-compatible, Anthropic, Gemini, Bedrock
  Converse, something new)?

If the provider bills something no existing dimension or condition can represent, extend
the core deliberately (see step 5) — do not approximate.

## 1. Provider spec (`openai_cost_calculator/providers/registry.py`)

```python
ProviderSpec(
    id="acme",
    display_name="Acme AI",
    pricing_url="https://acme.example/pricing",
    aliases=("acme-ai",),
    # Only if needed:
    # strip_prefixes=("models/",),              # path prefixes in model strings
    # parse_model=parse_acme_model,             # identifier formats that imply conditions
    # default_regions=("global", "us"),         # region preference when the caller gives none
    # service_tier_aliases={"turbo": "priority"},
),
```

Most providers need nothing beyond `id`, `display_name` and `pricing_url`: dated snapshot
suffixes, `vendor/` namespaces and punctuation differences are handled generically.

## 2. Source parser (`openai_cost_calculator/sync/sources/acme.py`)

```python
from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult, get_json
from .common import per_token_to_per_million, present, price_set

URL = "https://api.acme.example/v1/models"
SOURCE = Source(id="acme-models-api", kind="official_api", url=URL, description="Acme public models API")


def parse(payload) -> SourceResult:
    result = SourceResult()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        result.issue("response has no 'data' list; the API format may have changed")
        return result
    for item in models:
        try:
            standard = {
                "input": per_token_to_per_million(item["price"]["input"]),
                "output": per_token_to_per_million(item["price"]["output"]),
            }
        except (KeyError, TypeError, ArithmeticError) as exc:
            result.issue(f"unparseable price: {exc}", item.get("id"))
            continue
        sets = present([price_set(standard)])
        if sets:
            result.add(ModelPricing(id=item["id"], prices=sets, source=SOURCE.id, vendor="acme", canonical_id=f"acme/{item['id']}"))
    return result


class AcmeSource:
    provider = "acme"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(get_json(fetcher, URL))
```

Rules for parsers:

* **Never guess.** If a value, row or rule isn't understood, call `result.issue(...)`
  (with the model id when known) and skip it. Use `result.note(...)` only for input you
  handle conservatively on purpose.
* Store prices in USD per 1M tokens (per unit for `web_search`/`request`) as `Decimal`;
  convert from the source's unit explicitly.
* Emit one entry per row if convenient — duplicates are merged; emit `service_tier="standard"`
  implicitly (the default) and only non-standard tiers as conditions.
* Parse rule-bearing sentences (batch discounts, regional multipliers) and flag an issue if
  the sentence disappears, instead of hard-coding the multiplier.
* Keep any hand-maintained knowledge (aliases the page doesn't show) as a small documented
  table in the parser module.

Register it in `sync/sources/__init__.py` (`all_sources`).

## 3. Tests

1. Save the real page or API response to `tests/fixtures/sources/acme_models.json` (trim to
   a handful of models, preserving structure).
2. Add contract tests (see `tests/test_sources_api.py`): expected rates for a few models,
   every entry passes `validate_model`, and malformed inputs (missing keys, unexpected shape,
   unparseable values) produce issues rather than entries.
3. Add at least one end-to-end case to `tests/test_pricing_regression.py` with an amount
   computed by hand from the published price.
4. If the provider has identifier quirks, test `ProviderSpec.hints` in `tests/test_api.py`.

## 4. Generate the data

```bash
openai-cost-calculator pricing sync --provider acme --report-md report.md
openai-cost-calculator pricing cost acme some-model --input 1000 --output 100
pytest && ruff check . && mypy
```

The first run adds every model (official sources only). Commit the parser, tests and the
generated `openai_cost_calculator/data/pricing/acme.json` together. Add the provider to the
README table; the weekly workflow picks it up automatically.

## 5. When the core needs to change

Extend the core only for a variation that cannot be represented otherwise, and add it in one
place:

* **New dimension** → `catalog/dimensions.py` (with a fallback only if that is genuinely how
  providers bill it) and the matching `Usage` field in `catalog/costing.py`; a test asserts
  the two stay in sync.
* **New condition key** → `CONDITION_KEYS` in `catalog/model.py`; decide whether its absence
  means "any value" or a specific default (`IMPLICIT_CONDITIONS`).
* **New usage wire format** → an extractor in `usage.py` and detection in `extract_usage`.
* **New provider behaviour** (e.g. a time-based rule) → a `ProviderSpec` hook, not a branch
  in the core.
