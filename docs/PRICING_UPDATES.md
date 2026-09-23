# Automated pricing updates

Prices change often and without notice. This project keeps them current with a pipeline
that runs weekly in GitHub Actions (`.github/workflows/pricing-sync.yml`) and can be run
locally:

```bash
openai-cost-calculator pricing sync --dry-run --report-md report.md   # look, don't touch
openai-cost-calculator pricing sync --provider anthropic               # update one provider
openai-cost-calculator pricing export-legacy                           # regenerate the legacy CSV
```

## Sources

Every provider is fetched from an **official** source; structured APIs are preferred over
documentation pages, and documentation is read from a machine-friendly rendition where one
exists.

| Provider | Source | Kind |
| --- | --- | --- |
| OpenAI | `developers.openai.com/api/docs/pricing.md` | Markdown docs |
| Anthropic | `platform.claude.com/docs/en/about-claude/pricing.md` | Markdown docs |
| Gemini | `ai.google.dev/gemini-api/docs/pricing.md.txt` | Markdown docs |
| OpenRouter | `openrouter.ai/api/v1/models` | JSON API |
| Azure | `prices.azure.com/api/retail/prices` (Foundry Models meters) | JSON API |
| Bedrock | AWS Price List API, `AmazonBedrock` + `AmazonBedrockFoundationModels` offers | JSON API |
| Vertex AI | `cloud.google.com/vertex-ai/generative-ai/pricing` | HTML tables |
| DeepSeek | `api-docs.deepseek.com/quick_start/pricing` | HTML table |
| Together | `docs.together.ai/docs/serverless/models.md` | Markdown docs |
| Groq | `console.groq.com/docs/models.md` | Markdown docs |
| Fireworks | `docs.fireworks.ai/serverless/pricing.md` | Markdown docs |
| DeepInfra | `api.deepinfra.com/models/list` | JSON API |
| Mistral | `mistral.ai/pricing/api/` | HTML |

LiteLLM's community table is fetched as an independent **corroborator** only.

## Pipeline

For each provider:

1. **Fetch** through a `Fetcher` (timeouts, retries, descriptive user agent).
2. **Parse** into catalog entries. Parsers are strict: a cell, row or sentence they do not
   fully understand becomes an *issue* instead of a guess. Issues tied to a model block that
   model's update; *notes* record input deliberately handled conservatively (for example a
   footnoted per-image price in a per-token column, or contradictory duplicate rows on a page)
   and do not block.
3. **Merge** rows a page lists more than once (a model in both a chat and a vision table);
   consistent duplicates are absorbed, contradictory ones fail validation.
4. **Validate** every entry with the same structural checks used at load time, plus sanity
   checks (implausible magnitudes, cached price above input price).
5. **Diff** against the checked-in data, per price set and dimension.
6. **Decide** with the fail-closed policy (below).
7. **Write** canonical JSON only if something was applied; re-stamp `verified_at` when data
   changed or every 30 days, so a quiet week produces no diff.
8. **Report** in Markdown (PR body and job summary) and JSON.

## What is applied automatically

A change is applied only if **every** check passes:

| Guard | Rule | Catches |
| --- | --- | --- |
| Extraction | the source returns at least half as many models as the catalog holds for it | page redesigns, empty responses |
| Validation | entry passes structural validation and sanity checks | unit errors (per-token vs per-1M), negative/NaN values |
| Magnitude | no price moves by more than 3× either way | misread columns, shifted cells |
| Parser issues | the parser raised no blocking issue for the model | unrecognized wording |
| Corroboration | LiteLLM does not still report the *old* price | a parser regression that looks like a price change |
| Removals | never automatic; the model is kept and listed | renames, retirements, missed rows |
| Additions | automatic only from official sources | |

Rule-bearing sentences are parsed and verified too, not assumed: Anthropic's data-residency
multiplier, Fireworks' and Mistral's batch/cache/regional rules, and DeepSeek's peak-hour
schedule. The DeepSeek check is special: the schedule itself is implemented in code
(`providers.registry.deepseek_period`), and the parser flags a review if the published
schedule stops matching it, so code cannot silently drift from the rule.

## When extraction is ambiguous

* **One model can't be parsed** → it keeps its checked-in price; the report lists it under
  *Needs review*; other models update normally.
* **A whole source fails** (network error, redesign, schema change, parser exception) → that
  provider's data is left untouched, other providers still update, and the workflow opens the
  PR and then fails, so maintainers are notified.
* **Merged data would be invalid** (for example two entries claiming one identifier) → the
  provider is treated as failed and nothing is written.
* **Stale data** → `pricing stale --max-age-days 45` fails the workflow if any provider has
  not been verified recently.

## Reviewing a pricing PR

The PR body is the sync report. For each provider it lists applied changes (old → new), items
held for review with reasons, and parser notes. Maintainers typically:

1. Spot-check applied changes against the linked official source.
2. For held items, open the source page. If the held change is right (a genuine large price
   cut, or a deliberate parser change that restructures data), run
   `openai-cost-calculator pricing sync --provider <id> --accept-review` locally and push.
   `--accept-review` applies held changes but still rejects invalid data and never deletes a
   model.
3. For a genuinely removed model, delete its entry by hand after confirming retirement.

## Updating a parser after a page redesign

1. Save the new page into `tests/fixtures/sources/` (trim JSON to the models the tests use;
   strip scripts and styles from HTML).
2. Update the parser until its contract tests pass against the new fixture; add a test for
   the new shape.
3. Run `pricing sync --provider <id> --dry-run` against the live page and read the report.

## Hand-maintained entries

Entries with source kind `manual` (currently the `legacy-csv` entries: preview and retired
OpenAI and Gemini models from the pre-catalog CSV) are never modified by the sync and are
reported by `pricing stale` as not covered by an automated source. If an official source
starts listing the same model, its entry replaces the manual one automatically.
