# data/

`gpt_pricing_data.csv` is **generated** — do not edit it by hand.

It is the legacy three-bucket view (input / cached input / output per 1M tokens) of the
OpenAI and Gemini entries in the pricing catalog
(`openai_cost_calculator/data/pricing/*.json`), produced by:

```bash
openai-cost-calculator pricing export-legacy
```

Installed releases of this package download this file from the `main` branch at run time, so
its columns and row keys are a public interface. `tests/test_legacy_compat.py` checks it
against the parser those releases ship and against every row of the last hand-maintained
version, and CI fails if it is out of date with the catalog.

To change prices, update the catalog (normally via `openai-cost-calculator pricing sync`;
see `docs/PRICING_UPDATES.md`) and regenerate this file.
