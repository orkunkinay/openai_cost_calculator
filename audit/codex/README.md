# Codex accounting audit

From the repository root, install the development dependencies and the proxy extra once.

```sh
python -m pip install -r requirements-dev.txt -e '.[proxy]'
```

Run the offline audit and existing suite with no API credentials or paid requests.

```sh
python -m pytest -q tests audit/codex
```

Opt in to one live Codex turn. This uses the selected login's allowance or API budget. The fixture contains only model and token-count records. Use a new fixture destination to preserve the checked-in evidence.

```sh
python scripts/self_test_codex_integration.py --auth-mode chatgpt --model gpt-6.2-sol --accounting-fixture /tmp/occ-codex-rollout.json
```

Add `--resume-once` to send one additional fixed prompt via `codex exec resume`, record the combined rollout, and check that the proxy counts exactly one new request.

Run the independent fixture calculation separately.

```sh
PYTHONPATH=audit/codex python -c 'import json, sys; from reference_calculator.cost import isolated_rollout; print(isolated_rollout(json.load(open(sys.argv[1]))))' /tmp/occ-codex-rollout.json
```

Opt in to a read-only Costs API comparison with an existing organization admin key in `OPENAI_ADMIN_KEY`. Use an isolated API project, a completed UTC date with settled data and a tolerance justified by the reporting precision. The example values are placeholders.

```sh
python audit/codex/experiments/reconcile_billing.py 2026-11-24 proj_REPLACE_ME 0.028716 --tolerance-usd 0.000001
```

See [audit report](AUDIT_REPORT.md), [methodology](METHODOLOGY.md), [pricing sources](PRICING_SOURCES.md) and [machine-readable evidence](results/evidence.json).
