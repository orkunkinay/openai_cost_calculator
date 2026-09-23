# Upstream pricing fixtures

Verbatim (or trimmed) copies of official pricing sources, retrieved on
2026-09-23, used by the source contract tests in `tests/test_sources_*.py`.

They pin parser behaviour against real page layouts. When a provider changes
its page format, refresh the fixture from the live page and update the
parser and its expectations together:

```bash
openai-cost-calculator pricing sync --provider <id> --dry-run
```

Trimmed fixtures keep only the entries the tests assert on, to keep the
repository small; trimming preserves the upstream structure exactly.
