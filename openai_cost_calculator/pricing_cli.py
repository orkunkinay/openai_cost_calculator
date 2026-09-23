"""``openai-cost-calculator pricing ...`` subcommands.

    pricing cost bedrock anthropic/claude-sonnet-4-5 --input 12000 --output 800
    pricing models anthropic
    pricing providers
    pricing sync [--provider P ...] [--dry-run] [--report-md PATH]
    pricing stale [--max-age-days 45]
    pricing export-legacy [--check]
    pricing validate

Exit codes for ``sync``: 0 success (with or without updates), 2 something
needs human review, 3 a source failed (its checked-in data was kept).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

EXIT_REVIEW = 2
EXIT_SOURCE_FAILED = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CSV = REPO_ROOT / "data" / "gpt_pricing_data.csv"


def add_pricing_parser(subparsers: argparse._SubParsersAction) -> None:
    pricing = subparsers.add_parser("pricing", help="Pricing data operations")
    sub = pricing.add_subparsers(dest="pricing_command", required=True)

    validate = sub.add_parser("validate", help="Validate the pricing catalog and legacy CSV")
    validate.add_argument("--file", default=str(LEGACY_CSV), help="legacy pricing CSV to validate")
    validate.add_argument("--data-dir", help="catalog directory (default: bundled data)")

    cost = sub.add_parser("cost", help="Price one request")
    cost.add_argument("provider")
    cost.add_argument("model")
    cost.add_argument("--input", type=int, default=0, help="uncached input tokens")
    cost.add_argument("--output", type=int, default=0, help="output tokens")
    cost.add_argument("--cached", type=int, default=0, help="cache-read input tokens")
    cost.add_argument("--cache-write", type=int, default=0, help="cache-write input tokens")
    cost.add_argument("--cache-write-1h", type=int, default=0, help="1-hour cache-write tokens")
    cost.add_argument("--service-tier")
    cost.add_argument("--region")
    cost.add_argument("--period")
    cost.add_argument("--json", action="store_true")

    models = sub.add_parser("models", help="List priced models")
    models.add_argument("provider", nargs="?")
    models.add_argument("--search", help="substring filter on model ids")

    sub.add_parser("providers", help="List supported providers")

    sync = sub.add_parser("sync", help="Fetch official pricing and update the catalog")
    sync.add_argument("--provider", action="append", default=[], help="provider id (repeatable; default: all)")
    sync.add_argument("--data-dir", help="catalog directory to update (default: bundled data)")
    sync.add_argument("--dry-run", action="store_true", help="report changes without writing")
    sync.add_argument("--report-md", help="write a Markdown report to this path")
    sync.add_argument("--report-json", help="write a JSON report to this path")
    sync.add_argument("--no-corroborate", action="store_true", help="skip the LiteLLM cross-check")
    sync.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)

    stale = sub.add_parser("stale", help="Report providers whose data was not verified recently")
    stale.add_argument("--max-age-days", type=int, default=45)
    stale.add_argument("--data-dir")

    export = sub.add_parser("export-legacy", help="Regenerate the legacy OpenAI pricing CSV")
    export.add_argument("--output", default=str(LEGACY_CSV))
    export.add_argument("--check", action="store_true", help="fail if the file is out of date")


def _catalog(data_dir: object):
    from .catalog import PricingCatalog, bundled_catalog

    return PricingCatalog.from_directory(Path(str(data_dir))) if data_dir else bundled_catalog()


def run(args: argparse.Namespace) -> int:
    handler = {
        "validate": _validate,
        "cost": _cost,
        "models": _models,
        "providers": _providers,
        "sync": _sync,
        "stale": _stale,
        "export-legacy": _export_legacy,
    }[args.pricing_command]
    return handler(args)


def _validate(args: argparse.Namespace) -> int:
    from .catalog import PricingError
    from .pricing import validate_pricing_file

    try:
        catalog = _catalog(args.data_dir)
        models = sum(len(data.models) for data in map(catalog.get, catalog.provider_ids()) if data)
        rows = validate_pricing_file(args.file)
    except (OSError, ValueError, PricingError) as exc:
        print(f"pricing validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Pricing catalog valid: {models} models across {len(catalog.provider_ids())} providers")
    print(f"Pricing data valid: {rows} model/date entries in {args.file}")
    return 0


def _cost(args: argparse.Namespace) -> int:
    from .api import calculate_cost
    from .catalog import PricingError
    from .catalog.io import format_decimal

    try:
        cost = calculate_cost(
            args.provider,
            args.model,
            input_tokens=args.input,
            output_tokens=args.output,
            cached_input_tokens=args.cached,
            cache_write_tokens=args.cache_write,
            cache_write_1h_tokens=args.cache_write_1h,
            service_tier=args.service_tier,
            region=args.region,
            period=args.period,
        )
    except PricingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "provider": cost.provider,
                    "model": cost.resolved_model,
                    "total_usd": format_decimal(cost.total),
                    "conditions": dict(cost.conditions),
                    "items": {i.dimension: format_decimal(i.cost) for i in cost.items},
                    "assumptions": list(cost.assumptions),
                    "source": cost.source_url,
                    "verified_at": cost.verified_at.isoformat() if cost.verified_at else None,
                },
                indent=2,
            )
        )
        return 0
    print(f"{cost.provider} / {cost.resolved_model}: ${cost.rounded(8)}")
    for item in cost.items:
        print(
            f"  {item.dimension:<18} {item.quantity:>12,} x "
            f"${format_decimal(item.unit_price)}/{item.unit} = ${format_decimal(item.cost)}"
        )
    conditions = ", ".join(f"{k}={v}" for k, v in cost.conditions.items())
    print(f"  conditions: {conditions}")
    for assumption in cost.assumptions:
        print(f"  note: {assumption}")
    print(f"  source: {cost.source_url} (verified {cost.verified_at})")
    return 0


def _models(args: argparse.Namespace) -> int:
    from .api import list_models

    for provider, model in list_models(args.provider):
        if args.search and args.search.lower() not in model.id.lower():
            continue
        status = "" if model.status == "active" else f" [{model.status}]"
        print(f"{provider:<11} {model.id}{status}")
    return 0


def _providers(args: argparse.Namespace) -> int:
    from .api import list_providers
    from .catalog import bundled_catalog

    catalog = bundled_catalog()
    for spec in list_providers():
        data = catalog.get(spec.id)
        count = len(data.models) if data else 0
        verified = data.verified_at.isoformat() if data and data.verified_at else "never"
        aliases = f" (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
        print(f"{spec.id:<11} {count:>4} models, verified {verified}  {spec.display_name}{aliases}")
    return 0


def _sync(args: argparse.Namespace) -> int:
    from .catalog import BUNDLED_DATA_DIR
    from .sync.base import HttpFetcher, SourceError
    from .sync.corroborate import LiteLLMCorroborator
    from .sync.report import overall_status, render_json, render_markdown
    from .sync.runner import run_sync
    from .sync.sources import sources_for

    today = args.today or date.today()
    data_dir = Path(args.data_dir) if args.data_dir else BUNDLED_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    fetcher = HttpFetcher()
    corroborate = None
    if not args.no_corroborate:
        try:
            corroborate = LiteLLMCorroborator.fetch(fetcher)
        except SourceError as exc:
            print(f"warning: cross-check unavailable ({exc}); continuing without it", file=sys.stderr)
    try:
        sources = sources_for(args.provider)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    outcomes = run_sync(sources, data_dir, fetcher, today=today, dry_run=args.dry_run, corroborate=corroborate)
    markdown = render_markdown(outcomes, today=today)
    if args.report_md:
        Path(args.report_md).write_text(markdown, encoding="utf-8")
    if args.report_json:
        Path(args.report_json).write_text(render_json(outcomes, today=today), encoding="utf-8")
    for outcome in outcomes:
        detail = f" - {outcome.error}" if outcome.error else ""
        print(
            f"{outcome.provider:<11} {outcome.status:<13} fetched={outcome.fetched_models:<4} "
            f"applied={len(outcome.applied):<3} "
            f"review={len(outcome.needs_review) + len(outcome.blocking_issues)}{detail}"
        )
    status = overall_status(outcomes)
    return {"failed": EXIT_SOURCE_FAILED, "needs_review": EXIT_REVIEW}.get(status, 0)


def _stale(args: argparse.Namespace) -> int:
    catalog = _catalog(args.data_dir)
    cutoff = date.today() - timedelta(days=args.max_age_days)
    stale: List[str] = []
    for provider in catalog.provider_ids():
        data = catalog.get(provider)
        assert data is not None
        if data.verified_at is None or data.verified_at < cutoff:
            stale.append(f"{provider} (verified {data.verified_at or 'never'})")
        unverified = [m.id for m in data.models if m.status == "unverified"]
        if unverified:
            print(f"{provider}: {len(unverified)} hand-maintained entries are not covered by an automated source")
    if stale:
        print(f"stale pricing data (older than {args.max_age_days} days): {', '.join(stale)}", file=sys.stderr)
        return 1
    print(f"all providers verified within {args.max_age_days} days")
    return 0


def _export_legacy(args: argparse.Namespace) -> int:
    from .legacy import render_legacy_csv

    text = render_legacy_csv()
    target = Path(args.output)
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != text:
            print(f"{target} is out of date; run `openai-cost-calculator pricing export-legacy`", file=sys.stderr)
            return 1
        print(f"{target} is up to date")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return 0
