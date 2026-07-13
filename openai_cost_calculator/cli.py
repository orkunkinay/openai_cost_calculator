from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="openai-cost-calculator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    proxy_parser = subparsers.add_parser("proxy", help="Run the local cost proxy")
    proxy_parser.add_argument("--host", default="127.0.0.1")
    proxy_parser.add_argument("--port", default=8100, type=int)
    proxy_parser.add_argument("--upstream")
    proxy_parser.add_argument(
        "--auth-mode",
        choices=["auto", "api-key", "chatgpt"],
        default="auto",
        help="Select the matching OpenAI authentication domain (default: auto)",
    )
    proxy_parser.add_argument(
        "--ledger",
        help="Persist accounting to this JSON file (single proxy process only)",
    )

    install_parser = subparsers.add_parser("install", help="Install agent UI adapters")
    install_subparsers = install_parser.add_subparsers(dest="target", required=True)
    claude_install = install_subparsers.add_parser("claude-code")
    claude_install.add_argument("--scope", choices=["user", "project"], default="user")
    codex_install = install_subparsers.add_parser("codex")
    codex_install.add_argument("--proxy-url", default="http://127.0.0.1:8100")
    codex_install.add_argument("--session", default="default")

    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall agent UI adapters")
    uninstall_subparsers = uninstall_parser.add_subparsers(dest="target", required=True)
    claude_uninstall = uninstall_subparsers.add_parser("claude-code")
    claude_uninstall.add_argument("--scope", choices=["user", "project"], default="user")
    uninstall_subparsers.add_parser("codex")

    status_parser = subparsers.add_parser(
        "status", help="Inspect proxy totals, latest calls, and diagnostics"
    )
    _add_proxy_client_arguments(status_parser)
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument(
        "--diagnostics", action="store_true", help="Show accounting diagnostics"
    )

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="Consume and print the next session checkpoint"
    )
    _add_proxy_client_arguments(checkpoint_parser)
    checkpoint_parser.add_argument("--json", action="store_true")

    reset_parser = subparsers.add_parser("reset", help="Reset proxy accounting state")
    reset_parser.add_argument("--proxy-url", default=_default_proxy_url())
    reset_parser.add_argument("--yes", action="store_true", help="Confirm destructive reset")

    ledger_parser = subparsers.add_parser(
        "ledger", help="Inspect or reset an offline durable ledger"
    )
    ledger_subparsers = ledger_parser.add_subparsers(dest="ledger_command", required=True)
    ledger_inspect = ledger_subparsers.add_parser("inspect")
    ledger_inspect.add_argument("path")
    ledger_inspect.add_argument("--json", action="store_true")
    ledger_reset = ledger_subparsers.add_parser("reset")
    ledger_reset.add_argument("path")
    ledger_reset.add_argument("--yes", action="store_true")

    pricing_parser = subparsers.add_parser("pricing", help="Pricing data operations")
    pricing_subparsers = pricing_parser.add_subparsers(dest="pricing_command", required=True)
    pricing_validate = pricing_subparsers.add_parser("validate")
    pricing_validate.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parents[1] / "data" / "gpt_pricing_data.csv"),
    )

    args = parser.parse_args(argv)
    if args.command == "proxy":
        return _run_proxy(
            args.host,
            args.port,
            args.upstream,
            auth_mode=args.auth_mode,
            ledger=args.ledger,
        )
    if args.command == "install":
        return _install(args)
    if args.command == "uninstall":
        return _uninstall(args)
    if args.command == "status":
        return _status(args)
    if args.command == "checkpoint":
        return _checkpoint(args)
    if args.command == "reset":
        return _reset(args)
    if args.command == "ledger":
        return _ledger(args)
    if args.command == "pricing":
        return _pricing(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_proxy(
    host: str,
    port: int,
    upstream: Optional[str],
    *,
    auth_mode: str,
    ledger: Optional[str],
) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The proxy CLI requires optional web dependencies. Install with "
            "`pip install openai-cost-calculator[proxy]`."
        ) from exc

    from openai_cost_calculator.proxy.app import create_app
    from openai_cost_calculator.proxy.upstreams import (
        UpstreamSelectionError,
        resolve_upstream,
    )

    try:
        selection = resolve_upstream(auth_mode=auth_mode, upstream=upstream)
        app = create_app(upstream_selection=selection, ledger_path=ledger)
    except (UpstreamSelectionError, OSError, RuntimeError, ValueError) as exc:
        print(f"proxy configuration error: {exc}", file=sys.stderr)
        return 2
    base = f"http://{host}:{port}"
    print(f"OpenAI-compatible base_url: {base}/v1")
    print(f"Cost summary: {base}/_occ/costs")
    print(
        "Routing: "
        f"auth={selection.auth_mode}, upstream={selection.category}, "
        f"override={'yes' if selection.explicit_override else 'no'}"
    )
    if ledger:
        print(f"Durable ledger: {Path(ledger).expanduser()}")
    uvicorn.run(app, host=host, port=port)
    return 0


def _install(args: argparse.Namespace) -> int:
    from openai_cost_calculator.adapters.install import (
        install_claude_code,
        install_codex,
    )

    if args.target == "claude-code":
        messages = install_claude_code(args.scope)
    elif args.target == "codex":
        messages = install_codex(args.proxy_url, args.session)
    else:
        raise AssertionError(args.target)
    for message in messages:
        print(message)
    print("Undo with: openai-cost-calculator uninstall", args.target)
    return 0


def _uninstall(args: argparse.Namespace) -> int:
    from openai_cost_calculator.adapters.install import (
        uninstall_claude_code,
        uninstall_codex,
    )

    if args.target == "claude-code":
        messages = uninstall_claude_code(args.scope)
    elif args.target == "codex":
        messages = uninstall_codex()
    else:
        raise AssertionError(args.target)
    for message in messages:
        print(message)
    return 0


def _add_proxy_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proxy-url", default=_default_proxy_url())
    parser.add_argument("--session")


def _default_proxy_url() -> str:
    return os.environ.get("OCC_PROXY_URL", "http://127.0.0.1:8100")


def _status(args: argparse.Namespace) -> int:
    from openai_cost_calculator.adapters.codex import notifier_diagnostics

    local_diagnostics = notifier_diagnostics()
    query = {"session": args.session} if args.session else None
    try:
        costs = _request_json(args.proxy_url, "/_occ/costs", query=query)
        health = _request_json(args.proxy_url, "/_occ/health")
    except RuntimeError as exc:
        print(f"status unavailable: {exc}", file=sys.stderr)
        if args.diagnostics and local_diagnostics:
            _print_notifier_diagnostics(local_diagnostics)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "costs": costs,
                    "health": health,
                    "notifier_diagnostics": local_diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    _print_summary(costs, include_diagnostics=args.diagnostics)
    routing = health.get("routing", {})
    persistence = health.get("persistence", {})
    print(
        "Routing: "
        f"auth={routing.get('auth_mode', 'unknown')}, "
        f"upstream={routing.get('upstream_category', 'unknown')}, "
        f"override={'yes' if routing.get('explicit_override') else 'no'}"
    )
    if persistence.get("enabled"):
        state = "healthy" if persistence.get("healthy") else "ERROR"
        print(f"Persistence: {state} ({persistence.get('path')})")
    else:
        print("Persistence: disabled (in-memory only)")
    if args.diagnostics:
        _print_notifier_diagnostics(local_diagnostics)
    return 0


def _checkpoint(args: argparse.Namespace) -> int:
    query = {"session": args.session} if args.session else None
    try:
        payload = _request_json(
            args.proxy_url,
            "/_occ/checkpoint",
            method="POST",
            query=query,
        )
    except RuntimeError as exc:
        print(f"checkpoint failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{payload.get('session', 'default')}: "
            f"{payload.get('num_calls', 0)} calls, "
            f"${payload.get('total_cost', '0.00000000')}"
        )
    return 0


def _reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("refusing to reset accounting without --yes", file=sys.stderr)
        return 2
    try:
        _request_json(args.proxy_url, "/_occ/reset", method="POST")
    except RuntimeError as exc:
        print(f"reset failed: {exc}", file=sys.stderr)
        return 1
    print("Proxy accounting reset.")
    return 0


def _ledger(args: argparse.Namespace) -> int:
    from openai_cost_calculator.proxy.ledger import LedgerError
    from openai_cost_calculator.proxy.registry import TrackerRegistry

    try:
        registry = TrackerRegistry(ledger_path=args.path)
    except LedgerError as exc:
        print(f"ledger unavailable: {exc}", file=sys.stderr)
        return 1
    try:
        if args.ledger_command == "inspect":
            summary = registry.summary()
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_summary(summary, include_diagnostics=True)
            return 0
        if not args.yes:
            print("refusing to reset the ledger without --yes", file=sys.stderr)
            return 2
        registry.reset()
        print(f"Durable ledger reset: {Path(args.path).expanduser()}")
        return 0
    except LedgerError as exc:
        print(f"ledger operation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        registry.close()


def _pricing(args: argparse.Namespace) -> int:
    from openai_cost_calculator.pricing import validate_pricing_file

    try:
        count = validate_pricing_file(args.file)
    except (OSError, ValueError) as exc:
        print(f"pricing validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Pricing data valid: {count} model/date entries in {args.file}")
    return 0


def _request_json(
    proxy_url: str,
    path: str,
    *,
    method: str = "GET",
    query: Optional[dict[str, str]] = None,
) -> dict:
    url = f"{proxy_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"status": exc.code}
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("proxy returned a non-object JSON response")
    return payload


def _print_summary(payload: dict, *, include_diagnostics: bool) -> None:
    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict) or not sessions:
        print("No accounting sessions recorded.")
        return
    for session_id, session in sessions.items():
        if not isinstance(session, dict):
            continue
        latest = session.get("latest_call")
        latest_text = "none"
        if isinstance(latest, dict):
            latest_text = (
                f"{latest.get('model', 'unknown')} "
                f"${latest.get('cost', {}).get('total_cost', 'unknown')}"
            )
        print(
            f"{session_id}: total ${session.get('session_total', '0.00000000')} "
            f"(historical ${session.get('historical_total', '0.00000000')}, "
            f"process ${session.get('process_total', '0.00000000')}), latest {latest_text}"
        )
        if include_diagnostics:
            for error in session.get("errors", []):
                if isinstance(error, dict):
                    print(f"  diagnostic {error.get('code')}: {error.get('message')}")


def _print_notifier_diagnostics(diagnostics: list[dict]) -> None:
    for diagnostic in diagnostics:
        print(
            "Notifier diagnostic "
            f"{diagnostic.get('code')}: {diagnostic.get('message')}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
