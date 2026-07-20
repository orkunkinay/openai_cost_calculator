from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import stat
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
        "--protocol",
        choices=["openai-responses", "anthropic-messages"],
        default="openai-responses",
        help="Wire protocol to account (default: openai-responses for Codex)",
    )
    proxy_parser.add_argument(
        "--auth-mode",
        choices=["auto", "api-key", "chatgpt"],
        default="auto",
        help="Select the matching OpenAI authentication domain (default: auto)",
    )
    proxy_parser.add_argument(
        "--ledger",
        help="Persist accounting to a legacy JSON file (single proxy process only)",
    )
    proxy_parser.add_argument(
        "--database",
        help="Persist accounting to a concurrent SQLite database (recommended)",
    )
    proxy_parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit a non-loopback bind; requires an administrative token",
    )
    proxy_parser.add_argument(
        "--admin-token-file",
        help="Read the administrative bearer token from a protected file",
    )

    install_parser = subparsers.add_parser("install", help="Install agent UI adapters")
    install_subparsers = install_parser.add_subparsers(dest="target", required=True)
    claude_install = install_subparsers.add_parser("claude-code")
    claude_install.add_argument("--scope", choices=["user", "project"], default="user")
    codex_install = install_subparsers.add_parser("codex")
    codex_install.add_argument("--proxy-url", default="http://127.0.0.1:8100")
    codex_install.add_argument("--session", default="default")
    codex_install.add_argument(
        "--websockets",
        action="store_true",
        help="Opt in to Codex Responses WebSockets (HTTP is the cost-safe default)",
    )

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
    reset_parser.add_argument("--admin-token-file")
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

    database_parser = subparsers.add_parser(
        "database", help="Inspect or reset a SQLite accounting database"
    )
    database_subparsers = database_parser.add_subparsers(
        dest="database_command", required=True
    )
    database_inspect = database_subparsers.add_parser("inspect")
    database_inspect.add_argument("path")
    database_inspect.add_argument("--json", action="store_true")
    database_reset = database_subparsers.add_parser("reset")
    database_reset.add_argument("path")
    database_reset.add_argument("--yes", action="store_true")

    pricing_parser = subparsers.add_parser("pricing", help="Pricing data operations")
    pricing_subparsers = pricing_parser.add_subparsers(dest="pricing_command", required=True)
    pricing_validate = pricing_subparsers.add_parser("validate")
    pricing_validate.add_argument(
        "--file",
        default=str(Path(__file__).resolve().parents[1] / "data" / "gpt_pricing_data.csv"),
    )

    _add_claude_parser(subparsers)

    args = parser.parse_args(argv)
    if args.command == "claude":
        return _claude(args)
    if args.command == "proxy":
        return _run_proxy(
            args.host,
            args.port,
            args.upstream,
            auth_mode=args.auth_mode,
            protocol=args.protocol,
            ledger=args.ledger,
            database=args.database,
            allow_remote=args.allow_remote,
            admin_token_file=args.admin_token_file,
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
    if args.command == "database":
        return _database(args)
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
    protocol: str = "openai-responses",
    ledger: Optional[str],
    database: Optional[str],
    allow_remote: bool,
    admin_token_file: Optional[str],
) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The proxy CLI requires optional web dependencies. Install with "
            "`pip install openai-cost-calculator[proxy]`."
        ) from exc

    from openai_cost_calculator.proxy.app import create_app

    if protocol == "anthropic-messages":
        return _run_anthropic_proxy(
            host,
            port,
            upstream,
            ledger=ledger,
            database=database,
            allow_remote=allow_remote,
            admin_token_file=admin_token_file,
            uvicorn=uvicorn,
            create_app=create_app,
        )

    from openai_cost_calculator.proxy.upstreams import (
        UpstreamSelectionError,
        resolve_upstream,
    )

    try:
        if ledger and database:
            raise ValueError("pass either --ledger or --database, not both")
        admin_token = _load_admin_token(admin_token_file)
        _validate_proxy_exposure(host, allow_remote, admin_token)
        selection = resolve_upstream(auth_mode=auth_mode, upstream=upstream)
        app = create_app(
            upstream_selection=selection,
            ledger_path=ledger,
            database_path=database,
            admin_token=admin_token,
        )
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
    if not _is_loopback_host(host):
        print(
            "WARNING: proxy is remotely reachable; protect it with TLS and trusted network controls",
            file=sys.stderr,
        )
    if ledger:
        print(f"Durable ledger: {Path(ledger).expanduser()}")
    if database:
        print(f"SQLite database: {Path(database).expanduser()}")
    uvicorn.run(app, host=host, port=port)
    return 0


def _run_anthropic_proxy(
    host: str,
    port: int,
    upstream: Optional[str],
    *,
    ledger: Optional[str],
    database: Optional[str],
    allow_remote: bool,
    admin_token_file: Optional[str],
    uvicorn,
    create_app,
) -> int:
    from openai_cost_calculator.anthropic.resolve import (
        ClaudeResolutionError,
        resolve_claude,
    )

    base = f"http://{host}:{port}"
    try:
        if ledger and database:
            raise ValueError("pass either --ledger or --database, not both")
        admin_token = _load_admin_token(admin_token_file)
        _validate_proxy_exposure(host, allow_remote, admin_token)
        resolution = resolve_claude(
            env=os.environ,
            upstream=upstream,
            proxy_self_urls={base, f"{base}/", f"{base}/v1"},
        )
        if resolution.support_level == "unsupported":
            raise ValueError(
                f"unsupported Claude provider '{resolution.provider_category}': "
                "cost cannot be observed through the Anthropic Messages proxy"
            )
        semantics = {
            "api-equivalent": "api-equivalent",
            "billed-estimate": "billed-estimate",
            "unavailable": "unavailable",
        }.get(resolution.pricing_semantics, "billed-estimate")
        app = create_app(
            upstream=resolution.resolved_upstream,
            protocol="anthropic-messages",
            pricing_semantics=semantics,
            ledger_path=ledger,
            database_path=database,
            admin_token=admin_token,
        )
    except (ClaudeResolutionError, OSError, RuntimeError, ValueError) as exc:
        print(f"proxy configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Anthropic Messages base_url: {base}")
    print(f"Claude status: {base}/_occ/claude/status")
    print(
        "Routing: "
        f"auth={resolution.auth_mode}, provider={resolution.provider_category}, "
        f"upstream={resolution.upstream_category}, "
        f"pricing={resolution.pricing_semantics}, support={resolution.support_level}"
    )
    if not _is_loopback_host(host):
        print(
            "WARNING: proxy is remotely reachable; protect it with TLS and trusted network controls",
            file=sys.stderr,
        )
    if ledger:
        print(f"Durable ledger: {Path(ledger).expanduser()}")
    if database:
        print(f"SQLite database: {Path(database).expanduser()}")
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
        messages = install_codex(
            args.proxy_url,
            args.session,
            supports_websockets=args.websockets,
        )
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
    parser.add_argument("--admin-token-file")


def _default_proxy_url() -> str:
    return os.environ.get("OCC_PROXY_URL", "http://127.0.0.1:8100")


def _status(args: argparse.Namespace) -> int:
    from openai_cost_calculator.adapters.codex import notifier_diagnostics

    local_diagnostics = notifier_diagnostics()
    query = {"session": args.session} if args.session else None
    try:
        admin_token = _load_admin_token(args.admin_token_file)
        costs = _request_json(
            args.proxy_url,
            "/_occ/costs",
            query=query,
            admin_token=admin_token,
        )
        health = _request_json(
            args.proxy_url,
            "/_occ/health",
            admin_token=admin_token,
        )
    except (RuntimeError, ValueError) as exc:
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
        admin_token = _load_admin_token(args.admin_token_file)
        payload = _request_json(
            args.proxy_url,
            "/_occ/checkpoint",
            method="POST",
            query=query,
            admin_token=admin_token,
        )
    except (RuntimeError, ValueError) as exc:
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
        admin_token = _load_admin_token(args.admin_token_file)
        _request_json(
            args.proxy_url,
            "/_occ/reset",
            method="POST",
            admin_token=admin_token,
        )
    except (RuntimeError, ValueError) as exc:
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


def _database(args: argparse.Namespace) -> int:
    from openai_cost_calculator.proxy.ledger import LedgerError
    from openai_cost_calculator.proxy.registry import TrackerRegistry

    try:
        registry = TrackerRegistry(database_path=args.path)
    except LedgerError as exc:
        print(f"database unavailable: {exc}", file=sys.stderr)
        return 1
    try:
        if args.database_command == "inspect":
            summary = registry.summary()
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_summary(summary, include_diagnostics=True)
            return 0
        if not args.yes:
            print("refusing to reset the database without --yes", file=sys.stderr)
            return 2
        registry.reset()
        print(f"SQLite accounting database reset: {Path(args.path).expanduser()}")
        return 0
    except LedgerError as exc:
        print(f"database operation failed: {exc}", file=sys.stderr)
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
    admin_token: Optional[str] = None,
) -> dict:
    url = f"{proxy_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    headers = (
        {"Authorization": f"Bearer {admin_token}"}
        if admin_token is not None
        else {}
    )
    request = urllib.request.Request(url, method=method, headers=headers)
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


def _load_admin_token(path: Optional[str]) -> Optional[str]:
    environment_token = os.environ.get("OCC_ADMIN_TOKEN")
    environment_file = os.environ.get("OCC_ADMIN_TOKEN_FILE")
    selected_path = path or environment_file
    if path and environment_file and Path(path).expanduser() != Path(environment_file).expanduser():
        raise ValueError(
            "administrative token file is ambiguous between CLI and environment"
        )
    if selected_path and environment_token:
        raise ValueError(
            "set either OCC_ADMIN_TOKEN or an administrative token file, not both"
        )
    if selected_path:
        token_path = Path(selected_path).expanduser()
        try:
            mode = stat.S_IMODE(token_path.stat().st_mode)
            if mode & 0o077:
                raise ValueError(
                    "administrative token file must not be accessible by group or others"
                )
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError("cannot read administrative token file") from exc
    else:
        token = environment_token
    if token is not None and len(token) < 32:
        raise ValueError("administrative token must contain at least 32 characters")
    return token


def _validate_proxy_exposure(
    host: str,
    allow_remote: bool,
    admin_token: Optional[str],
) -> None:
    if _is_loopback_host(host):
        return
    if not allow_remote:
        raise ValueError(
            "non-loopback proxy binding requires explicit --allow-remote"
        )
    if admin_token is None:
        raise ValueError(
            "non-loopback proxy binding requires OCC_ADMIN_TOKEN or --admin-token-file"
        )


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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


def _add_claude_parser(subparsers) -> None:
    claude = subparsers.add_parser(
        "claude", help="Proxy-backed Claude Code cost observation"
    )
    claude_sub = claude.add_subparsers(dest="claude_command", required=True)

    install = claude_sub.add_parser("install", help="Install the Claude settings.json integration")
    install.add_argument("--proxy-url", default="http://127.0.0.1:8100")
    install.add_argument("--upstream")
    install.add_argument(
        "--replace-statusline",
        action="store_true",
        help="Replace an existing status line instead of refusing",
    )
    claude_sub.add_parser("uninstall", help="Remove the Claude settings.json integration")
    claude_sub.add_parser("check", help="Report effective configuration and conflicts")

    status = claude_sub.add_parser("status", help="Show current-turn and session cost")
    _add_proxy_client_arguments(status)
    status.add_argument("--json", action="store_true")
    status.add_argument("--diagnostics", action="store_true")

    checkpoint = claude_sub.add_parser("checkpoint", help="Consume the next session checkpoint")
    _add_proxy_client_arguments(checkpoint)
    checkpoint.add_argument("--json", action="store_true")

    reset_session = claude_sub.add_parser("reset-session", help="Reset all proxy accounting")
    reset_session.add_argument("--proxy-url", default=_default_proxy_url())
    reset_session.add_argument("--admin-token-file")
    reset_session.add_argument("--yes", action="store_true")

    pricing = claude_sub.add_parser("pricing", help="Anthropic pricing operations")
    pricing_sub = pricing.add_subparsers(dest="claude_pricing_command", required=True)
    pricing_sub.add_parser("validate")

    self_test = claude_sub.add_parser("self-test", help="Run the opt-in real Claude self-test")
    self_test.add_argument("passthrough", nargs=argparse.REMAINDER)


def _claude(args: argparse.Namespace) -> int:
    if args.claude_command == "install":
        from openai_cost_calculator.adapters.install import install_claude

        try:
            messages = install_claude(
                args.proxy_url,
                replace_statusline=args.replace_statusline,
                upstream=args.upstream,
            )
        except ValueError as exc:
            print(f"install failed: {exc}", file=sys.stderr)
            return 2
        for message in messages:
            print(message)
        print("Undo with: openai-cost-calculator claude uninstall")
        return 0
    if args.claude_command == "uninstall":
        from openai_cost_calculator.adapters.install import uninstall_claude

        for message in uninstall_claude():
            print(message)
        return 0
    if args.claude_command == "check":
        from openai_cost_calculator.adapters.install import check_claude

        _print_claude_check(check_claude())
        return 0
    if args.claude_command == "status":
        return _claude_status(args)
    if args.claude_command == "checkpoint":
        return _checkpoint(args)
    if args.claude_command == "reset-session":
        return _reset(args)
    if args.claude_command == "pricing":
        from openai_cost_calculator.anthropic.pricing import (
            AnthropicPricingError,
            validate_anthropic_pricing,
        )

        try:
            count = validate_anthropic_pricing()
        except AnthropicPricingError as exc:
            print(f"pricing validation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Anthropic pricing valid: {count} model/date tiers")
        return 0
    if args.claude_command == "self-test":
        import subprocess

        script = Path(__file__).resolve().parents[1] / "scripts" / "self_test_claude_integration.py"
        return subprocess.call([sys.executable, str(script), *args.passthrough])
    return 2


def _claude_status(args: argparse.Namespace) -> int:
    query = {"session": args.session} if args.session else None
    try:
        admin_token = _load_admin_token(args.admin_token_file)
        status = _request_json(
            args.proxy_url, "/_occ/claude/status", query=query, admin_token=admin_token
        )
    except (RuntimeError, ValueError) as exc:
        print(f"status unavailable: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    turn = status.get("turn") or {}
    session_id = str(status.get("session", "default"))
    print(f"Session: {_abbreviate(session_id)}")
    print(f"Turn: {turn.get('state', 'none') if turn else 'none'}")
    print()
    print(f"Turn cost:              ${turn.get('total_cost', '0.00000000')}")
    print(f"Session cost:           ${status.get('session_total', '0.00000000')}")
    print(f"Turn requests:          {turn.get('num_calls', 0)}")
    print(f"Session requests:       {status.get('session_requests', 0)}")
    print(f"Accounting:             {status.get('accounting', 'unknown')}")
    print(f"Pricing semantics:      {status.get('pricing_semantics', 'unknown')}")
    persistence = status.get("persistence", {})
    enabled = isinstance(persistence, dict) and persistence.get("enabled")
    print(f"Persistence:            {'enabled' if enabled else 'disabled'}")
    if args.diagnostics:
        for error in status.get("errors", []):
            if isinstance(error, dict):
                print(f"  diagnostic {error.get('code')}: {error.get('message')}")
    return 0


def _print_claude_check(report: dict) -> None:
    print(f"Config path:          {report.get('config_path')}")
    print(f"Settings present:     {report.get('settings_exists')}")
    print(f"ANTHROPIC_BASE_URL:   {report.get('anthropic_base_url')}")
    print(f"OCC_PROXY_URL:        {report.get('occ_proxy_url')}")
    print(f"Status line:          {'installed' if report.get('statusline_installed') else 'absent'}")
    print(f"Hook events:          {', '.join(report.get('hook_events_installed', [])) or 'none'}")
    print(f"Manifest present:     {report.get('manifest_present')}")
    for conflict in report.get("conflicts", []):
        print(f"  conflict: {conflict}")


def _abbreviate(identifier: str) -> str:
    if len(identifier) <= 12:
        return identifier
    return f"{identifier[:4]}…{identifier[-4:]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
