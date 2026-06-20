from __future__ import annotations

import argparse
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="openai-cost-calculator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    proxy_parser = subparsers.add_parser("proxy", help="Run the local cost proxy")
    proxy_parser.add_argument("--host", default="127.0.0.1")
    proxy_parser.add_argument("--port", default=8100, type=int)
    proxy_parser.add_argument("--upstream", default="https://api.openai.com/v1")

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

    args = parser.parse_args(argv)
    if args.command == "proxy":
        return _run_proxy(args.host, args.port, args.upstream)
    if args.command == "install":
        return _install(args)
    if args.command == "uninstall":
        return _uninstall(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_proxy(host: str, port: int, upstream: str) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The proxy CLI requires optional web dependencies. Install with "
            "`pip install openai-cost-calculator[proxy]`."
        ) from exc

    from openai_cost_calculator.proxy.app import create_app

    app = create_app(upstream=upstream)
    base = f"http://{host}:{port}"
    print(f"OpenAI-compatible base_url: {base}/v1")
    print(f"Cost summary: {base}/_occ/costs")
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
