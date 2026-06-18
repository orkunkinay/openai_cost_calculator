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

    args = parser.parse_args(argv)
    if args.command == "proxy":
        return _run_proxy(args.host, args.port, args.upstream)

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
