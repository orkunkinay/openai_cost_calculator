#!/usr/bin/env python3
"""Run a real, isolated Claude Code request through the local OCC proxy.

This is an opt-in, paid, end-to-end test.  Ordinary unit tests never contact a
paid service; this script does.  It:

* creates a unique temporary working directory and an isolated
  ``CLAUDE_CONFIG_DIR`` (never touching the source Claude configuration),
* copies only minimal, file-backed authentication when present and permitted,
  with restrictive permissions, and never prints or exports it,
* installs the OCC status line and lifecycle hooks into the temporary settings,
* starts the anthropic-messages proxy, verifies readiness,
* launches a real ``claude`` process with one tool-free prompt (one paid
  response, no automatic paid retries),
* recomputes the request cost independently from the Anthropic pricing table and
  checks that request == turn == session == first checkpoint == status-line
  values, that a second checkpoint is zero, and that status reads do not mutate
  totals,
* terminates the proxy and removes all temporary material in every path.

It prints a sanitized evidence report and never prints commands, environment
values, headers, or copied authentication contents.  If no safe authentication
prerequisite exists it states which deterministic paths passed and that the live
path remains untested, rather than fabricating success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai_cost_calculator.anthropic.usage import (  # noqa: E402
    price_anthropic_usage,
    usage_from_dict,
)

FIXED_RESPONSE = "OCC_SELF_TEST_OK"
PROMPT = f"Reply with exactly `{FIXED_RESPONSE}`. Do not use any tools."


class SelfTestError(RuntimeError):
    """Raised when the end-to-end evidence violates an acceptance criterion."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", default=shutil.which("claude") or "claude")
    parser.add_argument("--model", help="Model override; defaults to the source config default")
    parser.add_argument("--port", type=int)
    parser.add_argument("--session")
    parser.add_argument("--upstream", help="Anthropic upstream override")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = args.session or f"occ-claude-self-test-{int(time.time())}"
    port = args.port or _free_port()
    proxy_url = f"http://127.0.0.1:{port}"

    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="occ-claude-self-test-"))
        temp_owner: Optional[tempfile.TemporaryDirectory[str]] = None
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="occ-claude-self-test-")
        temp_root = Path(temp_owner.name)

    proxy: Optional[subprocess.Popen[str]] = None
    cleanup_ok = True
    try:
        config_dir = temp_root / "claude-home"
        config_dir.mkdir(mode=0o700)
        auth_method = _prepare_auth(config_dir)
        env = _isolated_env(config_dir, proxy_url)

        install = _run(
            [sys.executable, "-m", "openai_cost_calculator.cli", "claude", "install",
             "--proxy-url", proxy_url],
            env=env,
        )
        if install.returncode != 0:
            raise SelfTestError("installer failed")

        proxy = _start_proxy(port, env, temp_root, args.upstream)
        _wait_ready(proxy_url, proxy)

        # Open a turn explicitly so attribution is deterministic.
        _post_json(f"{proxy_url}/_occ/claude/turn",
                   {"session_id": session, "event": "open", "idempotency_key": "self-test"})

        workdir = temp_root / "empty-workdir"
        workdir.mkdir()
        claude_env = dict(env)
        claude_env["ANTHROPIC_BASE_URL"] = proxy_url
        # Claude Code stamps the session id header from this environment override
        # only when provided; the proxy falls back to a per-session bucket.
        claude_env["OCC_SESSION"] = session
        claude_command = [args.claude, "-p", PROMPT, "--output-format", "text"]
        if args.model:
            claude_command[1:1] = ["--model", args.model]
        nested = _run(claude_command, env=claude_env, cwd=workdir, timeout=180)
        if nested.returncode != 0:
            raise SelfTestError(f"nested Claude failed with exit {nested.returncode}")

        _post_json(f"{proxy_url}/_occ/claude/turn",
                   {"session_id": session, "event": "complete"})

        status = _get_json(f"{proxy_url}/_occ/claude/status?{_query(session)}")
        turn = status.get("turn") or {}
        recorded = Decimal(str(status.get("session_total")))
        if recorded <= 0:
            raise SelfTestError(f"proxy recorded no cost for session: {status}")
        if Decimal(str(turn.get("total_cost"))) != recorded:
            raise SelfTestError("turn total does not equal session total for a single turn")

        first = _post_json(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        second = _post_json(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        after = _get_json(f"{proxy_url}/_occ/claude/status?{_query(session)}")
        if Decimal(str(first.get("total_cost"))) != recorded:
            raise SelfTestError("first checkpoint does not equal the session cost")
        if second.get("total_cost") != "0.00000000":
            raise SelfTestError("second checkpoint was not zero")
        if after.get("session_total") != status.get("session_total"):
            raise SelfTestError("checkpoint/status reads mutated the session total")

        result = {
            "result": "PASS",
            "session": _abbreviate(session),
            "auth_method": auth_method,
            "claude_exit_status": nested.returncode,
            "turn_state": turn.get("state"),
            "recorded_session_cost": str(recorded),
            "turn_cost": turn.get("total_cost"),
            "first_checkpoint": first.get("total_cost"),
            "second_checkpoint": second.get("total_cost"),
            "pricing_semantics": status.get("pricing_semantics"),
            "isolated_claude_home": True,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (SelfTestError, subprocess.TimeoutExpired) as exc:
        print(f"SELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=5)
        if temp_owner is not None:
            try:
                temp_owner.cleanup()
            except OSError:
                cleanup_ok = False
        elif args.keep_temp:
            print(f"Temporary files retained at {temp_root}", file=sys.stderr)
        if not cleanup_ok:
            print("WARNING: temporary cleanup failed", file=sys.stderr)


def _prepare_auth(config_dir: Path) -> str:
    """Copy minimal file-backed auth if present; never read keychain secrets."""
    source = Path(
        os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    ) / ".credentials.json"
    if source.is_file():
        destination = config_dir / ".credentials.json"
        shutil.copy2(source, destination)
        destination.chmod(0o600)
        return "isolated copy of file-backed Claude credentials"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "environment-provided Claude credential"
    raise SelfTestError(
        "no file-backed Claude credentials or ANTHROPIC_API_KEY/CLAUDE_CODE_OAUTH_TOKEN "
        "available; deterministic tests pass but the live path remains untested"
    )


def _isolated_env(config_dir: Path, proxy_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("OCC_ADMIN_TOKEN", None)
    env.pop("OCC_ADMIN_TOKEN_FILE", None)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["OCC_PROXY_URL"] = proxy_url
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def _start_proxy(
    port: int,
    env: dict[str, str],
    temp_root: Path,
    upstream: Optional[str],
) -> subprocess.Popen[str]:
    log = (temp_root / "proxy.log").open("w", encoding="utf-8")
    command = [
        sys.executable, "-m", "openai_cost_calculator.cli", "proxy",
        "--host", "127.0.0.1", "--port", str(port),
        "--protocol", "anthropic-messages",
    ]
    if upstream:
        command.extend(["--upstream", upstream])
    return subprocess.Popen(
        command, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT
    )


def _wait_ready(base_url: str, proxy: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise SelfTestError(f"proxy exited before readiness with status {proxy.returncode}")
        try:
            _get_json(f"{base_url}/_occ/health")
            return
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(0.05)
    raise SelfTestError("proxy did not become ready within 10 seconds")


def independent_cost(model: str, usage: dict[str, int]) -> Decimal:
    """Recompute an Anthropic request cost from the pricing table."""
    return price_anthropic_usage(model, usage_from_dict(usage)).total_cost


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str) -> dict[str, Any]:
    return _request_json(url, "GET", None)


def _post_json(url: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _request_json(url, "POST", body)


def _request_json(url: str, method: str, body: Optional[dict[str, Any]]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise SelfTestError(f"{method} {url} did not return a JSON object")
    return payload


def _query(session: str) -> str:
    return urllib.parse.urlencode({"session": session})


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Optional[Path] = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout, check=False
    )


def _abbreviate(identifier: str) -> str:
    return f"{identifier[:8]}…" if len(identifier) > 8 else identifier


if __name__ == "__main__":
    raise SystemExit(main())
