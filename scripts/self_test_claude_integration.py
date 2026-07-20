#!/usr/bin/env python3
"""Run a real, isolated Claude Code request through the local OCC proxy.

This is an opt-in, paid, end-to-end test.  Ordinary unit tests never contact a
paid service; this script does.  It:

* starts the anthropic-messages proxy on a free port with a temporary SQLite
  ledger and verifies readiness,
* opens a turn for a fixed session id via the proxy admin endpoint,
* launches a real ``claude -p`` request with ``--session-id`` and
  ``ANTHROPIC_BASE_URL`` pointed at the proxy (one tool-free prompt, no
  automatic paid retries),
* finalizes the turn, then checks that turn cost == session cost == first
  checkpoint, that the second checkpoint is zero, that status reads do not
  mutate totals, and that the ``occ-claude-statusline`` executable's displayed
  turn and session values match the ledger,
* terminates the proxy and removes all temporary material in every path.

Authentication is used read-only and never printed or exported.  When a
file-backed credential or ``ANTHROPIC_API_KEY`` is available the request runs in
an isolated ``CLAUDE_CONFIG_DIR``; a subscription (Keychain) login is used
against the real config directory read-only, because the login is not visible
inside an empty isolated home.  The source configuration is never modified.
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
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

FIXED_RESPONSE = "OCC_SELF_TEST_OK"
PROMPT = f"Reply with exactly `{FIXED_RESPONSE}` and nothing else. Do not use any tools."


class SelfTestError(RuntimeError):
    """Raised when the end-to-end evidence violates an acceptance criterion."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", default=shutil.which("claude") or "claude")
    parser.add_argument("--model", help="Model override; defaults to the account default")
    parser.add_argument("--port", type=int)
    parser.add_argument("--upstream", help="Anthropic upstream override")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = str(uuid.uuid4())
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
        auth_method, isolate = _detect_auth()
        claude_env = _claude_env(temp_root, proxy_url, isolate=isolate, auth_method=auth_method)

        proxy = _start_proxy(port, temp_root, args.upstream)
        _wait_ready(proxy_url, proxy)

        _post(f"{proxy_url}/_occ/claude/turn",
              {"session_id": session, "event": "open", "idempotency_key": "self-test"})

        workdir = temp_root / "empty-workdir"
        workdir.mkdir()
        command = [args.claude, "-p", PROMPT, "--session-id", session, "--output-format", "text"]
        if args.model:
            command[1:1] = ["--model", args.model]
        nested = subprocess.run(
            command, cwd=workdir, env=claude_env, text=True, capture_output=True, timeout=180
        )
        if nested.returncode != 0:
            raise SelfTestError(f"nested Claude failed with exit {nested.returncode}")
        if FIXED_RESPONSE not in nested.stdout:
            raise SelfTestError(f"unexpected nested response: {nested.stdout.strip()!r}")

        _post(f"{proxy_url}/_occ/claude/turn", {"session_id": session, "event": "complete"})

        status = _get(f"{proxy_url}/_occ/claude/status?{_query(session)}")
        turn = status.get("turn") or {}
        session_total = Decimal(str(status.get("session_total")))
        if session_total <= 0:
            raise SelfTestError(f"proxy recorded no cost for the session: {status}")
        if Decimal(str(turn.get("total_cost"))) != session_total:
            raise SelfTestError("turn total does not equal session total for a single turn")
        if turn.get("state") != "completed":
            raise SelfTestError(f"turn did not finalize as completed: {turn}")

        first = _post(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        second = _post(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        after = _get(f"{proxy_url}/_occ/claude/status?{_query(session)}")
        if Decimal(str(first.get("total_cost"))) != session_total:
            raise SelfTestError("first checkpoint does not equal the session cost")
        if second.get("total_cost") != "0.00000000":
            raise SelfTestError("second checkpoint was not zero")
        if after.get("session_total") != status.get("session_total"):
            raise SelfTestError("checkpoint/status reads mutated the session total")

        statusline = _run_statusline(proxy_url, session)
        if _money(session_total) not in statusline:
            raise SelfTestError(
                f"status-line output {statusline!r} does not show the session cost"
            )

        result = {
            "result": "PASS",
            "session": _abbreviate(session),
            "auth_method": auth_method,
            "isolated_claude_home": isolate,
            "claude_exit_status": nested.returncode,
            "claude_response": nested.stdout.strip(),
            "model": _observed_model(proxy_url, session),
            "turn_state": turn.get("state"),
            "turn_requests": turn.get("num_calls"),
            "session_cost": str(session_total),
            "turn_cost": turn.get("total_cost"),
            "first_checkpoint": first.get("total_cost"),
            "second_checkpoint": second.get("total_cost"),
            "pricing_semantics": status.get("pricing_semantics"),
            "statusline": statusline,
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


def _detect_auth() -> tuple[str, bool]:
    """Return (description, isolate). Never reads or prints credential values."""
    source = Path(
        os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
    ) / ".credentials.json"
    if source.is_file():
        return "isolated copy of file-backed Claude credentials", True
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "environment-provided Claude credential", True
    # A subscription (Keychain) login is not visible inside an empty isolated
    # home, so use the real config directory read-only.
    return "existing subscription login (used read-only)", False


def _claude_env(
    temp_root: Path,
    proxy_url: str,
    *,
    isolate: bool,
    auth_method: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = proxy_url
    env.pop("OCC_ADMIN_TOKEN", None)
    env.pop("OCC_ADMIN_TOKEN_FILE", None)
    if isolate:
        iso_home = temp_root / "claude-home"
        iso_home.mkdir(mode=0o700, exist_ok=True)
        source = Path(
            os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"
        ) / ".credentials.json"
        if source.is_file():
            destination = iso_home / ".credentials.json"
            shutil.copy2(source, destination)
            destination.chmod(0o600)
        env["CLAUDE_CONFIG_DIR"] = str(iso_home)
    return env


def _start_proxy(port: int, temp_root: Path, upstream: Optional[str]) -> subprocess.Popen[str]:
    log = (temp_root / "proxy.log").open("w", encoding="utf-8")
    env = dict(os.environ)
    command = [
        sys.executable, "-m", "openai_cost_calculator.cli", "proxy",
        "--host", "127.0.0.1", "--port", str(port),
        "--protocol", "anthropic-messages",
        "--database", str(temp_root / "ledger.db"),
    ]
    if upstream:
        command.extend(["--upstream", upstream])
    return subprocess.Popen(
        command, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT
    )


def _run_statusline(proxy_url: str, session: str) -> str:
    env = dict(os.environ)
    env["OCC_PROXY_URL"] = proxy_url
    executable = Path(sys.executable).parent / "occ-claude-statusline"
    result = subprocess.run(
        [str(executable)] if executable.exists() else
        [sys.executable, "-c",
         "from openai_cost_calculator.adapters.claude_proxy import statusline_main;"
         "raise SystemExit(statusline_main())"],
        input=json.dumps({"session_id": session}),
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=10,
    )
    return result.stdout.strip()


def _observed_model(proxy_url: str, session: str) -> Optional[str]:
    """The priced model(s) for the session, read from the cost summary."""
    costs = _get(f"{proxy_url}/_occ/costs?{_query(session)}")
    session_data = costs.get("sessions", {}).get(session, {})
    models: set[str] = set()
    for turn in session_data.get("turns", []):
        if isinstance(turn, dict):
            models.update((turn.get("cost_by_model") or {}).keys())
    return ", ".join(sorted(models)) or None


def _wait_ready(base_url: str, proxy: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise SelfTestError(f"proxy exited before readiness with status {proxy.returncode}")
        try:
            _get(f"{base_url}/_occ/health")
            return
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(0.05)
    raise SelfTestError("proxy did not become ready within 10 seconds")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str) -> dict[str, Any]:
    return _request(url, "GET")


def _post(url: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _request(url, "POST", body)


def _request(url: str, method: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.0001'))}"


def _abbreviate(identifier: str) -> str:
    return f"{identifier[:8]}…" if len(identifier) > 8 else identifier


if __name__ == "__main__":
    raise SystemExit(main())
