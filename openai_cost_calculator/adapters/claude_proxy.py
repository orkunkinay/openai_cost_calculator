"""Proxy-backed Claude Code status line and lifecycle hook.

These executables talk to the local OCC proxy's Claude admin endpoints so that
the *current-turn* and *cumulative-session* cost the proxy has independently
observed are visible directly inside Claude Code.  They are distinct from the
transcript-based ``occ-cc-*`` adapters, which read Claude Code's own cost field.

The status line performs only non-mutating reads; the hook drives turn
boundaries.  Neither ever prints stack traces, credentials, full session ids, or
sensitive paths, and a failure in either must never break Claude Code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from openai_cost_calculator.adapters.common import (
    MONEY,
    SEP,
    admin_headers,
    decimal_from,
    format_money,
    record_jsonl_diagnostic,
)


DEFAULT_PROXY_URL = "http://127.0.0.1:8100"
UNAVAILABLE = f"{MONEY} OCC cost unavailable {SEP} inspect diagnostics"


# --------------------------------------------------------------------------- #
#   Status line                                                               #
# --------------------------------------------------------------------------- #
def statusline_text(payload: dict[str, Any], *, proxy_url: Optional[str] = None) -> str:
    session_id = _session_id(payload)
    data = _get_status(proxy_url, session_id)
    if data is None:
        return UNAVAILABLE
    return _render_status(data)


def _render_status(data: dict[str, Any]) -> str:
    accounting = data.get("accounting")
    if accounting == "unavailable":
        return UNAVAILABLE

    api_equivalent = data.get("pricing_semantics") == "api-equivalent"
    prefix = "API-eq " if api_equivalent else ""
    session_total = decimal_from(data.get("session_total"))
    turn = data.get("turn") if isinstance(data.get("turn"), dict) else None
    turn_active = bool(data.get("turn_is_active"))
    turn_word = "Turn" if turn_active or turn is None else "Last turn"
    turn_total = decimal_from(turn.get("total_cost")) if turn else Decimal("0")

    line = (
        f"{MONEY} OCC {turn_word} {prefix}{_money(turn_total)} "
        f"{SEP} Session {prefix}{_money(session_total)}"
    )
    if accounting == "partial":
        line = f"{line} {SEP} inspect diagnostics"
    return line


def compose_statusline_text(
    raw_stdin: str,
    previous_command: str,
    *,
    proxy_url: Optional[str] = None,
) -> str:
    """Render an existing status line composed with the OCC status line.

    The previous command is run the same way Claude Code runs it (through the
    shell, with the identical stdin), so no new unsafe evaluation is introduced.
    The two segments fail independently: a failure in one never suppresses the
    other, and the OCC segment is always appended after the existing output.
    """
    previous_output = _run_previous_statusline(previous_command, raw_stdin)
    try:
        payload = json.loads(raw_stdin or "{}")
        if not isinstance(payload, dict):
            payload = {}
        occ_output = statusline_text(payload, proxy_url=proxy_url)
    except Exception:
        occ_output = UNAVAILABLE
    segments = [segment for segment in (previous_output, occ_output) if segment]
    return f" {SEP} ".join(segments) if segments else UNAVAILABLE


def _run_previous_statusline(previous_command: str, raw_stdin: str) -> str:
    if not previous_command:
        return ""
    try:
        completed = subprocess.run(
            previous_command,
            shell=True,
            input=raw_stdin,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def statusline_main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--compose":
        return _compose_statusline_main(argv[1])
    try:
        payload = _read_stdin_json()
        print(statusline_text(payload))
    except Exception:
        # Never surface a traceback into the Claude status bar.
        print(UNAVAILABLE)
    return 0


def _compose_statusline_main(encoded_previous: str) -> int:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    try:
        previous_command = base64.urlsafe_b64decode(encoded_previous.encode("ascii")).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        previous_command = ""
    try:
        print(compose_statusline_text(raw, previous_command))
    except Exception:
        print(UNAVAILABLE)
    return 0


def encode_previous_statusline(previous_command: str) -> str:
    """Encode a previous status-line command for the ``--compose`` argument."""
    return base64.urlsafe_b64encode(previous_command.encode("utf-8")).decode("ascii")


# --------------------------------------------------------------------------- #
#   Lifecycle hook                                                            #
# --------------------------------------------------------------------------- #
_TURN_EVENTS = {
    "UserPromptSubmit": ("open", None),
    "Stop": ("complete", None),
    "SessionEnd": ("interrupt", None),
}


def hook_output(payload: dict[str, Any], *, proxy_url: Optional[str] = None) -> dict[str, Any]:
    """Translate a Claude hook event into a proxy turn-lifecycle call.

    Returns a small result dict for testing; the executable prints nothing so
    that ``UserPromptSubmit`` context is never polluted.
    """
    event_name = payload.get("hook_event_name")
    session_id = _session_id(payload)
    if _truthy(os.environ.get("OCC_CLAUDE_HOOK_DEBUG")):
        _record_diagnostic(
            "hook_invoked", f"event={event_name} session={session_id[:8]}…"
        )
    if event_name not in _TURN_EVENTS:
        return {"handled": False, "event": event_name}
    action, _ = _TURN_EVENTS[event_name]
    body: dict[str, Any] = {"session_id": session_id, "event": action}
    if action == "open":
        body["idempotency_key"] = _prompt_key(session_id, payload.get("prompt"))
    result = _post_turn(proxy_url, body)
    return {"handled": True, "event": event_name, "action": action, "result": result}


def hook_main() -> int:
    try:
        payload = _read_stdin_json()
        hook_output(payload)
    except Exception as exc:
        _record_diagnostic("hook_failed", f"Claude hook failed: {type(exc).__name__}")
    # Emit nothing on stdout and always succeed so Claude Code is never blocked.
    return 0


# --------------------------------------------------------------------------- #
#   Helpers                                                                   #
# --------------------------------------------------------------------------- #
def _read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    payload = json.loads(raw or "{}")
    return payload if isinstance(payload, dict) else {}


def _session_id(payload: dict[str, Any]) -> str:
    value = payload.get("session_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unscoped-claude"


def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def _prompt_key(session_id: str, prompt: Any) -> str:
    material = f"{session_id}:{prompt if isinstance(prompt, str) else ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _base_url(proxy_url: Optional[str]) -> str:
    return (proxy_url or os.environ.get("OCC_PROXY_URL") or DEFAULT_PROXY_URL).rstrip("/")


def _get_status(proxy_url: Optional[str], session_id: str) -> Optional[dict[str, Any]]:
    url = (
        f"{_base_url(proxy_url)}/_occ/claude/status?"
        + urllib.parse.urlencode({"session": session_id})
    )
    request = urllib.request.Request(url, method="GET", headers=admin_headers())
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _post_turn(proxy_url: Optional[str], body: dict[str, Any]) -> Optional[dict[str, Any]]:
    url = f"{_base_url(proxy_url)}/_occ/claude/turn"
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", **admin_headers()}
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        _record_diagnostic(
            "hook_failed", f"turn lifecycle request failed: {type(exc).__name__}"
        )
        return None
    return payload if isinstance(payload, dict) else None


def _money(value: Decimal) -> str:
    return format_money(value)


def _diagnostic_path() -> Path:
    root = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(root) if root else Path.home() / ".claude"
    return base / "occ-claude-hook-diagnostics.jsonl"


def _record_diagnostic(code: str, message: str) -> None:
    record_jsonl_diagnostic(_diagnostic_path(), code, message)


def hook_diagnostics() -> list[dict[str, Any]]:
    from openai_cost_calculator.adapters.common import read_jsonl_diagnostics

    return read_jsonl_diagnostics(_diagnostic_path())
