from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
import re

from openai_cost_calculator.adapters.common import (
    MONEY,
    SEP,
    compact_tokens,
    decimal_from,
    format_money,
)


def checkpoint_text(
    payload: dict[str, Any],
    *,
    proxy_url: Optional[str] = None,
    session: Optional[str] = None,
    on_error=None,
) -> Optional[str]:
    session_id = session or os.environ.get("OCC_SESSION") or "default"
    data = _post_json(
        _url(proxy_url, "/_occ/checkpoint", {"session": session_id}),
        timeout=1.0,
        on_error=on_error,
    )
    if not data:
        return None
    cost = decimal_from(data.get("total_cost"))
    if cost <= 0:
        return None
    model = _primary_model(data)
    prompt_tokens = int(data.get("prompt_tokens") or 0)
    completion_tokens = int(data.get("completion_tokens") or 0)
    return (
        f"{MONEY} Turn {format_money(cost)} {SEP} "
        f"{model} {compact_tokens(prompt_tokens)}->{compact_tokens(completion_tokens)}"
    )


def notify_main() -> int:
    try:
        raw_notification = _notification_arg()
        notification = json.loads(raw_notification) if raw_notification else {}
        if not isinstance(notification, dict):
            return 0
        if notification.get("type") != "agent-turn-complete":
            _run_previous_notify(
                raw_notification,
                on_error=_record_notifier_diagnostic,
            )
            return 0
        settings = _codex_adapter_settings()
        session = (
            os.environ.get("OCC_SESSION")
            or settings.get("session")
            or "default"
        )
        line = checkpoint_text(
            notification,
            proxy_url=settings.get("proxy_url"),
            session=session,
            on_error=_record_notifier_diagnostic,
        )
        if line:
            print(line)
        _run_previous_notify(
            raw_notification,
            previous_notify=settings.get("previous_notify"),
            on_error=_record_notifier_diagnostic,
        )
    except Exception as exc:
        _record_notifier_diagnostic(
            "notifier_failed",
            f"notification handling failed: {type(exc).__name__}",
        )
        return 0
    return 0


def statusline_text(
    *,
    proxy_url: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    settings = _codex_adapter_settings()
    session_id = (
        session
        or os.environ.get("OCC_SESSION")
        or settings.get("session")
        or "default"
    )
    try:
        data = _get_json(
            _url(
                proxy_url or settings.get("proxy_url"),
                "/_occ/costs",
                {"session": session_id},
            ),
            timeout=1.0,
        )
        if data is None:
            return f"{MONEY} cost offline"
        session_data = _session_data(data, session_id)
        if not session_data:
            return f"{MONEY} $0.0000 session {SEP} last $0.0000"
        total = decimal_from(session_data.get("session_total"))
        turns = session_data.get("turns")
        last_cost = Decimal("0")
        if isinstance(turns, list) and turns:
            last = turns[-1]
            if isinstance(last, dict):
                last_cost = decimal_from(last.get("total_cost"))
        return (
            f"{MONEY} {format_money(total)} session {SEP} "
            f"last {format_money(last_cost)}"
        )
    except Exception:
        return f"{MONEY} cost offline"


def statusline_main() -> int:
    print(statusline_text())
    return 0


def _url(
    proxy_url: Optional[str],
    path: str,
    query: Optional[dict[str, str]] = None,
) -> str:
    base = (proxy_url or os.environ.get("OCC_PROXY_URL") or "http://127.0.0.1:8100").rstrip("/")
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    return url


def _get_json(url: str, *, timeout: float) -> Optional[dict[str, Any]]:
    request = urllib.request.Request(url, method="GET")
    return _read_json(request, timeout=timeout)


def _post_json(
    url: str,
    *,
    timeout: float,
    on_error=None,
) -> Optional[dict[str, Any]]:
    request = urllib.request.Request(url, method="POST")
    return _read_json(request, timeout=timeout, on_error=on_error)


def _read_json(
    request: urllib.request.Request,
    *,
    timeout: float,
    on_error=None,
) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if on_error is not None:
            on_error(
                "checkpoint_unavailable",
                f"proxy checkpoint request failed: {type(exc).__name__}",
            )
        return None
    return payload if isinstance(payload, dict) else None


def _primary_model(data: dict[str, Any]) -> str:
    models = data.get("models")
    if isinstance(models, dict) and models:
        def model_cost(item: tuple[str, Any]) -> Decimal:
            value = item[1]
            if isinstance(value, dict):
                return decimal_from(value.get("total_cost"))
            return Decimal("0")

        return max(models.items(), key=model_cost)[0]
    costs = data.get("cost_by_model")
    if isinstance(costs, dict) and costs:
        return max(costs.items(), key=lambda item: decimal_from(item[1]))[0]
    return "model"


def _session_data(
    data: Optional[dict[str, Any]],
    session_id: str,
) -> Optional[dict[str, Any]]:
    if not data:
        return None
    if "session_total" in data:
        return data
    sessions = data.get("sessions")
    if isinstance(sessions, dict):
        value = sessions.get(session_id)
        if isinstance(value, dict):
            return value
    return None


def _notification_arg() -> Optional[str]:
    for value in reversed(sys.argv[1:]):
        value = value.strip()
        if value.startswith("{") and value.endswith("}"):
            return value
    return None


def _codex_adapter_settings() -> dict[str, str]:
    path = (
        Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        / "config.toml"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    settings: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "# >>> openai-cost-calculator":
            in_block = True
            continue
        if stripped == "# <<< openai-cost-calculator":
            break
        if not in_block:
            continue
        if stripped.startswith("# previous_notify = "):
            settings["previous_notify"] = stripped.removeprefix(
                "# previous_notify = "
            ).strip()
        elif stripped.startswith("occ_codex_proxy_url"):
            settings["proxy_url"] = _toml_string_value(stripped)
        elif stripped.startswith("occ_codex_session"):
            settings["session"] = _toml_string_value(stripped)
    return settings


def _toml_string_value(line: str) -> str:
    _, _, value = line.partition("=")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def _run_previous_notify(
    notification: Optional[str],
    *,
    previous_notify: Optional[str] = None,
    on_error=None,
) -> None:
    if not notification:
        return
    previous_notify = (
        previous_notify or _codex_adapter_settings().get("previous_notify")
    )
    command = _parse_notify_command(previous_notify)
    if not command or _is_self_notify(command[0]):
        return
    try:
        completed = subprocess.run(
            [*command, notification],
            check=False,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
        if completed.returncode != 0 and on_error is not None:
            on_error(
                "previous_notifier_failed",
                f"previous notifier exited with status {completed.returncode}",
            )
    except Exception as exc:
        if on_error is not None:
            on_error(
                "previous_notifier_failed",
                f"previous notifier could not run: {type(exc).__name__}",
            )
        return


def _parse_notify_command(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    try:
        import tomllib  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - Python < 3.11 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except Exception:
            tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        try:
            parsed = tomllib.loads(value)
            notify = parsed.get("notify")
            if isinstance(notify, list) and all(
                isinstance(item, str) for item in notify
            ):
                return notify
        except Exception:
            return None
    return None


def _is_self_notify(command: str) -> bool:
    return Path(command).name in {"occ-codex-notify", "occ-codex-statusline"}


def notifier_diagnostics() -> list[dict[str, Any]]:
    path = _notifier_diagnostic_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    diagnostics = []
    for line in lines[-100:]:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            diagnostics.append(payload)
    return diagnostics


def _record_notifier_diagnostic(code: str, message: str) -> None:
    path = _notifier_diagnostic_path()
    diagnostic = {
        "code": _safe_diagnostic_text(code, 64),
        "message": _safe_diagnostic_text(message, 500),
        "timestamp": time.time(),
    }
    diagnostics = notifier_diagnostics()
    diagnostics.append(diagnostic)
    diagnostics = diagnostics[-100:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in diagnostics),
            encoding="utf-8",
        )
        path.chmod(0o600)
    except OSError:
        return


def _notifier_diagnostic_path() -> Path:
    return (
        Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        / "occ-notifier-diagnostics.jsonl"
    )


_SECRET_RE = re.compile(r"(?i)(?:bearer\s+)[^\s,;]+|(?:sk-[a-z0-9_-]{8,})")


def _safe_diagnostic_text(value: object, limit: int) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = "".join(character if character.isprintable() else "?" for character in text)
    return _SECRET_RE.sub("[REDACTED]", text)[:limit]
