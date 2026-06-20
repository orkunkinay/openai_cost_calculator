from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from typing import Any, Optional

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
) -> Optional[str]:
    session_id = session or os.environ.get("OCC_SESSION") or "default"
    data = _post_json(
        _url(proxy_url, "/_occ/checkpoint", {"session": session_id}),
        timeout=1.0,
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
        notification = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        if not isinstance(notification, dict):
            return 0
        if notification.get("type") != "agent-turn-complete":
            return 0
        session = os.environ.get("OCC_SESSION") or notification.get("thread-id")
        line = checkpoint_text(notification, session=session)
        if line:
            print(line)
    except Exception:
        return 0
    return 0


def statusline_text(
    *,
    proxy_url: Optional[str] = None,
    session: Optional[str] = None,
) -> str:
    session_id = session or os.environ.get("OCC_SESSION") or "default"
    try:
        data = _get_json(
            _url(proxy_url, "/_occ/costs", {"session": session_id}),
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


def _post_json(url: str, *, timeout: float) -> Optional[dict[str, Any]]:
    request = urllib.request.Request(url, method="POST")
    return _read_json(request, timeout=timeout)


def _read_json(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Optional[dict[str, Any]]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
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
