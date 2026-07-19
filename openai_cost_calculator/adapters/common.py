from __future__ import annotations

import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional


MONEY = "\U0001f4b0"
SEP = "\u00b7"

_SECRET_RE = re.compile(r"(?i)(?:bearer\s+)[^\s,;]+|(?:sk-[a-z0-9_-]{8,})")


def sanitize_diagnostic(value: object, limit: int = 500) -> str:
    """Return a bounded, single-line, secret-redacted diagnostic string."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = "".join(character if character.isprintable() else "?" for character in text)
    return _SECRET_RE.sub("[REDACTED]", text)[:limit]


def admin_headers() -> dict[str, str]:
    """Bearer header for the proxy admin endpoints, if a token is configured."""
    token = os.environ.get("OCC_ADMIN_TOKEN")
    token_file = os.environ.get("OCC_ADMIN_TOKEN_FILE")
    if token is None and token_file:
        try:
            token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            token = None
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def record_jsonl_diagnostic(path: Path, code: str, message: str, *, limit: int = 100) -> None:
    """Append a bounded, sanitized diagnostic to a JSONL file (best effort)."""
    diagnostics = read_jsonl_diagnostics(path)
    diagnostics.append(
        {
            "code": sanitize_diagnostic(code, 64),
            "message": sanitize_diagnostic(message),
            "timestamp": time.time(),
        }
    )
    diagnostics = diagnostics[-limit:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in diagnostics),
            encoding="utf-8",
        )
        path.chmod(0o600)
    except OSError:
        return


def read_jsonl_diagnostics(path: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    diagnostics: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            diagnostics.append(payload)
    return diagnostics


def decimal_from(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def int_from(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def compact_tokens(tokens: int) -> str:
    if abs(tokens) >= 1_000_000:
        value = Decimal(tokens) / Decimal("1000000")
        return f"{value:.1f}M"
    if abs(tokens) >= 1_000:
        value = Decimal(tokens) / Decimal("1000")
        return f"{value:.1f}k"
    return str(tokens)


def format_money(value: Decimal, places: str = "0.0001") -> str:
    return f"${value.quantize(Decimal(places))}"


def nested(data: Any, *keys: str) -> Optional[Any]:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

