from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


PLATFORM_UPSTREAM = "https://api.openai.com/v1"
CHATGPT_CODEX_UPSTREAM = "https://chatgpt.com/backend-api/codex"
AUTH_MODES = {"auto", "api-key", "chatgpt"}


class UpstreamSelectionError(ValueError):
    """Raised when authentication and upstream settings are ambiguous or incompatible."""


@dataclass(frozen=True)
class UpstreamSelection:
    auth_mode: str
    category: str
    url: str
    explicit_override: bool
    detection_source: str


def resolve_upstream(
    *,
    auth_mode: str = "auto",
    upstream: Optional[str] = None,
    codex_home: str | Path | None = None,
) -> UpstreamSelection:
    if auth_mode not in AUTH_MODES:
        raise UpstreamSelectionError(
            f"authentication mode must be one of: {', '.join(sorted(AUTH_MODES))}"
        )

    selected_mode = auth_mode
    source = "explicit CLI option"
    if auth_mode == "auto":
        detected, source = detect_codex_auth_mode(codex_home)
        selected_mode = detected or "unspecified"

    explicit_override = upstream is not None
    if upstream is None:
        upstream = (
            CHATGPT_CODEX_UPSTREAM
            if selected_mode == "chatgpt"
            else PLATFORM_UPSTREAM
        )
    normalized = _normalize_upstream(upstream)
    category = classify_upstream(normalized)

    if selected_mode == "chatgpt" and category == "platform":
        raise UpstreamSelectionError(
            "ChatGPT-backed Codex authentication is incompatible with the Platform API upstream"
        )
    if selected_mode == "api-key" and category == "chatgpt-codex":
        raise UpstreamSelectionError(
            "Platform API-key authentication is incompatible with the ChatGPT Codex upstream"
        )

    return UpstreamSelection(
        auth_mode=selected_mode,
        category=category,
        url=normalized,
        explicit_override=explicit_override,
        detection_source=source,
    )


def detect_codex_auth_mode(
    codex_home: str | Path | None = None,
) -> tuple[Optional[str], str]:
    home = Path(codex_home) if codex_home is not None else Path(
        os.environ.get("CODEX_HOME") or Path.home() / ".codex"
    )
    auth_path = home / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = None
    except (OSError, ValueError) as exc:
        raise UpstreamSelectionError(
            f"cannot determine Codex authentication mode from {auth_path}: invalid auth metadata"
        ) from exc

    if isinstance(payload, dict):
        mode = payload.get("auth_mode")
        if mode == "api":
            return "api-key", "Codex auth metadata"
        if mode == "chatgpt":
            return "chatgpt", "Codex auth metadata"
        raise UpstreamSelectionError(
            "Codex auth metadata has an unsupported or missing authentication mode"
        )
    if os.environ.get("OPENAI_API_KEY"):
        return "api-key", "OPENAI_API_KEY environment presence"
    return None, "no Codex authentication metadata found"


def classify_upstream(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if host == "api.openai.com" and path == "/v1":
        return "platform"
    if host == "chatgpt.com" and path == "/backend-api/codex":
        return "chatgpt-codex"
    return "custom"


def _normalize_upstream(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpstreamSelectionError("upstream override must be a non-empty HTTP(S) URL")
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UpstreamSelectionError("upstream override must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise UpstreamSelectionError("upstream override must not contain credentials")
    if parsed.query or parsed.fragment:
        raise UpstreamSelectionError("upstream override must not contain a query or fragment")
    return normalized
