from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_cost_calculator.proxy.upstreams import (
    CHATGPT_CODEX_UPSTREAM,
    PLATFORM_UPSTREAM,
    UpstreamSelectionError,
    resolve_upstream,
)


def _auth_home(tmp_path: Path, mode: str) -> Path:
    home = tmp_path / mode
    home.mkdir()
    (home / "auth.json").write_text(
        json.dumps({"auth_mode": mode, "tokens": {"access_token": "not-read"}}),
        encoding="utf-8",
    )
    return home


def test_auto_selection_distinguishes_chatgpt_and_api_key_auth(tmp_path: Path):
    chatgpt = resolve_upstream(codex_home=_auth_home(tmp_path, "chatgpt"))
    assert chatgpt.auth_mode == "chatgpt"
    assert chatgpt.category == "chatgpt-codex"
    assert chatgpt.url == CHATGPT_CODEX_UPSTREAM
    assert chatgpt.explicit_override is False

    api_key = resolve_upstream(codex_home=_auth_home(tmp_path, "api"))
    assert api_key.auth_mode == "api-key"
    assert api_key.category == "platform"
    assert api_key.url == PLATFORM_UPSTREAM


def test_explicit_custom_upstream_is_observable_and_credentials_are_rejected():
    selection = resolve_upstream(
        auth_mode="api-key",
        upstream="https://gateway.example/openai/v1/",
    )
    assert selection.category == "custom"
    assert selection.explicit_override is True
    assert selection.url == "https://gateway.example/openai/v1"

    with pytest.raises(UpstreamSelectionError, match="must not contain credentials"):
        resolve_upstream(
            auth_mode="api-key",
            upstream="https://user:secret@gateway.example/v1",
        )


@pytest.mark.parametrize(
    "auth_mode, upstream, expected",
    [
        ("chatgpt", PLATFORM_UPSTREAM, "ChatGPT-backed"),
        ("api-key", CHATGPT_CODEX_UPSTREAM, "API-key"),
    ],
)
def test_known_incompatible_authentication_domains_are_rejected(
    auth_mode,
    upstream,
    expected,
):
    with pytest.raises(UpstreamSelectionError, match=expected):
        resolve_upstream(auth_mode=auth_mode, upstream=upstream)


def test_auto_without_auth_metadata_keeps_platform_compatible_default(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selection = resolve_upstream(codex_home=tmp_path)
    assert selection.auth_mode == "unspecified"
    assert selection.category == "platform"
    assert selection.url == PLATFORM_UPSTREAM
