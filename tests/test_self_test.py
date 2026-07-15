from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "self_test_codex_integration.py"
SPEC = importlib.util.spec_from_file_location("occ_self_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
self_test = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(self_test)


def test_self_test_has_toml_parser_on_all_supported_python_versions():
    assert self_test.tomllib.loads('model = "gpt-test"') == {"model": "gpt-test"}


def test_explicit_api_key_mode_uses_stdin_and_ignores_source_chatgpt_login(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    (source / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "token": "source-secret"}),
        encoding="utf-8",
    )
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-secret")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    method, mode = self_test._prepare_auth(
        isolated,
        Path("codex"),
        requested_mode="api-key",
    )

    assert mode == "api-key"
    assert method == "OPENAI_API_KEY imported into isolated Codex login"
    assert calls[0][0] == ["codex", "login", "--with-api-key"]
    assert calls[0][1]["input"] == "api-key-secret"
    assert "api-key-secret" not in calls[0][0]
    assert not (isolated / "auth.json").exists()


def test_explicit_chatgpt_mode_requires_and_copies_chatgpt_login(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    auth = source / "auth.json"
    auth.write_text(json.dumps({"auth_mode": "chatgpt"}), encoding="utf-8")
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    method, mode = self_test._prepare_auth(
        isolated,
        Path("codex"),
        requested_mode="chatgpt",
    )
    assert mode == "chatgpt"
    assert method == "isolated copy of existing Codex login"
    assert (isolated / "auth.json").stat().st_mode & 0o777 == 0o600


def test_explicit_api_key_mode_fails_cleanly_without_key(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(self_test.SelfTestError, match="requires OPENAI_API_KEY"):
        self_test._prepare_auth(
            isolated,
            Path("codex"),
            requested_mode="api-key",
        )
