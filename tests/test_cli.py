from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_cost_calculator.cli import (
    _load_admin_token,
    _validate_proxy_exposure,
    main,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_status_exposes_latest_call_diagnostics_and_routing(monkeypatch, capsys):
    costs = {
        "sessions": {
            "s1": {
                "session_total": "0.01000000",
                "historical_total": "0.00400000",
                "process_total": "0.00600000",
                "latest_call": {
                    "model": "gpt-test",
                    "cost": {"total_cost": "0.00600000"},
                },
                "errors": [{"code": "missing_usage", "message": "no usage"}],
            }
        },
        "grand_total": "0.01000000",
        "persistence": {"enabled": False, "healthy": True},
    }
    health = {
        "ok": True,
        "routing": {
            "auth_mode": "api-key",
            "upstream_category": "platform",
            "explicit_override": False,
        },
        "persistence": {"enabled": False, "healthy": True},
    }

    monkeypatch.setenv("OCC_ADMIN_TOKEN", "s" * 32)

    def urlopen(request, timeout):
        assert request.headers["Authorization"] == f"Bearer {'s' * 32}"
        return _Response(health if request.full_url.endswith("/_occ/health") else costs)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert main(["status", "--session", "s1", "--diagnostics"]) == 0
    output = capsys.readouterr().out
    assert "s1: total $0.01000000" in output
    assert "latest gpt-test $0.00600000" in output
    assert "diagnostic missing_usage: no usage" in output
    assert "auth=api-key, upstream=platform" in output


def test_ledger_cli_inspects_and_resets_offline_state(tmp_path: Path, capsys):
    path = tmp_path / "ledger.json"
    assert main(["ledger", "inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"] == {}

    assert main(["ledger", "reset", str(path)]) == 2
    assert "without --yes" in capsys.readouterr().err
    assert main(["ledger", "reset", str(path), "--yes"]) == 0


def test_database_cli_inspects_and_resets_concurrent_state(tmp_path: Path, capsys):
    path = tmp_path / "accounting.sqlite3"
    assert main(["database", "inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"] == {}
    assert payload["persistence"]["backend"] == "sqlite"

    assert main(["database", "reset", str(path)]) == 2
    assert "without --yes" in capsys.readouterr().err
    assert main(["database", "reset", str(path), "--yes"]) == 0


def test_pricing_validate_cli_and_incompatible_proxy_config(capsys):
    assert main(["pricing", "validate"]) == 0
    assert "Pricing data valid" in capsys.readouterr().out

    assert main(
        [
            "proxy",
            "--auth-mode",
            "chatgpt",
            "--upstream",
            "https://api.openai.com/v1",
        ]
    ) == 2
    assert "incompatible" in capsys.readouterr().err


def test_remote_binding_requires_explicit_permission_and_admin_token():
    with pytest.raises(ValueError, match="--allow-remote"):
        _validate_proxy_exposure("0.0.0.0", False, None)
    with pytest.raises(ValueError, match="requires OCC_ADMIN_TOKEN"):
        _validate_proxy_exposure("::", True, None)

    _validate_proxy_exposure("0.0.0.0", True, "x" * 32)
    _validate_proxy_exposure("127.0.0.1", False, None)
    _validate_proxy_exposure("::1", False, None)


def test_admin_token_file_must_be_protected(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OCC_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("OCC_ADMIN_TOKEN_FILE", raising=False)
    token_file = tmp_path / "admin-token"
    token_file.write_text("t" * 32 + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    assert _load_admin_token(str(token_file)) == "t" * 32

    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        _load_admin_token(str(token_file))
