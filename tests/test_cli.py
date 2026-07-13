from __future__ import annotations

import json
from pathlib import Path

from openai_cost_calculator.cli import main


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

    def urlopen(request, timeout):
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
