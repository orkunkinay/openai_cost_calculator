from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_cost_calculator.adapters.claude_code import (
    statusline_text as claude_statusline_text,
    stop_hook_output,
)
from openai_cost_calculator.adapters.codex import (
    checkpoint_text,
    statusline_text as codex_statusline_text,
)
from openai_cost_calculator.adapters.install import (
    install_claude_code,
    install_codex,
    uninstall_claude_code,
    uninstall_codex,
)
from openai_cost_calculator.pricing import (
    add_pricing_entry,
    clear_local_pricing,
    set_offline_mode,
)
from openai_cost_calculator.proxy.app import create_app
from openai_cost_calculator.proxy.registry import TrackerRegistry


@pytest.fixture(autouse=True)
def pinned_pricing():
    clear_local_pricing()
    set_offline_mode(True)
    add_pricing_entry(
        "gpt-test",
        "2025-01-01",
        input_price=1.0,
        output_price=2.0,
        cached_input_price=0.5,
    )
    yield
    clear_local_pricing()
    set_offline_mode(False)


def test_claude_statusline_formats_tokens_and_handles_null_usage():
    payload = {
        "model": {"display_name": "Sonnet 4.6"},
        "cost": {"total_cost_usd": "0.01234"},
        "context_window": {
            "total_input_tokens": 28_000,
            "context_window_size": 200_000,
            "current_usage": {
                "input_tokens": 8_500,
                "output_tokens": 1_200,
                "cache_read_input_tokens": 2_000,
            },
        },
    }

    assert claude_statusline_text(payload) == (
        "💰 $0.0123 session · last 8.5k->1.2k tok "
        "(cache 2.0k) · Sonnet 4.6 · ctx 14%"
    )

    payload["context_window"]["current_usage"] = None
    assert "last -- tok" in claude_statusline_text(payload)
    assert claude_statusline_text({}).startswith("💰 $0.0000 session")


def test_claude_stop_hook_costs_latest_turn_and_dedupes(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "usage": {"input_tokens": 1_000, "output_tokens": 500},
                "cost_usd": "0.0041",
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    payload = {"session_id": "s1", "transcript_path": str(transcript)}

    first = stop_hook_output(payload, cache_dir=tmp_path / "cache")
    assert first == {"systemMessage": "💰 This turn cost $0.0041 (1.0k in / 500 out)"}
    second = stop_hook_output(payload, cache_dir=tmp_path / "cache")
    assert second == {}


def test_codex_adapters_render_and_silently_handle_network_errors(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        url = request.full_url
        if "/_occ/checkpoint" in url:
            return Response(
                {
                    "total_cost": "0.00320000",
                    "prompt_tokens": 1_200,
                    "completion_tokens": 500,
                    "models": {"gpt-5.5": {"total_cost": "0.00320000"}},
                }
            )
        return Response(
            {
                "sessions": {
                    "s1": {
                        "session_total": "0.01000000",
                        "turns": [{"total_cost": "0.00320000"}],
                    }
                },
                "grand_total": "0.01000000",
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert checkpoint_text({}, session="s1") == "💰 Turn $0.0032 · gpt-5.5 1.2k->500"
    assert codex_statusline_text(session="s1") == "💰 $0.0100 session · last $0.0032"

    def failing_urlopen(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    assert checkpoint_text({}, session="s1") is None
    assert codex_statusline_text(session="s1") == "💰 cost offline"


def test_proxy_checkpoint_advances_cursor_and_costs_remain_cumulative():
    calls = [
        {
            "model": "gpt-test-2025-01-01",
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 0},
        },
        {
            "model": "gpt-test-2025-01-01",
            "usage": {"prompt_tokens": 0, "completion_tokens": 1_000},
        },
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json=calls.pop(0))

    app = create_app(
        upstream="https://upstream.example/v1",
        transport=httpx.MockTransport(handler),
        registry=TrackerRegistry(),
    )
    client = TestClient(app)
    for _ in range(2):
        client.post("/v1/responses", json={}, headers={"x-occ-session": "s1"})

    first = client.post("/_occ/checkpoint?session=s1").json()
    assert first["total_cost"] == "0.00300000"
    assert first["num_calls"] == 2
    assert first["models"]["gpt-test-2025-01-01"]["completion_tokens"] == 1_000
    second = client.post("/_occ/checkpoint?session=s1").json()
    assert second["total_cost"] == "0.00000000"
    assert client.get("/_occ/costs?session=s1").json()["grand_total"] == "0.00300000"


def test_installers_are_idempotent_and_reversible(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    claude_settings = claude_dir / "settings.json"
    claude_settings.write_text('{"unrelated": true}\n', encoding="utf-8")

    install_claude_code()
    install_claude_code()
    settings = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert settings["unrelated"] is True
    assert settings["statusLine"]["command"] == "occ-cc-statusline"
    stop_hooks = settings["hooks"]["Stop"]
    assert json.dumps(stop_hooks).count("occ-cc-stop-hook") == 1
    uninstall_claude_code()
    settings = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert settings == {"unrelated": True}

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    codex_config = codex_dir / "config.toml"
    codex_config.write_text(
        'notify = ["existing-notifier"]\nmodel = "gpt-test"\n',
        encoding="utf-8",
    )
    install_codex("http://127.0.0.1:8100", "s1")
    install_codex("http://127.0.0.1:8100", "s1")
    text = codex_config.read_text(encoding="utf-8")
    assert text.count("openai-cost-calculator") == 2
    assert text.count("occ-codex-notify") == 1
    assert "occ-codex-statusline" in text
    active_notify_lines = [
        line for line in text.splitlines() if line.startswith('notify = ["')
    ]
    assert active_notify_lines == ['notify = ["occ-codex-notify"]']
    assert "# previous_notify = notify = [\"existing-notifier\"]" in text
    assert 'model = "gpt-test"' in text
    uninstall_codex()
    assert codex_config.read_text(encoding="utf-8").strip() == (
        'notify = ["existing-notifier"]\nmodel = "gpt-test"'
    )
