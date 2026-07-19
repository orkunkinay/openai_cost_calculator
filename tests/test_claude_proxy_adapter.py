from __future__ import annotations

import json
from pathlib import Path

from openai_cost_calculator.adapters import claude_proxy


def _status(**overrides):
    base = {
        "session": "sess-1",
        "session_total": "0.08370000",
        "turn": {"label": "turn-2", "state": "active", "total_cost": "0.01240000", "num_calls": 1},
        "turn_is_active": True,
        "latest_turn": {"label": "turn-2", "state": "active", "total_cost": "0.01240000"},
        "accounting": "complete",
        "pricing_semantics": "billed-estimate",
    }
    base.update(overrides)
    return base


def test_render_active_turn_billed():
    assert claude_proxy._render_status(_status()) == "💰 OCC Turn $0.0124 · Session $0.0837"


def test_render_api_equivalent_labeling():
    line = claude_proxy._render_status(_status(pricing_semantics="api-equivalent"))
    assert line == "💰 OCC Turn API-eq $0.0124 · Session API-eq $0.0837"


def test_render_last_turn_when_not_active():
    line = claude_proxy._render_status(
        _status(turn_is_active=False, turn={"label": "turn-1", "state": "completed", "total_cost": "0.01240000"})
    )
    assert line == "💰 OCC Last turn $0.0124 · Session $0.0837"


def test_render_zero_active_turn_is_legitimate_zero():
    line = claude_proxy._render_status(
        _status(turn={"label": "turn-3", "state": "active", "total_cost": "0.00000000"})
    )
    assert line == "💰 OCC Turn $0.0000 · Session $0.0837"


def test_render_partial_appends_inspect_diagnostics():
    line = claude_proxy._render_status(_status(accounting="partial"))
    assert line.endswith("· inspect diagnostics")


def test_render_unavailable():
    assert claude_proxy._render_status(_status(accounting="unavailable")) == claude_proxy.UNAVAILABLE


def test_statusline_offline_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(OSError("offline")),
    )
    assert claude_proxy.statusline_text({"session_id": "s1"}) == claude_proxy.UNAVAILABLE


def test_statusline_reads_session_and_renders(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return json.dumps(_status()).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    line = claude_proxy.statusline_text({"session_id": "sess-1"})
    assert "session=sess-1" in captured["url"]
    assert line == "💰 OCC Turn $0.0124 · Session $0.0837"


def test_hook_maps_events_to_turn_lifecycle(monkeypatch):
    posts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return b'{"ok":true,"turn":"turn-1"}'

    def fake_urlopen(request, timeout):
        posts.append((request.full_url, json.loads(request.data.decode("utf-8"))))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = claude_proxy.hook_output(
        {"hook_event_name": "UserPromptSubmit", "session_id": "sess-1", "prompt": "hello"}
    )
    assert result["action"] == "open"
    assert posts[-1][1]["event"] == "open"
    key = posts[-1][1]["idempotency_key"]

    # Same prompt yields the same idempotency key.
    claude_proxy.hook_output(
        {"hook_event_name": "UserPromptSubmit", "session_id": "sess-1", "prompt": "hello"}
    )
    assert posts[-1][1]["idempotency_key"] == key

    claude_proxy.hook_output({"hook_event_name": "Stop", "session_id": "sess-1"})
    assert posts[-1][1]["event"] == "complete"
    claude_proxy.hook_output({"hook_event_name": "SessionEnd", "session_id": "sess-1", "reason": "exit"})
    assert posts[-1][1]["event"] == "interrupt"


def test_hook_ignores_unrelated_events():
    result = claude_proxy.hook_output({"hook_event_name": "PreToolUse", "session_id": "s1"})
    assert result == {"handled": False, "event": "PreToolUse"}


def test_hook_records_bounded_diagnostic_on_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(OSError("Bearer sk-secret-value")),
    )
    claude_proxy.hook_output({"hook_event_name": "Stop", "session_id": "s1"})
    diagnostics = claude_proxy.hook_diagnostics()
    assert diagnostics[-1]["code"] == "hook_failed"
    assert "secret" not in diagnostics[-1]["message"]
    log = tmp_path / "occ-claude-hook-diagnostics.jsonl"
    assert log.stat().st_mode & 0o777 == 0o600
