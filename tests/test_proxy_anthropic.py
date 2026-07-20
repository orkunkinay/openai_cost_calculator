from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from asgi_client import ASGITestClient

from openai_cost_calculator.anthropic.usage import (
    price_anthropic_usage,
    usage_from_dict,
)
from openai_cost_calculator.proxy.app import create_app
from openai_cost_calculator.proxy.registry import TrackerRegistry


class AsyncChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def _app(handler, registry=None):
    registry = registry or TrackerRegistry()
    app = create_app(
        upstream="https://api.anthropic.com",
        transport=httpx.MockTransport(handler),
        registry=registry,
        protocol="anthropic-messages",
    )
    return ASGITestClient(app), registry


def _messages_body(usage: dict, model: str = "claude-opus-4-8") -> bytes:
    return json.dumps(
        {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "usage": usage}
    ).encode("utf-8")


def _expected_total(usage: dict, model: str = "claude-opus-4-8") -> str:
    cost = price_anthropic_usage(model, usage_from_dict(usage))
    return f"{cost.total_cost:.8f}"


def _headers(session: str = "sess-1", **extra: str) -> dict:
    return {"x-claude-code-session-id": session, **extra}


def test_non_streaming_message_records_cost_and_forwards_body_unchanged():
    usage = {"input_tokens": 1_000, "output_tokens": 500, "cache_read_input_tokens": 200}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers.get("x-claude-code-session-id") == "sess-1"
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    response = client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8", "stream": False}), headers=_headers())

    assert response.content == body
    status = registry.claude_status("sess-1")
    assert status["session_total"] == _expected_total(usage)
    assert status["turn"]["state"] == "active"
    assert status["turn"]["num_calls"] == 1
    assert status["accounting"] == "complete"


def test_streaming_message_assembles_usage_and_costs_once():
    chunks = [
        b'event: message_start\n',
        b'data: {"type":"message_start","message":{"model":"claude-opus-4-8",'
        b'"usage":{"input_tokens":1000,"cache_read_input_tokens":0,"output_tokens":1}}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":250}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":500}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=AsyncChunkStream(chunks)
        )

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    response = client.post(
        "/v1/messages",
        content=json.dumps({"model": "claude-opus-4-8", "stream": True}),
        headers=_headers(),
    )
    assert b"message_stop" in response.content  # bytes forwarded intact

    usage = {"input_tokens": 1_000, "output_tokens": 500}
    assert registry.claude_status("sess-1")["session_total"] == _expected_total(usage)


def test_count_tokens_and_model_discovery_are_not_costed():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("count_tokens"):
            return httpx.Response(200, json={"input_tokens": 42})
        return httpx.Response(200, json={"data": []})

    client, registry = _app(handler)
    client.post("/v1/messages/count_tokens", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    client.get("/v1/models", headers=_headers())
    assert registry.claude_status("sess-1")["session_total"] == "0.00000000"
    assert registry.summary()["sessions"] == {}


def test_head_root_probe_is_forwarded_without_accounting():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200)

    client, registry = _app(handler)
    response = client.request("HEAD", "/")
    assert response.status_code == 200
    assert seen == {"method": "HEAD", "path": "/"}
    assert registry.summary()["sessions"] == {}


def test_request_without_active_turn_uses_synthetic_turn_and_keeps_session_total():
    usage = {"input_tokens": 400, "output_tokens": 100}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    client, registry = _app(handler)
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())

    status = registry.claude_status("sess-1")
    assert status["session_total"] == _expected_total(usage)
    assert status["turn"]["label"] == "unattributed"
    codes = {e["code"] for e in status["errors"]}
    assert "turn_unattributed" in codes


def test_subagent_request_aggregates_into_same_turn_and_session():
    usage = {"input_tokens": 100, "output_tokens": 50}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    # Main agent then a subagent, same session, different agent ids.
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers(**{"x-claude-code-agent-id": "main"}))
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers(**{"x-claude-code-agent-id": "sub", "x-claude-code-parent-agent-id": "main"}))

    status = registry.claude_status("sess-1")
    single = Decimal(_expected_total(usage))
    assert Decimal(status["session_total"]) == single * 2
    assert status["turn"]["num_calls"] == 2


def test_upstream_error_is_forwarded_and_not_costed():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "error", "error": {"type": "overloaded_error"}})

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    response = client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    assert response.status_code == 429
    assert registry.claude_status("sess-1")["session_total"] == "0.00000000"


def test_missing_usage_on_success_records_diagnostic_not_zero_cost():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b'{"model":"claude-opus-4-8"}')

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    status = registry.claude_status("sess-1")
    assert {e["code"] for e in status["errors"]} == {"usage_missing"}
    assert status["accounting"] == "partial"


def test_turn_lifecycle_admin_endpoints_open_aggregate_and_finalize():
    usage = {"input_tokens": 100, "output_tokens": 50}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    client, registry = _app(handler)
    opened = client.post("/_occ/claude/turn", json={"session_id": "sess-1", "event": "open", "idempotency_key": "k1"}).json()
    assert opened["turn"] == "turn-1"
    # Duplicate open is idempotent.
    again = client.post("/_occ/claude/turn", json={"session_id": "sess-1", "event": "open", "idempotency_key": "k1"}).json()
    assert again["turn"] == "turn-1"

    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())

    finalized = client.post("/_occ/claude/turn", json={"session_id": "sess-1", "event": "complete"}).json()
    assert finalized["state"] == "completed"

    status = client.get("/_occ/claude/status?session=sess-1").json()
    single = Decimal(_expected_total(usage))
    assert Decimal(status["turn"]["total_cost"]) == single * 2
    assert Decimal(status["session_total"]) == single * 2
    assert status["turn"]["state"] == "completed"
    assert status["turn_is_active"] is False
    assert status["pricing_semantics"] == "api-equivalent"

    # A new empty turn does not reset the session total.
    client.post("/_occ/claude/turn", json={"session_id": "sess-1", "event": "open", "idempotency_key": "k2"})
    status2 = client.get("/_occ/claude/status?session=sess-1").json()
    assert status2["turn"]["total_cost"] == "0.00000000"
    assert Decimal(status2["session_total"]) == single * 2


def test_status_reads_do_not_mutate_and_checkpoint_is_idempotent():
    usage = {"input_tokens": 100, "output_tokens": 50}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    client, registry = _app(handler)
    registry.open_turn("sess-1", "k1")
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())

    first = client.get("/_occ/claude/status?session=sess-1").json()["session_total"]
    second = client.get("/_occ/claude/status?session=sess-1").json()["session_total"]
    assert first == second == _expected_total(usage)

    cp1 = client.post("/_occ/checkpoint?session=sess-1").json()
    cp2 = client.post("/_occ/checkpoint?session=sess-1").json()
    assert cp1["total_cost"] == _expected_total(usage)
    assert cp2["total_cost"] == "0.00000000"
    # Cumulative totals unchanged by checkpoint reads.
    assert client.get("/_occ/claude/status?session=sess-1").json()["session_total"] == _expected_total(usage)


def test_configurable_session_header(monkeypatch):
    usage = {"input_tokens": 100, "output_tokens": 50}
    body = _messages_body(usage)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    monkeypatch.setenv("OCC_CLAUDE_SESSION_HEADER", "x-my-session")
    client, registry = _app(handler)
    registry.open_turn("custom-1", "k1")
    client.post(
        "/v1/messages",
        content=json.dumps({"model": "claude-opus-4-8"}),
        headers={"x-my-session": "custom-1"},
    )
    assert registry.claude_status("custom-1")["session_total"] == _expected_total(usage)


def test_debug_header_capture_records_names_not_values(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=_messages_body({"input_tokens": 1, "output_tokens": 1}))

    monkeypatch.setenv("OCC_CLAUDE_DEBUG_HEADERS", "1")
    client, registry = _app(handler)
    registry.open_turn("sess-1", "k1")
    client.post(
        "/v1/messages",
        content=json.dumps({"model": "claude-opus-4-8"}),
        headers=_headers(**{"x-claude-code-agent-id": "main", "authorization": "Bearer sk-secret"}),
    )
    errors = registry.claude_status("sess-1")["errors"]
    captured = [e for e in errors if e["code"] == "claude_headers_seen"]
    assert len(captured) == 1
    message = captured[0]["message"]
    assert "x-claude-code-session-id" in message
    assert "x-claude-code-agent-id" in message
    assert "sk-secret" not in message  # values are never captured


def test_two_request_turn_with_cache_aggregates(monkeypatch):
    # A realistic turn: an initial request that writes a prompt cache, then a
    # tool-use continuation that reads it. Both aggregate into one turn.
    first_usage = {
        "input_tokens": 2_000,
        "output_tokens": 300,
        "cache_creation_input_tokens": 1_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1_000},
    }
    second_usage = {
        "input_tokens": 200,
        "output_tokens": 150,
        "cache_read_input_tokens": 1_000,
    }
    bodies = [_messages_body(first_usage), _messages_body(second_usage)]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=bodies.pop(0))

    client, registry = _app(handler)
    registry.open_turn("sess-1", "prompt-key")
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    client.post("/v1/messages", content=json.dumps({"model": "claude-opus-4-8"}), headers=_headers())
    registry.finalize_turn("sess-1", "completed")

    status = registry.claude_status("sess-1")
    total = Decimal(_expected_total(first_usage)) + Decimal(_expected_total(second_usage))
    assert Decimal(status["session_total"]) == total
    assert status["turn"]["num_calls"] == 2
    assert status["turn"]["state"] == "completed"
