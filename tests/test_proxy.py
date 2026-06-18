from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_cost_calculator.parser import extract_usage_from_payload
from openai_cost_calculator.pricing import (
    add_pricing_entry,
    clear_local_pricing,
    set_offline_mode,
)
from openai_cost_calculator.proxy.app import create_app
from openai_cost_calculator.proxy.registry import TrackerRegistry


class AsyncChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


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


def _client(handler):
    transport = httpx.MockTransport(handler)
    app = create_app(
        upstream="https://upstream.example/v1",
        transport=transport,
        registry=TrackerRegistry(),
    )
    return TestClient(app)


def test_extract_usage_from_payload_supports_chat_and_responses_shapes():
    chat = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 25},
        }
    }
    responses = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 25},
        }
    }

    expected = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cached_tokens": 25,
    }
    assert extract_usage_from_payload(chat) == expected
    assert extract_usage_from_payload(responses) == expected
    assert extract_usage_from_payload({"id": "no-usage"}) is None


def test_non_streaming_post_returns_body_unchanged_and_records_session_cost():
    upstream_body = json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "gpt-test-2025-01-01",
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 2_000,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://upstream.example/v1/chat/completions?beta=1"
        assert request.headers["authorization"] == "Bearer test-token"
        assert await request.aread() == b'{"model":"gpt-test-2025-01-01"}'
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=upstream_body,
        )

    client = _client(handler)
    response = client.post(
        "/v1/chat/completions?beta=1",
        content=b'{"model":"gpt-test-2025-01-01"}',
        headers={
            "authorization": "Bearer test-token",
            "x-occ-session": "agent-a",
        },
    )

    assert response.content == upstream_body
    costs = client.get("/_occ/costs").json()
    assert costs["grand_total"] == "0.00495000"
    assert costs["sessions"]["agent-a"]["session_total"] == "0.00495000"


def test_streaming_sse_passes_body_through_and_costs_final_usage_event():
    chunks = [
        b'data: {"id":"chunk-1","model":"gpt-test-2025-01-01"}\n\n',
        (
            b'data: {"id":"chunk-2","model":"gpt-test-2025-01-01",'
            b'"usage":{"prompt_tokens":2000,"completion_tokens":3000,'
            b'"prompt_tokens_details":{"cached_tokens":500}}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads((await request.aread()).decode("utf-8"))["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncChunkStream(chunks),
        )

    client = _client(handler)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-test-2025-01-01",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"x-occ-session": "stream-session"},
    )

    assert response.content == b"".join(chunks)
    costs = client.get("/_occ/costs").json()
    assert costs["sessions"]["stream-session"]["session_total"] == "0.00775000"


def test_costs_endpoint_reflects_turn_grouping():
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
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=calls.pop(0),
        )

    client = _client(handler)
    for _ in range(2):
        client.post(
            "/v1/responses",
            json={"model": "gpt-test-2025-01-01"},
            headers={"x-occ-session": "agent-b", "x-occ-turn": "turn-1"},
        )

    costs = client.get("/_occ/costs").json()
    turn = costs["sessions"]["agent-b"]["turns"][0]
    assert turn["label"] == "turn-1"
    assert turn["num_calls"] == 2
    assert turn["prompt_tokens"] == 1_000
    assert turn["completion_tokens"] == 1_000
    assert costs["sessions"]["agent-b"]["session_total"] == "0.00300000"


def test_upstream_error_and_missing_usage_record_nothing():
    responses = [
        httpx.Response(
            500,
            headers={"content-type": "application/json"},
            json={"model": "gpt-test-2025-01-01", "usage": {"prompt_tokens": 100}},
        ),
        httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"model": "gpt-test-2025-01-01"},
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    client = _client(handler)
    assert client.post("/v1/chat/completions").status_code == 500
    assert client.post("/v1/chat/completions").status_code == 200
    assert client.get("/_occ/costs").json() == {
        "sessions": {},
        "grand_total": "0.00000000",
    }


def test_costing_failure_is_swallowed_and_response_is_unchanged():
    upstream_body = b'{"model":"unknown-2099-01-01","usage":{"prompt_tokens":1}}'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=upstream_body,
        )

    client = _client(handler)
    response = client.post("/v1/chat/completions", headers={"x-occ-session": "bad"})

    assert response.content == upstream_body
    assert client.get("/_occ/costs").json() == {
        "sessions": {},
        "grand_total": "0.00000000",
    }
