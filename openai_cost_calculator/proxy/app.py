from __future__ import annotations

import json
import codecs
from collections import deque
import hashlib
import hmac
import asyncio
from typing import Any, AsyncIterator, Optional

import httpx

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
    raise ImportError(
        "The proxy requires optional web dependencies. Install with "
        "`pip install openai-cost-calculator[proxy]`."
    ) from exc

from openai_cost_calculator.anthropic.stream import AnthropicStreamAccountant
from openai_cost_calculator.parser import extract_usage_from_payload
from openai_cost_calculator.proxy.anthropic_accounting import (
    UNATTRIBUTED_TURN,
    record_anthropic_response,
    usage_from_response_payload,
)
from openai_cost_calculator.proxy.ledger import LedgerError
from openai_cost_calculator.proxy.registry import TrackerRegistry, default_registry
from openai_cost_calculator.proxy.upstreams import (
    PLATFORM_UPSTREAM,
    UpstreamSelection,
    classify_upstream,
)


ANTHROPIC_PROTOCOL = "anthropic-messages"
OPENAI_PROTOCOL = "openai-responses"


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def create_app(
    *,
    upstream: str = PLATFORM_UPSTREAM,
    registry: Optional[TrackerRegistry] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    ledger_path: Optional[str] = None,
    database_path: Optional[str] = None,
    upstream_selection: Optional[UpstreamSelection] = None,
    admin_token: Optional[str] = None,
    websocket_connector=None,
    protocol: str = OPENAI_PROTOCOL,
    pricing_semantics: Optional[str] = None,
) -> FastAPI:
    if registry is not None and (ledger_path is not None or database_path is not None):
        raise ValueError("pass either registry or a persistence path, not both")
    if ledger_path is not None and database_path is not None:
        raise ValueError("pass either ledger_path or database_path, not both")
    app = FastAPI()
    if upstream_selection is not None:
        upstream = upstream_selection.url
    app.state.occ_upstream = upstream.rstrip("/")
    app.state.occ_upstream_selection = upstream_selection or UpstreamSelection(
        auth_mode="unspecified",
        category=classify_upstream(app.state.occ_upstream),
        url=app.state.occ_upstream,
        explicit_override=True,
        detection_source="application configuration",
    )
    app.state.occ_registry = (
        registry
        if registry is not None
        else TrackerRegistry(ledger_path=ledger_path, database_path=database_path)
        if ledger_path is not None or database_path is not None
        else default_registry
    )
    app.state.occ_transport = transport
    app.state.occ_admin_token = admin_token
    app.state.occ_websocket_connector = websocket_connector
    if protocol not in {OPENAI_PROTOCOL, ANTHROPIC_PROTOCOL}:
        raise ValueError(f"unsupported proxy protocol: {protocol!r}")
    app.state.occ_protocol = protocol
    app.state.occ_pricing_semantics = pricing_semantics or (
        "api-equivalent" if protocol == ANTHROPIC_PROTOCOL else "billed-estimate"
    )

    @app.get("/_occ/health")
    async def health(request: Request) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        persistence = app.state.occ_registry.persistence_status()
        status = 200 if persistence["healthy"] else 503
        selection = app.state.occ_upstream_selection
        return JSONResponse(
            {
                "ok": status == 200,
                "persistence": persistence,
                "routing": {
                    "auth_mode": selection.auth_mode,
                    "upstream_category": selection.category,
                    "explicit_override": selection.explicit_override,
                    "detection_source": selection.detection_source,
                },
            },
            status_code=status,
        )

    @app.get("/_occ/costs")
    async def costs(request: Request, session: Optional[str] = None) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        return JSONResponse(app.state.occ_registry.summary(session))

    @app.post("/_occ/checkpoint")
    async def checkpoint(request: Request, session: Optional[str] = None) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        try:
            payload = app.state.occ_registry.checkpoint(session)
        except LedgerError as exc:
            return JSONResponse(
                {"error": {"code": "ledger_write_failed", "message": str(exc)}},
                status_code=503,
            )
        return JSONResponse(payload)

    @app.post("/_occ/reset")
    async def reset(request: Request) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        try:
            app.state.occ_registry.reset()
        except LedgerError as exc:
            return JSONResponse(
                {"error": {"code": "ledger_write_failed", "message": str(exc)}},
                status_code=503,
            )
        return JSONResponse({"ok": True})

    @app.get("/_occ/claude/status")
    async def claude_status(request: Request, session: Optional[str] = None) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        session_id = _bounded_identifier(session, 128) or "unscoped-claude"
        payload = app.state.occ_registry.claude_status(session_id)
        payload["pricing_semantics"] = app.state.occ_pricing_semantics
        return JSONResponse(payload)

    @app.post("/_occ/claude/turn")
    async def claude_turn(request: Request) -> JSONResponse:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        try:
            body = _json_from_bytes(await request.body())
        except Exception:
            body = {}
        session_id = _bounded_identifier(body.get("session_id"), 128) or "unscoped-claude"
        event = body.get("event")
        idem_key = _bounded_identifier(body.get("idempotency_key"), 128) or "default"
        registry = app.state.occ_registry
        if event == "open":
            label = registry.open_turn(session_id, idem_key)
            return JSONResponse({"ok": True, "turn": label, "event": event})
        if event in {"complete", "fail", "interrupt"}:
            state = {"complete": "completed", "fail": "failed", "interrupt": "interrupted"}[event]
            label = registry.finalize_turn(session_id, state)
            return JSONResponse({"ok": True, "turn": label, "state": state, "event": event})
        return JSONResponse(
            {"error": {"code": "turn_lifecycle_conflict", "message": "unknown turn event"}},
            status_code=400,
        )

    @app.get("/_occ/costs/stream")
    async def cost_stream(request: Request) -> Response:
        unauthorized = _admin_auth_error(app, request)
        if unauthorized is not None:
            return unauthorized
        queue = app.state.occ_registry.subscribe()

        async def events() -> AsyncIterator[bytes]:
            try:
                yield _sse_data(app.state.occ_registry.summary())
                while True:
                    summary = await queue.get()
                    yield _sse_data(summary)
            finally:
                app.state.occ_registry.unsubscribe(queue)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root_probe(request: Request) -> Response:
        # Claude Code issues a connectivity probe against the root; forward it
        # faithfully without recording it as a model request.
        if app.state.occ_protocol != ANTHROPIC_PROTOCOL:
            return Response(status_code=404)
        return await _forward_passthrough(
            app, f"{app.state.occ_upstream}/", request, await request.body()
        )

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "HEAD"])
    async def forward(path: str, request: Request) -> Response:
        body = await request.body()
        request_payload = _json_from_bytes(body)

        if app.state.occ_protocol == ANTHROPIC_PROTOCOL:
            return await _forward_anthropic(app, path, request, body, request_payload)

        session_id = _bounded_identifier(request.headers.get("x-occ-session"), 128)
        turn_label = _bounded_identifier(request.headers.get("x-occ-turn"), 256)

        if request.method == "POST" and request_payload.get("stream") is True:
            return await _forward_streaming(
                app,
                path,
                request,
                body,
                session_id=session_id,
                turn_label=turn_label,
                request_payload=request_payload,
            )

        return await _forward_buffered(
            app,
            path,
            request,
            body,
            session_id=session_id,
            turn_label=turn_label,
        )

    @app.websocket("/v1/{path:path}")
    async def forward_websocket(websocket: WebSocket, path: str) -> None:
        await _forward_websocket(app, path, websocket)

    return app


async def _forward_buffered(
    app: FastAPI,
    path: str,
    request: Request,
    body: bytes,
    *,
    session_id: Optional[str],
    turn_label: Optional[str],
) -> Response:
    async with _client(app) as client:
        upstream_response = await client.request(
            request.method,
            _upstream_url(app, path, request),
            headers=_forward_headers(request),
            content=body,
        )

    _record_from_payload(
        app.state.occ_registry,
        session_id,
        turn_label,
        upstream_response.status_code,
        upstream_response.headers.get("content-type"),
        upstream_response.content,
    )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers),
    )


async def _forward_streaming(
    app: FastAPI,
    path: str,
    request: Request,
    body: bytes,
    *,
    session_id: Optional[str],
    turn_label: Optional[str],
    request_payload: dict,
) -> StreamingResponse:
    client = _client(app)
    upstream_request = client.build_request(
        request.method,
        _upstream_url(app, path, request),
        headers=_forward_headers(request),
        content=body,
    )
    upstream_response = await client.send(upstream_request, stream=True)
    parser = _SSEUsageParser(default_model=request_payload.get("model"))

    async def chunks() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in upstream_response.aiter_raw():
                parser.feed(chunk)
                yield chunk
            completed = True
        except Exception as exc:
            app.state.occ_registry.record_error(
                session_id,
                "stream_interrupted",
                f"upstream stream ended unexpectedly: {type(exc).__name__}",
            )
            raise
        finally:
            parser.close()
            if completed and upstream_response.status_code < 400:
                usage_payload = parser.usage_payload
                if usage_payload is not None:
                    _record_json_payload(
                        app.state.occ_registry,
                        session_id,
                        turn_label,
                        usage_payload,
                    )
                else:
                    app.state.occ_registry.record_error(
                        session_id,
                        "missing_usage",
                        "successful streaming response ended without final usage data",
                    )
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers, streaming=True),
    )


async def _forward_passthrough(
    app: FastAPI,
    url: str,
    request: Request,
    body: bytes,
) -> Response:
    """Forward a request verbatim without any accounting (probes, discovery)."""
    async with _client(app) as client:
        upstream_response = await client.request(
            request.method,
            url,
            headers=_forward_headers(request),
            content=body,
        )
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers),
    )


def _anthropic_upstream_url(app: FastAPI, path: str, request: Request) -> str:
    # Claude Code targets ``<base>/v1/<path>``; the route captures only the
    # portion after ``/v1/``, so re-add the ``/v1`` segment when forwarding.
    url = f"{app.state.occ_upstream}/v1/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


def _anthropic_session(request: Request) -> str:
    session = _bounded_identifier(request.headers.get("x-claude-code-session-id"), 128)
    if session:
        return session
    session = _bounded_identifier(request.headers.get("x-occ-session"), 128)
    return session or "unscoped-claude"


async def _forward_anthropic(
    app: FastAPI,
    path: str,
    request: Request,
    body: bytes,
    request_payload: dict,
) -> Response:
    normalized = path.strip("/")
    accountable = request.method == "POST" and normalized == "messages"
    if not accountable:
        # count_tokens, model discovery, and probes are forwarded without cost.
        return await _forward_passthrough(
            app, _anthropic_upstream_url(app, path, request), request, body
        )

    registry = app.state.occ_registry
    session_id = _anthropic_session(request)
    turn_label = registry.active_turn_label(session_id) or UNATTRIBUTED_TURN

    if request_payload.get("stream") is True:
        return await _forward_anthropic_streaming(
            app, path, request, body, session_id, turn_label, request_payload
        )
    return await _forward_anthropic_buffered(
        app, path, request, body, session_id, turn_label, request_payload
    )


async def _forward_anthropic_buffered(
    app: FastAPI,
    path: str,
    request: Request,
    body: bytes,
    session_id: str,
    turn_label: str,
    request_payload: dict,
) -> Response:
    async with _client(app) as client:
        upstream_response = await client.request(
            request.method,
            _anthropic_upstream_url(app, path, request),
            headers=_forward_headers(request),
            content=body,
        )

    if upstream_response.status_code < 400:
        content_type = upstream_response.headers.get("content-type") or ""
        if "json" in content_type.lower():
            payload = _json_from_bytes(upstream_response.content)
            record_anthropic_response(
                app.state.occ_registry,
                session_id,
                turn_label,
                usage=usage_from_response_payload(payload),
                raw_usage=payload.get("usage") if isinstance(payload, dict) else None,
                response_model=payload.get("model") if isinstance(payload, dict) else None,
                request_model=request_payload.get("model"),
            )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers),
    )


async def _forward_anthropic_streaming(
    app: FastAPI,
    path: str,
    request: Request,
    body: bytes,
    session_id: str,
    turn_label: str,
    request_payload: dict,
) -> StreamingResponse:
    client = _client(app)
    upstream_request = client.build_request(
        request.method,
        _anthropic_upstream_url(app, path, request),
        headers=_forward_headers(request),
        content=body,
    )
    upstream_response = await client.send(upstream_request, stream=True)
    accountant = AnthropicStreamAccountant(default_model=request_payload.get("model"))

    async def chunks() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in upstream_response.aiter_raw():
                accountant.feed(chunk)
                yield chunk
            completed = True
        except Exception as exc:
            app.state.occ_registry.record_error(
                session_id,
                "stream_interrupted",
                f"Anthropic stream ended unexpectedly: {type(exc).__name__}",
            )
            raise
        finally:
            accountant.close()
            if completed and upstream_response.status_code < 400:
                if accountant.error_code is not None:
                    app.state.occ_registry.record_error(
                        session_id,
                        "stream_malformed_event",
                        f"Anthropic stream reported an error event: {accountant.error_code}",
                    )
                else:
                    record_anthropic_response(
                        app.state.occ_registry,
                        session_id,
                        turn_label,
                        usage=accountant.usage,
                        raw_usage=accountant.last_usage_dict,
                        response_model=accountant.model,
                        request_model=request_payload.get("model"),
                    )
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers, streaming=True),
    )


async def _forward_websocket(app: FastAPI, path: str, websocket: WebSocket) -> None:
    session_id = _bounded_identifier(websocket.headers.get("x-occ-session"), 128)
    turn_label = _bounded_identifier(websocket.headers.get("x-occ-turn"), 256)
    observer = _WebSocketAccountingObserver(
        app.state.occ_registry,
        session_id=session_id,
        turn_label=turn_label,
    )
    connector = app.state.occ_websocket_connector
    if connector is None:
        try:
            from websockets.asyncio.client import connect as connector
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            app.state.occ_registry.record_error(
                session_id,
                "websocket_dependency_missing",
                "WebSocket forwarding requires the proxy optional dependencies",
            )
            await websocket.close(code=1011)
            return

    subprotocols = _websocket_subprotocols(websocket)
    try:
        async with connector(
            _websocket_upstream_url(app, path, websocket),
            additional_headers=_websocket_forward_headers(websocket),
            subprotocols=subprotocols or None,
            origin=websocket.headers.get("origin"),
            max_size=None,
        ) as upstream:
            selected_subprotocol = getattr(upstream, "subprotocol", None)
            await websocket.accept(subprotocol=selected_subprotocol)
            client_task = asyncio.create_task(
                _relay_client_websocket(websocket, upstream, observer)
            )
            upstream_task = asyncio.create_task(
                _relay_upstream_websocket(websocket, upstream, observer)
            )
            done, pending = await asyncio.wait(
                {client_task, upstream_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            result = next(iter(done)).result()
            if result["source"] == "upstream":
                observer.upstream_closed()
                await websocket.close(
                    code=_websocket_close_code(result.get("code")),
                    reason=_websocket_close_reason(result.get("reason")),
                )
            else:
                await upstream.close(
                    code=_websocket_close_code(result.get("code")),
                    reason=_websocket_close_reason(result.get("reason")),
                )
    except Exception as exc:
        app.state.occ_registry.record_error(
            session_id,
            "websocket_upstream_failed",
            f"WebSocket upstream connection failed: {type(exc).__name__}",
        )
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


async def _relay_client_websocket(
    websocket: WebSocket,
    upstream: Any,
    observer: "_WebSocketAccountingObserver",
) -> dict[str, Any]:
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return {
                    "source": "client",
                    "code": message.get("code", 1000),
                    "reason": message.get("reason", ""),
                }
            if message.get("text") is not None:
                payload: str | bytes = message["text"]
            elif message.get("bytes") is not None:
                payload = message["bytes"]
            else:
                continue
            observer.observe_client(payload)
            await upstream.send(payload)
    except WebSocketDisconnect as exc:
        return {"source": "client", "code": exc.code, "reason": ""}


async def _relay_upstream_websocket(
    websocket: WebSocket,
    upstream: Any,
    observer: "_WebSocketAccountingObserver",
) -> dict[str, Any]:
    try:
        async for payload in upstream:
            observer.observe_upstream(payload)
            if isinstance(payload, bytes):
                await websocket.send_bytes(payload)
            else:
                await websocket.send_text(payload)
        return {"source": "upstream", "code": 1000, "reason": ""}
    except Exception as exc:
        return {
            "source": "upstream",
            "code": getattr(exc, "code", 1011),
            "reason": getattr(exc, "reason", ""),
        }


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=None,
        transport=app.state.occ_transport,
    )


def _websocket_upstream_url(app: FastAPI, path: str, websocket: WebSocket) -> str:
    base = app.state.occ_upstream
    if base.startswith("https://"):
        base = f"wss://{base.removeprefix('https://')}"
    elif base.startswith("http://"):
        base = f"ws://{base.removeprefix('http://')}"
    url = f"{base}/{path}"
    if websocket.url.query:
        url = f"{url}?{websocket.url.query}"
    return url


def _websocket_forward_headers(websocket: WebSocket) -> list[tuple[str, str]]:
    excluded = set(_HOP_BY_HOP_HEADERS)
    excluded.update(
        {
            "host",
            "content-length",
            "origin",
            "x-occ-session",
            "x-occ-turn",
        }
    )
    for name in websocket.headers:
        if name.lower().startswith("sec-websocket-"):
            excluded.add(name.lower())
    for value in websocket.headers.getlist("connection"):
        excluded.update(token.strip().lower() for token in value.split(","))
    return [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in websocket.headers.raw
        if name.decode("latin-1").lower() not in excluded
    ]


def _websocket_subprotocols(websocket: WebSocket) -> list[str]:
    protocols = []
    for value in websocket.headers.getlist("sec-websocket-protocol"):
        protocols.extend(item.strip() for item in value.split(",") if item.strip())
    return protocols


def _upstream_url(app: FastAPI, path: str, request: Request) -> str:
    url = f"{app.state.occ_upstream}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


def _forward_headers(request: Request) -> list[tuple[bytes, bytes]]:
    excluded = {name.encode("ascii") for name in _HOP_BY_HOP_HEADERS}
    excluded.update({b"host", b"content-length", b"x-occ-session", b"x-occ-turn"})
    for value in request.headers.getlist("connection"):
        excluded.update(token.strip().lower().encode("ascii") for token in value.split(","))
    return [
        (name, value)
        for name, value in request.headers.raw
        if name.lower() not in excluded
    ]


def _response_headers(
    headers: httpx.Headers,
    *,
    streaming: bool = False,
) -> dict[str, str]:
    excluded = set(_HOP_BY_HOP_HEADERS)
    for value in headers.get_list("connection"):
        excluded.update(token.strip().lower() for token in value.split(","))
    if streaming:
        excluded.add("content-length")
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in excluded
    }


def _json_from_bytes(body: bytes) -> dict:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_from_payload(
    registry: TrackerRegistry,
    session_id: Optional[str],
    turn_label: Optional[str],
    status_code: int,
    content_type: Optional[str],
    content: bytes,
) -> None:
    if status_code >= 400:
        return
    if content_type is not None and "json" not in content_type.lower():
        return
    payload = _json_from_bytes(content)
    _record_json_payload(registry, session_id, turn_label, payload)


def _record_json_payload(
    registry: TrackerRegistry,
    session_id: Optional[str],
    turn_label: Optional[str],
    payload: dict,
) -> None:
    usage = extract_usage_from_payload(payload)
    model = payload.get("model")
    if usage is None:
        registry.record_error(
            session_id,
            "missing_usage",
            "successful JSON response did not contain a usage object",
        )
        return
    if not isinstance(model, str):
        registry.record_error(
            session_id,
            "missing_model",
            "successful JSON response with usage did not identify a model",
        )
        return
    registry.record_call(session_id, model, usage, turn_label=turn_label)


def _sse_data(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


class _SSEUsageParser:
    def __init__(self, *, default_model: Any = None) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending_cr = False
        self._default_model = default_model
        self.usage_payload: Optional[dict] = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._append_text(self._decoder.decode(chunk))
        while True:
            event = self._pop_event()
            if event is None:
                return
            self._handle_event(event)

    def close(self) -> None:
        self._append_text(self._decoder.decode(b"", final=True), final=True)
        if self._buffer.strip():
            self._handle_event(self._buffer)
        self._buffer = ""

    def _append_text(self, text: str, *, final: bool = False) -> None:
        if self._pending_cr:
            text = "\r" + text
            self._pending_cr = False
        if text.endswith("\r") and not final:
            text = text[:-1]
            self._pending_cr = True
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")

    def _pop_event(self) -> Optional[str]:
        index = self._buffer.find("\n\n")
        if index == -1:
            return None
        event = self._buffer[:index]
        self._buffer = self._buffer[index + 2 :]
        return event

    def _handle_event(self, event: str) -> None:
        data_lines = []
        for line in event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            return
        payload = _json_from_bytes(raw_data.encode("utf-8"))
        if not payload:
            return
        usage_payload = payload
        if extract_usage_from_payload(usage_payload) is None:
            nested_response = payload.get("response")
            if isinstance(nested_response, dict):
                usage_payload = nested_response
        if extract_usage_from_payload(usage_payload) is None:
            return
        if "model" not in usage_payload and isinstance(self._default_model, str):
            usage_payload = {**usage_payload, "model": self._default_model}
        self.usage_payload = usage_payload


class _WebSocketAccountingObserver:
    def __init__(
        self,
        registry: TrackerRegistry,
        *,
        session_id: Optional[str],
        turn_label: Optional[str],
    ) -> None:
        self._registry = registry
        self._session_id = session_id
        self._turn_label = turn_label
        self._pending_models: deque[Optional[str]] = deque()
        self._terminal_ids: deque[str] = deque()
        self._terminal_id_set: set[str] = set()

    def observe_client(self, message: str | bytes) -> None:
        payload = _websocket_json(message)
        if payload is None or payload.get("type") != "response.create":
            return
        response = payload.get("response")
        model = response.get("model") if isinstance(response, dict) else None
        if not isinstance(model, str):
            model = payload.get("model")
        self._pending_models.append(model if isinstance(model, str) else None)

    def observe_upstream(self, message: str | bytes) -> None:
        payload = _websocket_json(message)
        if payload is None:
            return
        event_type = payload.get("type")
        if event_type not in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            return
        response = payload.get("response")
        response_payload = response if isinstance(response, dict) else payload
        terminal_id = response_payload.get("id")
        if not isinstance(terminal_id, str):
            terminal_id = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        if terminal_id in self._terminal_id_set:
            return
        self._remember_terminal(terminal_id)
        model = self._pending_models.popleft() if self._pending_models else None

        if event_type == "response.completed":
            if "model" not in response_payload and model is not None:
                response_payload = {**response_payload, "model": model}
            _record_json_payload(
                self._registry,
                self._session_id,
                self._turn_label,
                response_payload,
            )
            return
        self._registry.record_error(
            self._session_id,
            "websocket_response_failed",
            f"WebSocket response ended with terminal event {event_type}",
        )

    def upstream_closed(self) -> None:
        if self._pending_models:
            self._registry.record_error(
                self._session_id,
                "websocket_missing_terminal",
                "WebSocket upstream closed with unfinished response requests",
            )
            self._pending_models.clear()

    def _remember_terminal(self, terminal_id: str) -> None:
        self._terminal_ids.append(terminal_id)
        self._terminal_id_set.add(terminal_id)
        if len(self._terminal_ids) > 1_024:
            removed = self._terminal_ids.popleft()
            self._terminal_id_set.discard(removed)


def _websocket_json(message: str | bytes) -> Optional[dict]:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _websocket_close_code(value: Any) -> int:
    return value if isinstance(value, int) and 1000 <= value <= 4999 else 1011


def _websocket_close_reason(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    encoded = value.encode("utf-8")[:123]
    return encoded.decode("utf-8", errors="ignore")


def _bounded_identifier(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    cleaned = "".join(character if character.isprintable() else "?" for character in value)
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: limit - 13]}#{digest}"


def _admin_auth_error(app: FastAPI, request: Request) -> Optional[JSONResponse]:
    expected = app.state.occ_admin_token
    if expected is None:
        return None
    authorization = request.headers.get("authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected)
    ):
        return JSONResponse(
            {
                "error": {
                    "code": "admin_auth_required",
                    "message": "valid administrative authentication is required",
                }
            },
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


app = create_app()
