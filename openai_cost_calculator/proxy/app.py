from __future__ import annotations

import json
import codecs
import hashlib
from typing import Any, AsyncIterator, Optional

import httpx

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
    raise ImportError(
        "The proxy requires optional web dependencies. Install with "
        "`pip install openai-cost-calculator[proxy]`."
    ) from exc

from openai_cost_calculator.parser import extract_usage_from_payload
from openai_cost_calculator.proxy.registry import TrackerRegistry, default_registry


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
    upstream: str = "https://api.openai.com/v1",
    registry: Optional[TrackerRegistry] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> FastAPI:
    app = FastAPI()
    app.state.occ_upstream = upstream.rstrip("/")
    app.state.occ_registry = registry or default_registry
    app.state.occ_transport = transport

    @app.get("/_occ/costs")
    async def costs(session: Optional[str] = None) -> JSONResponse:
        return JSONResponse(app.state.occ_registry.summary(session))

    @app.post("/_occ/checkpoint")
    async def checkpoint(session: Optional[str] = None) -> JSONResponse:
        return JSONResponse(app.state.occ_registry.checkpoint(session))

    @app.post("/_occ/reset")
    async def reset() -> JSONResponse:
        app.state.occ_registry.reset()
        return JSONResponse({"ok": True})

    @app.get("/_occ/costs/stream")
    async def cost_stream() -> StreamingResponse:
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

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Response:
        body = await request.body()
        session_id = _bounded_identifier(request.headers.get("x-occ-session"), 128)
        turn_label = _bounded_identifier(request.headers.get("x-occ-turn"), 256)
        request_payload = _json_from_bytes(body)

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


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=None,
        transport=app.state.occ_transport,
    )


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


def _bounded_identifier(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    cleaned = "".join(character if character.isprintable() else "?" for character in value)
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: limit - 13]}#{digest}"


app = create_app()
