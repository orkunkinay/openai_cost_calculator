"""Byte-safe accounting observer for the Anthropic Messages streaming protocol.

Anthropic streams usage across multiple Server-Sent Events:

* ``message_start`` carries input and cache token counts and an initial output
  token count.
* Each ``message_delta`` carries a *cumulative* ``output_tokens`` total, so the
  final value must be taken (never summed).
* ``message_stop`` terminates a successful response; ``error`` events terminate
  a failed one.

This observer consumes a copy of the raw response bytes without altering them.
It tolerates arbitrary chunk boundaries (including UTF-8 code points and JSON
fields split across chunks), CRLF or LF delimiters, multi-line ``data:`` fields,
comments, keep-alives, ping events, unknown future events, duplicate
``message_start`` events, and missing terminal events.
"""

from __future__ import annotations

import codecs
import json
from typing import Any, Optional

from openai_cost_calculator.anthropic.usage import AnthropicUsage, AnthropicUsageError


class _SSEEventReader:
    """Incremental Server-Sent Events splitter tolerant of chunk boundaries."""

    def __init__(self) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending_cr = False

    def feed(self, chunk: bytes) -> list[str]:
        if not chunk:
            return []
        self._append(self._decoder.decode(chunk))
        return self._drain()

    def close(self) -> list[str]:
        self._append(self._decoder.decode(b"", final=True), final=True)
        events = self._drain()
        if self._buffer.strip():
            events.append(self._buffer)
        self._buffer = ""
        return events

    def _append(self, text: str, *, final: bool = False) -> None:
        if self._pending_cr:
            text = "\r" + text
            self._pending_cr = False
        if text.endswith("\r") and not final:
            text = text[:-1]
            self._pending_cr = True
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")

    def _drain(self) -> list[str]:
        events: list[str] = []
        while True:
            index = self._buffer.find("\n\n")
            if index == -1:
                break
            events.append(self._buffer[:index])
            self._buffer = self._buffer[index + 2 :]
        return events


class AnthropicStreamAccountant:
    """Accumulates Anthropic streaming usage from raw response bytes."""

    def __init__(self, *, default_model: Optional[str] = None) -> None:
        self._reader = _SSEEventReader()
        self._model = default_model if isinstance(default_model, str) else None
        self._input: Optional[int] = None
        self._cache_read: Optional[int] = None
        self._cache_creation: Optional[int] = None
        self._write_5m: Optional[int] = None
        self._write_1h: Optional[int] = None
        self._output = 0
        self._saw_output = False
        self.saw_message_start = False
        self.saw_message_stop = False
        self.malformed = False
        self.error_code: Optional[str] = None
        self.error_message: Optional[str] = None
        self.last_usage_dict: Optional[dict] = None

    def feed(self, chunk: bytes) -> None:
        for event in self._reader.feed(chunk):
            self._handle_event(event)

    def close(self) -> None:
        for event in self._reader.close():
            self._handle_event(event)

    @property
    def model(self) -> Optional[str]:
        return self._model

    @property
    def usage(self) -> Optional[AnthropicUsage]:
        """Assembled usage, or ``None`` if no accounting-relevant usage was seen."""
        if self._input is None and not self._saw_output and self._cache_read is None:
            return None
        return AnthropicUsage(
            input_tokens=self._input or 0,
            cache_read_input_tokens=self._cache_read or 0,
            cache_creation_input_tokens=self._cache_creation or 0,
            cache_creation_5m_input_tokens=self._write_5m or 0,
            cache_creation_1h_input_tokens=self._write_1h or 0,
            output_tokens=self._output,
        )

    def _handle_event(self, event: str) -> None:
        data_lines: list[str] = []
        for line in event.split("\n"):
            if line.startswith(":"):
                continue  # SSE comment / keep-alive
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        if not raw or raw == "[DONE]":
            return
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            self.malformed = True
            return
        if not isinstance(payload, dict):
            return
        self._dispatch(payload)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "message_start":
            self.saw_message_start = True
            message = payload.get("message")
            if isinstance(message, dict):
                model = message.get("model")
                if isinstance(model, str) and model:
                    self._model = model
                self._absorb_usage(message.get("usage"))
        elif event_type == "message_delta":
            self._absorb_usage(payload.get("usage"))
        elif event_type == "message_stop":
            self.saw_message_stop = True
        elif event_type == "error":
            error = payload.get("error")
            if isinstance(error, dict):
                self.error_code = _safe_str(error.get("type")) or "error"
                self.error_message = _safe_str(error.get("message"))
            else:
                self.error_code = "error"

    def _absorb_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        self.last_usage_dict = usage
        self._input = _merge_int(self._input, usage.get("input_tokens"))
        self._cache_read = _merge_int(
            self._cache_read, usage.get("cache_read_input_tokens")
        )
        self._cache_creation = _merge_int(
            self._cache_creation, usage.get("cache_creation_input_tokens")
        )
        breakdown = usage.get("cache_creation")
        if isinstance(breakdown, dict):
            self._write_5m = _merge_int(
                self._write_5m, breakdown.get("ephemeral_5m_input_tokens")
            )
            self._write_1h = _merge_int(
                self._write_1h, breakdown.get("ephemeral_1h_input_tokens")
            )
        output = usage.get("output_tokens")
        if output is not None:
            try:
                self._output = _valid_int(output)
                self._saw_output = True
            except AnthropicUsageError:
                self.malformed = True


def _merge_int(current: Optional[int], value: Any) -> Optional[int]:
    if value is None:
        return current
    try:
        return _valid_int(value)
    except AnthropicUsageError:
        return current


def _valid_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnthropicUsageError("token count must be a non-negative integer")
    if value < 0:
        raise AnthropicUsageError("token count must be non-negative")
    return value


def _safe_str(value: Any, limit: int = 200) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value)
    return cleaned[:limit]


__all__ = ["AnthropicStreamAccountant"]
