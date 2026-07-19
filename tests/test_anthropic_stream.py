from __future__ import annotations

import json

from openai_cost_calculator.anthropic.stream import AnthropicStreamAccountant


def _event(event_type: str, data: dict, *, newline: str = "\n") -> str:
    body = json.dumps(data)
    return f"event: {event_type}{newline}data: {body}{newline}{newline}"


def _message_start(**usage) -> str:
    return _event(
        "message_start",
        {"type": "message_start", "message": {"model": "claude-opus-4-8", "usage": usage}},
    )


def _message_delta(output_tokens: int) -> str:
    return _event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}},
    )


def _feed_all(accountant: AnthropicStreamAccountant, text: str, *, chunk_size: int | None = None) -> None:
    data = text.encode("utf-8")
    if chunk_size is None:
        accountant.feed(data)
    else:
        for start in range(0, len(data), chunk_size):
            accountant.feed(data[start : start + chunk_size])
    accountant.close()


def test_basic_stream_assembles_input_and_final_output():
    stream = (
        _message_start(input_tokens=25, cache_read_input_tokens=5, cache_creation_input_tokens=0, output_tokens=1)
        + _message_delta(10)
        + _message_delta(42)
        + _event("message_stop", {"type": "message_stop"})
    )
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    usage = accountant.usage
    assert accountant.model == "claude-opus-4-8"
    assert usage is not None
    assert usage.input_tokens == 25
    assert usage.cache_read_input_tokens == 5
    assert usage.output_tokens == 42  # cumulative last value, never summed
    assert accountant.saw_message_stop is True


def test_output_is_not_summed_across_deltas():
    stream = _message_start(input_tokens=10, output_tokens=1) + _message_delta(5) + _message_delta(20)
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.usage.output_tokens == 20


def test_duplicate_message_start_takes_last():
    stream = (
        _message_start(input_tokens=10, output_tokens=1)
        + _message_start(input_tokens=99, output_tokens=1)
        + _message_delta(7)
    )
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.usage.input_tokens == 99


def test_byte_by_byte_fragmentation_is_stable():
    stream = _message_start(input_tokens=123, output_tokens=1) + _message_delta(456)
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream, chunk_size=1)
    assert accountant.usage.input_tokens == 123
    assert accountant.usage.output_tokens == 456


def test_utf8_split_across_chunks_does_not_break_parsing():
    # A multibyte character inside content is split; usage still parses.
    content_event = _event(
        "content_block_delta",
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "héllo—世界"}},
    )
    stream = _message_start(input_tokens=8, output_tokens=1) + content_event + _message_delta(3)
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream, chunk_size=1)
    assert accountant.usage.input_tokens == 8
    assert accountant.usage.output_tokens == 3


def test_multiple_events_in_one_chunk_and_crlf_delimiters():
    stream = (
        _message_start(input_tokens=4, output_tokens=1, newline="\r\n")
        + _message_delta(9)
    ).replace("\n", "\r\n")
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)  # single chunk, CRLF throughout
    assert accountant.usage.input_tokens == 4
    assert accountant.usage.output_tokens == 9


def test_multiline_data_field_is_joined():
    # Per the SSE spec, multiple data: lines are concatenated with newlines;
    # a pretty-printed JSON payload spans several data: lines.
    pretty = json.dumps({"type": "message_delta", "usage": {"output_tokens": 15}}, indent=2)
    event = "event: message_delta\n" + "".join(f"data: {line}\n" for line in pretty.split("\n")) + "\n"
    stream = _message_start(input_tokens=2, output_tokens=1) + event
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.usage.output_tokens == 15
    assert accountant.malformed is False


def test_comments_pings_and_unknown_events_are_ignored():
    stream = (
        ": keep-alive comment\n\n"
        + _event("ping", {"type": "ping"})
        + _message_start(input_tokens=6, output_tokens=1)
        + _event("some_future_event", {"type": "some_future_event", "data": {"x": 1}})
        + _message_delta(11)
    )
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.usage.input_tokens == 6
    assert accountant.usage.output_tokens == 11
    assert accountant.malformed is False


def test_missing_terminal_event_still_yields_usage():
    stream = _message_start(input_tokens=6, output_tokens=1) + _message_delta(11)
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.saw_message_stop is False
    assert accountant.usage.output_tokens == 11


def test_error_event_records_error_and_no_double_usage():
    stream = _message_start(input_tokens=6, output_tokens=1) + _event(
        "error", {"type": "error", "error": {"type": "overloaded_error", "message": "slow down"}}
    )
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.error_code == "overloaded_error"
    assert accountant.error_message == "slow down"


def test_malformed_accounting_json_sets_flag_without_crashing():
    stream = _message_start(input_tokens=6, output_tokens=1) + "event: message_delta\ndata: {not json}\n\n"
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.malformed is True
    assert accountant.usage.input_tokens == 6  # earlier usage preserved


def test_stream_without_usage_reports_none():
    stream = _event("ping", {"type": "ping"}) + _event(
        "content_block_delta", {"type": "content_block_delta", "delta": {}}
    )
    accountant = AnthropicStreamAccountant()
    _feed_all(accountant, stream)
    assert accountant.usage is None
