from decimal import Decimal

import pytest

from openai_cost_calculator import CostTracker
from openai_cost_calculator.pricing import (
    add_pricing_entry,
    clear_local_pricing,
    set_offline_mode,
)
from openai_cost_calculator.tracker import CallRecord


class _Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Resource:
    def __init__(self, handler):
        self.handler = handler

    def create(self, **kwargs):
        return self.handler(**kwargs)


class _FakeClient:
    def __init__(self, chat_handler=None, responses_handler=None):
        self.chat = _Struct(completions=_Resource(chat_handler or (lambda **_: None)))
        self.responses = _Resource(responses_handler or (lambda **_: None))


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


def _chat_response(prompt_t, completion_t, cached_t=0, model="gpt-test-2025-01-01"):
    usage = _Struct(
        prompt_tokens=prompt_t,
        completion_tokens=completion_t,
        prompt_tokens_details=_Struct(cached_tokens=cached_t),
    )
    return _Struct(model=model, usage=usage)


def _responses_response(input_t, output_t, cached_t=0, model="gpt-test-2025-01-01"):
    usage = _Struct(
        input_tokens=input_t,
        output_tokens=output_t,
        input_tokens_details=_Struct(cached_tokens=cached_t),
    )
    return _Struct(model=model, usage=usage)


def test_turn_aggregates_multiple_record_calls():
    tracker = CostTracker()

    with tracker.turn("Implement feature") as turn:
        first = tracker.record(_chat_response(1_000, 2_000, 100))
        tracker.record(_responses_response(2_000, 1_000, 500))

    assert isinstance(first, CallRecord)
    assert turn.num_calls == 2
    assert turn.prompt_tokens == 3_000
    assert turn.completion_tokens == 3_000
    assert turn.cached_tokens == 600
    assert turn.total_cost == Decimal("0.00870")
    assert turn.cost_by_model == {"gpt-test-2025-01-01": Decimal("0.00870")}
    assert turn.as_dict()["total_cost"] == "0.00870000"
    assert turn.as_dict(stringify=False)["total_cost"] == Decimal("0.00870")


def test_session_total_accumulates_across_multiple_turns():
    tracker = CostTracker()

    with tracker.turn("First"):
        tracker.record(_chat_response(1_000, 0, 0))

    with tracker.turn("Second"):
        tracker.record(_chat_response(0, 2_000, 0))

    assert len(tracker.turns) == 2
    assert tracker.session_total == Decimal("0.005")


def test_wrap_auto_records_non_streaming_calls_and_returns_original_response():
    response = _chat_response(1_000, 2_000, 100)
    client = _FakeClient(chat_handler=lambda **kwargs: response)
    tracker = CostTracker()

    wrapped = tracker.wrap(client)
    with tracker.turn("Chat") as turn:
        returned = wrapped.chat.completions.create(model="gpt-test")

    assert returned is response
    assert turn.num_calls == 1
    assert tracker.session_total == Decimal("0.00495")


def test_wrap_auto_records_responses_create():
    response = _responses_response(1_000, 2_000, 100)
    client = _FakeClient(responses_handler=lambda **kwargs: response)
    tracker = CostTracker()

    tracker.wrap(client)
    with tracker.turn("Responses") as turn:
        returned = client.responses.create(model="gpt-test")

    assert returned is response
    assert turn.prompt_tokens == 1_000
    assert turn.total_cost == Decimal("0.00495")


def test_streaming_call_passes_chunks_through_and_records_last_usage_chunk():
    chunks = [
        _Struct(model="gpt-test-2025-01-01", delta="start"),
        _chat_response(1_000, 0, 0),
        _chat_response(2_000, 3_000, 500),
    ]
    client = _FakeClient(chat_handler=lambda **kwargs: iter(chunks))
    tracker = CostTracker()

    tracker.wrap(client)
    with tracker.turn("Stream") as turn:
        stream = client.chat.completions.create(stream=True)
        returned_chunks = list(stream)

    assert returned_chunks == chunks
    assert turn.num_calls == 1
    assert turn.prompt_tokens == 2_000
    assert turn.completion_tokens == 3_000
    assert turn.cached_tokens == 500
    assert turn.total_cost == Decimal("0.00775")


def test_nested_turns_attach_to_innermost_and_restore_outer():
    tracker = CostTracker()

    with tracker.turn("Outer") as outer:
        tracker.record(_chat_response(1_000, 0, 0))
        with tracker.turn("Inner") as inner:
            tracker.record(_chat_response(0, 1_000, 0))
        tracker.record(_chat_response(2_000, 0, 0))

    assert outer.num_calls == 2
    assert outer.total_cost == Decimal("0.003")
    assert inner.num_calls == 1
    assert inner.total_cost == Decimal("0.002")
    assert tracker.turns == [inner, outer]
    assert tracker.session_total == Decimal("0.005")


def test_costing_error_in_wrapped_call_does_not_propagate_and_triggers_on_error():
    errors = []
    response = _chat_response(10, 10, 0, model="unknown-model-2099-01-01")
    client = _FakeClient(chat_handler=lambda **kwargs: response)
    tracker = CostTracker(on_error=errors.append)

    tracker.wrap(client)
    with tracker.turn("Unknown") as turn:
        returned = client.chat.completions.create(model="unknown-model")

    assert returned is response
    assert turn.num_calls == 0
    assert tracker.session_total == Decimal("0")
    assert len(errors) == 1


def test_double_wrap_does_not_double_count():
    response = _chat_response(1_000, 0, 0)
    client = _FakeClient(chat_handler=lambda **kwargs: response)
    tracker = CostTracker()

    tracker.wrap(client)
    tracker.wrap(client)
    with tracker.turn("Double wrap") as turn:
        client.chat.completions.create()

    assert turn.num_calls == 1
    assert tracker.session_total == Decimal("0.001")
