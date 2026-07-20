from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .core import calculate_cost_typed
from .estimate import _find_rates, estimate_cost_typed
from .parser import extract_model_details, extract_usage
from .types import CostBreakdown


_WRAPPED_SENTINEL = "_openai_cost_calculator_wrapped"


@dataclass(frozen=True)
class CallRecord:
    """One costed OpenAI API call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost: CostBreakdown
    timestamp: float


class Turn:
    """Aggregated cost and token usage for one agent turn."""

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label
        self._records: List[CallRecord] = []

    @property
    def records(self) -> List[CallRecord]:
        return list(self._records)

    @property
    def num_calls(self) -> int:
        return len(self._records)

    @property
    def total_cost(self) -> Decimal:
        return sum((record.cost.total_cost for record in self._records), Decimal("0"))

    @property
    def prompt_tokens(self) -> int:
        return sum(record.prompt_tokens for record in self._records)

    @property
    def completion_tokens(self) -> int:
        return sum(record.completion_tokens for record in self._records)

    @property
    def cached_tokens(self) -> int:
        return sum(record.cached_tokens for record in self._records)

    @property
    def cost_by_model(self) -> Dict[str, Decimal]:
        costs: Dict[str, Decimal] = {}
        for record in self._records:
            costs[record.model] = costs.get(record.model, Decimal("0")) + record.cost.total_cost
        return costs

    def add(self, record: CallRecord) -> None:
        self._records.append(record)

    def as_dict(self, stringify: bool = True) -> dict:
        total_cost: Union[Decimal, str]
        cost_by_model: Union[Dict[str, Decimal], Dict[str, str]]
        if stringify:
            total_cost = f"{self.total_cost:.8f}"
            cost_by_model = {
                model: f"{cost:.8f}" for model, cost in self.cost_by_model.items()
            }
        else:
            total_cost = self.total_cost
            cost_by_model = self.cost_by_model

        return {
            "label": self.label,
            "num_calls": self.num_calls,
            "total_cost": total_cost,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_by_model": cost_by_model,
        }


def _record_source(response: Any) -> Any:
    if not hasattr(response, "__iter__") or hasattr(response, "model"):
        return response

    last_usage_chunk = None
    for chunk in response:
        if hasattr(chunk, "usage"):
            last_usage_chunk = chunk

    if last_usage_chunk is None:
        return response
    return last_usage_chunk


class _TrackedStream:
    def __init__(self, stream: Any, tracker: "CostTracker") -> None:
        self._stream = stream
        self._tracker = tracker
        self._iterator = iter(stream)
        self._last_usage_chunk: Optional[Any] = None

    def __iter__(self) -> "_TrackedStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
        except StopIteration:
            if self._last_usage_chunk is not None:
                self._tracker._record_safely(self._last_usage_chunk)
                self._last_usage_chunk = None
            raise

        if hasattr(chunk, "usage") and getattr(chunk, "usage") is not None:
            self._last_usage_chunk = chunk
        return chunk

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "_TrackedStream":
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            entered = enter()
            self._stream = entered
            self._iterator = iter(entered)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        exit_ = getattr(self._stream, "__exit__", None)
        if exit_ is not None:
            return exit_(exc_type, exc, tb)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class CostTracker:
    """Stateful cost tracker for multi-call agent turns."""

    def __init__(self, on_error: Optional[Callable[[Exception], None]] = None) -> None:
        self.session_total = Decimal("0")
        self.turns: List[Turn] = []
        self.on_error = on_error
        self._active_turn: Optional[Turn] = None

    @contextmanager
    def turn(self, label: Optional[str] = None) -> Iterator[Turn]:
        current = Turn(label)
        previous = self._active_turn
        self._active_turn = current
        try:
            yield current
        finally:
            self._active_turn = previous
            self.turns.append(current)

    def record(self, response: Any) -> CallRecord:
        source = _record_source(response)
        cost = estimate_cost_typed(source)
        usage = extract_usage(source)
        record = CallRecord(
            model=source.model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=usage["cached_tokens"],
            cost=cost,
            timestamp=time.time(),
        )
        if self._active_turn is not None:
            self._active_turn.add(record)
        self.session_total += record.cost.total_cost
        return record

    def record_call(
        self,
        model: str,
        usage: dict,
        *,
        turn_label: Optional[str] = None,
    ) -> Optional[CallRecord]:
        try:
            return self._record_call(model, usage, turn_label=turn_label)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(exc)
            return None

    def wrap(self, client: Any) -> Any:
        chat_completions = getattr(getattr(client, "chat", None), "completions", None)
        if chat_completions is not None:
            self._wrap_create(chat_completions)

        responses = getattr(client, "responses", None)
        if responses is not None:
            self._wrap_create(responses)

        return client

    def reset(self) -> None:
        self.turns.clear()
        self.session_total = Decimal("0")
        self._active_turn = None

    def restore_turn(self, records: List[CallRecord], *, label: Optional[str]) -> None:
        """Restore one previously costed turn without repricing its calls."""
        turn = Turn(label)
        for record in records:
            turn.add(record)
            self.session_total += record.cost.total_cost
        self.turns.append(turn)

    def ensure_turn(self, label: Optional[str]) -> Turn:
        """Return the turn with ``label``, creating an empty one if needed."""
        return self._find_or_create_turn(label)

    def add_costed_call(
        self,
        model: str,
        tokens: Dict[str, int],
        cost: CostBreakdown,
        *,
        turn_label: Optional[str] = None,
    ) -> CallRecord:
        """Record an already-priced call without repricing it.

        Used for protocols (such as Anthropic Messages) whose pricing is
        computed outside the OpenAI three-bucket estimator.
        """
        record = CallRecord(
            model=model,
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            cached_tokens=tokens["cached_tokens"],
            cost=cost,
            timestamp=time.time(),
        )
        turn = self._active_turn or self._find_or_create_turn(turn_label)
        turn.add(record)
        self.session_total += record.cost.total_cost
        return record

    def _find_or_create_turn(self, label: Optional[str]) -> Turn:
        for turn in self.turns:
            if turn.label == label:
                return turn
        turn = Turn(label)
        self.turns.append(turn)
        return turn

    def _record_call(
        self,
        model: str,
        usage: dict,
        *,
        turn_label: Optional[str] = None,
    ) -> CallRecord:
        details = extract_model_details(model)
        rates = _find_rates(
            details["model_name"],
            details["model_date"],
            usage["prompt_tokens"],
        )
        cost = calculate_cost_typed(usage, rates)
        record = CallRecord(
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=usage["cached_tokens"],
            cost=cost,
            timestamp=time.time(),
        )
        turn = self._active_turn or self._find_or_create_turn(turn_label)
        turn.add(record)
        self.session_total += record.cost.total_cost
        return record

    def _wrap_create(self, owner: Any) -> None:
        original = getattr(owner, "create", None)
        if original is None or getattr(original, _WRAPPED_SENTINEL, False):
            return

        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            response = original(*args, **kwargs)
            if kwargs.get("stream") is True:
                return _TrackedStream(response, self)

            self._record_safely(response)
            return response

        setattr(wrapped_create, _WRAPPED_SENTINEL, True)
        setattr(owner, "create", wrapped_create)

    def _record_safely(self, response: Any) -> Optional[CallRecord]:
        try:
            return self.record(response)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(exc)
            return None
