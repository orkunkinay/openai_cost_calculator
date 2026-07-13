from __future__ import annotations

import asyncio
from decimal import Decimal
import re
from threading import RLock
import time
from typing import Dict, Iterable, Optional

from openai_cost_calculator.tracker import CallRecord, CostTracker


_MAX_ERRORS_PER_SESSION = 100
_MAX_DIAGNOSTIC_LENGTH = 500
_SECRET_RE = re.compile(
    r"(?i)(?:bearer\s+)[^\s,;]+|(?:sk-[a-z0-9_-]{8,})"
)


class TrackerRegistry:
    def __init__(self, on_error=None) -> None:
        self._trackers: Dict[str, CostTracker] = {}
        self._checkpoint_cursors: Dict[str, int] = {}
        self._errors: Dict[str, list[dict]] = {}
        self._subscribers: list[asyncio.Queue] = []
        self._lock = RLock()
        self._on_error = on_error

    def get(self, session_id: Optional[str]) -> CostTracker:
        key = session_id or "default"
        with self._lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                tracker = CostTracker(
                    on_error=lambda exc: self.record_error(
                        key,
                        "cost_estimation_failed",
                        str(exc),
                    )
                )
                self._trackers[key] = tracker
            return tracker

    def record_call(
        self,
        session_id: Optional[str],
        model: str,
        usage: dict,
        *,
        turn_label: Optional[str] = None,
    ) -> Optional[CallRecord]:
        key = session_id or "default"
        tracker = self.get(key)
        with self._lock:
            record = tracker.record_call(
                model,
                usage,
                turn_label=turn_label,
            )
        if record is not None:
            self.notify()
        elif tracker.session_total == Decimal("0") and not tracker.turns:
            with self._lock:
                if self._trackers.get(key) is tracker:
                    del self._trackers[key]
        return record

    def record_error(self, session_id: Optional[str], code: str, message: str) -> None:
        key = session_id or "default"
        error = {
            "code": _diagnostic_text(code, limit=64),
            "message": _diagnostic_text(message),
            "timestamp": time.time(),
        }
        with self._lock:
            errors = self._errors.setdefault(key, [])
            errors.append(error)
            del errors[:-_MAX_ERRORS_PER_SESSION]
        if self._on_error is not None:
            self._on_error(RuntimeError(f"{code}: {message}"))
        self.notify()

    def reset(self) -> None:
        with self._lock:
            self._trackers.clear()
            self._checkpoint_cursors.clear()
            self._errors.clear()
        self.notify()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def notify(self) -> None:
        summary = self.summary()
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(summary)

    def summary(self, session_id: Optional[str] = None) -> dict:
        with self._lock:
            if session_id is None:
                trackers = dict(self._trackers)
                errors = {key: list(value) for key, value in self._errors.items()}
            else:
                key = session_id or "default"
                tracker = self._trackers.get(key)
                trackers = {key: tracker} if tracker is not None else {}
                errors = {key: list(self._errors.get(key, []))} if key in self._errors else {}

        sessions = {}
        grand_total = Decimal("0")
        for current_session in sorted(set(trackers) | set(errors)):
            tracker = trackers.get(current_session)
            session_total = tracker.session_total if tracker is not None else Decimal("0")
            grand_total += session_total
            sessions[current_session] = {
                "session_total": f"{session_total:.8f}",
                "turns": [turn.as_dict() for turn in tracker.turns] if tracker is not None else [],
                "errors": errors.get(current_session, []),
            }

        return {
            "sessions": sessions,
            "grand_total": f"{grand_total:.8f}",
        }

    def checkpoint(self, session_id: Optional[str]) -> dict:
        key = session_id or "default"
        with self._lock:
            tracker = self._trackers.get(key)
            records = _tracker_records(tracker) if tracker is not None else []
            cursor = min(self._checkpoint_cursors.get(key, 0), len(records))
            new_records = records[cursor:]
            self._checkpoint_cursors[key] = len(records)

        return _records_summary(key, new_records)


def _tracker_records(tracker: Optional[CostTracker]) -> list[CallRecord]:
    if tracker is None:
        return []
    records: list[CallRecord] = []
    for turn in tracker.turns:
        records.extend(turn.records)
    records.sort(key=lambda record: record.timestamp)
    return records


def _records_summary(session_id: str, records: Iterable[CallRecord]) -> dict:
    total_cost = Decimal("0")
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    models: dict[str, dict] = {}
    num_calls = 0

    for record in records:
        num_calls += 1
        total_cost += record.cost.total_cost
        prompt_tokens += record.prompt_tokens
        completion_tokens += record.completion_tokens
        cached_tokens += record.cached_tokens
        model = models.setdefault(
            record.model,
            {
                "num_calls": 0,
                "total_cost": Decimal("0"),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
            },
        )
        model["num_calls"] += 1
        model["total_cost"] += record.cost.total_cost
        model["prompt_tokens"] += record.prompt_tokens
        model["completion_tokens"] += record.completion_tokens
        model["cached_tokens"] += record.cached_tokens

    stringified_models = {
        model_name: {
            **{key: value for key, value in model.items() if key != "total_cost"},
            "total_cost": f"{model['total_cost']:.8f}",
        }
        for model_name, model in models.items()
    }

    return {
        "session": session_id,
        "num_calls": num_calls,
        "total_cost": f"{total_cost:.8f}",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "models": stringified_models,
        "cost_by_model": {
            model_name: model["total_cost"]
            for model_name, model in stringified_models.items()
        },
    }


default_registry = TrackerRegistry()


def _diagnostic_text(value: object, *, limit: int = _MAX_DIAGNOSTIC_LENGTH) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = "".join(character if character.isprintable() else "?" for character in text)
    text = _SECRET_RE.sub("[REDACTED]", text)
    return text[:limit]
