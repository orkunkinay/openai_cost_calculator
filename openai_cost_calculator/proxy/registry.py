from __future__ import annotations

import asyncio
from decimal import Decimal
from threading import RLock
from typing import Dict, Iterable, Optional

from openai_cost_calculator.tracker import CallRecord, CostTracker


class TrackerRegistry:
    def __init__(self, on_error=None) -> None:
        self._trackers: Dict[str, CostTracker] = {}
        self._checkpoint_cursors: Dict[str, int] = {}
        self._subscribers: list[asyncio.Queue] = []
        self._lock = RLock()
        self._on_error = on_error

    def get(self, session_id: Optional[str]) -> CostTracker:
        key = session_id or "default"
        with self._lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                tracker = CostTracker(on_error=self._on_error)
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

    def reset(self) -> None:
        with self._lock:
            self._trackers.clear()
            self._checkpoint_cursors.clear()
        self.notify()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
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
            queue.put_nowait(summary)

    def summary(self, session_id: Optional[str] = None) -> dict:
        with self._lock:
            if session_id is None:
                trackers = dict(self._trackers)
            else:
                key = session_id or "default"
                tracker = self._trackers.get(key)
                trackers = {key: tracker} if tracker is not None else {}

        sessions = {}
        grand_total = Decimal("0")
        for session_id, tracker in trackers.items():
            grand_total += tracker.session_total
            sessions[session_id] = {
                "session_total": f"{tracker.session_total:.8f}",
                "turns": [turn.as_dict() for turn in tracker.turns],
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
