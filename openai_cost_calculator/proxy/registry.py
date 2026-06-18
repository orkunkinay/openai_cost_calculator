from __future__ import annotations

import asyncio
from decimal import Decimal
from threading import RLock
from typing import Dict, Optional

from openai_cost_calculator.tracker import CallRecord, CostTracker


class TrackerRegistry:
    def __init__(self, on_error=None) -> None:
        self._trackers: Dict[str, CostTracker] = {}
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

    def summary(self) -> dict:
        with self._lock:
            trackers = dict(self._trackers)

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


default_registry = TrackerRegistry()
