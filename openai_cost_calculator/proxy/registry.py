from __future__ import annotations

import asyncio
from decimal import Decimal
import re
from threading import RLock
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from openai_cost_calculator.tracker import CallRecord, CostTracker
from openai_cost_calculator.types import CostBreakdown
from openai_cost_calculator.proxy.ledger import DurableLedger, LedgerError, SQLiteLedger


_MAX_ERRORS_PER_SESSION = 100
_MAX_DIAGNOSTIC_LENGTH = 500
_MAX_SESSIONS = 1_024
_MAX_CALLS_PER_SESSION = 50_000
_TURN_STATES = {"active", "completed", "failed", "interrupted", "unknown"}
# Diagnostic codes that mark a session's monetary total as possibly incomplete.
_ACCOUNTING_INCOMPLETE_CODES = {
    "missing_usage",
    "usage_missing",
    "usage_malformed",
    "missing_model",
    "model_missing",
    "model_unknown",
    "pricing_unavailable",
    "pricing_ambiguous",
    "server_tool_unpriced",
    "stream_incomplete",
    "stream_interrupted",
    "cost_estimation_failed",
}
_SECRET_RE = re.compile(
    r"(?i)(?:bearer\s+)[^\s,;]+|(?:sk-[a-z0-9_-]{8,})"
)


class TrackerRegistry:
    def __init__(
        self,
        on_error=None,
        *,
        ledger_path: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        if ledger_path is not None and database_path is not None:
            raise ValueError("pass either ledger_path or database_path, not both")
        self._trackers: Dict[str, CostTracker] = {}
        self._checkpoint_cursors: Dict[str, int] = {}
        self._errors: Dict[str, list[dict]] = {}
        self._initial_totals: Dict[str, Decimal] = {}
        # Claude turn lifecycle state (in-memory; scoped to this proxy process).
        self._active_turns: Dict[str, str] = {}
        self._turn_states: Dict[str, Dict[str, str]] = {}
        self._turn_keys: Dict[str, str] = {}
        self._turn_order: Dict[str, list[str]] = {}
        self._turn_counters: Dict[str, int] = {}
        self._subscribers: list[asyncio.Queue] = []
        self._lock = RLock()
        self._on_error = on_error
        self._ledger = (
            SQLiteLedger(database_path)
            if database_path is not None
            else DurableLedger(ledger_path) if ledger_path is not None else None
        )
        self._ledger_healthy = True
        self._ledger_error: Optional[str] = None
        self._sqlite_generation: Optional[int] = None
        if isinstance(self._ledger, SQLiteLedger):
            self._initial_totals = self._ledger.totals()
            self._sqlite_generation = self._ledger.generation()
        elif self._ledger is not None:
            self._restore(self._ledger.load())

    def get(self, session_id: Optional[str]) -> CostTracker:
        key = session_id or "default"
        with self._lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                if len(set(self._trackers) | set(self._errors)) >= _MAX_SESSIONS:
                    raise RegistryCapacityError("accounting session capacity reached")
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
        try:
            tracker = self.get(key)
        except RegistryCapacityError:
            self.record_error(
                "__registry__",
                "accounting_capacity_reached",
                "additional session was not recorded because the registry session limit was reached",
            )
            return None
        with self._lock:
            if (
                not isinstance(self._ledger, SQLiteLedger)
                and len(_tracker_records(tracker)) >= _MAX_CALLS_PER_SESSION
            ):
                self.record_error(
                    key,
                    "accounting_capacity_reached",
                    "call was not recorded because the per-session call limit was reached",
                )
                return None
            record = tracker.record_call(
                model,
                usage,
                turn_label=turn_label,
            )
            if record is not None:
                self._persist_call_locked(key, turn_label, record)
        if record is not None:
            self.notify()
        elif tracker.session_total == Decimal("0") and not tracker.turns:
            with self._lock:
                if self._trackers.get(key) is tracker:
                    del self._trackers[key]
        return record

    def record_costed_call(
        self,
        session_id: Optional[str],
        model: str,
        tokens: dict,
        cost: CostBreakdown,
        *,
        turn_label: Optional[str] = None,
    ) -> Optional[CallRecord]:
        """Record an already-priced call (for example Anthropic Messages).

        The cost breakdown is stored verbatim; it is not recomputed with the
        OpenAI estimator.  Attribution, aggregation, persistence, and capacity
        limits are identical to :meth:`record_call`.
        """
        key = session_id or "default"
        try:
            tracker = self.get(key)
        except RegistryCapacityError:
            self.record_error(
                "__registry__",
                "accounting_capacity_reached",
                "additional session was not recorded because the registry session limit was reached",
            )
            return None
        with self._lock:
            if (
                not isinstance(self._ledger, SQLiteLedger)
                and len(_tracker_records(tracker)) >= _MAX_CALLS_PER_SESSION
            ):
                self.record_error(
                    key,
                    "accounting_capacity_reached",
                    "call was not recorded because the per-session call limit was reached",
                )
                return None
            try:
                record = tracker.add_costed_call(
                    model, tokens, cost, turn_label=turn_label
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.record_error(key, "ledger_write_failed", str(exc))
                return None
            self._persist_call_locked(key, turn_label, record)
        self.notify()
        return record

    def open_turn(self, session_id: Optional[str], idem_key: str) -> Optional[str]:
        """Open (or re-confirm) the active turn for a Claude session.

        Idempotent: a repeated call with the same key while the turn is still
        active returns the same label without creating a second turn.  A key
        that matches an already-finalized turn opens a genuinely new turn.
        """
        key = session_id or "default"
        with self._lock:
            active = self._active_turns.get(key)
            if active is not None and self._turn_keys.get(active) == idem_key:
                return active
            try:
                tracker = self.get(key)
            except RegistryCapacityError:
                self.record_error(
                    "__registry__",
                    "accounting_capacity_reached",
                    "turn was not opened because the registry session limit was reached",
                )
                return None
            counter = self._turn_counters.get(key, 0) + 1
            self._turn_counters[key] = counter
            label = f"turn-{counter}"
            tracker.ensure_turn(label)
            self._active_turns[key] = label
            self._turn_keys[label] = idem_key
            self._turn_states.setdefault(key, {})[label] = "active"
            self._turn_order.setdefault(key, []).append(label)
            # The SQLite ledger derives turns from recorded calls, so an empty
            # opened turn needs no snapshot there; the JSON ledger persists it.
            if not isinstance(self._ledger, SQLiteLedger):
                self._persist_locked()
        self.notify()
        return label

    def finalize_turn(
        self,
        session_id: Optional[str],
        state: str,
        *,
        idem_key: Optional[str] = None,
    ) -> Optional[str]:
        """Finalize the active turn for a session with ``state``.

        Idempotent: finalizing when no turn is active is a no-op and never
        mutates cost.
        """
        key = session_id or "default"
        normalized = state if state in _TURN_STATES else "unknown"
        with self._lock:
            active = self._active_turns.get(key)
            if active is None:
                return None
            if idem_key is not None and self._turn_keys.get(active) != idem_key:
                return None
            self._turn_states.setdefault(key, {})[active] = normalized
            del self._active_turns[key]
        self.notify()
        return active

    def active_turn_label(self, session_id: Optional[str]) -> Optional[str]:
        key = session_id or "default"
        with self._lock:
            return self._active_turns.get(key)

    def claude_status(self, session_id: Optional[str]) -> dict:
        """A Claude-oriented, non-mutating view of turn and session totals."""
        key = session_id or "default"
        summary = self.summary(key)
        session = summary.get("sessions", {}).get(key, {})
        turn_dicts = {
            turn.get("label"): turn
            for turn in session.get("turns", [])
            if isinstance(turn, dict)
        }
        with self._lock:
            active_label = self._active_turns.get(key)
            order = list(self._turn_order.get(key, []))
            states = dict(self._turn_states.get(key, {}))
        # Include record-derived turns (for example synthetic "unattributed"
        # turns) that were never opened through the lifecycle hooks.
        for label in turn_dicts:
            if label is not None and label not in order:
                order.append(label)

        def _view(label: Optional[str]) -> Optional[dict]:
            if label is None:
                return None
            turn = turn_dicts.get(label, {})
            return {
                "label": label,
                "state": states.get(label, "unknown"),
                "total_cost": turn.get("total_cost", "0.00000000"),
                "num_calls": turn.get("num_calls", 0),
            }

        latest_label = order[-1] if order else None
        errors = session.get("errors", [])
        accounting = "complete"
        if any(
            isinstance(error, dict) and error.get("code") in _ACCOUNTING_INCOMPLETE_CODES
            for error in errors
        ):
            accounting = "partial"
        persistence = summary.get("persistence", {})
        if isinstance(persistence, dict) and persistence.get("enabled") and not persistence.get("healthy"):
            accounting = "unavailable"
        session_requests = sum(
            int(turn.get("num_calls", 0)) for turn in turn_dicts.values()
        )
        return {
            "session": key,
            "session_total": session.get("session_total", "0.00000000"),
            "historical_total": session.get("historical_total", "0.00000000"),
            "process_total": session.get("process_total", "0.00000000"),
            "active_turn": active_label,
            "turn": _view(active_label) if active_label is not None else _view(latest_label),
            "turn_is_active": active_label is not None,
            "latest_turn": _view(latest_label),
            "num_turns": len(order),
            "session_requests": session_requests,
            "accounting": accounting,
            "errors": errors,
            "persistence": persistence,
        }

    def record_error(self, session_id: Optional[str], code: str, message: str) -> None:
        key = session_id or "default"
        with self._lock:
            if (
                key not in self._trackers
                and key not in self._errors
                and len(set(self._trackers) | set(self._errors)) >= _MAX_SESSIONS
            ):
                key = "__registry__"
        error = {
            "code": _diagnostic_text(code, limit=64),
            "message": _diagnostic_text(message),
            "timestamp": time.time(),
        }
        with self._lock:
            errors = self._errors.setdefault(key, [])
            errors.append(error)
            del errors[:-_MAX_ERRORS_PER_SESSION]
            if isinstance(self._ledger, SQLiteLedger):
                try:
                    self._ledger.append_error(
                        key,
                        error,
                        max_sessions=_MAX_SESSIONS,
                        max_errors=_MAX_ERRORS_PER_SESSION,
                    )
                except LedgerError as exc:
                    self._mark_ledger_error(exc)
                else:
                    self._mark_ledger_healthy()
            else:
                self._persist_locked()
        if self._on_error is not None:
            self._on_error(RuntimeError(f"{code}: {message}"))
        self.notify()

    def reset(self) -> None:
        with self._lock:
            if self._ledger is not None:
                try:
                    if isinstance(self._ledger, SQLiteLedger):
                        self._ledger.reset()
                    else:
                        self._ledger.save({"schema_version": 1, "sessions": {}})
                except LedgerError as exc:
                    self._mark_ledger_error(exc)
                    raise
            self._trackers.clear()
            self._checkpoint_cursors.clear()
            self._errors.clear()
            self._initial_totals.clear()
            self._active_turns.clear()
            self._turn_states.clear()
            self._turn_keys.clear()
            self._turn_order.clear()
            self._turn_counters.clear()
            if isinstance(self._ledger, SQLiteLedger):
                self._sqlite_generation = self._ledger.generation()
            self._ledger_healthy = True
            self._ledger_error = None
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
        with self._lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        summary = self.summary()
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(summary)

    def summary(self, session_id: Optional[str] = None) -> dict:
        with self._lock:
            if isinstance(self._ledger, SQLiteLedger) and self._ledger_healthy:
                generation = self._ledger.generation()
                if generation != self._sqlite_generation:
                    self._initial_totals = self._ledger.totals()
                    self._sqlite_generation = generation
                summary = self._ledger.summary(session_id, self._initial_totals)
                summary["persistence"] = self.persistence_status()
                return summary
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
                "historical_total": f"{self._initial_totals.get(current_session, Decimal('0')):.8f}",
                "process_total": f"{(session_total - self._initial_totals.get(current_session, Decimal('0'))):.8f}",
                "turns": [turn.as_dict() for turn in tracker.turns] if tracker is not None else [],
                "errors": errors.get(current_session, []),
                "latest_call": _latest_call(tracker),
            }

        return {
            "sessions": sessions,
            "grand_total": f"{grand_total:.8f}",
            "persistence": self.persistence_status(),
        }

    def checkpoint(self, session_id: Optional[str]) -> dict:
        key = session_id or "default"
        with self._lock:
            if isinstance(self._ledger, SQLiteLedger):
                try:
                    payload = self._ledger.checkpoint(key)
                except LedgerError as exc:
                    self._mark_ledger_error(exc)
                    raise
                self._mark_ledger_healthy()
                return payload
            tracker = self._trackers.get(key)
            records = _tracker_records(tracker) if tracker is not None else []
            cursor = min(self._checkpoint_cursors.get(key, 0), len(records))
            new_records = records[cursor:]
            self._checkpoint_cursors[key] = len(records)
            if not self._persist_locked():
                self._checkpoint_cursors[key] = cursor
                raise LedgerError(
                    "checkpoint was not consumed because the durable ledger write failed"
                )

        return _records_summary(key, new_records)

    def persistence_status(self) -> dict:
        backend = (
            "sqlite"
            if isinstance(self._ledger, SQLiteLedger)
            else "json" if isinstance(self._ledger, DurableLedger) else None
        )
        return {
            "enabled": self._ledger is not None,
            "path": str(self._ledger.path) if self._ledger is not None else None,
            "healthy": self._ledger_healthy,
            "error": self._ledger_error,
            "backend": backend,
            "concurrency": (
                "multi-proxy"
                if isinstance(self._ledger, SQLiteLedger)
                else "single-proxy" if self._ledger is not None else None
            ),
        }

    def close(self) -> None:
        if self._ledger is not None:
            self._ledger.close()

    def _persist_locked(self) -> bool:
        if self._ledger is None:
            return True
        try:
            self._ledger.save(self._snapshot_locked())
        except LedgerError as exc:
            self._mark_ledger_error(exc)
            return False
        self._mark_ledger_healthy()
        return True

    def _persist_call_locked(
        self,
        session: str,
        turn_label: Optional[str],
        record: CallRecord,
    ) -> bool:
        if not isinstance(self._ledger, SQLiteLedger):
            return self._persist_locked()
        try:
            self._ledger.append_call(
                session,
                turn_label,
                _record_payload(record),
                max_sessions=_MAX_SESSIONS,
                max_calls_per_session=None,
            )
        except LedgerError as exc:
            self._mark_ledger_error(exc)
            return False
        self._mark_ledger_healthy()
        return True

    def _mark_ledger_error(self, exc: Exception) -> None:
        self._ledger_healthy = False
        self._ledger_error = _diagnostic_text(exc)

    def _mark_ledger_healthy(self) -> None:
        self._ledger_healthy = True
        self._ledger_error = None

    def _snapshot_locked(self) -> dict[str, Any]:
        sessions: dict[str, Any] = {}
        for key in sorted(set(self._trackers) | set(self._errors)):
            tracker = self._trackers.get(key)
            turns = []
            if tracker is not None:
                for turn in tracker.turns:
                    turns.append(
                        {
                            "label": turn.label,
                            "records": [_record_payload(record) for record in turn.records],
                        }
                    )
            sessions[key] = {
                "turns": turns,
                "errors": list(self._errors.get(key, [])),
                "checkpoint_cursor": self._checkpoint_cursors.get(key, 0),
            }
        return {"schema_version": 1, "sessions": sessions}

    def _restore(self, payload: dict[str, Any], *, set_initial: bool = True) -> None:
        try:
            sessions = payload["sessions"]
            if len(sessions) > _MAX_SESSIONS + 1:
                raise ValueError("ledger exceeds supported session capacity")
            for key, session in sessions.items():
                if not isinstance(key, str) or not isinstance(session, dict):
                    raise ValueError("invalid session entry")
                tracker = CostTracker(
                    on_error=lambda exc, session_key=key: self.record_error(
                        session_key, "cost_estimation_failed", str(exc)
                    )
                )
                for turn_payload in session.get("turns", []):
                    if not isinstance(turn_payload, dict):
                        raise ValueError("invalid turn entry")
                    label = turn_payload.get("label")
                    if label is not None and not isinstance(label, str):
                        raise ValueError("invalid turn label")
                    record_payloads = turn_payload.get("records", [])
                    if not isinstance(record_payloads, list):
                        raise ValueError("invalid turn records")
                    tracker.restore_turn(
                        [_record_from_payload(item) for item in record_payloads],
                        label=label,
                    )
                errors = session.get("errors", [])
                if not isinstance(errors, list) or not all(
                    isinstance(error, dict) for error in errors
                ):
                    raise ValueError("invalid diagnostics entry")
                cursor = session.get("checkpoint_cursor", 0)
                if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
                    raise ValueError("invalid checkpoint cursor")
                if tracker.turns:
                    if len(_tracker_records(tracker)) > _MAX_CALLS_PER_SESSION:
                        raise ValueError("session exceeds supported call capacity")
                    self._trackers[key] = tracker
                self._errors[key] = list(errors[-_MAX_ERRORS_PER_SESSION:])
                self._checkpoint_cursors[key] = min(
                    cursor, len(_tracker_records(tracker))
                )
                if set_initial:
                    self._initial_totals[key] = tracker.session_total
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"durable ledger has invalid accounting data: {exc}") from exc


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


def _record_payload(record: CallRecord) -> dict[str, Any]:
    return {
        "model": record.model,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "cached_tokens": record.cached_tokens,
        "cost": record.cost.as_dict(stringify=True),
        "timestamp": record.timestamp,
    }


def _record_from_payload(payload: Any) -> CallRecord:
    if not isinstance(payload, dict):
        raise ValueError("invalid call record")
    model = payload.get("model")
    timestamp = payload.get("timestamp")
    if not isinstance(model, str) or not isinstance(timestamp, (int, float)):
        raise ValueError("invalid call record identity")
    token_values = {}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid call record {key}")
        token_values[key] = value
    cost = payload.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("invalid call record cost")
    try:
        breakdown = CostBreakdown(
            prompt_cost_uncached=Decimal(cost["prompt_cost_uncached"]),
            prompt_cost_cached=Decimal(cost["prompt_cost_cached"]),
            completion_cost=Decimal(cost["completion_cost"]),
            total_cost=Decimal(cost["total_cost"]),
        )
    except Exception as exc:
        raise ValueError("invalid call record cost") from exc
    if any(value < 0 or not value.is_finite() for value in breakdown.as_dict(False).values()):
        raise ValueError("invalid call record cost")
    if breakdown.total_cost != (
        breakdown.prompt_cost_uncached
        + breakdown.prompt_cost_cached
        + breakdown.completion_cost
    ):
        raise ValueError("call record cost components do not equal total")
    return CallRecord(cost=breakdown, timestamp=float(timestamp), model=model, **token_values)


def _latest_call(tracker: Optional[CostTracker]) -> Optional[dict[str, Any]]:
    records = _tracker_records(tracker)
    if not records:
        return None
    return _record_payload(records[-1])


default_registry = TrackerRegistry()


class RegistryCapacityError(RuntimeError):
    """Raised internally when bounded accounting state reaches capacity."""


def _diagnostic_text(value: object, *, limit: int = _MAX_DIAGNOSTIC_LENGTH) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = "".join(character if character.isprintable() else "?" for character in text)
    text = _SECRET_RE.sub("[REDACTED]", text)
    return text[:limit]
