from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


SCHEMA_VERSION = 1


class LedgerError(RuntimeError):
    """Raised when a durable ledger cannot be opened, validated, or written."""


class DurableLedger:
    """Single-process atomic JSON storage for proxy accounting state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._lock_handle = None
        self._acquire_lock()
        self._cleanup_temporary_files()

    def load(self) -> dict[str, Any]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"schema_version": SCHEMA_VERSION, "sessions": {}}
        except OSError as exc:
            raise LedgerError(f"cannot read durable ledger: {exc}") from exc

        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise LedgerError("durable ledger is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise LedgerError("durable ledger root must be a JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise LedgerError(
                f"unsupported durable ledger schema: {payload.get('schema_version')!r}"
            )
        if not isinstance(payload.get("sessions"), dict):
            raise LedgerError("durable ledger sessions must be a JSON object")
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        if self.path.is_symlink():
            raise LedgerError("refusing to replace a symlinked durable ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            self._fsync_directory()
        except (OSError, TypeError, ValueError) as exc:
            raise LedgerError(f"cannot write durable ledger: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._lock_handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._lock_handle = None

    def _acquire_lock(self) -> None:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - non-POSIX platforms
            raise LedgerError("durable ledgers currently require POSIX file locking") from exc

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LedgerError(
                "durable ledger is already in use by another proxy process"
            ) from exc
        self._lock_handle = handle

    def _cleanup_temporary_files(self) -> None:
        pattern = f".{self.path.name}.*.tmp"
        for temporary_path in self.path.parent.glob(pattern):
            try:
                temporary_path.unlink()
            except OSError:
                pass

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class SQLiteLedger:
    """Concurrent transactional accounting storage backed by stdlib SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if self.path.is_symlink():
            raise LedgerError("refusing to open a symlinked SQLite ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=5,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._initialize()
            self.path.chmod(0o600)
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot open SQLite ledger: {exc}") from exc

    def load(self) -> dict[str, Any]:
        sessions: dict[str, dict[str, Any]] = {}
        try:
            calls = self._connection.execute(
                """
                SELECT id, session, turn_label, model, prompt_tokens,
                       completion_tokens, cached_tokens, prompt_cost_uncached,
                       prompt_cost_cached, completion_cost, total_cost, timestamp
                FROM calls ORDER BY id
                """
            ).fetchall()
            turns_by_session: dict[str, dict[str | None, dict[str, Any]]] = {}
            for row in calls:
                session = row["session"]
                state = sessions.setdefault(
                    session,
                    {"turns": [], "errors": [], "checkpoint_cursor": 0},
                )
                by_label = turns_by_session.setdefault(session, {})
                label = row["turn_label"]
                turn = by_label.get(label)
                if turn is None:
                    turn = {"label": label, "records": []}
                    by_label[label] = turn
                    state["turns"].append(turn)
                turn["records"].append(self._call_payload(row))

            for row in self._connection.execute(
                "SELECT session, code, message, timestamp FROM diagnostics ORDER BY id"
            ):
                state = sessions.setdefault(
                    row["session"],
                    {"turns": [], "errors": [], "checkpoint_cursor": 0},
                )
                state["errors"].append(
                    {
                        "code": row["code"],
                        "message": row["message"],
                        "timestamp": row["timestamp"],
                    }
                )

            for row in self._connection.execute(
                "SELECT session, call_id FROM checkpoints"
            ):
                state = sessions.setdefault(
                    row["session"],
                    {"turns": [], "errors": [], "checkpoint_cursor": 0},
                )
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM calls WHERE session = ? AND id <= ?",
                    (row["session"], row["call_id"]),
                ).fetchone()[0]
                state["checkpoint_cursor"] = count
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot read SQLite ledger: {exc}") from exc
        return {"schema_version": SCHEMA_VERSION, "sessions": sessions}

    def append_call(
        self,
        session: str,
        turn_label: str | None,
        payload: dict[str, Any],
        *,
        max_sessions: int,
        max_calls_per_session: int,
    ) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._enforce_capacity(
                session,
                max_sessions=max_sessions,
                max_calls_per_session=max_calls_per_session,
            )
            cost = payload["cost"]
            self._connection.execute(
                """
                INSERT INTO calls (
                    session, turn_label, model, prompt_tokens, completion_tokens,
                    cached_tokens, prompt_cost_uncached, prompt_cost_cached,
                    completion_cost, total_cost, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session,
                    turn_label,
                    payload["model"],
                    payload["prompt_tokens"],
                    payload["completion_tokens"],
                    payload["cached_tokens"],
                    cost["prompt_cost_uncached"],
                    cost["prompt_cost_cached"],
                    cost["completion_cost"],
                    cost["total_cost"],
                    payload["timestamp"],
                ),
            )
            self._connection.execute("COMMIT")
        except LedgerError:
            self._rollback()
            raise
        except (KeyError, sqlite3.Error) as exc:
            self._rollback()
            raise LedgerError(f"cannot append SQLite call: {exc}") from exc

    def append_error(
        self,
        session: str,
        error: dict[str, Any],
        *,
        max_sessions: int,
        max_errors: int,
    ) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._enforce_session_capacity(session, max_sessions)
            self._connection.execute(
                "INSERT INTO diagnostics (session, code, message, timestamp) VALUES (?, ?, ?, ?)",
                (session, error["code"], error["message"], error["timestamp"]),
            )
            self._connection.execute(
                """
                DELETE FROM diagnostics
                WHERE session = ? AND id NOT IN (
                    SELECT id FROM diagnostics WHERE session = ?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (session, session, max_errors),
            )
            self._connection.execute("COMMIT")
        except LedgerError:
            self._rollback()
            raise
        except (KeyError, sqlite3.Error) as exc:
            self._rollback()
            raise LedgerError(f"cannot append SQLite diagnostic: {exc}") from exc

    def checkpoint(self, session: str) -> list[dict[str, Any]]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT call_id FROM checkpoints WHERE session = ?",
                (session,),
            ).fetchone()
            cursor = int(row[0]) if row is not None else 0
            rows = self._connection.execute(
                """
                SELECT id, session, turn_label, model, prompt_tokens,
                       completion_tokens, cached_tokens, prompt_cost_uncached,
                       prompt_cost_cached, completion_cost, total_cost, timestamp
                FROM calls WHERE session = ? AND id > ? ORDER BY id
                """,
                (session, cursor),
            ).fetchall()
            next_cursor = int(rows[-1]["id"]) if rows else cursor
            self._connection.execute(
                """
                INSERT INTO checkpoints (session, call_id) VALUES (?, ?)
                ON CONFLICT(session) DO UPDATE SET call_id = excluded.call_id
                """,
                (session, next_cursor),
            )
            self._connection.execute("COMMIT")
            return [self._call_payload(item) for item in rows]
        except sqlite3.Error as exc:
            self._rollback()
            raise LedgerError(f"cannot consume SQLite checkpoint: {exc}") from exc

    def reset(self) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute("DELETE FROM calls")
            self._connection.execute("DELETE FROM diagnostics")
            self._connection.execute("DELETE FROM checkpoints")
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback()
            raise LedgerError(f"cannot reset SQLite ledger: {exc}") from exc

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                turn_label TEXT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL CHECK(prompt_tokens >= 0),
                completion_tokens INTEGER NOT NULL CHECK(completion_tokens >= 0),
                cached_tokens INTEGER NOT NULL CHECK(cached_tokens >= 0),
                prompt_cost_uncached TEXT NOT NULL,
                prompt_cost_cached TEXT NOT NULL,
                completion_cost TEXT NOT NULL,
                total_cost TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS calls_session_id ON calls(session, id);
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                code TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS diagnostics_session_id
                ON diagnostics(session, id);
            CREATE TABLE IF NOT EXISTS checkpoints (
                session TEXT PRIMARY KEY,
                call_id INTEGER NOT NULL CHECK(call_id >= 0)
            );
            """
        )
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif row[0] != str(SCHEMA_VERSION):
            raise LedgerError(f"unsupported SQLite ledger schema: {row[0]!r}")

    def _enforce_capacity(
        self,
        session: str,
        *,
        max_sessions: int,
        max_calls_per_session: int,
    ) -> None:
        self._enforce_session_capacity(session, max_sessions)
        count = self._connection.execute(
            "SELECT COUNT(*) FROM calls WHERE session = ?",
            (session,),
        ).fetchone()[0]
        if count >= max_calls_per_session:
            raise LedgerError("per-session SQLite call capacity reached")

    def _enforce_session_capacity(self, session: str, max_sessions: int) -> None:
        exists = self._connection.execute(
            """
            SELECT 1 FROM calls WHERE session = ?
            UNION ALL SELECT 1 FROM diagnostics WHERE session = ? LIMIT 1
            """,
            (session, session),
        ).fetchone()
        if exists is not None:
            return
        count = self._connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT session FROM calls UNION SELECT session FROM diagnostics
            )
            """
        ).fetchone()[0]
        if count >= max_sessions:
            raise LedgerError("SQLite session capacity reached")

    @staticmethod
    def _call_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "model": row["model"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "cached_tokens": row["cached_tokens"],
            "cost": {
                "prompt_cost_uncached": row["prompt_cost_uncached"],
                "prompt_cost_cached": row["prompt_cost_cached"],
                "completion_cost": row["completion_cost"],
                "total_cost": row["total_cost"],
            },
            "timestamp": row["timestamp"],
        }

    def _rollback(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
