from __future__ import annotations

import json
import os
from pathlib import Path
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
