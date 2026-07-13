from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_cost_calculator.pricing import (
    add_pricing_entry,
    clear_local_pricing,
    set_offline_mode,
)
from openai_cost_calculator.proxy.ledger import DurableLedger, LedgerError
from openai_cost_calculator.proxy.registry import TrackerRegistry


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


def test_durable_ledger_restores_costs_diagnostics_and_checkpoint(tmp_path: Path):
    path = tmp_path / "accounting.json"
    first = TrackerRegistry(ledger_path=path)
    first.record_call(
        "session-a",
        "gpt-test-2025-01-01",
        {"prompt_tokens": 1000, "completion_tokens": 500, "cached_tokens": 100},
        turn_label="turn-1",
    )
    first.record_error("session-a", "test_diagnostic", "observable")
    consumed = first.checkpoint("session-a")
    assert consumed["num_calls"] == 1
    first.close()

    second = TrackerRegistry(ledger_path=path)
    restored = second.summary("session-a")
    session = restored["sessions"]["session-a"]
    assert session["session_total"] == "0.00195000"
    assert session["historical_total"] == "0.00195000"
    assert session["process_total"] == "0.00000000"
    assert session["latest_call"]["model"] == "gpt-test-2025-01-01"
    assert session["errors"][0]["code"] == "test_diagnostic"
    assert second.checkpoint("session-a")["num_calls"] == 0

    second.record_call(
        "session-a",
        "gpt-test-2025-01-01",
        {"prompt_tokens": 0, "completion_tokens": 1000, "cached_tokens": 0},
        turn_label="turn-2",
    )
    updated = second.summary("session-a")["sessions"]["session-a"]
    assert updated["historical_total"] == "0.00195000"
    assert updated["process_total"] == "0.00200000"
    assert second.checkpoint("session-a")["num_calls"] == 1
    second.close()


def test_durable_ledger_uses_atomic_replacement_and_cleans_interrupted_temp(
    tmp_path: Path,
):
    path = tmp_path / "accounting.json"
    path.write_text('{"schema_version":1,"sessions":{}}\n', encoding="utf-8")
    interrupted = tmp_path / ".accounting.json.abandoned.tmp"
    interrupted.write_text("partial", encoding="utf-8")

    ledger = DurableLedger(path)
    assert not interrupted.exists()
    ledger.save({"schema_version": 1, "sessions": {"safe": {}}})
    assert json.loads(path.read_text(encoding="utf-8"))["sessions"] == {"safe": {}}
    assert path.stat().st_mode & 0o777 == 0o600
    ledger.close()


def test_durable_ledger_rejects_concurrent_writer_and_corrupt_data(tmp_path: Path):
    path = tmp_path / "accounting.json"
    first = DurableLedger(path)
    with pytest.raises(LedgerError, match="already in use"):
        DurableLedger(path)
    first.close()

    path.write_text("not json", encoding="utf-8")
    corrupt = DurableLedger(path)
    with pytest.raises(LedgerError, match="not valid JSON"):
        corrupt.load()
    corrupt.close()


def test_failed_ledger_write_is_visible_and_does_not_consume_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    registry = TrackerRegistry(ledger_path=tmp_path / "accounting.json")

    def fail_save(payload):
        raise LedgerError("simulated interrupted write")

    monkeypatch.setattr(registry._ledger, "save", fail_save)
    record = registry.record_call(
        "session-a",
        "gpt-test-2025-01-01",
        {"prompt_tokens": 1000, "completion_tokens": 0, "cached_tokens": 0},
    )
    assert record is not None
    assert registry.persistence_status()["healthy"] is False
    with pytest.raises(LedgerError, match="not consumed"):
        registry.checkpoint("session-a")
    assert registry._checkpoint_cursors["session-a"] == 0
    registry.close()


def test_reset_is_durable_across_restart(tmp_path: Path):
    path = tmp_path / "accounting.json"
    registry = TrackerRegistry(ledger_path=path)
    registry.record_call(
        "session-a",
        "gpt-test-2025-01-01",
        {"prompt_tokens": 1000, "completion_tokens": 0, "cached_tokens": 0},
    )
    registry.reset()
    registry.close()

    restored = TrackerRegistry(ledger_path=path)
    assert restored.summary()["sessions"] == {}
    restored.close()


def test_registry_capacity_failure_is_explicit(monkeypatch):
    import openai_cost_calculator.proxy.registry as registry_module

    monkeypatch.setattr(registry_module, "_MAX_CALLS_PER_SESSION", 1)
    registry = TrackerRegistry()
    usage = {"prompt_tokens": 1000, "completion_tokens": 0, "cached_tokens": 0}
    assert registry.record_call("bounded", "gpt-test-2025-01-01", usage) is not None
    assert registry.record_call("bounded", "gpt-test-2025-01-01", usage) is None

    session = registry.summary("bounded")["sessions"]["bounded"]
    assert session["session_total"] == "0.00100000"
    assert session["errors"][-1]["code"] == "accounting_capacity_reached"
