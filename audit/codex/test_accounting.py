import json
import random
from decimal import Decimal
from pathlib import Path

import pytest

from openai_cost_calculator import pricing
from openai_cost_calculator.estimate import CostEstimateError
from openai_cost_calculator.proxy.app import _record_json_payload
from openai_cost_calculator.proxy.registry import TrackerRegistry
from reference_calculator.cost import estimate, isolated_rollout

ROOT = Path(__file__).parent


@pytest.fixture
def pin_pricing(monkeypatch):
    csv_path = ROOT.parents[1] / "data/gpt_pricing_data.csv"
    monkeypatch.setattr(pricing, "_CACHE", pricing._parse_csv(csv_path.read_text()))
    monkeypatch.setattr(pricing, "_CACHE_TS", pricing.time.time())
    monkeypatch.setattr(pricing, "_OFFLINE_ONLY", False)
    monkeypatch.setattr(pricing, "_LOCAL_OVERRIDES", {})


def wire_usage(input_tokens, cached, output, reasoning):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output,
    }


@pytest.mark.parametrize("model,max_input", [("gpt-5.3-codex", 200000), ("gpt-6.2-sol", 922000)])
def test_differential_exact_cost_and_aggregates(pin_pricing, model, max_input):
    rng = random.Random(20261125)
    registry = TrackerRegistry()
    expected = {f"session-{index}": Decimal(0) for index in range(5)}
    inputs = [0, 1, *[n for n in (271999, 272000, 272001) if n <= max_input]]
    inputs.extend(rng.randint(0, max_input) for _ in range(2500))
    for index, input_tokens in enumerate(inputs):
        cached = rng.randint(0, input_tokens)
        output = rng.randint(0, 120000)
        reasoning = rng.randint(0, output)
        session = f"session-{index % 5}"
        value = estimate(model, input_tokens, cached, output, reasoning)
        _record_json_payload(registry, session, str(index), {
            "model": model, "usage": wire_usage(input_tokens, cached, output, reasoning)
        })
        tracker = registry.get(session)
        record = tracker.turns[-1]._records[-1]
        assert record.cost.total_cost == value
        assert (record.model, record.prompt_tokens, record.cached_tokens, record.completion_tokens) == (
            model, input_tokens, cached, output,
        )
        expected[session] += value
    summary = registry.summary()
    assert Decimal(summary["grand_total"]) == sum(expected.values()).quantize(Decimal("0.00000001"))
    for session, value in expected.items():
        assert Decimal(summary["sessions"][session]["session_total"]) == value.quantize(Decimal("0.00000001"))


@pytest.mark.parametrize("filename,cost", [
    ("live-single-turn.json", "0.028710"),
    ("live-single-turn-after.json", "0.028716"),
    ("live-single-turn-diagnostic.json", "0.028704"),
])
def test_rollout_cumulative_snapshot_is_not_an_extra_call(filename, cost):
    fixture = json.loads((ROOT / "fixtures" / filename).read_text())
    calls = isolated_rollout(fixture)
    assert len(calls) == 1
    assert calls[0][-1] == Decimal(cost)
    fixture["events"].append(fixture["events"][-1])
    assert isolated_rollout(fixture) == calls


def test_unknown_model_has_diagnostic(pin_pricing):
    registry = TrackerRegistry()
    _record_json_payload(registry, "unknown", None, {
        "model": "never-priced", "usage": wire_usage(1, 0, 1, 0),
    })
    session = registry.summary()["sessions"]["unknown"]
    assert session["turns"] == []
    assert session["errors"][0]["code"] == "cost_estimation_failed"


def test_bundled_rate_can_fill_missing_remote_model(pin_pricing, monkeypatch):
    from openai_cost_calculator.estimate import _find_rates

    pinned = {key: value for key, value in pricing._CACHE.items() if key[0] != "gpt-6.2-sol"}
    monkeypatch.setattr(pricing, "_CACHE", pinned)
    assert _find_rates("gpt-6.2-sol", "2026-11-25", 272000)["input_price"] == 2
    assert _find_rates("gpt-6.2-sol", "2026-11-25", 272001)["input_price"] == 4
    with pytest.raises(CostEstimateError):
        _find_rates("never-priced", "2026-11-25", 1)


@pytest.mark.parametrize("backend", ["ledger_path", "database_path"])
def test_durable_cost_rounds_after_aggregation(pin_pricing, tmp_path, backend):
    options = {backend: tmp_path / "ledger"}
    registry = TrackerRegistry(**options)
    for _ in range(20):
        _record_json_payload(registry, "s", None, {
            "model": "gpt-5.3-codex", "usage": wire_usage(1, 1, 0, 0),
        })
    registry.close()
    restored = TrackerRegistry(**options)
    try:
        assert Decimal(restored.summary()["grand_total"]) == estimate("gpt-5.3-codex", 20, 20, 0)
    finally:
        restored.close()


def test_additional_price_dimensions_mark_standard_subtotal_incomplete(pin_pricing):
    registry = TrackerRegistry()
    usage = wire_usage(10, 0, 1, 0)
    usage["input_tokens_details"]["cache_write_tokens"] = 5
    _record_json_payload(registry, "s", None, {
        "model": "gpt-6.2-sol", "service_tier": "priority", "usage": usage,
    })
    assert registry.summary()["sessions"]["s"]["errors"][0]["code"] == "unsupported_price_dimensions"


def test_live_rollout_replayed_through_proxy_http(pin_pricing):
    import asyncio

    import httpx
    from openai_cost_calculator.proxy.app import create_app

    fixture = json.loads((ROOT / "fixtures/live-single-turn-after.json").read_text())
    [(model, input_tokens, cached, output, reasoning, expected)] = isolated_rollout(fixture)
    payload = {"type": "response.completed", "response": {
        "model": model, "usage": wire_usage(input_tokens, cached, output, reasoning),
    }}
    body = f"data: {json.dumps(payload)}\n\n".encode()

    class TerminalStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body

    async def fake_upstream(request):
        return httpx.Response(200, stream=TerminalStream())

    async def replay():
        app = create_app(upstream="https://example.invalid/v1",
                         transport=httpx.MockTransport(fake_upstream), registry=TrackerRegistry())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://proxy") as client:
            response = await client.post("/v1/responses", json={"stream": True},
                                         headers={"X-OCC-Session": "fixture"})
            assert response.content == body
            summary = (await client.get("/_occ/costs?session=fixture")).json()
            assert summary["cost_basis"] == "standard_api_equivalent_token_estimate"
            assert summary["has_accounting_diagnostics"] is False
            assert Decimal(summary["grand_total"]) == expected
            checkpoint = await client.post("/_occ/checkpoint?session=fixture")
            assert checkpoint.json()["num_calls"] == 1
            repeated = await client.post("/_occ/checkpoint?session=fixture")
            assert repeated.json()["num_calls"] == 0

    asyncio.run(replay())


def test_reference_rejects_rollout_gap():
    fixture = json.loads((ROOT / "fixtures/live-single-turn-after.json").read_text())
    fixture["events"][-1]["info"]["total_token_usage"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="non-contiguous"):
        isolated_rollout(fixture)


def test_real_resumption_cumulative_counts_are_incremented_once():
    fixture = json.loads((ROOT / "fixtures/live-resume.json").read_text())
    first, resumed = isolated_rollout(fixture)
    assert first[1:5] == (14304, 0, 9, 0)
    assert resumed[1:5] == (20009, 0, 9, 0)
    assert first[-1] + resumed[-1] == Decimal("0.06880600")
