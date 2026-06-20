from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

from openai_cost_calculator.adapters.anthropic_pricing import seed_anthropic_pricing
from openai_cost_calculator.adapters.common import (
    MONEY,
    SEP,
    compact_tokens,
    decimal_from,
    format_money,
    int_from,
    nested,
)
from openai_cost_calculator.core import calculate_cost_typed
from openai_cost_calculator.estimate import _find_rates
from openai_cost_calculator.parser import extract_model_details


@dataclass
class _AssistantCost:
    cost: Decimal
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def statusline_text(payload: dict[str, Any]) -> str:
    cost = decimal_from(nested(payload, "cost", "total_cost_usd"))
    model = (
        nested(payload, "model", "display_name")
        or nested(payload, "model", "id")
        or "model"
    )
    context = payload.get("context_window") if isinstance(payload, dict) else {}
    context = context if isinstance(context, dict) else {}

    current_usage = context.get("current_usage")
    if isinstance(current_usage, dict):
        input_tokens = int_from(current_usage.get("input_tokens"))
        output_tokens = int_from(current_usage.get("output_tokens"))
        cached_tokens = int_from(current_usage.get("cache_read_input_tokens")) + int_from(
            current_usage.get("cache_creation_input_tokens")
        )
        last = (
            f"last {compact_tokens(input_tokens)}->{compact_tokens(output_tokens)} tok"
        )
        if cached_tokens:
            last = f"{last} (cache {compact_tokens(cached_tokens)})"
    else:
        last = "last -- tok"

    ctx_pct = _context_percent(context)
    ctx = f"ctx {ctx_pct}%" if ctx_pct is not None else "ctx --"
    return (
        f"{MONEY} {format_money(cost)} session {SEP} {last} "
        f"{SEP} {model} {SEP} {ctx}"
    )


def statusline_main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        print(statusline_text(payload))
    except Exception:
        print(f"{MONEY} $0.0000 session {SEP} ctx --")
    return 0


def stop_hook_output(
    payload: dict[str, Any],
    *,
    cache_dir: Optional[Path] = None,
) -> dict[str, str]:
    session_id = str(payload.get("session_id") or "default")
    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return {}

    records = _read_jsonl(Path(os.path.expanduser(transcript_path)))
    if not records:
        return {}

    cache_path = _cache_path(session_id, transcript_path, cache_dir)
    last_index = _read_cache_index(cache_path)
    start_index = _turn_start_index(records, last_index)
    selected = [
        (index, record)
        for index, record in enumerate(records)
        if index >= start_index and _role(record) == "assistant"
    ]
    if not selected:
        _write_cache_index(cache_path, len(records))
        return {}

    total = _sum_assistant_costs((record for _, record in selected), payload)
    _write_cache_index(cache_path, len(records))
    if total is None or total.cost <= 0:
        return {}

    message = (
        f"{MONEY} This turn cost {format_money(total.cost)} "
        f"({compact_tokens(total.prompt_tokens)} in / "
        f"{compact_tokens(total.completion_tokens)} out)"
    )
    return {"systemMessage": message}


def stop_hook_main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        output = stop_hook_output(payload)
    except Exception:
        output = {}
    print(json.dumps(output, separators=(",", ":")))
    return 0


def _context_percent(context: dict[str, Any]) -> Optional[int]:
    size = int_from(context.get("context_window_size"))
    total = int_from(context.get("total_input_tokens"))
    if size > 0 and total >= 0:
        return int((Decimal(total) / Decimal(size) * Decimal("100")).quantize(Decimal("1")))
    used = context.get("used_percentage")
    if used is not None:
        return int(decimal_from(used).quantize(Decimal("1")))
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except Exception:
        return []
    return records


def _cache_path(
    session_id: str,
    transcript_path: str,
    cache_dir: Optional[Path],
) -> Path:
    root = cache_dir or Path(
        os.environ.get("OCC_CACHE_DIR")
        or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "occ"
    digest = hashlib.sha256(f"{session_id}:{transcript_path}".encode("utf-8")).hexdigest()
    return root / f"claude-{digest[:24]}.json"


def _read_cache_index(path: Path) -> Optional[int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("last_index"), int):
        return int(payload["last_index"])
    return None


def _write_cache_index(path: Path, index: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_index": index}), encoding="utf-8")
    except Exception:
        return


def _turn_start_index(records: list[dict[str, Any]], last_index: Optional[int]) -> int:
    if last_index is not None:
        return min(max(last_index, 0), len(records))
    last_assistant = max(
        (index for index, record in enumerate(records) if _role(record) == "assistant"),
        default=-1,
    )
    if last_assistant == -1:
        return len(records)
    for index in range(last_assistant - 1, -1, -1):
        if _role(records[index]) == "user":
            return index + 1
    return 0


def _role(record: dict[str, Any]) -> Optional[str]:
    for value in (
        nested(record, "message", "role"),
        record.get("role"),
        record.get("type"),
    ):
        if value in {"assistant", "user"}:
            return str(value)
    return None


def _sum_assistant_costs(
    records: Iterable[dict[str, Any]],
    hook_payload: dict[str, Any],
) -> Optional[_AssistantCost]:
    total = _AssistantCost(Decimal("0"), 0, 0, 0)
    usage_records: list[tuple[str, dict[str, int]]] = []

    for record in records:
        cost = _message_cost(record)
        usage = _message_usage(record)
        if usage is not None:
            total.prompt_tokens += usage["prompt_tokens"]
            total.completion_tokens += usage["completion_tokens"]
            total.cached_tokens += usage["cached_tokens"]
        if cost is not None:
            total.cost += cost
        elif usage is not None:
            model = _message_model(record) or nested(hook_payload, "model", "id")
            if isinstance(model, str) and model:
                usage_records.append((model, usage))

    if total.cost == 0 and usage_records:
        seed_anthropic_pricing()
        for model, usage in usage_records:
            try:
                details = extract_model_details(model)
                rates = _find_rates(
                    details["model_name"],
                    details["model_date"],
                    usage["prompt_tokens"],
                )
                total.cost += calculate_cost_typed(usage, rates).total_cost
            except Exception:
                continue

    if total.cost == 0 and total.prompt_tokens == 0 and total.completion_tokens == 0:
        return None
    return total


def _message_cost(record: dict[str, Any]) -> Optional[Decimal]:
    candidates = (
        record.get("cost_usd"),
        record.get("total_cost_usd"),
        nested(record, "cost", "total_cost_usd"),
        nested(record, "message", "cost_usd"),
        nested(record, "message", "total_cost_usd"),
        nested(record, "message", "cost", "total_cost_usd"),
        nested(record, "message", "usage", "cost_usd"),
    )
    for candidate in candidates:
        if candidate is not None:
            cost = decimal_from(candidate)
            if cost > 0:
                return cost
    return None


def _message_usage(record: dict[str, Any]) -> Optional[dict[str, int]]:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = nested(record, "message", "usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = int_from(usage.get("input_tokens") or usage.get("prompt_tokens"))
    output_tokens = int_from(usage.get("output_tokens") or usage.get("completion_tokens"))
    cache_read = int_from(usage.get("cache_read_input_tokens"))
    cache_write = int_from(usage.get("cache_creation_input_tokens"))
    cached_tokens = int_from(
        nested(usage, "input_tokens_details", "cached_tokens")
        or nested(usage, "prompt_tokens_details", "cached_tokens")
        or cache_read
    )
    prompt_tokens = input_tokens + cache_write
    if prompt_tokens == 0 and output_tokens == 0 and cached_tokens == 0:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }


def _message_model(record: dict[str, Any]) -> Optional[str]:
    for candidate in (
        record.get("model"),
        nested(record, "message", "model"),
        nested(record, "message", "model", "id"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None

