"""Independent standard token-only API-equivalent reference."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

RATES = json.loads((Path(__file__).parent / "rates.json").read_text())


def estimate(model, input_tokens, cached_tokens, output_tokens, reasoning_tokens=0):
    counts = input_tokens, cached_tokens, output_tokens, reasoning_tokens
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("expected nonnegative integer token counts")
    if cached_tokens > input_tokens or reasoning_tokens > output_tokens:
        raise ValueError("token detail exceeds its total")
    rate = RATES["models"][model]
    if input_tokens > rate["maximum_input"]:
        raise ValueError("outside verified input range")
    prefix = "long_" if input_tokens > rate.get("long_after", rate["maximum_input"]) else ""
    with localcontext() as context:
        context.prec = 60
        return (
            (input_tokens - cached_tokens) * Decimal(rate[prefix + "input"])
            + cached_tokens * Decimal(rate[prefix + "cached_input"])
            + output_tokens * Decimal(rate[prefix + "output"])
        ).scaleb(-6)


def isolated_rollout(fixture):
    """Read a fresh single-model rollout; reject ambiguous deltas."""
    names = ("input_tokens", "cached_input_tokens", "output_tokens",
             "reasoning_output_tokens", "total_tokens")
    previous = (0,) * len(names)
    model = None
    calls = []
    for event in fixture["events"]:
        if event["type"] == "turn_context":
            if model is not None and model != event["model"]:
                raise ValueError("model switch requires per-response attribution")
            model = event["model"]
        elif event["type"] == "token_count" and event["info"] is not None:
            info = event["info"]
            total = tuple(info["total_token_usage"][name] for name in names)
            if total == previous:
                continue
            last = tuple(info["last_token_usage"][name] for name in names)
            if tuple(current - base for current, base in zip(total, previous)) != last:
                raise ValueError("non-contiguous cumulative usage")
            if model is None or last[0] + last[2] != last[4]:
                raise ValueError("missing model or inconsistent total")
            calls.append((model, *last[:4], estimate(model, *last[:4])))
            previous = total
    if not calls:
        raise ValueError("no usable token records")
    return calls
