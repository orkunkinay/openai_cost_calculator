"""
Lightweight helpers that **only** look at the public attributes exposed by
the OpenAI Python SDK (both the classic `chat.completions.create` and
the `responses.create` flavours).
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
#   Model string → {"name": "...", "date": "..."}                             #
# --------------------------------------------------------------------------- #
_MODEL_RE = re.compile(r"^(.*?)(?:-(\d{4}-\d{2}-\d{2}))?$")


def extract_model_details(model: str) -> Dict[str, str]:
    """
    Accepts:  "gpt-4o-mini-2024-07-18"
    Returns:  {"name": "gpt-4o-mini", "date": "2024-07-18"}

    If the date tag is missing, today's date is used (so pricing falls back
    to “latest available” when the CSV is behind the model rollout).
    """
    if not isinstance(model, str) or not model:
        raise ValueError("`model` must be a non-empty str")

    m = _MODEL_RE.match(model)
    if not m:
        raise ValueError(f"Cannot parse model string: {model!r}")

    name, date = m.groups()
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {"model_name": name, "model_date": date}


# --------------------------------------------------------------------------- #
#   Usage parser                                                              #
# --------------------------------------------------------------------------- #
def extract_usage(obj: Any) -> Dict[str, int]:
    """
    Works with BOTH usage schemas:

    * Respones API (`responses.create`)
          - usage.input_tokens
          - usage.output_tokens
          - usage.input_tokens_details.cached_tokens

    * Chat Completion API (`chat.completions.create`)
          - usage.prompt_tokens
          - usage.completion_tokens
          - usage.prompt_tokens_details.cached_tokens
    """
    if not hasattr(obj, "usage"):
        raise AttributeError("Response / chunk has no `.usage` attribute")

    u = obj.usage

    # ----------- Find prompt / completion tokens ---------------------------
    if hasattr(u, "input_tokens"):            # new schema
        prompt_tokens = getattr(u, "input_tokens", 0) or 0
        completion_tokens = getattr(u, "output_tokens", 0) or 0
        cached_tokens = (
            getattr(getattr(u, "input_tokens_details", None) or {}, "cached_tokens", 0)
            or 0
        )
    else:                                     # classic schema
        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
        completion_tokens = getattr(u, "completion_tokens", 0) or 0
        cached_tokens = (
            getattr(getattr(u, "prompt_tokens_details", None) or {}, "cached_tokens", 0)
            or 0
        )

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_tokens": int(cached_tokens),
    }


def extract_usage_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Return validated OpenAI wire token totals, or None for unusable usage."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    if "input_tokens" in usage or "output_tokens" in usage:
        input_key, output_key = "input_tokens", "output_tokens"
    else:
        input_key, output_key = "prompt_tokens", "completion_tokens"
    input_tokens, output_tokens = usage.get(input_key), usage.get(output_key)
    input_details = usage.get(f"{input_key}_details")
    output_details = usage.get(f"{output_key}_details")
    if input_details is None:
        input_details = {}
    if output_details is None:
        output_details = {}
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        return None

    cached = input_details.get("cached_tokens", 0)
    reasoning = output_details.get("reasoning_tokens", 0)
    total = usage.get("total_tokens", 0)
    counts = (input_tokens, output_tokens, cached, reasoning, total)
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    if cached > input_tokens or reasoning > output_tokens:
        return None
    if "total_tokens" in usage and total != input_tokens + output_tokens:
        return None

    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "cached_tokens": cached,
    }
