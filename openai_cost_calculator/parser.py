"""
Lightweight helpers that **only** look at the public attributes exposed by
the OpenAI Python SDK (both the classic `chat.completions.create` and
the `responses.create` flavours).
"""

import re
from datetime import datetime
from typing import Any, Dict


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
        date = datetime.utcnow().strftime("%Y-%m-%d")

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
