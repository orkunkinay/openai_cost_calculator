"""Bridge between observed Anthropic Messages usage and the proxy ledger.

These helpers price a single observed request and record it (or a sanitized
diagnostic) without ever altering the bytes forwarded to Claude Code.  All
monetary arithmetic uses :class:`Decimal` via the Anthropic pricing modules.
"""

from __future__ import annotations

from typing import Any, Optional

from openai_cost_calculator.anthropic.pricing import AnthropicPricingError
from openai_cost_calculator.anthropic.usage import (
    AnthropicUsage,
    AnthropicUsageError,
    price_anthropic_usage,
    to_cost_breakdown,
    to_ledger_tokens,
    unpriced_usage_fields,
    usage_from_dict,
)
from openai_cost_calculator.proxy.registry import TrackerRegistry


UNATTRIBUTED_TURN = "unattributed"


def record_anthropic_response(
    registry: TrackerRegistry,
    session_id: Optional[str],
    turn_label: Optional[str],
    *,
    usage: Optional[AnthropicUsage],
    raw_usage: Optional[dict[str, Any]],
    response_model: Optional[str],
    request_model: Optional[str],
) -> None:
    """Price and record one observed Anthropic request.

    ``usage`` is the assembled usage (``None`` when none was observed).  The
    response model is preferred for pricing; the requested model is a recorded
    fallback.  Nothing here can raise into the forwarding path.
    """
    if usage is None:
        registry.record_error(
            session_id,
            "usage_missing",
            "successful Anthropic response did not report usage",
        )
        return

    resolved_model = _clean_model(response_model)
    used_fallback = False
    if resolved_model is None:
        resolved_model = _clean_model(request_model)
        used_fallback = resolved_model is not None
    if resolved_model is None:
        registry.record_error(
            session_id,
            "model_missing",
            "Anthropic response with usage did not identify a model",
        )
        return

    try:
        cost = price_anthropic_usage(resolved_model, usage)
    except (AnthropicPricingError, AnthropicUsageError):
        registry.record_error(
            session_id,
            "pricing_unavailable",
            f"no Anthropic pricing for observed model {_safe_model(resolved_model)!r}",
        )
        return

    registry.record_costed_call(
        session_id,
        resolved_model,
        to_ledger_tokens(usage),
        to_cost_breakdown(cost),
        turn_label=turn_label,
    )

    if used_fallback:
        registry.record_error(
            session_id,
            "model_fallback",
            "priced request using the requested model; response omitted its model",
        )
    if turn_label == UNATTRIBUTED_TURN:
        registry.record_error(
            session_id,
            "turn_unattributed",
            "request had no active turn; recorded under the session-only synthetic turn",
        )
    for field in unpriced_usage_fields(raw_usage or {}):
        registry.record_error(
            session_id,
            "server_tool_unpriced",
            f"Anthropic usage reported unpriced billable field {field!r}",
        )


def usage_from_response_payload(payload: Any) -> Optional[AnthropicUsage]:
    """Assemble usage from a non-streaming Anthropic JSON body."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        return usage_from_dict(usage)
    except AnthropicUsageError:
        return None


def _clean_model(model: Any) -> Optional[str]:
    return model if isinstance(model, str) and model else None


def _safe_model(model: Any) -> str:
    """A bounded, printable model identifier safe to store in a diagnostic."""
    text = str(model)
    text = "".join(character if character.isprintable() else "?" for character in text)
    return text[:100]
