"""
Public façade – import **one function** and you're done:

    from openai_cost_calc import estimate_cost
    cost = estimate_cost(openai_response)
"""

from __future__ import annotations

from typing import Iterable, Any, Dict, Tuple, Optional

from .core import calculate_cost_typed
from .parser import extract_model_details, extract_usage
from .pricing import load_pricing
from .types import CostBreakdown
from .telemetry import merge_tags, emit_to_sinks


class CostEstimateError(RuntimeError):
    """All cost-estimation-related errors collapse to this one type."""


def _pick_last_chunk(response: Iterable[Any]) -> Any:
    """Walk a stream and return the **last** chunk that has `.usage`."""
    last = None
    for chunk in response:
        if hasattr(chunk, "usage"):
            last = chunk
    if last is None:
        raise CostEstimateError("Stream contained zero usable chunks")
    return last


def _find_rates(model_name: str, model_date: str) -> Dict[str, float]:
    pricing = load_pricing()
    key_exact: Tuple[str, str] = (model_name, model_date)
    if key_exact in pricing:
        return pricing[key_exact]

    # Fallback: latest entry for that model (max date ≤ requested)
    candidates = [
        (date, rates)
        for (name, date), rates in pricing.items()
        if name == model_name and date <= model_date
    ]
    if not candidates:
        raise CostEstimateError(
            f"No pricing data for model '{model_name}' (date {model_date})"
        )
    # pick the newest among the older dates
    selected_date, rates = max(candidates, key=lambda t: t[0])
    return rates


# --------------------------------------------------------------------------- #
#   PUBLIC: estimate_cost_typed                                               #
# --------------------------------------------------------------------------- #
def estimate_cost_typed(
    response: Any,
    *,
    tags: Optional[Dict[str, str]] = None,
) -> CostBreakdown:
    """
    Estimate costs and return a strongly-typed CostBreakdown dataclass.

    Parameters
    ----------
    response
        * a single `ChatCompletion`
        * **or** an iterator / generator of streamed `ChatCompletionChunk`s
        * **or** the `Response` object
    tags
        Optional dict of metadata tags (project, feature, user_id, cost_center…).
        Merged with global default tags; per-call keys override global ones.

    Returns
    -------
    CostBreakdown
        Strongly-typed dataclass with Decimal fields containing:
        - prompt_cost_uncached: Decimal
        - prompt_cost_cached: Decimal
        - completion_cost: Decimal
        - total_cost: Decimal
        - metadata: Dict[str, str] (may be empty)

    Raises
    ------
    CostEstimateError
        for every recoverable problem (bad input, missing attrs, …)
    """
    try:
        # -------------------------------------------------------------- usage
        if hasattr(response, "__iter__") and not hasattr(response, "model"):
            # stream → look at the LAST chunk that carried `usage`
            chunk = _pick_last_chunk(response)
        else:
            chunk = response

        usage = extract_usage(chunk)

        # ------------------------------------------------------------- model
        details = extract_model_details(chunk.model)
        rates = _find_rates(details["model_name"], details["model_date"])

        # -------------------------------------------------------------- cost
        base = calculate_cost_typed(usage, rates)

        # Merge global + per-call tags (and validate)
        merged = merge_tags(tags)

        if merged:
            result = CostBreakdown(
                prompt_cost_uncached=base.prompt_cost_uncached,
                prompt_cost_cached=base.prompt_cost_cached,
                completion_cost=base.completion_cost,
                total_cost=base.total_cost,
                metadata=merged,
            )
        else:
            result = base

        # Notify sinks (best-effort; exceptions swallowed)
        emit_to_sinks(result)
        return result

    except Exception as exc:
        raise CostEstimateError(str(exc)) from exc


# --------------------------------------------------------------------------- #
#   PUBLIC: estimate_cost                                                     #
# --------------------------------------------------------------------------- #
def estimate_cost(
    response: Any,
    *,
    tags: Optional[Dict[str, str]] = None,
    include_metadata: bool = False,
) -> Dict[str, str]:
    """
    Parameters
    ----------
    response
        * a single `ChatCompletion`
        * **or** an iterator / generator of streamed `ChatCompletionChunk`s
        * **or** the `Response` object
    tags
        Optional dict of metadata tags (merged with global defaults).
    include_metadata
        If metadata exists (global tags or per-call tags), it will be included automatically.

    Returns
    -------
    dict
        Same numeric keys as :pyfunc:`openai_cost_calculator.core.calculate_cost`,
        with optional "metadata" if requested.

    Raises
    ------
    CostEstimateError
        for every recoverable problem (bad input, missing attrs, …)

    Note
    ----
    This function is maintained for backward compatibility. For new code,
    consider using estimate_cost_typed() which returns a strongly-typed
    CostBreakdown dataclass with Decimal precision.
    """
    cost_breakdown = estimate_cost_typed(response, tags=tags)
    return cost_breakdown.as_dict(stringify=True, include_metadata=bool(cost_breakdown.metadata))
