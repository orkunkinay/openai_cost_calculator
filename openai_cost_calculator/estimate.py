"""
Public façade – import **one function** and you're done:

    from openai_cost_calc import estimate_cost
    cost = estimate_cost(openai_response)
"""

from __future__ import annotations

from typing import Iterable, Any, Dict, Tuple

from .core import calculate_cost
from .parser import extract_model_details, extract_usage
from .pricing import load_pricing


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
#   PUBLIC: estimate_cost                                                     #
# --------------------------------------------------------------------------- #
def estimate_cost(response: Any) -> Dict[str, str]:
    """
    Parameters
    ----------
    response
        * a single `ChatCompletion`
        * **or** an iterator / generator of streamed `ChatCompletionChunk`s
        * **or** the `Response` object

    Returns
    -------
    dict
        Same keys as :pyfunc:`openai_cost_calc.core.calculate_cost`.

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
        return calculate_cost(usage, rates)

    except Exception as exc:
        raise CostEstimateError(str(exc)) from exc
