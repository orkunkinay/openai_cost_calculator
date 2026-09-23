"""The vocabulary of billable quantities.

Every provider bills some subset of these dimensions.  Usage counts are
*disjoint*: a token is counted in exactly one dimension (for example
``input`` excludes cache reads and cache writes).  Provider-specific usage
adapters are responsible for converting overlapping provider counters into
disjoint buckets, so the cost engine never needs provider knowledge.

A dimension may declare a *fallback*: the dimension whose rate is charged when
a model has no explicit rate for it.  Fallbacks exist only where the
substitution is how providers actually bill (a cache write on a provider
without explicit write pricing is billed as ordinary input) or where it yields
a conservative upper bound (cache reads charged at the uncached rate).  Every
other missing rate is an explicit error rather than a silent guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple

PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Dimension:
    name: str
    #: Number of usage units one quoted price covers (1M tokens, or 1 call).
    unit_size: Decimal
    unit_label: str
    #: ``"input"`` dimensions count toward long-context tier thresholds.
    side: str
    fallback: Optional[str] = None
    description: str = ""


def _token(name: str, side: str, description: str, fallback: Optional[str] = None) -> Dimension:
    return Dimension(name, PER_MILLION, "1M tokens", side, fallback, description)


def _unit(name: str, label: str, description: str) -> Dimension:
    return Dimension(name, Decimal(1), label, "other", None, description)


_ALL: Tuple[Dimension, ...] = (
    _token("input", "input", "Uncached text input tokens."),
    _token("cached_input", "input", "Input tokens served from a prompt cache (cache reads/hits).", "input"),
    _token(
        "cache_write",
        "input",
        "Input tokens written to a prompt cache (Anthropic: 5-minute TTL).",
        "input",
    ),
    _token("cache_write_1h", "input", "Input tokens written to a 1-hour prompt cache."),
    _token("input_audio", "input", "Uncached audio input tokens."),
    _token("cached_input_audio", "input", "Cached audio input tokens.", "input_audio"),
    _token("input_image", "input", "Image input tokens, where priced separately from text."),
    _token("output", "output", "Output tokens, including reasoning/thinking tokens."),
    _token("output_audio", "output", "Audio output tokens."),
    _token("output_image", "output", "Image output tokens."),
    _unit("web_search", "call", "Web-search / grounding tool calls."),
    _unit("request", "request", "Flat per-request fee."),
)

DIMENSIONS: Dict[str, Dimension] = {dimension.name: dimension for dimension in _ALL}
TOKEN_DIMENSIONS: Tuple[str, ...] = tuple(d.name for d in _ALL if d.unit_size == PER_MILLION)
INPUT_DIMENSIONS: Tuple[str, ...] = tuple(d.name for d in _ALL if d.side == "input")


def get_dimension(name: str) -> Dimension:
    try:
        return DIMENSIONS[name]
    except KeyError:
        known = ", ".join(sorted(DIMENSIONS))
        raise KeyError(f"unknown pricing dimension {name!r}; known dimensions: {known}") from None
