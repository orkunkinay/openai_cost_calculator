"""Provider specifications: the single home for provider-specific behaviour.

A :class:`ProviderSpec` answers three questions the provider-agnostic core
cannot:

* Which catalog identifiers might a caller's model string refer to?
  (Bedrock inference-profile prefixes and ARNs, Vertex ``@version`` suffixes,
  Gemini ``models/`` paths, dated snapshots.)
* What do the caller's model string and request time imply about pricing
  conditions?  (A Bedrock ``global.`` profile implies global-endpoint pricing;
  a DeepSeek request at 07:00 UTC on a weekday is peak-priced.)
* Which condition values apply when the caller specifies none?

Everything else - arithmetic, tier selection, lookup - is shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Preferences = Dict[str, Tuple[str, ...]]

_SNAPSHOT_SUFFIXES = (
    re.compile(r"@(?:\d{8}|\d{3}|latest|default)$"),  # vertex: claude-x@20250929, gemini@001
    re.compile(r"-v\d+(?::\d+)?$"),  # bedrock: -v1:0 / -v1
    re.compile(r":\d+$"),  # bedrock throughput suffix
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),  # openai: gpt-4o-2024-08-06
    re.compile(r"-\d{8}$"),  # anthropic: claude-sonnet-4-5-20250929
    re.compile(r"-0\d\d$"),  # gemini stable versions: gemini-2.0-flash-001
)


def _dedupe(items: Sequence[str]) -> Tuple[str, ...]:
    seen: Dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return tuple(seen)


def generic_candidates(model: str, strip_prefixes: Sequence[str] = ()) -> Tuple[str, ...]:
    """Candidate catalog identifiers for ``model``, most specific first.

    Tries the string as given, then without known path prefixes, then without a
    ``vendor/`` namespace, then with snapshot/version suffixes removed one at
    a time.  The catalog index accepts the first candidate that matches, so an
    exact dated entry always beats its undated family.
    """
    model = model.strip()
    candidates: List[str] = [model]
    current = model
    for prefix in strip_prefixes:
        if current.startswith(prefix):
            current = current[len(prefix) :]
            candidates.append(current)
    if "/" in current:
        candidates.append(current.rsplit("/", 1)[-1])
    stripped = candidates[-1]
    changed = True
    while changed:
        changed = False
        for pattern in _SNAPSHOT_SUFFIXES:
            shorter = pattern.sub("", stripped)
            if shorter != stripped and shorter:
                stripped = shorter
                candidates.append(stripped)
                changed = True
    return _dedupe(candidates)


@dataclass(frozen=True)
class ModelHints:
    """What a model string itself reveals, beyond the catalog identifier."""

    candidates: Tuple[str, ...]
    preferences: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    #: condition key -> explanation, reported when that condition affects the price.
    notes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    display_name: str
    pricing_url: str
    aliases: Tuple[str, ...] = ()
    #: Region preference order when the caller does not choose one.
    default_regions: Tuple[str, ...] = ("global",)
    #: Other condition defaults (e.g. DeepSeek's billing period).
    default_preferences: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Path prefixes stripped from model strings (``models/``, ``accounts/...``).
    strip_prefixes: Tuple[str, ...] = ()
    #: Provider-specific model-string parsing; defaults to :func:`generic_candidates`.
    parse_model: Optional[Callable[[str], ModelHints]] = None
    #: Conditions implied by the request time (e.g. peak/off-peak).
    time_conditions: Optional[Callable[[datetime], Tuple[Preferences, Mapping[str, str]]]] = None
    #: Acceptable region values for a caller-chosen region, most specific first
    #: (Vertex: ``us-central1`` falls back to the "regional" price).
    expand_region: Optional[Callable[[str], Tuple[str, ...]]] = None
    notes: str = ""

    def regions_for(self, region: str) -> Tuple[str, ...]:
        return self.expand_region(region) if self.expand_region is not None else (region,)

    def hints(self, model: str) -> ModelHints:
        if self.parse_model is not None:
            return self.parse_model(model)
        return ModelHints(candidates=generic_candidates(model, self.strip_prefixes))
