"""Decide which fetched changes are safe to apply automatically.

The guiding rule is *fail closed*: a change is applied only when every check
passes; anything surprising is reported for human review and the checked-in
value is kept.  Specifically:

* **Extraction guard** - if a source suddenly returns far fewer models than
  the catalog holds for it, the parser has probably broken (page redesign);
  nothing from that source is applied.
* **Validation** - fetched entries must pass the same structural validation as
  bundled data; implausible values (unit-conversion errors) need review.
* **Magnitude guard** - a price moving by more than :data:`MAX_RATIO` in
  either direction needs review; real price changes are rarely that large,
  parser bugs often are.
* **Corroboration** - when an independent source still reports the *old*
  price, the change needs review.
* **Parser issues** - a model the parser flagged as ambiguous is not updated.
* **Removals** are never automatic: a model missing upstream is kept and
  reported (it may be retired, renamed, or the parser may have missed it).
* **Additions** are automatic only from official sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence

from ..catalog.errors import CatalogValidationError
from ..catalog.model import ModelPricing
from ..catalog.validation import sanity_warnings, validate_model
from .base import Issue
from .diff import ModelDiff

MAX_RATIO = Decimal(3)
MIN_COVERAGE = Decimal("0.5")
MIN_MODELS_FOR_COVERAGE_GUARD = 4

#: Returns independent base-tier rates for a model, or None if unknown.
Corroborator = Callable[[str, ModelPricing], Optional[Dict[str, Decimal]]]


@dataclass
class Decision:
    diff: ModelDiff
    apply: bool
    reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def coverage_problem(previous_count: int, fetched_count: int) -> Optional[str]:
    if previous_count < MIN_MODELS_FOR_COVERAGE_GUARD:
        return None
    if Decimal(fetched_count) < MIN_COVERAGE * previous_count:
        return (
            f"source returned {fetched_count} models but the catalog holds {previous_count}; "
            "the upstream format probably changed, so no updates were applied"
        )
    return None


def _validation_reasons(model: ModelPricing, provider: str) -> List[str]:
    try:
        validate_model(model, provider=provider)
    except CatalogValidationError as exc:
        return [f"invalid: {exc}"]
    return sanity_warnings(model, provider=provider)


def _corroboration(diff: ModelDiff, provider: str, corroborate: Optional[Corroborator]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {"reasons": [], "notes": []}
    if corroborate is None or diff.new is None:
        return result
    independent = corroborate(provider, diff.new)
    if not independent:
        return result
    for change in diff.rate_changes:
        if change.price != "standard" or change.dimension not in independent:
            continue
        other = independent[change.dimension]
        if change.new is not None and other == change.new:
            result["notes"].append(f"{change.dimension} change corroborated by an independent source")
        elif change.old is not None and other == change.old:
            result["reasons"].append(
                f"{change.dimension}: an independent source still reports the old price {change.old}"
            )
    return result


def decide(
    diffs: Sequence[ModelDiff],
    *,
    provider: str,
    official: bool,
    issues: Sequence[Issue] = (),
    corroborate: Optional[Corroborator] = None,
) -> List[Decision]:
    flagged = {issue.model_id: issue.message for issue in issues if issue.model_id and issue.blocking}
    decisions: List[Decision] = []
    for diff in diffs:
        if diff.kind == "unchanged":
            continue
        decision = Decision(diff, apply=True)
        if diff.kind == "removed":
            decision.apply = False
            decision.reasons.append("no longer listed upstream; kept until a maintainer confirms retirement")
            decisions.append(decision)
            continue
        if diff.model_id in flagged:
            decision.reasons.append(f"parser flagged this model: {flagged[diff.model_id]}")
        assert diff.new is not None
        decision.reasons += _validation_reasons(diff.new, provider)
        if diff.kind == "added" and not official:
            decision.reasons.append("new model from a non-official source")
        for change in diff.rate_changes:
            ratio = change.ratio
            if ratio is not None and (ratio > MAX_RATIO or ratio < 1 / MAX_RATIO):
                decision.reasons.append(f"large change ({ratio:.2f}x) in {change}")
            elif diff.kind == "changed" and change.old is not None and change.new is None:
                decision.reasons.append(f"price disappeared upstream: {change}")
        corroboration = _corroboration(diff, provider, corroborate)
        decision.reasons += corroboration["reasons"]
        decision.notes += corroboration["notes"]
        decision.apply = not decision.reasons
        decisions.append(decision)
    return decisions
