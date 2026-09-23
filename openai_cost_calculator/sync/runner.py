"""Orchestrate fetch -> validate -> diff -> decide -> write for each provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..catalog.errors import CatalogValidationError
from ..catalog.io import dump_provider, load_provider_file
from ..catalog.model import ModelPricing, ProviderPricing
from ..catalog.validation import validate_provider
from .base import Fetcher, Issue, PricingSource, SourceError, merge_duplicate_models
from .diff import diff_models
from .policy import Corroborator, Decision, coverage_problem, decide

#: Re-stamp ``verified_at`` at most this often when prices are unchanged, so a
#: successful weekly run does not produce a data diff every week.
REVERIFY_DAYS = 30


@dataclass
class ProviderOutcome:
    provider: str
    source_id: str
    source_url: str
    status: str  # "unchanged" | "updated" | "needs_review" | "failed"
    decisions: List[Decision] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    error: Optional[str] = None
    data: Optional[ProviderPricing] = None
    fetched_models: int = 0
    written: bool = False

    @property
    def applied(self) -> List[Decision]:
        return [d for d in self.decisions if d.apply]

    @property
    def needs_review(self) -> List[Decision]:
        return [d for d in self.decisions if not d.apply]

    @property
    def blocking_issues(self) -> List[Issue]:
        return [i for i in self.issues if i.blocking]

    @property
    def notes(self) -> List[Issue]:
        return [i for i in self.issues if not i.blocking]


def _merge(
    current: Optional[ProviderPricing],
    source: PricingSource,
    decisions: Sequence[Decision],
    owned: Sequence[ModelPricing],
) -> List[ModelPricing]:
    by_id = {m.id: m for m in owned}
    for decision in decisions:
        if decision.apply and decision.diff.new is not None:
            by_id[decision.diff.model_id] = decision.diff.new
    # A source-backed entry supersedes a hand-maintained one for the same model.
    claimed = {i.lower() for m in by_id.values() for i in m.identifiers}
    others = [
        m
        for m in (current.models if current else ())
        if m.source != source.source.id and not claimed.intersection(i.lower() for i in m.identifiers)
    ]
    return others + list(by_id.values())


def sync_provider(
    source: PricingSource,
    current: Optional[ProviderPricing],
    fetcher: Fetcher,
    *,
    today: date,
    corroborate: Optional[Corroborator] = None,
) -> ProviderOutcome:
    outcome = ProviderOutcome(
        provider=source.provider,
        source_id=source.source.id,
        source_url=source.source.url,
        status="unchanged",
        data=current,
    )
    owned = [m for m in (current.models if current else ()) if m.source == source.source.id]
    try:
        result = merge_duplicate_models(source.fetch(fetcher))
    except SourceError as exc:
        outcome.status, outcome.error = "failed", str(exc)
        return outcome
    except Exception as exc:  # a parser bug must not take down other providers
        outcome.status, outcome.error = "failed", f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.fetched_models = len(result.models)
    outcome.issues = list(result.issues)
    problem = coverage_problem(len(owned), len(result.models))
    if problem or not result.models:
        outcome.status, outcome.error = "failed", problem or "source returned no models"
        return outcome

    fetched = [
        ModelPricing(
            id=m.id,
            prices=m.prices,
            source=source.source.id,
            vendor=m.vendor,
            canonical_id=m.canonical_id,
            display_name=m.display_name,
            aliases=m.aliases,
            status=m.status,
        )
        for m in result.models
    ]
    kind = source.source.kind
    outcome.decisions = decide(
        diff_models(owned, fetched),
        provider=source.provider,
        official=kind.startswith("official"),
        issues=result.issues,
        corroborate=corroborate,
    )

    models = _merge(current, source, outcome.decisions, owned)
    sources = tuple(s for s in (current.sources if current else ()) if s.id != source.source.id) + (source.source,)
    changed = bool(outcome.applied) or current is None or (current.source(source.source.id) != source.source)
    verified_at = current.verified_at if current else None
    if changed or verified_at is None or today - verified_at >= timedelta(days=REVERIFY_DAYS):
        verified_at = today
    data = ProviderPricing(
        provider=source.provider,
        sources=sources,
        models=tuple(models),
        currency=current.currency if current else "USD",
        verified_at=verified_at,
    )
    try:
        validate_provider(data)
    except CatalogValidationError as exc:
        outcome.status, outcome.error = "failed", f"merged data is invalid, nothing written: {exc}"
        return outcome
    outcome.data = data
    if outcome.needs_review or outcome.blocking_issues:
        outcome.status = "needs_review"
    elif outcome.applied:
        outcome.status = "updated"
    return outcome


def run_sync(
    sources: Iterable[PricingSource],
    data_dir: Path,
    fetcher: Fetcher,
    *,
    today: date,
    dry_run: bool = False,
    corroborate: Optional[Corroborator] = None,
) -> List[ProviderOutcome]:
    outcomes = []
    for source in sources:
        path = data_dir / f"{source.provider}.json"
        current = load_provider_file(path) if path.exists() else None
        outcome = sync_provider(source, current, fetcher, today=today, corroborate=corroborate)
        if outcome.data is not None and outcome.status != "failed" and not dry_run:
            text = dump_provider(outcome.data)
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
                outcome.written = True
        outcomes.append(outcome)
    return outcomes
