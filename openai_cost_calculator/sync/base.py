"""Interfaces shared by every upstream pricing source."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

from ..catalog.model import ModelPricing, Source

USER_AGENT = "openai-cost-calculator-pricing-sync (+https://github.com/orkunkinay/openai_cost_calculator)"


class SourceError(RuntimeError):
    """An upstream source could not be fetched or parsed at all."""


class Fetcher(Protocol):
    def get_text(self, url: str, *, params: Optional[Mapping[str, str]] = None) -> str: ...


def get_json(fetcher: Fetcher, url: str, *, params: Optional[Mapping[str, str]] = None) -> Any:
    text = fetcher.get_text(url, params=params)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceError(f"{url} did not return valid JSON: {exc}") from exc


class HttpFetcher:
    """Real network access with timeouts, a descriptive user agent and retries."""

    def __init__(self, *, timeout: float = 60.0, attempts: int = 3, backoff: float = 2.0) -> None:
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff

    def get_text(self, url: str, *, params: Optional[Mapping[str, str]] = None) -> str:
        import requests

        last: Optional[Exception] = None
        for attempt in range(self.attempts):
            try:
                response = requests.get(
                    url, params=params, timeout=self.timeout, headers={"User-Agent": USER_AGENT}
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    time.sleep(self.backoff * (attempt + 1))
        raise SourceError(f"failed to fetch {url}: {last}")


class FixtureFetcher:
    """Serve recorded responses; used by tests and offline parser development."""

    def __init__(self, responses: Mapping[str, str]) -> None:
        self.responses = dict(responses)
        self.requested: List[str] = []

    def get_text(self, url: str, *, params: Optional[Mapping[str, str]] = None) -> str:
        key = url if not params else f"{url}?{json.dumps(dict(params), sort_keys=True)}"
        self.requested.append(key)
        for candidate in (key, url):
            if candidate in self.responses:
                return self.responses[candidate]
        raise SourceError(f"no fixture for {key}")


@dataclass(frozen=True)
class Issue:
    """Something a parser could not interpret confidently.

    Blocking issues are surfaced for human review and, when tied to a model,
    prevent that model from being auto-updated.  Non-blocking issues are notes
    about input the parser deliberately handled conservatively (for example a
    footnoted price it ignored); they appear in the report only.
    """

    message: str
    model_id: Optional[str] = None
    blocking: bool = True


@dataclass
class SourceResult:
    models: List[ModelPricing] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    def add(self, model: ModelPricing) -> None:
        self.models.append(model)

    def issue(self, message: str, model_id: Optional[str] = None) -> None:
        self.issues.append(Issue(message, model_id))

    def note(self, message: str, model_id: Optional[str] = None) -> None:
        self.issues.append(Issue(message, model_id, blocking=False))


class PricingSource(Protocol):
    """One authoritative upstream for one provider."""

    provider: str
    source: Source

    def fetch(self, fetcher: Fetcher) -> SourceResult: ...


def merge_duplicate_models(result: SourceResult) -> SourceResult:
    """Combine entries a parser emitted more than once for the same id.

    Price sets are concatenated (exact duplicates dropped, e.g. a model listed
    in two tables at the same price); aliases are unioned.  Parsers can then
    emit one entry per table row (standard, batch, flex...) without
    bookkeeping.  *Conflicting* duplicates are kept so validation rejects them.
    """
    merged: Dict[str, ModelPricing] = {}
    order: List[str] = []
    for model in result.models:
        existing = merged.get(model.id)
        if existing is None:
            merged[model.id] = model
            order.append(model.id)
            continue
        aliases: Tuple[str, ...] = tuple(dict.fromkeys(existing.aliases + model.aliases))
        merged[model.id] = ModelPricing(
            id=existing.id,
            prices=existing.prices + tuple(p for p in model.prices if p not in existing.prices),
            source=existing.source,
            vendor=existing.vendor or model.vendor,
            canonical_id=existing.canonical_id or model.canonical_id,
            display_name=existing.display_name or model.display_name,
            aliases=aliases,
            status=existing.status,
        )
    return SourceResult(models=[merged[i] for i in order], issues=list(result.issues))
