"""The in-memory pricing catalog and per-provider model lookup."""

from __future__ import annotations

import difflib
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .errors import AmbiguousModelError, UnknownModelError
from .io import load_directory
from .model import ModelPricing, ProviderPricing

BUNDLED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pricing"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def compact_key(identifier: str) -> str:
    """Punctuation- and case-insensitive key: ``Claude-Sonnet-4.5`` -> ``claudesonnet45``."""
    return _NON_ALNUM.sub("", identifier.lower())


class ModelIndex:
    """Resolve identifiers to one provider's offerings.

    Lookup tries, in order: exact id or alias (case-insensitive), canonical
    id, then a punctuation-insensitive match that must be unique.  The final
    step absorbs harmless formatting differences between vendors
    (``claude-sonnet-4.5`` vs ``claude-sonnet-4-5``) without guessing between
    genuinely different models.
    """

    def __init__(self, data: ProviderPricing) -> None:
        self.provider = data.provider
        self._exact: Dict[str, ModelPricing] = {}
        self._canonical: Dict[str, List[ModelPricing]] = {}
        self._compact: Dict[str, Set[str]] = {}
        self._by_id: Dict[str, ModelPricing] = {}
        for model in data.models:
            self._by_id[model.id] = model
            for identifier in model.identifiers:
                self._exact[identifier.lower()] = model
                self._compact.setdefault(compact_key(identifier), set()).add(model.id)
            if model.canonical_id:
                canonical = model.canonical_id.lower()
                self._canonical.setdefault(canonical, []).append(model)
                bare = canonical.split("/", 1)[-1]
                for key in {compact_key(canonical), compact_key(bare)}:
                    self._compact.setdefault(key, set()).add(model.id)

    def find(self, candidates: Sequence[str]) -> Optional[ModelPricing]:
        for candidate in candidates:
            model = self._exact.get(candidate.lower())
            if model is not None:
                return model
        for candidate in candidates:
            matches = self._canonical.get(candidate.lower(), [])
            if matches:
                return self._one(candidate, matches)
        for candidate in candidates:
            ids = self._compact.get(compact_key(candidate), set())
            if ids:
                return self._one(candidate, [self._by_id[i] for i in ids])
        return None

    def _one(self, candidate: str, matches: List[ModelPricing]) -> ModelPricing:
        """Pick the single offering for ``candidate`` or raise.

        Variants often share a base model's canonical id (OpenRouter's
        ``x`` and ``x:batch``); the entry whose own identifier matches the
        query is then the intended one.
        """
        if len(matches) == 1:
            return matches[0]
        key = compact_key(candidate.split("/", 1)[-1])
        own = [m for m in matches if any(compact_key(i.split("/", 1)[-1]) == key for i in m.identifiers)]
        if len(own) == 1:
            return own[0]
        raise AmbiguousModelError(self.provider, candidate, sorted(m.id for m in matches))

    def resolve(self, model: str, candidates: Sequence[str]) -> ModelPricing:
        found = self.find(candidates)
        if found is None:
            raise UnknownModelError(self.provider, model, self.suggestions(model))
        return found

    def suggestions(self, model: str, limit: int = 5) -> List[str]:
        names = sorted(self._by_id)
        by_lower = {name.lower(): name for name in names}
        close = difflib.get_close_matches(model.lower(), list(by_lower), n=limit, cutoff=0.6)
        return [by_lower[name] for name in close]


class PricingCatalog:
    """Immutable collection of provider pricing data."""

    def __init__(self, providers: Iterable[ProviderPricing]) -> None:
        self._providers: Dict[str, ProviderPricing] = {}
        for data in providers:
            if data.provider in self._providers:
                raise ValueError(f"duplicate pricing data for provider {data.provider!r}")
            self._providers[data.provider] = data
        self._indexes: Dict[str, ModelIndex] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_directory(cls, directory: Path) -> "PricingCatalog":
        return cls(load_directory(Path(directory)))

    def provider_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._providers))

    def get(self, provider: str) -> Optional[ProviderPricing]:
        return self._providers.get(provider)

    def index(self, provider: str) -> ModelIndex:
        with self._lock:
            index = self._indexes.get(provider)
            if index is None:
                index = ModelIndex(self._providers[provider])
                self._indexes[provider] = index
            return index


_BUNDLED: Optional[PricingCatalog] = None
_BUNDLED_LOCK = threading.Lock()


def bundled_catalog() -> PricingCatalog:
    """The catalog shipped with the package (loaded and validated once)."""
    global _BUNDLED
    with _BUNDLED_LOCK:
        if _BUNDLED is None:
            _BUNDLED = PricingCatalog.from_directory(BUNDLED_DATA_DIR)
        return _BUNDLED
