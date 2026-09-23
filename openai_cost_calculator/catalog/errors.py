"""Exception hierarchy for provider-agnostic pricing.

All errors derive from :class:`PricingError` (a ``ValueError``) so callers can
catch one type, while tests and applications can still distinguish the
failure modes that need different handling (an unknown model is usually a
configuration problem; a missing rate means the catalog cannot price a usage
dimension the provider reported).
"""

from __future__ import annotations

from typing import Iterable, Sequence


class PricingError(ValueError):
    """Base class for every provider-agnostic pricing failure."""


class CatalogValidationError(PricingError):
    """Pricing data is malformed or internally inconsistent."""


class UnknownProviderError(PricingError):
    def __init__(self, provider: str, known: Iterable[str]) -> None:
        self.provider = provider
        self.known = tuple(sorted(known))
        super().__init__(
            f"unknown provider {provider!r}; supported providers: {', '.join(self.known)}"
        )


class UnknownModelError(PricingError):
    def __init__(self, provider: str, model: str, suggestions: Sequence[str] = ()) -> None:
        self.provider = provider
        self.model = model
        self.suggestions = tuple(suggestions)
        hint = f"; did you mean: {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(f"no pricing for model {model!r} on provider {provider!r}{hint}")


class AmbiguousModelError(PricingError):
    def __init__(self, provider: str, model: str, candidates: Sequence[str]) -> None:
        self.provider = provider
        self.model = model
        self.candidates = tuple(candidates)
        super().__init__(
            f"model {model!r} on provider {provider!r} is ambiguous; "
            f"use one of: {', '.join(self.candidates)}"
        )


class PricingUnavailableError(PricingError):
    """The model is known but no price applies to the requested conditions."""


class MissingRateError(PricingError):
    """Usage includes a dimension the resolved price set does not price."""


class UsageError(PricingError):
    """Usage counts are malformed (negative, non-integer, inconsistent)."""
