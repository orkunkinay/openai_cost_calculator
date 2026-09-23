"""Billing-provider specifications (identifier formats, default conditions)."""

from .base import ModelHints, ProviderSpec, generic_candidates
from .registry import PROVIDERS, get_provider, iter_providers, provider_ids

__all__ = [
    "PROVIDERS",
    "ModelHints",
    "ProviderSpec",
    "generic_candidates",
    "get_provider",
    "iter_providers",
    "provider_ids",
]
