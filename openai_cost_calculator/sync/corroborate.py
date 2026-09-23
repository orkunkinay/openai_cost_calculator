"""Independent cross-check against LiteLLM's community pricing table.

The original checker treated LiteLLM's JSON as the source of truth.  It is
now used only as a *second opinion*: an official source remains
authoritative, but when a parsed price changes and LiteLLM still reports the
old value, the change is held for review (see :mod:`.policy`).  This catches
parser regressions - a misread column looks exactly like a price change -
without letting a community table overwrite official data.

Matching is deliberately strict: a model is corroborated only when one of its
identifiers matches a LiteLLM key for the same provider exactly (ignoring
punctuation), so a near-miss never produces false confidence.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Tuple

from ..catalog.catalog import compact_key
from ..catalog.model import ModelPricing
from .base import Fetcher, get_json

URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

#: catalog provider -> (LiteLLM ``litellm_provider`` values, key prefix)
PROVIDERS: Mapping[str, Tuple[Tuple[str, ...], str]] = {
    "openai": (("openai",), ""),
    "anthropic": (("anthropic",), ""),
    "gemini": (("gemini",), "gemini/"),
    "deepseek": (("deepseek",), "deepseek/"),
    "mistral": (("mistral",), "mistral/"),
    "groq": (("groq",), "groq/"),
    "together": (("together_ai",), "together_ai/"),
    "fireworks": (("fireworks_ai",), "fireworks_ai/"),
    "deepinfra": (("deepinfra",), "deepinfra/"),
    "openrouter": (("openrouter",), "openrouter/"),
}
_FIELDS = {
    "input_cost_per_token": "input",
    "output_cost_per_token": "output",
    "cache_read_input_token_cost": "cached_input",
}
_PER_MILLION = Decimal(1_000_000)


class LiteLLMCorroborator:
    def __init__(self, table: Mapping[str, Any]) -> None:
        self._index: Dict[Tuple[str, str], Dict[str, Decimal]] = {}
        for provider, (litellm_providers, prefix) in PROVIDERS.items():
            for key, entry in table.items():
                if not isinstance(entry, dict) or entry.get("litellm_provider") not in litellm_providers:
                    continue
                name = key[len(prefix) :] if prefix and key.startswith(prefix) else key
                rates = self._rates(entry)
                if rates:
                    self._index[(provider, compact_key(name))] = rates

    @staticmethod
    def _rates(entry: Mapping[str, Any]) -> Dict[str, Decimal]:
        rates: Dict[str, Decimal] = {}
        for field, dimension in _FIELDS.items():
            value = entry.get(field)
            if value is None or isinstance(value, bool):
                continue
            try:
                rates[dimension] = (Decimal(str(value)) * _PER_MILLION).normalize()
            except InvalidOperation:
                continue
        return rates

    @classmethod
    def fetch(cls, fetcher: Fetcher) -> "LiteLLMCorroborator":
        return cls(get_json(fetcher, URL))

    def __call__(self, provider: str, model: ModelPricing) -> Optional[Dict[str, Decimal]]:
        for identifier in model.identifiers:
            found = self._index.get((provider, compact_key(identifier)))
            if found:
                return found
        return None
