"""One module per upstream pricing source, and the provider -> source registry.

Adding a provider means adding a source here (see ``docs/ADDING_A_PROVIDER.md``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..base import PricingSource
from .anthropic import AnthropicSource
from .azure import AzureSource
from .bedrock import BedrockSource
from .deepinfra import DeepInfraSource
from .deepseek import DeepSeekSource
from .gemini import GeminiSource
from .hosts import FireworksSource, GroqSource, TogetherSource
from .mistral import MistralSource
from .openai import OpenAISource
from .openrouter import OpenRouterSource
from .vertex import VertexSource


def all_sources() -> List[PricingSource]:
    return [
        OpenAISource(),
        AnthropicSource(),
        GeminiSource(),
        OpenRouterSource(),
        AzureSource(),
        BedrockSource(),
        VertexSource(),
        DeepSeekSource(),
        TogetherSource(),
        GroqSource(),
        FireworksSource(),
        DeepInfraSource(),
        MistralSource(),
    ]


def sources_for(providers: Sequence[str] = ()) -> List[PricingSource]:
    """Sources for the given provider ids (all when empty)."""
    registry: Dict[str, PricingSource] = {s.provider: s for s in all_sources()}
    if not providers:
        return list(registry.values())
    unknown = [p for p in providers if p not in registry]
    if unknown:
        raise KeyError(f"no pricing source for {unknown}; known: {sorted(registry)}")
    return [registry[p] for p in providers]


__all__ = ["all_sources", "sources_for"]
