"""The supported billing providers and their identifier/condition quirks."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from ..catalog.errors import UnknownProviderError
from .base import ModelHints, Preferences, ProviderSpec, generic_candidates

# --------------------------------------------------------------------------- Amazon Bedrock

#: Source region assumed for a geographic cross-region inference profile.
#: Geographic profiles bill at the source region's (regional) rate.
BEDROCK_GEO_SOURCE_REGION = {"us": "us-east-1", "eu": "eu-central-1"}
BEDROCK_DEFAULT_REGION = "us-east-1"

_BEDROCK_ARN = re.compile(
    r"^arn:aws[\w-]*:bedrock:(?P<region>[a-z0-9-]+):\d*:"
    r"(?:inference-profile|foundation-model|application-inference-profile)/(?P<id>.+)$"
)
_BEDROCK_PROFILE = re.compile(r"^(?P<geo>global|us|eu|apac|jp|au|ca|us-gov)\.(?P<rest>.+)$")
_BEDROCK_VENDOR_ID = re.compile(r"^(?P<vendor>[a-z][a-z0-9-]*)\.(?P<rest>[a-z].*)$")


def parse_bedrock_model(model: str) -> ModelHints:
    """Understand Bedrock model ids, inference profiles and ARNs.

    * ``global.anthropic.claude-...`` -> global-endpoint pricing;
    * ``us.anthropic.claude-...`` -> regional pricing of the source region;
    * ``anthropic.claude-...`` (in-region) -> regional pricing;
    * ``arn:aws:bedrock:<region>:...`` -> that region.
    """
    text = model.strip()
    arn_region: Optional[str] = None
    arn = _BEDROCK_ARN.match(text)
    if arn:
        arn_region, text = arn.group("region"), arn.group("id")

    preferences: Preferences = {}
    notes: Dict[str, str] = {}
    profile = _BEDROCK_PROFILE.match(text)
    base = profile.group("rest") if profile else text
    if profile and profile.group("geo") == "global":
        preferences["region"] = ("global",)
    elif arn_region:
        preferences["region"] = (arn_region,)
    elif profile:
        geo = profile.group("geo")
        source = BEDROCK_GEO_SOURCE_REGION.get(geo)
        if source:
            preferences["region"] = (source,)
            notes["region"] = (
                f"'{geo}.' inference profile billed at the {source} regional rate; "
                "pass region=... for another source region"
            )
    elif _BEDROCK_VENDOR_ID.match(text):
        preferences["region"] = (BEDROCK_DEFAULT_REGION, "global")
        notes["region"] = (
            f"in-region model id billed at the {BEDROCK_DEFAULT_REGION} rate; "
            "pass region=... for another region"
        )
    # Anything else (a canonical "vendor/model" id) gets the provider defaults.

    candidates: List[str] = list(generic_candidates(text))
    for candidate in generic_candidates(base):
        candidates.append(candidate)
        # anthropic.claude-sonnet-4-5 -> anthropic/claude-sonnet-4-5 (canonical form)
        vendor_id = _BEDROCK_VENDOR_ID.match(candidate)
        if vendor_id:
            candidates.append(f"{vendor_id.group('vendor')}/{vendor_id.group('rest')}")
    return ModelHints(candidates=tuple(dict.fromkeys(candidates)), preferences=preferences, notes=notes)


# --------------------------------------------------------------------------- DeepSeek


def deepseek_period(at: datetime) -> Tuple[Preferences, Mapping[str, str]]:
    """DeepSeek bills peak rates Mon-Fri 01:00-04:00 and 06:00-10:00 UTC."""
    utc = at.astimezone(timezone.utc) if at.tzinfo else at
    peak = utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)
    period = "peak" if peak else "off_peak"
    note = (
        f"{period} rate derived from request time {utc.strftime('%Y-%m-%d %H:%M')} UTC; "
        "Chinese public holidays (always off-peak) are not modelled"
    )
    return {"period": (period,)}, {"period": note}


# --------------------------------------------------------------------------- registry

PROVIDERS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        display_name="OpenAI",
        pricing_url="https://developers.openai.com/api/docs/pricing",
    ),
    ProviderSpec(
        id="anthropic",
        display_name="Anthropic (Claude API)",
        pricing_url="https://platform.claude.com/docs/en/about-claude/pricing",
        aliases=("claude",),
        notes="region='us' prices inference_geo='us' (1.1x) for models that support it.",
    ),
    ProviderSpec(
        id="gemini",
        display_name="Google Gemini API (AI Studio)",
        pricing_url="https://ai.google.dev/gemini-api/docs/pricing",
        aliases=("google-gemini", "google-ai-studio", "ai-studio"),
        strip_prefixes=("models/",),
    ),
    ProviderSpec(
        id="openrouter",
        display_name="OpenRouter",
        pricing_url="https://openrouter.ai/api/v1/models",
    ),
    ProviderSpec(
        id="azure",
        display_name="Azure OpenAI / Azure AI Foundry",
        pricing_url="https://prices.azure.com/api/retail/prices",
        aliases=("azure-openai", "azure-ai", "azure-foundry"),
        default_regions=("global",),
        notes="region is the deployment type: 'global', 'data-zone', or an Azure region for regional deployments.",
    ),
    ProviderSpec(
        id="bedrock",
        display_name="Amazon Bedrock",
        pricing_url="https://aws.amazon.com/bedrock/pricing/",
        aliases=("aws-bedrock", "amazon-bedrock", "aws"),
        default_regions=("global", BEDROCK_DEFAULT_REGION),
        parse_model=parse_bedrock_model,
        notes="Model ids, inference-profile prefixes and ARNs select global vs regional pricing.",
    ),
    ProviderSpec(
        id="vertex",
        display_name="Google Cloud Vertex AI",
        pricing_url="https://cloud.google.com/vertex-ai/generative-ai/pricing",
        aliases=("vertex-ai", "google-vertex", "gcp-vertex"),
        default_regions=("global",),
        strip_prefixes=("projects/", "publishers/google/models/", "publishers/anthropic/models/"),
        notes="region is 'global', a multi-region ('us', 'eu') or a Google Cloud region.",
    ),
    ProviderSpec(
        id="deepseek",
        display_name="DeepSeek",
        pricing_url="https://api-docs.deepseek.com/quick_start/pricing",
        # Used only when the request time is unknown (``at`` is a date): the
        # conservative choice is the higher peak rate.
        default_preferences={"period": ("peak",)},
        time_conditions=deepseek_period,
    ),
    ProviderSpec(
        id="together",
        display_name="Together AI",
        pricing_url="https://docs.together.ai/docs/serverless/models",
        aliases=("together-ai", "togetherai"),
    ),
    ProviderSpec(
        id="groq",
        display_name="Groq",
        pricing_url="https://console.groq.com/docs/models",
    ),
    ProviderSpec(
        id="fireworks",
        display_name="Fireworks AI",
        pricing_url="https://docs.fireworks.ai/serverless/pricing",
        aliases=("fireworks-ai",),
        strip_prefixes=("accounts/fireworks/models/",),
        notes="region='us' prices US-only serverless variants.",
    ),
    ProviderSpec(
        id="deepinfra",
        display_name="DeepInfra",
        pricing_url="https://deepinfra.com/pricing",
    ),
    ProviderSpec(
        id="mistral",
        display_name="Mistral AI",
        pricing_url="https://mistral.ai/pricing/api/",
        aliases=("mistral-ai", "mistralai"),
    ),
)

_BY_ID: Dict[str, ProviderSpec] = {spec.id: spec for spec in PROVIDERS}
_BY_ALIAS: Dict[str, ProviderSpec] = {}
for _spec in PROVIDERS:
    for _name in (_spec.id, *_spec.aliases):
        _BY_ALIAS[_name] = _spec


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def get_provider(name: str) -> ProviderSpec:
    """Resolve a provider id or alias (``"aws-bedrock"`` -> Bedrock)."""
    if not isinstance(name, str):
        raise UnknownProviderError(repr(name), _BY_ID)
    spec = _BY_ALIAS.get(_normalize(name))
    if spec is None:
        raise UnknownProviderError(name, _BY_ID)
    return spec


def provider_ids() -> Tuple[str, ...]:
    return tuple(_BY_ID)


def iter_providers() -> Iterable[ProviderSpec]:
    return iter(PROVIDERS)
