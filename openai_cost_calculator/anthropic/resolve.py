"""Centralized Claude authentication and upstream resolution.

Claude Code can authenticate and route in many ways (subscription OAuth, direct
Anthropic API key, bearer-token gateways, Amazon Bedrock, Google Vertex,
Microsoft Foundry, custom Anthropic-format gateways).  This module resolves an
observed environment into a single typed :class:`ClaudeResolution` that the
installer, proxy, CLI, and self-test all share, so authentication precedence and
provider support live in exactly one place.

The resolver never reads, stores, or returns credential *values* — only their
kind (for example ``x-api-key`` versus ``authorization-bearer``).  It is a pure
function of its inputs and performs no network or filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Optional
from urllib.parse import urlparse


ANTHROPIC_API_UPSTREAM = "https://api.anthropic.com"

# Recognised but not natively priced/proxied provider modes.
_PROVIDER_FLAGS = {
    "CLAUDE_CODE_USE_BEDROCK": "bedrock",
    "CLAUDE_CODE_USE_VERTEX": "vertex",
    "CLAUDE_CODE_USE_FOUNDRY": "foundry",
}

_FALSEY = {"", "0", "false", "no", "off"}


class ClaudeResolutionError(ValueError):
    """Raised when Claude authentication and upstream settings are unusable."""


@dataclass(frozen=True)
class ClaudeResolution:
    auth_mode: str
    provider_category: str
    upstream_category: str
    resolved_upstream: str
    explicit_override: bool
    credential_header_kind: str
    pricing_profile: str
    pricing_semantics: str
    support_level: str
    detection_source: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() not in _FALSEY


def _detect_provider(env: Mapping[str, str]) -> tuple[str, str]:
    active = [
        category for flag, category in _PROVIDER_FLAGS.items() if _truthy(env.get(flag))
    ]
    if len(active) > 1:
        raise ClaudeResolutionError(
            "conflicting Claude provider flags are set; enable at most one of "
            "Bedrock, Vertex, or Foundry"
        )
    if active:
        return active[0], f"{active[0]} provider flag"
    return "anthropic", "default Anthropic provider"


def _detect_auth(env: Mapping[str, str], *, api_key_helper: bool) -> tuple[str, str]:
    has_oauth = _truthy(env.get("CLAUDE_CODE_OAUTH_TOKEN"))
    has_api_key = _truthy(env.get("ANTHROPIC_API_KEY"))
    has_auth_token = _truthy(env.get("ANTHROPIC_AUTH_TOKEN"))

    if has_oauth and has_api_key:
        raise ClaudeResolutionError(
            "both CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_API_KEY are set; these "
            "belong to different billing domains and must not be mixed"
        )
    if has_oauth:
        return "subscription-oauth", "CLAUDE_CODE_OAUTH_TOKEN present"
    if has_api_key:
        return "api-key", "ANTHROPIC_API_KEY present"
    if has_auth_token:
        return "auth-token", "ANTHROPIC_AUTH_TOKEN present"
    if api_key_helper:
        return "api-key-helper", "settings apiKeyHelper configured"
    return "unspecified", "no Claude credential detected"


def _classify_upstream(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"api.anthropic.com"}:
        return "anthropic-api"
    if parsed.hostname in {"127.0.0.1", "::1", "localhost"}:
        return "loopback"
    return "custom-gateway"


def _normalize_upstream(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaudeResolutionError("upstream override must be a non-empty HTTP(S) URL")
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ClaudeResolutionError("upstream override must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ClaudeResolutionError("upstream override must not embed credentials")
    is_loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme == "http" and not is_loopback:
        raise ClaudeResolutionError(
            "a remote upstream must use HTTPS; plain HTTP is only permitted for "
            "loopback development endpoints"
        )
    return normalized


def resolve_claude(
    *,
    env: Optional[Mapping[str, str]] = None,
    upstream: Optional[str] = None,
    api_key_helper: bool = False,
    proxy_self_urls: Optional[set[str]] = None,
) -> ClaudeResolution:
    """Resolve Claude authentication and upstream routing from ``env``.

    ``upstream`` is an explicit override or an existing ``ANTHROPIC_BASE_URL``
    to chain behind.  ``proxy_self_urls`` are addresses the local proxy binds to,
    used to reject an override that would loop back onto the proxy.
    """
    env = env or {}
    provider_category, provider_source = _detect_provider(env)
    auth_mode, auth_source = _detect_auth(env, api_key_helper=api_key_helper)

    explicit_override = upstream is not None
    if not explicit_override and _truthy(env.get("ANTHROPIC_BASE_URL")):
        upstream = env["ANTHROPIC_BASE_URL"]
        explicit_override = True

    resolved = _normalize_upstream(upstream) if upstream else ANTHROPIC_API_UPSTREAM
    upstream_category = _classify_upstream(resolved)

    if proxy_self_urls and resolved.rstrip("/") in {u.rstrip("/") for u in proxy_self_urls}:
        raise ClaudeResolutionError(
            "upstream override points back at the local proxy, which would loop"
        )

    credential_header_kind = {
        "api-key": "x-api-key",
        "subscription-oauth": "authorization-bearer",
        "auth-token": "authorization-bearer",
        "api-key-helper": "x-api-key",
    }.get(auth_mode, "none")

    # Provider and pricing semantics.
    if provider_category != "anthropic":
        return ClaudeResolution(
            auth_mode=auth_mode,
            provider_category=provider_category,
            upstream_category=upstream_category,
            resolved_upstream=resolved,
            explicit_override=explicit_override,
            credential_header_kind=credential_header_kind,
            pricing_profile="none",
            pricing_semantics="unavailable",
            support_level="unsupported",
            detection_source=f"{provider_source}; {auth_source}",
        )

    if upstream_category == "custom-gateway":
        # An Anthropic-format gateway may re-price or re-route; without a known
        # pricing profile cost is reported as unavailable rather than assumed.
        pricing_profile = "none"
        pricing_semantics = "unavailable"
        support_level = "requires-approximation"
    elif auth_mode == "subscription-oauth":
        pricing_profile = "anthropic-first-party"
        pricing_semantics = "api-equivalent"
        support_level = "supported"
    elif auth_mode in {"api-key", "auth-token", "api-key-helper"}:
        pricing_profile = "anthropic-first-party"
        pricing_semantics = "billed-estimate"
        support_level = "supported"
    else:
        pricing_profile = "anthropic-first-party"
        pricing_semantics = "billed-estimate"
        support_level = "supported"

    return ClaudeResolution(
        auth_mode=auth_mode,
        provider_category=provider_category,
        upstream_category=upstream_category,
        resolved_upstream=resolved,
        explicit_override=explicit_override,
        credential_header_kind=credential_header_kind,
        pricing_profile=pricing_profile,
        pricing_semantics=pricing_semantics,
        support_level=support_level,
        detection_source=f"{provider_source}; {auth_source}",
    )


__all__ = [
    "ANTHROPIC_API_UPSTREAM",
    "ClaudeResolution",
    "ClaudeResolutionError",
    "resolve_claude",
]
