from __future__ import annotations

import pytest

from openai_cost_calculator.anthropic.resolve import (
    ANTHROPIC_API_UPSTREAM,
    ClaudeResolutionError,
    resolve_claude,
)


def test_direct_api_key_is_billed_estimate_via_anthropic_api():
    result = resolve_claude(env={"ANTHROPIC_API_KEY": "sk-ant-xxx"})
    assert result.auth_mode == "api-key"
    assert result.provider_category == "anthropic"
    assert result.upstream_category == "anthropic-api"
    assert result.resolved_upstream == ANTHROPIC_API_UPSTREAM
    assert result.credential_header_kind == "x-api-key"
    assert result.pricing_semantics == "billed-estimate"
    assert result.support_level == "supported"


def test_subscription_oauth_is_api_equivalent():
    result = resolve_claude(env={"CLAUDE_CODE_OAUTH_TOKEN": "oauth-xyz"})
    assert result.auth_mode == "subscription-oauth"
    assert result.credential_header_kind == "authorization-bearer"
    assert result.pricing_semantics == "api-equivalent"
    assert result.support_level == "supported"


def test_bearer_auth_token_gateway():
    result = resolve_claude(env={"ANTHROPIC_AUTH_TOKEN": "tok"})
    assert result.auth_mode == "auth-token"
    assert result.credential_header_kind == "authorization-bearer"


def test_api_key_helper_detected_when_no_env_credential():
    result = resolve_claude(env={}, api_key_helper=True)
    assert result.auth_mode == "api-key-helper"


def test_conflicting_oauth_and_api_key_are_rejected():
    with pytest.raises(ClaudeResolutionError, match="different billing domains"):
        resolve_claude(env={"CLAUDE_CODE_OAUTH_TOKEN": "o", "ANTHROPIC_API_KEY": "k"})


def test_conflicting_provider_flags_are_rejected():
    with pytest.raises(ClaudeResolutionError, match="conflicting Claude provider"):
        resolve_claude(
            env={"CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_USE_VERTEX": "1"}
        )


def test_unsupported_provider_is_bounded_not_zero_cost():
    result = resolve_claude(env={"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_API_KEY": "k"})
    assert result.provider_category == "bedrock"
    assert result.support_level == "unsupported"
    assert result.pricing_semantics == "unavailable"
    assert result.pricing_profile == "none"


def test_existing_base_url_is_chained_and_marked_override():
    result = resolve_claude(
        env={"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "https://gw.example.com/anthropic"}
    )
    assert result.explicit_override is True
    assert result.resolved_upstream == "https://gw.example.com/anthropic"
    assert result.upstream_category == "custom-gateway"
    assert result.support_level == "requires-approximation"
    assert result.pricing_semantics == "unavailable"


def test_remote_http_upstream_is_rejected_but_loopback_allowed():
    with pytest.raises(ClaudeResolutionError, match="must use HTTPS"):
        resolve_claude(env={"ANTHROPIC_API_KEY": "k"}, upstream="http://gw.example.com")
    loopback = resolve_claude(
        env={"ANTHROPIC_API_KEY": "k"}, upstream="http://127.0.0.1:9000"
    )
    assert loopback.upstream_category == "loopback"


def test_upstream_with_embedded_credentials_is_rejected():
    with pytest.raises(ClaudeResolutionError, match="must not embed credentials"):
        resolve_claude(env={"ANTHROPIC_API_KEY": "k"}, upstream="https://user:pw@gw.example.com")


def test_upstream_loop_back_to_proxy_is_rejected():
    with pytest.raises(ClaudeResolutionError, match="loop"):
        resolve_claude(
            env={"ANTHROPIC_API_KEY": "k"},
            upstream="http://127.0.0.1:8200",
            proxy_self_urls={"http://127.0.0.1:8200"},
        )
