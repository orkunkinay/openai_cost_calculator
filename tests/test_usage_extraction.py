"""Contract tests: provider usage payloads -> disjoint normalized Usage.

Payload shapes follow each provider's documented response schema.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from openai_cost_calculator.catalog import Usage, UsageError
from openai_cost_calculator.usage import extract_usage


def test_openai_chat_completions_subtracts_cached_and_audio():
    payload = {
        "model": "gpt-4o-2024-08-06",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 50, "audio_tokens": 40},
        },
    }
    extracted = extract_usage(payload)
    assert extracted.format == "openai" and extracted.model == "gpt-4o-2024-08-06"
    assert extracted.usage == Usage(
        input_tokens=700,
        input_audio_tokens=100,
        cached_input_tokens=200,
        output_tokens=210,
        reasoning_tokens=50,
        output_audio_tokens=40,
    )


def test_openai_responses_api_objects_with_cache_writes():
    response = NS(
        model="gpt-6-sol",
        usage=NS(
            input_tokens=10_000,
            output_tokens=500,
            input_tokens_details=NS(cached_tokens=6_000, cache_write_tokens=3_000),
            output_tokens_details=NS(reasoning_tokens=200),
        ),
    )
    usage = extract_usage(response).usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens) == (1_000, 6_000, 3_000)
    assert (usage.output_tokens, usage.reasoning_tokens) == (300, 200)
    assert usage.total_input_tokens == 10_000


def test_openrouter_reported_cost_is_preserved():
    extracted = extract_usage(
        {
            "model": "anthropic/claude-sonnet-4.5",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.000105},
        }
    )
    assert extracted.reported_cost == Decimal("0.000105")


def test_deepseek_cache_hit_and_miss():
    usage = extract_usage(
        {
            "model": "deepseek-flash",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 36,
            },
        }
    )
    assert usage.format == "deepseek"
    assert usage.usage == Usage(input_tokens=36, cached_input_tokens=64, output_tokens=7)


def test_anthropic_messages_cache_ttl_split_and_web_search():
    payload = {
        "type": "message",
        "model": "claude-sonnet-4-5-20250929",
        "usage": {
            "input_tokens": 50,
            "cache_read_input_tokens": 1_000,
            "cache_creation_input_tokens": 700,
            "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 400},
            "output_tokens": 90,
            "server_tool_use": {"web_search_requests": 2},
        },
    }
    extracted = extract_usage(payload)
    assert extracted.format == "anthropic"
    # 100 unattributed write tokens are billed at the default 5-minute rate.
    assert extracted.usage == Usage(
        input_tokens=50,
        cached_input_tokens=1_000,
        cache_write_tokens=300,
        cache_write_1h_tokens=400,
        output_tokens=90,
        web_search_calls=2,
    )


def test_gemini_usage_metadata_reports_thoughts_and_splits_audio():
    payload = {
        "modelVersion": "gemini-2.5-flash",
        "usageMetadata": {
            "promptTokenCount": 1_000,
            "cachedContentTokenCount": 400,
            "candidatesTokenCount": 120,
            "thoughtsTokenCount": 80,
            "toolUsePromptTokenCount": 10,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 700}, {"modality": "AUDIO", "tokenCount": 300}],
            "cacheTokensDetails": [{"modality": "AUDIO", "tokenCount": 100}, {"modality": "TEXT", "tokenCount": 300}],
        },
    }
    extracted = extract_usage(payload)
    assert extracted.format == "gemini" and extracted.model == "gemini-2.5-flash"
    assert extracted.usage == Usage(
        input_tokens=410,
        input_audio_tokens=200,
        cached_input_tokens=300,
        cached_input_audio_tokens=100,
        output_tokens=120,
        reasoning_tokens=80,
    )


def test_bedrock_converse_usage():
    extracted = extract_usage(
        {"usage": {"inputTokens": 20, "outputTokens": 5, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 30}}
    )
    assert extracted.format == "bedrock-converse"
    assert extracted.usage == Usage(input_tokens=20, cached_input_tokens=100, cache_write_tokens=30, output_tokens=5)


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"model": "x"}, "no usage"),
        (
            {"usage": {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 11}}},
            "exceeds",
        ),
        ({"usage": {"prompt_tokens": -1, "completion_tokens": 1}}, "non-negative"),
        ({"usage": {"prompt_tokens": "10", "completion_tokens": 1}}, "integer"),
        ({"usage": {"weird": 1}}, "unrecognized"),
    ],
)
def test_malformed_usage_is_rejected_with_a_clear_error(payload, message):
    with pytest.raises(UsageError, match=message):
        extract_usage(payload)
