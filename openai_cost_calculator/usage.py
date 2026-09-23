"""Convert provider response usage into normalized, disjoint :class:`Usage`.

Usage *wire formats* are orthogonal to billing providers: Azure, Groq,
Together, Fireworks, DeepInfra, Mistral and OpenRouter speak the OpenAI
schema; Claude on Bedrock (InvokeModel) and Vertex speaks Anthropic's; Gemini
on Vertex speaks Gemini's.  Extractors therefore key off the payload's shape,
and each owns one format's overlapping-counter quirks:

* OpenAI: ``prompt_tokens``/``input_tokens`` *include* cached tokens and cache
  writes; audio is a subset of the prompt/completion counts.
* DeepSeek: ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``.
* Anthropic: ``input_tokens`` *excludes* cache reads and writes; writes are
  split by TTL; web searches are counted separately.
* Gemini: ``promptTokenCount`` includes cached content; thinking tokens are
  reported apart from candidates but billed as output.
* Bedrock Converse: ``inputTokens`` excludes cache reads and writes.

Works with SDK objects (attribute access) and plain JSON dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional

from .catalog.costing import Usage
from .catalog.errors import UsageError


@dataclass(frozen=True)
class ExtractedUsage:
    usage: Usage
    format: str
    model: Optional[str] = None
    #: Cost the provider itself reported (OpenRouter), for reconciliation.
    reported_cost: Optional[Decimal] = None


_MISSING = object()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, _MISSING)
    return default if value is _MISSING else value


def _has(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj
    return obj is not None and getattr(obj, key, _MISSING) is not _MISSING and getattr(obj, key) is not None


def _int(obj: Any, *path: str) -> int:
    current = obj
    for key in path:
        current = _get(current, key)
        if current is None:
            return 0
    if isinstance(current, bool) or not isinstance(current, int):
        raise UsageError(f"usage field {'.'.join(path)!r} must be an integer, got {current!r}")
    if current < 0:
        raise UsageError(f"usage field {'.'.join(path)!r} must be non-negative")
    return current


def _subtract(total: int, part: int, what: str) -> int:
    if part > total:
        raise UsageError(f"{what} ({part}) exceeds the total it is part of ({total})")
    return total - part


def from_openai(usage: Any) -> Usage:
    """OpenAI Chat Completions or Responses usage (and OpenAI-compatible APIs)."""
    if _has(usage, "prompt_cache_hit_tokens") or _has(usage, "prompt_cache_miss_tokens"):
        return from_deepseek(usage)
    responses_api = _has(usage, "input_tokens")
    total_in = _int(usage, "input_tokens" if responses_api else "prompt_tokens")
    out = _int(usage, "output_tokens" if responses_api else "completion_tokens")
    in_details = "input_tokens_details" if responses_api else "prompt_tokens_details"
    out_details = "output_tokens_details" if responses_api else "completion_tokens_details"

    cached = _int(usage, in_details, "cached_tokens")
    written = _int(usage, in_details, "cache_write_tokens")
    audio_in = _int(usage, in_details, "audio_tokens")
    audio_out = _int(usage, out_details, "audio_tokens")

    uncached = _subtract(total_in, cached + written, "cached + cache-write tokens")
    # Audio details count all audio input (cached or not); attribute cached
    # tokens to text first, which is exact whenever no audio is cached.
    text_uncached = _subtract(uncached, min(audio_in, uncached), "audio input tokens")
    return Usage(
        input_tokens=text_uncached,
        input_audio_tokens=min(audio_in, uncached),
        cached_input_tokens=cached,
        cache_write_tokens=written,
        output_tokens=_subtract(out, audio_out, "audio output tokens"),
        output_audio_tokens=audio_out,
    )


def from_deepseek(usage: Any) -> Usage:
    hit = _int(usage, "prompt_cache_hit_tokens")
    miss = _int(usage, "prompt_cache_miss_tokens")
    total = _int(usage, "prompt_tokens")
    if total and hit + miss != total:
        miss = _subtract(total, hit, "cache-hit tokens")
    return Usage(input_tokens=miss, cached_input_tokens=hit, output_tokens=_int(usage, "completion_tokens"))


def from_anthropic(usage: Any) -> Usage:
    write_total = _int(usage, "cache_creation_input_tokens")
    write_5m = _int(usage, "cache_creation", "ephemeral_5m_input_tokens")
    write_1h = _int(usage, "cache_creation", "ephemeral_1h_input_tokens")
    # Tokens not covered by a TTL breakdown are billed at the default (5m) rate.
    write_5m += max(0, write_total - write_5m - write_1h)
    return Usage(
        input_tokens=_int(usage, "input_tokens"),
        cached_input_tokens=_int(usage, "cache_read_input_tokens"),
        cache_write_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
        output_tokens=_int(usage, "output_tokens"),
        web_search_calls=_int(usage, "server_tool_use", "web_search_requests"),
    )


def _modality_count(details: Any, modality: str) -> int:
    total = 0
    for entry in details or ():
        if str(_get(entry, "modality", "")).upper() == modality:
            total += _int(entry, "tokenCount") or _int(entry, "token_count")
    return total


def from_gemini(metadata: Any) -> Usage:
    def pick(camel: str, snake: str) -> int:
        return _int(metadata, camel) if _has(metadata, camel) else _int(metadata, snake)

    prompt = pick("promptTokenCount", "prompt_token_count") + pick("toolUsePromptTokenCount", "tool_use_prompt_token_count")
    cached = pick("cachedContentTokenCount", "cached_content_token_count")
    prompt_details = _get(metadata, "promptTokensDetails") or _get(metadata, "prompt_tokens_details")
    cache_details = _get(metadata, "cacheTokensDetails") or _get(metadata, "cache_tokens_details")
    audio_in = _modality_count(prompt_details, "AUDIO")
    audio_cached = _modality_count(cache_details, "AUDIO")
    uncached = _subtract(prompt, cached, "cached content tokens")
    audio_uncached = _subtract(audio_in, audio_cached, "cached audio tokens")
    output = pick("candidatesTokenCount", "candidates_token_count") + pick("thoughtsTokenCount", "thoughts_token_count")
    return Usage(
        input_tokens=_subtract(uncached, audio_uncached, "uncached audio tokens"),
        input_audio_tokens=audio_uncached,
        cached_input_tokens=_subtract(cached, audio_cached, "cached audio tokens"),
        cached_input_audio_tokens=audio_cached,
        output_tokens=output,
    )


def from_bedrock_converse(usage: Any) -> Usage:
    return Usage(
        input_tokens=_int(usage, "inputTokens"),
        cached_input_tokens=_int(usage, "cacheReadInputTokens"),
        cache_write_tokens=_int(usage, "cacheWriteInputTokens"),
        output_tokens=_int(usage, "outputTokens"),
    )


def _first(obj: Any, keys: Iterable[str]) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return None


def _decimal_or_none(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def extract_usage(response: Any) -> ExtractedUsage:
    """Detect the usage format of a response (object or JSON dict) and normalize it."""
    model = _first(response, ("model", "modelVersion", "model_version"))
    gemini = _first(response, ("usageMetadata", "usage_metadata"))
    if gemini is not None:
        return ExtractedUsage(from_gemini(gemini), "gemini", model)
    usage = _get(response, "usage")
    if usage is None:
        raise UsageError("response has no usage information (for streams, request usage in the final chunk)")
    if _has(usage, "inputTokens") or _has(usage, "outputTokens"):
        return ExtractedUsage(from_bedrock_converse(usage), "bedrock-converse", model)
    if (
        _has(usage, "cache_read_input_tokens")
        or _has(usage, "cache_creation_input_tokens")
        or (_get(response, "type") == "message" and _has(usage, "input_tokens"))
    ):
        return ExtractedUsage(from_anthropic(usage), "anthropic", model)
    if _has(usage, "prompt_cache_hit_tokens") or _has(usage, "prompt_cache_miss_tokens"):
        return ExtractedUsage(from_deepseek(usage), "deepseek", model)
    if _has(usage, "prompt_tokens") or _has(usage, "input_tokens"):
        return ExtractedUsage(from_openai(usage), "openai", model, _decimal_or_none(_get(usage, "cost")))
    raise UsageError("unrecognized usage format")
