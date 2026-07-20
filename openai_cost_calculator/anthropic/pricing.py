"""Anthropic first-party token pricing.

Prices are expressed in USD per one million tokens and stored as :class:`Decimal`
so that all downstream arithmetic is exact.  Each model maps to one or more
*dated* rows; a row may additionally declare a ``min_input_tokens`` threshold so
that long-context tiers (for example Sonnet above 200K input tokens) can be
priced differently from the base tier.

Anthropic separates four billable token categories:

``input``
    Uncached input tokens (Anthropic's ``input_tokens`` field already excludes
    cache reads and cache writes, so no subtraction is required).
``cache_write_5m`` / ``cache_write_1h``
    Cache-creation writes, priced by their time-to-live.  Anthropic bills these
    at 1.25x and 2x the base input rate respectively.
``cache_read``
    Cache-read (hit) input tokens, billed at 0.1x the base input rate.
``output``
    Generated output tokens.

The table is intentionally self-contained rather than derived from the OpenAI
pricing CSV, whose schema cannot express cache-write TTLs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional


MILLION = Decimal(1_000_000)

# Anthropic cache-pricing multipliers relative to the base input rate.
_CACHE_WRITE_5M_MULTIPLIER = Decimal("1.25")
_CACHE_WRITE_1H_MULTIPLIER = Decimal("2")
_CACHE_READ_MULTIPLIER = Decimal("0.1")

_DATED_MODEL_RE = re.compile(r"^(?P<name>.+)-(?P<date>\d{8})$")


class AnthropicPricingError(ValueError):
    """Raised when Anthropic pricing data is invalid or cannot be resolved."""


@dataclass(frozen=True)
class AnthropicRate:
    """Per-million-token prices for one Anthropic pricing tier."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    min_input_tokens: int = 0

    def validate(self, *, model: str, date: str) -> None:
        for name in ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise AnthropicPricingError(
                    f"{model} ({date}) rate {name!r} must be a Decimal"
                )
            if not value.is_finite() or value < 0:
                raise AnthropicPricingError(
                    f"{model} ({date}) rate {name!r} must be finite and non-negative"
                )
        if (
            not isinstance(self.min_input_tokens, int)
            or isinstance(self.min_input_tokens, bool)
            or self.min_input_tokens < 0
        ):
            raise AnthropicPricingError(
                f"{model} ({date}) min_input_tokens must be a non-negative integer"
            )


@dataclass(frozen=True)
class _PriceRow:
    date: str
    rate: AnthropicRate


def _rate(
    input_price: str,
    output_price: str,
    cache_read: str,
    *,
    cache_write_5m: Optional[str] = None,
    cache_write_1h: Optional[str] = None,
    min_input_tokens: int = 0,
) -> AnthropicRate:
    base = Decimal(input_price)
    return AnthropicRate(
        input=base,
        output=Decimal(output_price),
        cache_read=Decimal(cache_read),
        cache_write_5m=(
            Decimal(cache_write_5m)
            if cache_write_5m is not None
            else base * _CACHE_WRITE_5M_MULTIPLIER
        ),
        cache_write_1h=(
            Decimal(cache_write_1h)
            if cache_write_1h is not None
            else base * _CACHE_WRITE_1H_MULTIPLIER
        ),
        min_input_tokens=min_input_tokens,
    )


def _base_family(input_price: str, output_price: str, cache_read: str) -> list[AnthropicRate]:
    return [_rate(input_price, output_price, cache_read)]


# Effective date shared by the current published rows.  Undated model aliases
# (for example ``claude-opus-4-8``) resolve to the newest row whose effective
# date is on or before the request date.
_EFFECTIVE = "2026-06-20"

_RATES: dict[str, list[_PriceRow]] = {}


def _register(model: str, date: str, rates: list[AnthropicRate]) -> None:
    rows = _RATES.setdefault(model, [])
    for rate in rates:
        rows.append(_PriceRow(date=date, rate=rate))


# Opus 4.5/4.6/4.7/4.8 share the current $5 / $25 tier.
for _model in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5"):
    _register(_model, _EFFECTIVE, _base_family("5", "25", "0.5"))

# Earlier Opus generations remain at the $15 / $75 tier.
for _model in ("claude-opus-4-1", "claude-opus-4"):
    _register(_model, _EFFECTIVE, _base_family("15", "75", "1.5"))

# Sonnet 5 and Sonnet 4.x: base tier plus a long-context tier above 200K input.
for _model in ("claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4"):
    _register(
        _model,
        _EFFECTIVE,
        [
            _rate("3", "15", "0.3", min_input_tokens=0),
            _rate("6", "22.5", "0.6", min_input_tokens=200_001),
        ],
    )

_register("claude-haiku-4-5", _EFFECTIVE, _base_family("1", "5", "0.1"))
_register("claude-fable-5", _EFFECTIVE, _base_family("10", "50", "1"))
_register("claude-mythos-5", _EFFECTIVE, _base_family("10", "50", "1"))


def validate_anthropic_pricing() -> int:
    """Validate every pricing row; returns the number of (model, date) tiers."""
    tiers = 0
    for model, rows in _RATES.items():
        if not rows:
            raise AnthropicPricingError(f"{model} has no pricing rows")
        by_date: dict[str, list[int]] = {}
        for row in rows:
            _validate_date(row.date, model)
            row.rate.validate(model=model, date=row.date)
            thresholds = by_date.setdefault(row.date, [])
            if row.rate.min_input_tokens in thresholds:
                raise AnthropicPricingError(
                    f"{model} ({row.date}) has duplicate tier "
                    f"min_input_tokens={row.rate.min_input_tokens}"
                )
            thresholds.append(row.rate.min_input_tokens)
            tiers += 1
        for date, thresholds in by_date.items():
            if 0 not in thresholds:
                raise AnthropicPricingError(
                    f"{model} ({date}) has no base tier (min_input_tokens=0)"
                )
    return tiers


def _validate_date(date: str, model: str) -> None:
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise AnthropicPricingError(
            f"{model} pricing date must be 'YYYY-MM-DD', got {date!r}"
        ) from exc
    if parsed.strftime("%Y-%m-%d") != date:
        raise AnthropicPricingError(
            f"{model} pricing date must be 'YYYY-MM-DD', got {date!r}"
        )


def split_anthropic_model(model: str) -> tuple[str, str]:
    """Split an Anthropic model id into ``(name, request_date)``.

    Accepts undated aliases (``claude-opus-4-8``) and dated ids using
    Anthropic's ``-YYYYMMDD`` suffix (``claude-sonnet-4-5-20250929``).  When no
    date is embedded, today's UTC date is used so that undated aliases resolve
    to the newest applicable row.
    """
    if not isinstance(model, str) or not model:
        raise AnthropicPricingError("model must be a non-empty string")
    match = _DATED_MODEL_RE.match(model)
    if match:
        digits = match.group("date")
        try:
            parsed = datetime.strptime(digits, "%Y%m%d")
        except ValueError:
            parsed = None
        if parsed is not None:
            return match.group("name"), parsed.strftime("%Y-%m-%d")
    return model, datetime.now(timezone.utc).strftime("%Y-%m-%d")


def resolve_anthropic_rate(model: str, total_input_tokens: int) -> AnthropicRate:
    """Resolve the applicable rate for a model and request input-token count.

    ``total_input_tokens`` should be the sum of uncached input, cache-read, and
    cache-creation tokens, matching how Anthropic evaluates long-context tiers.
    """
    if not isinstance(total_input_tokens, int) or isinstance(total_input_tokens, bool):
        raise AnthropicPricingError("total_input_tokens must be an integer")
    if total_input_tokens < 0:
        raise AnthropicPricingError("total_input_tokens must be non-negative")

    name, _snapshot_date = split_anthropic_model(model)
    rows = _RATES.get(name)
    if not rows:
        raise AnthropicPricingError(f"no Anthropic pricing for model {name!r}")

    # Pricing is selected by the pricing row's effective date relative to today,
    # not by the model id's release-date suffix (a dated model id such as
    # ``claude-sonnet-4-5-20250929`` is still billed at the current rate).  If no
    # row is effective yet (rows dated in the future relative to the clock), fall
    # back to the model's rows so a known model always resolves.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    applicable_dates = [row.date for row in rows if row.date <= today]
    newest = max(applicable_dates) if applicable_dates else min(row.date for row in rows)
    tiers = sorted(
        (row.rate for row in rows if row.date == newest),
        key=lambda rate: rate.min_input_tokens,
    )
    selected = tiers[0]
    for tier in tiers:
        if tier.min_input_tokens <= total_input_tokens:
            selected = tier
        else:
            break
    return selected


# Fail fast at import time so that a malformed table never reaches production.
validate_anthropic_pricing()
