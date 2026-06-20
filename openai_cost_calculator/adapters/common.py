from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional


MONEY = "\U0001f4b0"
SEP = "\u00b7"


def decimal_from(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def int_from(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def compact_tokens(tokens: int) -> str:
    if abs(tokens) >= 1_000_000:
        value = Decimal(tokens) / Decimal("1000000")
        return f"{value:.1f}M"
    if abs(tokens) >= 1_000:
        value = Decimal(tokens) / Decimal("1000")
        return f"{value:.1f}k"
    return str(tokens)


def format_money(value: Decimal, places: str = "0.0001") -> str:
    return f"${value.quantize(Decimal(places))}"


def nested(data: Any, *keys: str) -> Optional[Any]:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

