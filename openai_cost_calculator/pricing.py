"""
Remote CSV → in-memory dict (with a tiny 24-hour cache).

Pricing CSV may contain tiered rows per (model, date) using `Minimum Tokens`.
If the column is missing, rows are treated as a single tier with minimum 0.
"""

from __future__ import annotations

import csv
import io
import time
import threading
from typing import Dict, Tuple, Iterable, Mapping, Optional, List
import requests

_PRICING_CSV_URL = (
    "https://raw.githubusercontent.com/orkunkinay/openai_cost_calculator/refs/heads/main/data/gpt_pricing_data.csv"
)

# Existing cache for remote CSV
_CACHE: Dict[Tuple[str, str], List[dict]] | None = None
_CACHE_TS = 0
_TTL = 60 * 60 * 24  # 24h

# local, in-process overrides (tiered by minimum_tokens)
# Keys are (model_name, model_date) with YYYY-MM-DD dates.
_LOCAL_OVERRIDES: Dict[Tuple[str, str], Dict[int, dict]] = {}
_LOCK = threading.RLock()
_OFFLINE_ONLY = False  # if True, never fetch remote CSV

def _validate_date_str(date: str) -> None:
    # very lightweight guard
    if not isinstance(date, str) or len(date) != 10:
        raise ValueError("model_date must be 'YYYY-MM-DD'")
    y, m, d = date.split("-")
    if not (y.isdigit() and m.isdigit() and d.isdigit()):
        raise ValueError("model_date must be 'YYYY-MM-DD'")

def _normalize_row(
    input_price: float,
    output_price: float,
    cached_input_price: Optional[float],
    *,
    minimum_tokens: int = 0,
) -> dict:
    if input_price < 0 or output_price < 0:
        raise ValueError("Prices must be non-negative")
    if not isinstance(minimum_tokens, int) or isinstance(minimum_tokens, bool) or minimum_tokens < 0:
        raise ValueError("minimum_tokens must be a non-negative integer")
    row = {
        "input_price": float(input_price),
        "output_price": float(output_price),
        "cached_input_price": float(cached_input_price) if cached_input_price not in (None, 0) else None,
        "minimum_tokens": int(minimum_tokens),
    }
    return row


def _coerce_minimum_tokens(raw_value: str | None) -> int:
    if raw_value is None:
        return 0
    s = str(raw_value).strip()
    if s == "":
        return 0
    value = int(s)
    if value < 0:
        raise ValueError("Minimum Tokens must be non-negative")
    return value


def _sorted_tiers(min_to_row: Mapping[int, dict]) -> List[dict]:
    return [dict(min_to_row[min_tokens]) for min_tokens in sorted(min_to_row.keys())]

def add_pricing_entry(
    model_name: str,
    model_date: str,
    *,
    input_price: float,
    output_price: float,
    cached_input_price: Optional[float] = None,
    minimum_tokens: int = 0,
    replace: bool = True,
) -> None:
    """
    Register or override a single pricing row that will be used by `load_pricing()`.
    Users call this at process start, before they estimate costs.

    Example:
        add_pricing_entry(
            "gpt-4o-mini", "2025-08-01",
            input_price=0.20, output_price=0.60, cached_input_price=0.04
        )
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name must be a non-empty string")
    _validate_date_str(model_date)
    row = _normalize_row(
        input_price,
        output_price,
        cached_input_price,
        minimum_tokens=minimum_tokens,
    )

    with _LOCK:
        key = (model_name, model_date)
        by_min = _LOCAL_OVERRIDES.setdefault(key, {})
        if not replace and minimum_tokens in by_min:
            raise KeyError(
                f"Pricing already exists for {key} at minimum_tokens={minimum_tokens}; "
                "set replace=True to override."
            )
        by_min[minimum_tokens] = row

def add_pricing_entries(
    entries: Iterable[tuple],
    *,
    replace: bool = True,
) -> None:
    """
    Bulk add. Supported tuple formats:
      - (model_name, model_date, input_price, output_price, cached_input_price)
      - (model_name, model_date, input_price, output_price, cached_input_price, minimum_tokens)
    """
    with _LOCK:
        for entry in entries:
            if len(entry) == 5:
                model_name, model_date, ip, op, cip = entry
                minimum_tokens = 0
            elif len(entry) == 6:
                model_name, model_date, ip, op, cip, minimum_tokens = entry
            else:
                raise ValueError(
                    "Each entry must be a 5-tuple or 6-tuple: "
                    "(model_name, model_date, input_price, output_price, "
                    "cached_input_price[, minimum_tokens])"
                )
            _validate_date_str(model_date)
            row = _normalize_row(ip, op, cip, minimum_tokens=minimum_tokens)
            key = (model_name, model_date)
            by_min = _LOCAL_OVERRIDES.setdefault(key, {})
            if not replace and minimum_tokens in by_min:
                raise KeyError(
                    f"Pricing already exists for {key} at minimum_tokens={minimum_tokens}; "
                    "set replace=True to override."
                )
            by_min[minimum_tokens] = row

def clear_local_pricing() -> None:
    """Remove all user-added overrides (remote CSV remains unaffected)."""
    with _LOCK:
        _LOCAL_OVERRIDES.clear()

def set_offline_mode(offline: bool = True) -> None:
    """
    If True, `load_pricing()` will NEVER fetch remote CSV—only local overrides are used.
    Useful for air-gapped or pinned environments.
    """
    global _OFFLINE_ONLY
    with _LOCK:
        _OFFLINE_ONLY = bool(offline)

# remote fetch
def _fetch_csv() -> Dict[Tuple[str, str], List[dict]]:
    resp = requests.get(_PRICING_CSV_URL, timeout=5)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    data: Dict[Tuple[str, str], Dict[int, dict]] = {}
    for row in reader:
        key = (row["Model Name"], row["Model Date"])
        minimum_tokens = _coerce_minimum_tokens(row.get("Minimum Tokens"))
        cached_raw = (row.get("Cached Input Price") or "").strip()
        cached_input_price = float(cached_raw) if cached_raw else None
        parsed = _normalize_row(
            input_price=float((row.get("Input Price") or "").strip()),
            output_price=float((row.get("Output Price") or "").strip()),
            cached_input_price=cached_input_price,
            minimum_tokens=minimum_tokens,
        )
        data.setdefault(key, {})[minimum_tokens] = parsed
    return {key: _sorted_tiers(by_min) for key, by_min in data.items()}


def load_pricing_tiered() -> Dict[Tuple[str, str], List[dict]]:
    """
    Authoritative tiered pricing map:
        - Remote CSV (cached ~24h) unless offline mode is enabled.
        - PLUS user overrides, where each (key, minimum_tokens) override wins.
    """
    global _CACHE, _CACHE_TS
    base: Dict[Tuple[str, str], List[dict]] = {}

    if not _OFFLINE_ONLY:
        now = time.time()
        if _CACHE is None or (now - _CACHE_TS) > _TTL:
            _CACHE = _fetch_csv()
            _CACHE_TS = now
        base = {key: [dict(tier) for tier in tiers] for key, tiers in _CACHE.items()}

    with _LOCK:
        for key, local_by_min in _LOCAL_OVERRIDES.items():
            merged_by_min = {int(t["minimum_tokens"]): dict(t) for t in base.get(key, [])}
            for min_tokens, local_row in local_by_min.items():
                merged_by_min[int(min_tokens)] = dict(local_row)
            base[key] = _sorted_tiers(merged_by_min)

    return base

def load_pricing() -> Dict[Tuple[str, str], dict]:
    """
    Backward-compatible flat pricing view.

    Returns one row per (model, date) by selecting the lowest tier (`minimum_tokens=0` if present).
    For tier-aware resolution, use `load_pricing_tiered()`.
    """
    tiered = load_pricing_tiered()
    flat: Dict[Tuple[str, str], dict] = {}
    for key, tiers in tiered.items():
        if not tiers:
            continue
        row = dict(tiers[0])
        row.pop("minimum_tokens", None)
        flat[key] = row
    return flat

def refresh_pricing() -> None:
    """
    Refresh remote CSV cache immediately. Local overrides are preserved.
    In offline mode this is a no-op for the remote side.
    """
    global _CACHE, _CACHE_TS
    if _OFFLINE_ONLY:
        return
    _CACHE = _fetch_csv()
    _CACHE_TS = time.time()
