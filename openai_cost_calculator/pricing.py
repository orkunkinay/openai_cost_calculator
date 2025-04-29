"""
Remote CSV → in-memory dict  (with a tiny 24-hour cache).
The CSV **no longer** has a Token-Type column – every row is Text.
"""

from __future__ import annotations

import csv
import io
import time
from typing import Dict, Tuple

import requests


_PRICING_CSV_URL = (
    "https://raw.githubusercontent.com/orkunkinay/openai_api_data/"
    "refs/heads/main/gpt_pricing_data.csv"
)
_CACHE: Dict[Tuple[str, str], dict] | None = None
_CACHE_TS = 0
_TTL = 60 * 60 * 24  # 24h


def _fetch_csv() -> Dict[Tuple[str, str], dict]:
    resp = requests.get(_PRICING_CSV_URL, timeout=5)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    data = {}
    for row in reader:
        key = (row["Model Name"], row["Model Date"])
        data[key] = {
            "input_price": float(row["Input Price"]),
            "cached_input_price": float(row["Cached Input Price"] or 0)
            or None,
            "output_price": float(row["Output Price"]),
        }
    return data


def load_pricing() -> Dict[Tuple[str, str], dict]:
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE is None or now - _CACHE_TS > _TTL:
        _CACHE = _fetch_csv()
        _CACHE_TS = now
    return _CACHE


# Convenience for users who want a manual refresh
def refresh_pricing() -> None:
    global _CACHE, _CACHE_TS
    _CACHE = _fetch_csv()
    _CACHE_TS = time.time()
