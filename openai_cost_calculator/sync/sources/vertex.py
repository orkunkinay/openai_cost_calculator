"""Google Cloud Vertex AI: official generative AI pricing page (HTML).

Three table families are parsed:

* **Gemini** tables: one row per (model, type, Global/Non-global region)
  with <=200K / >200K and cached columns; separate tables for Priority and
  Flex/Batch.  Model cells may carry a date qualifier ("...through December 31,
  2026", "...Starting January 1, 2027") that becomes an effective window.
* **Claude** tables: one table per region tab (Global, US and EU
  multi-regions, individual regions).  Tabs are matched to tables by order and
  the tab count is verified; a mismatch is flagged instead of guessed.
* **Partner model** tables (Grok, DeepSeek, Qwen, GLM, gpt-oss, Llama...):
  plain input/output prices.

"Non-global" Gemini prices are stored as ``region="regional"``; the Vertex
provider spec maps any concrete region (``us-central1``) to it.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from ...catalog.model import ModelPricing, PriceSet, Source
from ..base import Fetcher, SourceResult
from ..text import Table, html_tables, html_text, parse_money, slugify
from .common import price_set

URL = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
SOURCE = Source(id="vertex-pricing-page", kind="official_docs", url=URL, description="Vertex AI generative AI pricing page")

_CLAUDE_TYPES = {
    "input": ("standard", "input"),
    "output": ("standard", "output"),
    "5m cache write": ("standard", "cache_write"),
    "1h cache write": ("standard", "cache_write_1h"),
    "cache hit": ("standard", "cached_input"),
    "batch input": ("batch", "input"),
    "batch output": ("batch", "output"),
    "5m batch cache write": ("batch", "cache_write"),
    "1h batch cache write": ("batch", "cache_write_1h"),
    "batch cache hit": ("batch", "cached_input"),
    # Label variants that appear on the live page for some models.
    "cache write": ("standard", "cache_write"),
    "5m batch write": ("batch", "cache_write"),
    "5m batch cache hit": ("batch", "cached_input"),
    "1h batch cache hit": ("batch", "cached_input"),
}
_TAB_LABEL = re.compile(r"Global|US Multi-Region \(US\)|EU Multi-Region \(EU\)|[a-z]+-[a-z]+ ?\d+")
_DATE_QUALIFIER = re.compile(r"\*?\s*(?P<kind>through|starting)\s+(?P<date>[A-Z][a-z]+ \d{1,2}, \d{4})\s*$", re.I)
_LONG_CONTEXT = 200_001


def _tab_region(label: str) -> str:
    if label == "Global":
        return "global"
    if label.startswith("US Multi"):
        return "us"
    if label.startswith("EU Multi"):
        return "eu"
    return label.replace(" ", "")


def _money(cell: str) -> Optional[Decimal]:
    text = cell.strip()
    if not text or text.upper() == "N/A":
        return None
    return parse_money(text)


def _add(bucket: Dict[Tuple, Dict[str, Decimal]], key: Tuple, dimension: str, value: Optional[Decimal]) -> bool:
    """Record a price; return False if it conflicts with one already recorded."""
    if value is None:
        return True
    existing = bucket.setdefault(key, {}).get(dimension)
    if existing is not None and existing != value:
        return False
    bucket[key][dimension] = value
    return True


# --------------------------------------------------------------------------- Claude


def _claude_model_id(name: str) -> str:
    slug = slugify(name)
    return slug if slug.startswith("claude-") else f"claude-{slug}"


def parse_claude(tables: List[Table], labels: List[str], result: SourceResult) -> None:
    if len(tables) != len(labels):
        result.issue(
            f"found {len(tables)} Claude tables but {len(labels)} region tabs; "
            "only the global table was used"
        )
        tables, labels = tables[:1], ["Global"]
    rates: Dict[str, Dict[Tuple, Dict[str, Decimal]]] = {}
    conflicted = set()  # (model, region, tier) groups with contradictory rows
    for table, label in zip(tables, labels):
        region = _tab_region(label)
        model = ""
        for row in table.rows:
            if len(row) < 3:
                continue
            model = row[0] or model
            kind = _CLAUDE_TYPES.get(row[1].strip().lower())
            if not model or kind is None:
                if model and row[1].strip():
                    result.issue(f"unrecognized Claude price type {row[1]!r}", _claude_model_id(model))
                continue
            tier, dimension = kind
            try:
                short = _money(row[2])
                long = _money(row[3]) if len(row) > 3 else None
            except ValueError as exc:
                result.issue(str(exc), _claude_model_id(model))
                continue
            model_id = _claude_model_id(model)
            bucket = rates.setdefault(model_id, {})
            consistent = _add(bucket, (region, tier, 0), dimension, short)
            if long is not None and long != short:
                consistent &= _add(bucket, (region, tier, _LONG_CONTEXT), dimension, long)
            if not consistent and (model_id, region, tier) not in conflicted:
                conflicted.add((model_id, region, tier))
                result.note(
                    f"the {label} table lists contradictory {row[1]!r} prices; "
                    f"{tier} prices for region {region} were left out",
                    model_id,
                )
    for model_id, buckets in rates.items():
        sets: List[Optional[PriceSet]] = []
        for (region, tier, minimum), bucket in sorted(buckets.items()):
            if (model_id, region, tier) in conflicted:
                continue
            if minimum:
                bucket = {**buckets.get((region, tier, 0), {}), **bucket}
            sets.append(price_set(bucket, service_tier=tier, region=region, min_input_tokens=minimum))
        if not any(s is not None for s in sets):
            continue
        result.add(
            ModelPricing(
                id=model_id,
                prices=tuple(s for s in sets if s is not None),
                source=SOURCE.id,
                vendor="anthropic",
                canonical_id=f"anthropic/{model_id}",
            )
        )


# --------------------------------------------------------------------------- Gemini


def _model_slug(name: str) -> str:
    """"DeepSeek R1 (0528)" -> "deepseek-r1"; keeps version dots ("gemini-2.5-pro")."""
    name = re.sub(r"\([^)]*\)", "", name).strip()
    return re.sub(r"\s+", "-", name.lower())


def _gemini_model(cell: str) -> Tuple[str, Optional[date], Optional[date]]:
    text = cell.replace("*", " ").strip()
    start = end = None
    match = _DATE_QUALIFIER.search(text)
    if match:
        text = text[: match.start()].strip()
        when = datetime.strptime(match.group("date"), "%B %d, %Y").date()
        if match.group("kind").lower() == "through":
            end = when + timedelta(days=1)
        else:
            start = when
    else:
        text = re.sub(r"(Starting|through)$", "", text).strip()
    return _model_slug(text), start, end


def _gemini_dimensions(kind: str) -> Tuple[List[str], List[str]]:
    """(price dimensions, cached dimensions) for a Gemini row type."""
    lowered = kind.lower()
    if "output" in lowered:
        return (["output"] if "text" in lowered else []), []
    if "input" not in lowered:
        return [], []
    modalities = re.findall(r"text|image|video|audio", lowered) or ["text"]
    dims, cached = [], []
    if "text" in modalities or "video" in modalities:
        dims.append("input")
        cached.append("cached_input")
    if "image" in modalities:
        dims.append("input_image")
    if "audio" in modalities:
        dims.append("input_audio")
        cached.append("cached_input_audio")
    return dims, cached


def _gemini_columns(header: List[str]) -> Dict[str, int]:
    columns: Dict[str, int] = {}
    for index, name in enumerate(header):
        lowered = name.lower()
        if "price" not in lowered and "token" not in lowered:
            continue
        batch = "batch api" in lowered
        cached = "cached" in lowered
        long = ">" in lowered
        key = ("batch_" if batch else "") + ("cached_" if cached else "") + ("long" if long else "short")
        columns.setdefault(key, index)
    return columns


def parse_gemini_table(table: Table, result: SourceResult, rates: Dict[Tuple, Dict[str, Decimal]]) -> None:
    header = " ".join(table.header).lower()
    tiers = ["priority"] if "with priority" in header else ["flex", "batch"] if "with flex/batch" in header else ["standard"]
    columns = _gemini_columns(table.header)
    region_col = table.column("Region")
    model = kind = ""
    for row in table.rows:
        if len(row) < len(table.header):
            continue
        model = row[0] or model
        kind = row[1] or kind
        if not model:
            continue
        region = "global" if region_col is None or row[region_col].startswith("Global") else "regional"
        dims, cached_dims = _gemini_dimensions(kind)
        if not dims:
            continue
        for name in re.split(r",(?!\s*\d)", model):
            model_id, start, end = _gemini_model(name)
            try:
                values = {key: _money(row[index]) for key, index in columns.items()}
            except ValueError as exc:
                result.issue(str(exc), model_id)
                continue
            for tier_list, prefix in ((tiers, ""), (["batch"], "batch_")):
                if prefix and f"{prefix}short" not in values:
                    continue
                for tier in tier_list:
                    for suffix, minimum in (("short", 0), ("long", _LONG_CONTEXT)):
                        key = (model_id, region, tier, minimum, start, end)
                        for dimension in dims:
                            _add(rates, key, dimension, values.get(f"{prefix}{suffix}"))
                        for dimension in cached_dims:
                            _add(rates, key, dimension, values.get(f"{prefix}cached_{suffix}"))


def build_gemini(rates: Dict[Tuple, Dict[str, Decimal]], result: SourceResult) -> None:
    by_model: Dict[str, List[PriceSet]] = {}
    for (model_id, region, tier, minimum, start, end), bucket in sorted(rates.items(), key=lambda kv: str(kv[0])):
        if minimum:
            base = rates.get((model_id, region, tier, 0, start, end), {})
            if bucket == {k: base[k] for k in bucket if k in base}:
                continue  # long-context price identical to base: no tier
            bucket = {**base, **bucket}
        built = price_set(bucket, service_tier=tier, region=region, min_input_tokens=minimum, effective_from=start, effective_until=end)
        if built is not None and ("input" in built.rates or "output" in built.rates):
            by_model.setdefault(model_id, []).append(built)
    for model_id, sets in by_model.items():
        result.add(
            ModelPricing(id=model_id, prices=tuple(sets), source=SOURCE.id, vendor="google", canonical_id=f"google/{model_id}")
        )


# --------------------------------------------------------------------------- partner models


def parse_partner_table(table: Table, vendor: str, result: SourceResult) -> None:
    price_col = next((i for i, h in enumerate(table.header) if "price" in h.lower()), None)
    long_col = next((i for i, h in enumerate(table.header) if ">" in h), None)
    if price_col is None:
        return
    rates: Dict[str, Dict[Tuple[int], Dict[str, Decimal]]] = {}
    model = ""
    for row in table.rows:
        if len(row) <= price_col:
            continue
        model = row[0] or model
        kind = row[1].strip().lower()
        dimension = {"input": "input", "output": "output", "cache read": "cached_input", "cached input": "cached_input"}.get(kind)
        if not model or dimension is None:
            continue
        model_id = _model_slug(model)
        try:
            value = _money(row[price_col])
            long = _money(row[long_col]) if long_col is not None and long_col < len(row) else None
        except ValueError:
            continue  # per-page / per-character prices (OCR): outside the token catalog
        bucket = rates.setdefault(model_id, {})
        _add(bucket, (0,), dimension, value)
        if long is not None and long != value:
            _add(bucket, (_LONG_CONTEXT,), dimension, long)
    for model_id, buckets in rates.items():
        sets = []
        for (minimum,), bucket in sorted(buckets.items()):
            if minimum:
                bucket = {**buckets.get((0,), {}), **bucket}
            sets.append(price_set(bucket, min_input_tokens=minimum))
        sets = [s for s in sets if s is not None and "input" in s.rates]
        if sets:
            result.add(
                ModelPricing(id=model_id, prices=tuple(sets), source=SOURCE.id, vendor=vendor, canonical_id=f"{vendor}/{model_id}")
            )


_PARTNER_SECTIONS = {
    "xai's grok models": "xai", "deepseek's models": "deepseek", "minimax's models": "minimax",
    "moonshot's models": "moonshotai", "qwen's models": "qwen", "glm's models": "zai",
    "openai's models": "openai", "meta's llama models": "meta", "mistral ai’s models": "mistral",
}


def parse(document: str) -> SourceResult:
    result = SourceResult()
    tables = html_tables(document)
    text = html_text(document)

    claude_tables = [t for t in tables if t.headings and "Claude models" in t.headings[-1] and t.column("Type") is not None]
    labels_line = next((line for line in text.splitlines() if line.startswith("Global") and "Multi-Region" in line), "")
    labels = _TAB_LABEL.findall(labels_line)
    if claude_tables:
        parse_claude(claude_tables, labels, result)
    else:
        result.issue("Claude pricing tables not found")

    gemini_rates: Dict[Tuple, Dict[str, Decimal]] = {}
    for table in tables:
        section = table.headings[-1] if table.headings else ""
        header = " ".join(table.header)
        if section.startswith("Gemini") and table.column("Model") is not None and table.column("Type") is not None and "/1M" in header or "Token Price" in header:
            if "$/M char" in " ".join(r[1] for r in table.rows if len(r) > 1):
                continue  # character-priced legacy table
            parse_gemini_table(table, result, gemini_rates)
        vendor = _PARTNER_SECTIONS.get(section.lower())
        if vendor:
            parse_partner_table(table, vendor, result)
    if not gemini_rates:
        result.issue("Gemini pricing tables not found")
    build_gemini(gemini_rates, result)
    return result


class VertexSource:
    provider = "vertex"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
