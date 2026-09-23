"""OpenAI: official pricing documentation (Markdown rendition).

OpenAI publishes no pricing API, but its documentation site serves a Markdown
version of the pricing page (append ``.md``), whose tables are structured
enough to parse reliably:

* token-price tables with short/long-context and cache-write columns, grouped
  by a plain-text tier label ("Standard", "Batch", "Flex", "Fast mode");
* modality tables (Audio/Text/Image rows) for realtime, audio and image models.

Units other than per-1M-tokens (per minute, per character, per second) are
outside the catalog's scope and skipped deliberately; anything that looks
like a token table but does not parse is reported for review.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ...catalog.model import ModelPricing, PriceSet, Source
from ..base import Fetcher, SourceResult
from ..text import clean_cell, parse_money
from .common import present, price_set

URL = "https://developers.openai.com/api/docs/pricing.md"
SOURCE = Source(
    id="openai-pricing-docs",
    kind="official_docs",
    url="https://developers.openai.com/api/docs/pricing",
    description="OpenAI API pricing documentation (Markdown rendition)",
)

#: Long-context pricing applies at or above this many input tokens.  Rows that
#: carry a "(<272K context length)" note confirm it; rows with long-context
#: columns but no note use the same documented threshold.
LONG_CONTEXT_THRESHOLD = 272_000

#: Dated snapshot ids priced like their undated family.  OpenAI's page lists
#: undated names; these aliases keep dated ids (and the legacy CSV rows that
#: installed clients depend on) resolving to the right family rather than to an
#: older, differently priced snapshot such as gpt-4o-2024-05-13.
KNOWN_SNAPSHOTS = {
    "gpt-4o": ("gpt-4o-2024-08-06", "gpt-4o-2024-11-20"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18",),
    "gpt-4.1": ("gpt-4.1-2025-04-14",),
    "gpt-4.1-mini": ("gpt-4.1-mini-2025-04-14",),
    "gpt-4.1-nano": ("gpt-4.1-nano-2025-04-14",),
    "gpt-5": ("gpt-5-2025-08-07",),
    "gpt-5-mini": ("gpt-5-mini-2025-08-07",),
    "gpt-5-nano": ("gpt-5-nano-2025-08-07",),
    "o1": ("o1-2024-12-17",),
    "o1-pro": ("o1-pro-2025-03-19",),
    "o3": ("o3-2025-04-16",),
    "o3-mini": ("o3-mini-2025-01-31",),
    "o3-pro": ("o3-pro-2025-06-10",),
    "o4-mini": ("o4-mini-2025-04-16",),
}

_TIER_LABELS = {
    "standard": "standard",
    "batch": "batch",
    "flex": "flex",
    "fast mode": "priority",
    "priority": "priority",
}
_SKIPPED_SECTIONS = ("finetuning", "fine-tuning", "video generation", "transcription", "tools", "gpt-live")
_MODEL_CELL = re.compile(r"^(?P<id>[a-z0-9][a-z0-9.\-:]*)\s*(?:\((?P<note>[^)]*)\))?$")
_THRESHOLD_NOTE = re.compile(r"<\s*(\d+)\s*K", re.I)

_FLAGSHIP_COLUMNS = {
    "input": ("short context input", "input"),
    "cached_input": ("short context cached input", "cached input"),
    "cache_write": ("short context cache writes", "cache writes"),
    "output": ("short context output", "output"),
}
_LONG_COLUMNS = {
    "input": ("long context input",),
    "cached_input": ("long context cached input",),
    "cache_write": ("long context cache writes",),
    "output": ("long context output",),
}
_MODALITY_DIMENSIONS = {
    "text": ("input", "cached_input", "output"),
    "audio": ("input_audio", "cached_input_audio", "output_audio"),
    "image": ("input_image", None, "output_image"),
}


def _split_tables(text: str) -> Iterator[Tuple[str, str, List[str], List[List[str]]]]:
    """Yield (section, tier, header, rows) for every table, tracking plain-text labels."""
    section, tier = "", "standard"
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if raw.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-{3}", lines[i + 1].strip()):
            header = [clean_cell(c) for c in raw.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            yield section, tier, header, rows
            continue
        plain = clean_cell(raw)
        lowered = plain.lower()
        if lowered in _TIER_LABELS:
            tier = _TIER_LABELS[lowered]
        elif re.match(r"^###\s+(standard|batch|flex|fast)\s+pricing data$", raw, re.I):
            tier = _TIER_LABELS[raw.split()[1].lower().replace("fast", "fast mode")]
        elif (
            plain
            and not raw.startswith("#")
            and (lowered.endswith(" models") or lowered.endswith("sessions") or lowered in {"tools", "finetuning"})
        ):
            section, tier = lowered, "standard"
        i += 1


def _column(header: List[str], names: Sequence[str]) -> Optional[int]:
    lowered = [h.lower() for h in header]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return None


def _money(cells: List[str], index: Optional[int]) -> Optional[Decimal]:
    return parse_money(cells[index]) if index is not None and index < len(cells) else None


def _model(model_id: str, prices: Iterable[Optional[PriceSet]], display: Optional[str] = None) -> ModelPricing:
    return ModelPricing(
        id=model_id,
        prices=present(prices),
        source=SOURCE.id,
        vendor="openai",
        canonical_id=f"openai/{model_id}",
        display_name=display,
        aliases=KNOWN_SNAPSHOTS.get(model_id, ()),
    )


def _parse_token_table(tier: str, header: List[str], rows: List[List[str]], result: SourceResult) -> None:
    short_cols = {dim: _column(header, names) for dim, names in _FLAGSHIP_COLUMNS.items()}
    long_cols = {dim: _column(header, names) for dim, names in _LONG_COLUMNS.items()}
    model_col = _column(header, ("model",))
    for cells in rows:
        cell = clean_cell(cells[model_col]) if model_col is not None else ""
        match = _MODEL_CELL.match(cell)
        if not match:
            result.issue(f"unrecognized model cell {cell!r} in {tier} table")
            continue
        model_id, note = match.group("id"), match.group("note")
        threshold = LONG_CONTEXT_THRESHOLD
        if note:
            found = _THRESHOLD_NOTE.search(note)
            if not found:
                result.issue(f"unrecognized note {note!r}", model_id)
                continue
            threshold = int(found.group(1)) * 1000
        try:
            short = {dim: _money([clean_cell(c) for c in cells], col) for dim, col in short_cols.items()}
            long = {dim: _money([clean_cell(c) for c in cells], col) for dim, col in long_cols.items()}
        except ValueError as exc:
            result.issue(f"unparseable price in {tier} table: {exc}", model_id)
            continue
        if short.get("input") is None and short.get("output") is None:
            continue  # model not offered in this tier
        prices = [price_set(short, service_tier=tier)]
        if any(v is not None for v in long.values()):
            prices.append(price_set(long, service_tier=tier, min_input_tokens=threshold))
        result.add(_model(model_id, prices))


def _parse_modality_table(tier: str, header: List[str], rows: List[List[str]], result: SourceResult) -> None:
    model_col = _column(header, ("model",))
    modality_col = _column(header, ("modality",))
    input_col = _column(header, ("input",))
    cached_col = _column(header, ("cached input",))
    output_col = _column(header, ("output / cost", "output"))
    rates: Dict[str, Dict[str, Decimal]] = {}
    if model_col is None or modality_col is None:
        result.issue(f"modality table without model/modality columns: {header}")
        return
    for cells in rows:
        model_id = clean_cell(cells[model_col])
        modality = clean_cell(cells[modality_col]).lower()
        dimensions = _MODALITY_DIMENSIONS.get(modality)
        if dimensions is None or not _MODEL_CELL.match(model_id):
            result.issue(f"unrecognized modality row {model_id!r}/{modality!r}", model_id or None)
            continue
        values = []
        skip = False
        for col in (input_col, cached_col, output_col):
            text = clean_cell(cells[col]) if col is not None and col < len(cells) else "-"
            if "/" in text and "token" not in text.lower():
                skip = True  # per-character/per-minute pricing: out of scope
                break
            try:
                values.append(parse_money(text))
            except ValueError as exc:
                result.issue(f"unparseable modality price: {exc}", model_id)
                skip = True
                break
        if skip:
            continue
        entry = rates.setdefault(model_id, {})
        for dimension, value in zip(dimensions, values):
            if dimension is not None and value is not None:
                entry[dimension] = value
    for model_id, model_rates in rates.items():
        if model_rates:
            result.add(_model(model_id, [price_set(model_rates, service_tier=tier)]))


def parse(text: str) -> SourceResult:
    result = SourceResult()
    for section, tier, header, rows in _split_tables(text):
        if any(section.startswith(skipped) for skipped in _SKIPPED_SECTIONS):
            continue
        lowered = [h.lower() for h in header]
        if "modality" in lowered and "model" in lowered:
            _parse_modality_table(tier, header, rows, result)
        elif "model" in lowered and ("input" in lowered or "short context input" in lowered):
            _parse_token_table(tier, header, rows, result)
    if not result.models:
        result.issue("no priced models found; the page layout may have changed")
    return result


class OpenAISource:
    provider = "openai"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
