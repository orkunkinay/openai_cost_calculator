"""Google Gemini API (AI Studio): official pricing documentation (Markdown).

``ai.google.dev/gemini-api/docs/pricing.md.txt`` has one section per model
with Standard/Batch/Flex/Priority tables whose paid-tier cells are short
prose.  Each cell is split into *clauses*, one per dollar amount, and each
clause may carry:

* a modality qualifier - ``(text / image / video)``, ``(audio)``, ``(images)``;
* a context bound - ``prompts <= 200k tokens`` / ``prompts > 200k``;
* an effective-date bound - ``through December 31, 2026.`` /
  ``starting January 1, 2027.`` (scheduled price changes).

Clauses priced in other units (per image, per minute, storage per hour) are
outside the catalog's token model and skipped.  Any clause with leftover text
the grammar does not understand is reported, and that model is not updated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from ...catalog.model import ModelPricing, PriceSet, Source
from ..base import Fetcher, SourceResult
from ..text import clean_cell
from .common import price_set

URL = "https://ai.google.dev/gemini-api/docs/pricing.md.txt"
SOURCE = Source(
    id="gemini-pricing-docs",
    kind="official_docs",
    url="https://ai.google.dev/gemini-api/docs/pricing",
    description="Gemini Developer API pricing documentation (Markdown rendition)",
)

_TIERS = {"standard": "standard", "batch": "batch", "flex": "flex", "priority": "priority"}
_CLAUSE = re.compile(r"\$(?P<amount>\d+(?:\.\d+)?)(?P<rest>[^$]*)")
_NON_TOKEN_UNIT = re.compile(
    r"^\s*(/\s*min|/\s*[\d,]+\s*tokens per hour|per (image|song|second|frame|hour|[\d.]+k)|/\s*[\d,]+ (grounded|search))",
    re.I,
)
_MODALITY = re.compile(r"^\s*\(([a-z ,/]+)\)", re.I)
_CONTEXT = re.compile(r"^\s*,?\s*prompts\s*(?P<op><=|>)\s*(?P<k>\d+)k(?:\s*tokens?)?", re.I)
_THROUGH = re.compile(r"^\s*through (?P<date>[A-Z][a-z]+ \d{1,2}, \d{4})\.?", re.I)
_STARTING = re.compile(r"^\s*starting (?P<date>[A-Z][a-z]+ \d{1,2}, \d{4})\.?", re.I)
_EQUIVALENT = re.compile(r"(,\s*)?equivalent to .*$", re.I | re.S)
_IDS = re.compile(r"`([a-z0-9][a-z0-9.\-]*)`")

_ROW_KINDS = (
    ("context caching price", "cache", frozenset()),
    ("audio input price", "input", frozenset({"audio"})),
    ("image input price", "input", frozenset({"image"})),
    ("video input price", "input", frozenset({"video"})),
    ("text input price", "input", frozenset({"text"})),
    ("input price", "input", frozenset()),
    ("output price", "output", frozenset()),
)

#: (row kind, modality) -> dimension.  Video is billed like text in usage
#: metadata we can observe, so separately priced video clauses are ignored.
_DIMENSIONS = {
    ("input", "text"): "input",
    ("input", "image"): "input_image",
    ("input", "audio"): "input_audio",
    ("output", "text"): "output",
    ("output", "image"): "output_image",
    ("output", "audio"): "output_audio",
    ("cache", "text"): "cached_input",
    ("cache", "audio"): "cached_input_audio",
}


@dataclass(frozen=True)
class Clause:
    dimension: str
    amount: Decimal
    #: None: applies to every context size; else the tier's min_input_tokens.
    min_input_tokens: Optional[int]
    start: Optional[date]
    end: Optional[date]


def _date(text: str) -> date:
    return datetime.strptime(text, "%B %d, %Y").date()


def _modalities(text: str) -> FrozenSet[str]:
    words = set(re.split(r"[\s,/]+|\band\b", text.lower())) - {""}
    normalized = {"images": "image", "thinking": "text"}
    return frozenset(normalized.get(w, w) for w in words)


_FOOTNOTE = "\u2020"


def parse_cell(
    cell: str, kind: str, default_modalities: FrozenSet[str]
) -> Tuple[List[Clause], List[str], List[str]]:
    """Parse one paid-tier cell into ``(clauses, problems, notes)``.

    A price immediately followed by a footnote marker is quoted in a unit the
    footnote defines (per image), not per 1M tokens as the column says; it is
    dropped and noted rather than guessed.
    """
    text = clean_cell(cell)
    text = re.sub(r"\^[^^]*\^", _FOOTNOTE, text).replace("Same as Standard", "").strip()
    if not text or text.lower() in {"not available", "free of charge"}:
        return [], []
    raw = list(_CLAUSE.finditer(text))
    if not raw or text[: raw[0].start()].strip():
        return [], [f"unrecognized cell {cell!r}"]
    parsed: List[Tuple[Decimal, str]] = [(Decimal(m.group("amount")), m.group("rest")) for m in raw]

    clauses: List[Clause] = []
    problems: List[str] = []
    notes: List[str] = []
    carried_modalities: Optional[FrozenSet[str]] = None
    closes_parenthetical = False
    # Walk backwards so "$3.00 or $0.005/min (audio)" gives the first price the
    # modality written after its per-minute alternative.
    for amount, rest in reversed(parsed):
        rest = _EQUIVALENT.sub("", rest).replace("*", "")
        if _NON_TOKEN_UNIT.match(rest):
            closes_parenthetical = rest.replace(_FOOTNOTE, "").rstrip(" ,.").endswith(")")
            modality = re.search(r"\(([a-z ,/]+)\)[\s" + _FOOTNOTE + r"]*$", rest, re.I)
            carried_modalities = _modalities(modality.group(1)) if modality else None
            continue
        if re.match(r"^\s*(\([a-z ,/]+\))?\s*" + _FOOTNOTE, rest, re.I):
            notes.append(f"ignored footnoted price ${amount} in {clean_cell(cell)!r}")
            continue
        rest = rest.replace(_FOOTNOTE, "")
        if closes_parenthetical and rest.rstrip().endswith("("):
            rest = rest.rstrip()[:-1]  # "$0.45 ($0.00012 per image)": equivalence note
        closes_parenthetical = False
        modalities: Optional[FrozenSet[str]] = None
        min_tokens: Optional[int] = None
        start = end = None
        remaining = rest
        progress = True
        while progress:
            progress = False
            for pattern in (_MODALITY, _CONTEXT, _THROUGH, _STARTING):
                match = pattern.match(remaining)
                if not match:
                    continue
                progress = True
                remaining = remaining[match.end():]
                if pattern is _MODALITY:
                    modalities = _modalities(match.group(1))
                elif pattern is _CONTEXT:
                    k = int(match.group("k")) * 1000
                    min_tokens = 0 if match.group("op") == "<=" else k + 1
                elif pattern is _THROUGH:
                    end = _date(match.group("date")) + timedelta(days=1)
                else:
                    start = _date(match.group("date"))
        leftover = remaining.strip(" .,;")
        if leftover.lower() == "or" and carried_modalities is not None:
            modalities, leftover = carried_modalities, ""
        carried_modalities = None
        if leftover:
            problems.append(f"could not interpret {leftover!r} in {cell!r}")
            continue
        for modality in modalities or default_modalities or frozenset({"text"}):
            dimension = _DIMENSIONS.get((kind, modality))
            if dimension is not None:
                clauses.append(Clause(dimension, amount, min_tokens, start, end))
    return list(reversed(clauses)), problems, notes


def build_price_sets(clauses: List[Clause], tier: str) -> List[PriceSet]:
    """Expand clauses over every (effective window, context tier) combination."""
    bounds: Set[date] = {c.start for c in clauses if c.start} | {c.end for c in clauses if c.end}
    points = sorted(bounds)
    windows: List[Tuple[Optional[date], Optional[date]]] = []
    edges: List[Optional[date]] = [None, *points, None]
    for start, end in zip(edges, edges[1:]):
        windows.append((start, end))
    tiers = sorted({0} | {c.min_input_tokens for c in clauses if c.min_input_tokens})
    sets: List[PriceSet] = []
    for start, end in windows:
        for minimum in tiers:
            rates: Dict[str, Decimal] = {}
            for clause in clauses:
                active = (clause.start is None or (start is not None and clause.start <= start)) and (
                    clause.end is None or (end is not None and end <= clause.end)
                )
                applies = clause.min_input_tokens is None or clause.min_input_tokens == minimum
                if active and applies:
                    rates[clause.dimension] = clause.amount
            built = price_set(rates, service_tier=tier, min_input_tokens=minimum, effective_from=start, effective_until=end)
            if built is not None and ("input" in built.rates or "output" in built.rates or "input_audio" in built.rates):
                sets.append(built)
    return sets


def _row_kind(label: str) -> Optional[Tuple[str, FrozenSet[str]]]:
    lowered = clean_cell(re.sub(r"\^[^^]*\^", "", label)).lower()
    for prefix, kind, modalities in _ROW_KINDS:
        if lowered.startswith(prefix):
            return kind, modalities
    return None


def parse(text: str) -> SourceResult:
    result = SourceResult()
    ids: List[str] = []
    display = ""
    tier: Optional[str] = None
    in_table = False
    clauses_by_tier: Dict[str, List[Clause]] = {}
    problems: List[str] = []

    def flush() -> None:
        if not ids:
            return
        sets = [s for t, clauses in clauses_by_tier.items() for s in build_price_sets(clauses, t)]
        for model_id in ids:
            for problem in problems:
                result.issue(problem, model_id)
            if sets and not problems:
                result.add(
                    ModelPricing(
                        id=model_id,
                        prices=tuple(sets),
                        source=SOURCE.id,
                        vendor="google",
                        canonical_id=f"google/{model_id}",
                        display_name=display,
                    )
                )

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            ids, display, tier, problems = [], clean_cell(stripped[3:]), None, []
            clauses_by_tier = {}
            continue
        if not ids and stripped.startswith("*[`"):
            ids = _IDS.findall(stripped)
            continue
        if stripped.startswith("### "):
            tier = _TIERS.get(clean_cell(stripped[4:]).lower())
            in_table = False
            continue
        if not stripped.startswith("|") or tier is None or not ids:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3 or all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        if "free tier" in cells[1].lower():
            in_table = True
            continue
        kind = _row_kind(cells[0]) if in_table else None
        if kind is None:
            continue
        clauses, cell_problems, cell_notes = parse_cell(cells[2], *kind)
        problems.extend(f"{tier}: {p}" for p in cell_problems)
        for note in cell_notes:
            for model_id in ids:
                result.note(f"{tier}: {note}", model_id)
        clauses_by_tier.setdefault(tier, []).extend(clauses)
    flush()
    return result


class GeminiSource:
    provider = "gemini"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
