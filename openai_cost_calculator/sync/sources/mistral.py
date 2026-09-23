"""Mistral AI: official API pricing page (server-rendered HTML).

``mistral.ai/pricing/api/`` renders one card per model: a display name, a
license badge, a description, then lines such as ``Input (/M tokens) $1.5``
and ``Output (/M tokens) $7.5``.  Service-wide modifiers are stated once on
the page and parsed strictly:

* ``Batch ... at half price`` -> batch prices at 50%;
* ``Cached input tokens -90%`` -> cached input at 10% of input;
* ``Regional inference +10%`` -> ``region="eu"`` prices at 110%.

If a modifier sentence stops matching, the corresponding prices are not
generated and the run is flagged.  Cards priced per page, minute or
character are skipped.  The page shows display names only, so API ids are
slugs of them plus the ``-latest`` aliases in :data:`LATEST_ALIASES` - a
small, hand-maintained table because Mistral moves ``-latest`` pointers
between releases.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional

from ...catalog.model import ModelPricing, Source
from ..base import Fetcher, SourceResult
from ..text import html_text, slugify
from .common import price_set, scale

URL = "https://mistral.ai/pricing/api/"
SOURCE = Source(id="mistral-pricing-page", kind="official_docs", url=URL, description="Mistral AI API pricing page")

LATEST_ALIASES: Dict[str, tuple] = {
    "mistral-medium-3-5": ("mistral-medium-latest", "mistral-medium-3.5"),
    "mistral-small-4": ("mistral-small-latest",),
    "mistral-large-3": ("mistral-large-latest",),
    "codestral": ("codestral-latest",),
    "ministral-3-3b": ("ministral-3b-latest",),
    "ministral-3-8b": ("ministral-8b-latest",),
    "ministral-3-14b": ("ministral-14b-latest",),
    "mistral-embed": ("mistral-embed-latest",),
    "codestral-embed": ("codestral-embed-latest",),
}

_PRICE_LINE = re.compile(r"^(?P<kind>Input|Output) \(/M tokens\)\s*\$(?P<price>[\d.]+)$")
_LABEL = re.compile(r"^[A-Za-z ]+\((?:/M tokens|per [^)]*|/M tok[^)]*)\)$|^[A-Za-z ]+/min$")
_OTHER_PRICE = re.compile(r"\(per |/ 1000 pages|per 1k characters|/min", re.I)
_BADGES = {"Open", "Premier", "Labs", "New"}
_FIRST_PARTY = ("mistral", "ministral", "codestral", "devstral", "magistral", "pixtral", "voxtral", "leanstral")
_BATCH = re.compile(r"Batch\s*High-volume processing, at half price", re.I)
_CACHED = re.compile(r"Cached input tokens\s*-(?P<pct>\d+)%")
_REGIONAL = re.compile(r"Regional inference\s*\+(?P<pct>\d+)%")


def _join_prices(lines: List[str]) -> List[str]:
    """Merge a price label and its value ("Input (/M tokens)", "$1.5") into one line."""
    joined: List[str] = []
    for line in lines:
        if joined and re.fullmatch(r"\$[\d.]+", line) and _LABEL.match(joined[-1]):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return joined


def _is_description(line: str) -> bool:
    return len(line) > 45 or line.endswith(".")


def _card_name(lines: List[str], price_index: int) -> Optional[str]:
    """Walk back from a card's first price line, past tags and the description, to its title."""
    index = price_index - 1
    while index >= 0 and not _is_description(lines[index]):
        if _PRICE_LINE.match(lines[index]) or lines[index].startswith(("Learn more", "Read ")):
            return None  # reached the previous card without finding a description
        index -= 1  # capability tags and other price labels of this card
    index -= 1
    while index >= 0 and lines[index] in _BADGES:
        index -= 1
    if index < 0 or _is_description(lines[index]):
        return None
    return re.sub(r"New$", "", lines[index]).strip()


def model_id(display_name: str) -> str:
    """"Ministral 3 (8B)" -> "ministral-3-8b"."""
    return slugify(display_name)


def parse(document: str) -> SourceResult:
    result = SourceResult()
    lines = _join_prices(html_text(document).splitlines())
    text = " ".join(lines)
    batch = _BATCH.search(text)
    cached = _CACHED.search(text)
    regional = _REGIONAL.search(text)
    for found, what in ((batch, "batch"), (cached, "cached-input"), (regional, "regional inference")):
        if found is None:
            result.issue(f"{what} pricing rule not found; those prices were not generated")

    cards: Dict[str, Dict[str, Decimal]] = {}
    order: List[str] = []
    skipped = set()
    current: Optional[str] = None
    for index, line in enumerate(lines):
        match = _PRICE_LINE.match(line)
        if not match:
            continue
        follows_price = index > 0 and _PRICE_LINE.match(lines[index - 1])
        name = current if follows_price else _card_name(lines, index)
        current = name
        if name is None:
            if match.group("kind") == "Input":
                result.issue(f"price line without a model name near {line!r}")
            continue
        if any(_OTHER_PRICE.search(lines[j]) for j in range(max(0, index - 3), min(len(lines), index + 3))):
            skipped.add(name)  # e.g. Voxtral: audio priced per minute alongside tokens
            continue
        if name not in cards:
            order.append(name)
        cards.setdefault(name, {})[match.group("kind").lower()] = Decimal(match.group("price"))

    for name in order:
        if name in skipped:
            continue
        rates = cards[name]
        mid = model_id(name)
        if cached and "output" in rates:
            rates = {**rates, "cached_input": rates["input"] * (1 - Decimal(cached.group("pct")) / 100)}
        sets = [price_set(rates)]
        if batch:
            sets.append(price_set(scale(rates, Decimal("0.5")), service_tier="batch"))
        if regional:
            factor = 1 + Decimal(regional.group("pct")) / 100
            sets += [
                price_set(scale(dict(s.rates), factor), service_tier=s.conditions.get("service_tier", "standard"), region="eu")
                for s in list(sets)
                if s is not None
            ]
        first_party = name.lower().startswith(_FIRST_PARTY)
        result.add(
            ModelPricing(
                id=mid,
                prices=tuple(s for s in sets if s is not None),
                source=SOURCE.id,
                vendor="mistral" if first_party else None,
                canonical_id=f"mistral/{mid}" if first_party else None,
                display_name=name,
                aliases=LATEST_ALIASES.get(mid, ()),
            )
        )
    if not result.models:
        result.issue("no priced models found; the page layout may have changed")
    return result


class MistralSource:
    provider = "mistral"
    source = SOURCE

    def fetch(self, fetcher: Fetcher) -> SourceResult:
        return parse(fetcher.get_text(URL))
