"""Small, dependency-free parsing helpers for documentation pages.

Official pricing pages come as Markdown (OpenAI, Anthropic, Gemini, Together,
Groq, Fireworks) or server-rendered HTML (Vertex AI, DeepSeek, Mistral).
These helpers turn them into tables of cell strings; each source's parser
then interprets its own columns.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import List, Optional, Tuple

_MONEY = re.compile(r"^\$?\s*(\d[\d,]*(?:\.\d+)?)$")
_NOT_AVAILABLE = {"", "-", "—", "–", "n/a", "na", "not available", "contactsales", "contact sales"}


def clean_cell(text: str) -> str:
    """Strip Markdown links/images/emphasis, escapes and surrounding space."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> label
    text = text.replace("\\$", "$").replace("\\*", "*").replace("\\<", "<").replace("\\>", ">")
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_money(text: str) -> Optional[Decimal]:
    """``"$1.25"`` -> ``Decimal("1.25")``; ``"Free"`` -> 0; ``"-"`` -> ``None``.

    Raises ``ValueError`` for anything else so callers can flag the cell for
    review instead of guessing.
    """
    value = clean_cell(text).replace("/ MTok", "").replace("/MTok", "").strip()
    if value.lower() in _NOT_AVAILABLE:
        return None
    if value.lower() == "free":
        return Decimal(0)
    match = _MONEY.match(value)
    if not match:
        raise ValueError(f"not a plain price: {text!r}")
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation as exc:  # pragma: no cover - regex guarantees digits
        raise ValueError(f"not a plain price: {text!r}") from exc


@dataclass
class Table:
    header: List[str]
    rows: List[List[str]]
    #: Headings in effect where the table appears, outermost first.
    headings: List[str] = field(default_factory=list)
    #: Plain-text lines between the previous heading and the table.
    preamble: List[str] = field(default_factory=list)

    def column(self, *names: str) -> Optional[int]:
        lowered = [h.lower() for h in self.header]
        for name in names:
            for index, header in enumerate(lowered):
                if header == name.lower():
                    return index
        return None


def _split_row(line: str) -> List[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", body)]


_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def markdown_tables(text: str) -> List[Table]:
    """Every pipe table in a Markdown document, with its heading context."""
    tables: List[Table] = []
    headings: List[Tuple[int, str]] = []
    preamble: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, clean_cell(heading.group(2)))]
            preamble = []
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1].strip()):
            header = [clean_cell(c) for c in _split_row(line)]
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            tables.append(Table(header, rows, [h[1] for h in headings], list(preamble)))
            continue
        if line.strip():
            preamble.append(clean_cell(line))
        i += 1
    return tables


class _HtmlTables(HTMLParser):
    """Collect tables (as cell text) and the text of headings before them."""

    _SKIP = {"script", "style", "svg", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[Table] = []
        self.text_chunks: List[str] = []
        self._skip = 0
        self._heading: Optional[List[str]] = None
        self._headings: List[str] = []
        self._stack: List[Table] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._cell_is_header = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif self._skip:
            return
        elif tag in {"h1", "h2", "h3", "h4"}:
            self._heading = []
        elif tag == "table":
            self._stack.append(Table(header=[], rows=[], headings=list(self._headings[-4:])))
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._cell_is_header = tag == "th"
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif tag in {"h1", "h2", "h3", "h4"} and self._heading is not None:
            text = re.sub(r"\s+", " ", "".join(self._heading)).strip()
            if text:
                self._headings.append(text)
            self._heading = None
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._stack:
            table = self._stack[-1]
            if not table.header and self._cell_is_header:
                table.header = self._row
            elif self._row:
                table.rows.append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading is not None:
            self._heading.append(data)
        if self._cell is not None:
            self._cell.append(data)
        self.text_chunks.append(data)


def html_tables(document: str) -> List[Table]:
    parser = _HtmlTables()
    parser.feed(document)
    return parser.tables


def html_text(document: str) -> str:
    """Visible text of an HTML document, one block per line."""
    parser = _HtmlTables()
    parser.feed(re.sub(r"<(br|/p|/div|/li|/h\d|/tr)[^>]*>", "\n", document, flags=re.I))
    text = "".join(parser.text_chunks)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def slugify(name: str) -> str:
    """``"Claude Sonnet 4.5"`` -> ``"claude-sonnet-4-5"``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
