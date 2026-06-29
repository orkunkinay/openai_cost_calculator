#!/usr/bin/env python3
"""Check local model pricing against LiteLLM's structured pricing metadata.

OpenAI does not publish an official machine-readable pricing API. This checker
uses LiteLLM's community-maintained JSON pricing table instead of scraping the
HTML pricing page:
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json

The CSV stores prices as USD per 1M tokens. LiteLLM stores per-token prices, so
values are multiplied by 1,000,000 before comparison.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests


DEFAULT_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
UPSTREAM_URL = os.environ.get("PRICING_UPSTREAM_URL", DEFAULT_UPSTREAM_URL)
CSV_PATH = Path(os.environ.get("PRICING_CSV_PATH", "data/gpt_pricing_data.csv"))
SUMMARY_PATH = Path(os.environ.get("PRICING_DIFF_PATH", "pricing_diff.md"))
TOLERANCE = Decimal("1e-9")
PER_MILLION = Decimal("1000000")
ABOVE_272K_MIN_TOKENS = 272001
AUTO_ADD_GPT_MIN_MAJOR = int(os.environ.get("PRICING_AUTO_ADD_GPT_MIN_MAJOR", "5"))
DATED_UPSTREAM_KEY_RE = re.compile(r"^(?P<model>.+)-(?P<date>\d{4}-\d{2}-\d{2})$")
GPT_MODEL_RE = re.compile(r"^gpt-(?P<major>\d+)(?:[.\w-].*)?$")

PRICE_FIELDS = (
    ("Input Price", "input_cost_per_token", "input_cost_per_token_above_272k_tokens"),
    (
        "Cached Input Price",
        "cache_read_input_token_cost",
        "cache_read_input_token_cost_above_272k_tokens",
    ),
    ("Output Price", "output_cost_per_token", "output_cost_per_token_above_272k_tokens"),
)


@dataclass(frozen=True)
class FieldChange:
    field: str
    old: str
    new: str


@dataclass(frozen=True)
class RowChange:
    model_name: str
    model_date: str
    minimum_tokens: str
    upstream_key: str
    changes: list[FieldChange]


@dataclass(frozen=True)
class AddedRow:
    model_name: str
    model_date: str
    minimum_tokens: str
    upstream_key: str
    row: dict[str, str]


def fetch_upstream() -> dict[str, dict[str, Any]]:
    try:
        response = requests.get(UPSTREAM_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to fetch upstream pricing JSON from {UPSTREAM_URL}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"upstream pricing response was not valid JSON from {UPSTREAM_URL}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("upstream pricing JSON was not an object")

    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def detect_lineterminator(csv_text: str) -> str:
    if "\r\n" in csv_text:
        return "\r\n"
    if "\r" in csv_text:
        return "\r"
    return "\n"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open(newline="") as csv_file:
        sample = csv_file.read()
        csv_file.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
        reader = csv.DictReader(csv_file, dialect=dialect)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows, detect_lineterminator(sample)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], lineterminator: str) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)


def parse_price(value: str) -> Decimal | None:
    value = value.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError(f"invalid price value in CSV: {value!r}") from exc


def upstream_price(model_info: dict[str, Any], base_key: str, above_272k_key: str, minimum_tokens: int) -> Decimal | None:
    key = above_272k_key if minimum_tokens >= ABOVE_272K_MIN_TOKENS and above_272k_key in model_info else base_key
    raw_value = model_info.get(key)
    if raw_value is None:
        return None
    try:
        return Decimal(str(raw_value)) * PER_MILLION
    except InvalidOperation as exc:
        raise RuntimeError(f"invalid upstream price for {key}: {raw_value!r}") from exc


def decimal_places(value: str) -> int:
    if "." not in value:
        return 0
    return len(value.split(".", 1)[1])


def minimal_decimal_places(value: Decimal) -> int:
    normalized = value.normalize()
    exponent = normalized.as_tuple().exponent
    return max(0, -exponent)


def format_price(value: Decimal | None, old_value: str) -> str:
    if value is None:
        return ""

    old_value = old_value.strip()
    places = max(decimal_places(old_value), minimal_decimal_places(value))
    return f"{value:.{places}f}"


def minimum_tokens(row: dict[str, str]) -> int:
    raw_value = row.get("Minimum Tokens", "").strip()
    if not raw_value:
        return 0
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"invalid Minimum Tokens value in CSV: {raw_value!r}") from exc
    if value < 0:
        raise RuntimeError(f"Minimum Tokens must be non-negative, got {raw_value!r}")
    return value


def upstream_key_for_row(row: dict[str, str], upstream: dict[str, dict[str, Any]]) -> str | None:
    name = row["Model Name"].strip()
    model_date = row.get("Model Date", "").strip()
    if model_date:
        dated_key = f"{name}-{model_date}"
        if dated_key in upstream:
            return dated_key
    if name in upstream:
        return name
    return None


def split_dated_upstream_key(upstream_key: str) -> tuple[str, str]:
    match = DATED_UPSTREAM_KEY_RE.match(upstream_key)
    if match:
        return match.group("model"), match.group("date")
    return upstream_key, ""


def is_auto_addable_gpt_model(upstream_key: str, model_info: dict[str, Any]) -> bool:
    if model_info.get("litellm_provider") != "openai":
        return False

    model_name, _model_date = split_dated_upstream_key(upstream_key)
    match = GPT_MODEL_RE.match(model_name)
    if not match:
        return False

    if int(match.group("major")) < AUTO_ADD_GPT_MIN_MAJOR:
        return False

    return (
        upstream_price(model_info, "input_cost_per_token", "input_cost_per_token", 0) is not None
        and upstream_price(model_info, "output_cost_per_token", "output_cost_per_token", 0) is not None
    )


def has_above_272k_pricing(model_info: dict[str, Any]) -> bool:
    return any(above_272k_key in model_info for _csv_field, _base_key, above_272k_key in PRICE_FIELDS)


def new_price_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format_price(value, "0.0")


def build_new_row(
    model_name: str,
    model_date: str,
    minimum_tokens_value: int,
    upstream_key: str,
    model_info: dict[str, Any],
) -> AddedRow:
    prices = {
        csv_field: new_price_text(upstream_price(model_info, base_key, above_272k_key, minimum_tokens_value))
        for csv_field, base_key, above_272k_key in PRICE_FIELDS
    }
    row = {
        "Model Name": model_name,
        "Model Date": model_date,
        "Input Price": prices["Input Price"],
        "Cached Input Price": prices["Cached Input Price"],
        "Output Price": prices["Output Price"],
        "Minimum Tokens": str(minimum_tokens_value),
    }
    return AddedRow(model_name, model_date, str(minimum_tokens_value), upstream_key, row)


def new_gpt_rows(
    upstream: dict[str, dict[str, Any]], represented_models: set[str], existing_rows: list[dict[str, str]]
) -> list[AddedRow]:
    added_rows: list[AddedRow] = []
    existing_row_keys = {
        (
            row.get("Model Name", "").strip(),
            row.get("Model Date", "").strip(),
            str(minimum_tokens(row)),
        )
        for row in existing_rows
    }
    planned_model_names = set(represented_models)

    for upstream_key in sorted(upstream):
        model_info = upstream[upstream_key]
        if not is_auto_addable_gpt_model(upstream_key, model_info):
            continue

        model_name, model_date = split_dated_upstream_key(upstream_key)
        if upstream_key in planned_model_names or model_name in planned_model_names:
            continue

        minimum_token_values = [0]
        if has_above_272k_pricing(model_info):
            minimum_token_values.append(ABOVE_272K_MIN_TOKENS)

        row_additions: list[AddedRow] = []
        for min_tokens in minimum_token_values:
            row_key = (model_name, model_date, str(min_tokens))
            if row_key in existing_row_keys:
                continue
            row_additions.append(build_new_row(model_name, model_date, min_tokens, upstream_key, model_info))

        if not row_additions:
            continue

        added_rows.extend(row_additions)
        planned_model_names.add(upstream_key)
        planned_model_names.add(model_name)
        if model_date:
            planned_model_names.add(f"{model_name}-{model_date}")
        existing_row_keys.update((row.model_name, row.model_date, row.minimum_tokens) for row in row_additions)

    return added_rows


def price_changed(old_price: Decimal | None, new_price: Decimal | None) -> bool:
    if old_price is None or new_price is None:
        return old_price is not new_price
    return abs(old_price - new_price) > TOLERANCE


def apply_updates(
    rows: list[dict[str, str]], upstream: dict[str, dict[str, Any]]
) -> tuple[list[RowChange], set[str], set[str], list[dict[str, str]]]:
    changes: list[RowChange] = []
    matched_upstream_keys: set[str] = set()
    represented_models: set[str] = set()

    for row in rows:
        name = row["Model Name"].strip()
        model_date = row.get("Model Date", "").strip()
        represented_models.add(name)
        if model_date:
            represented_models.add(f"{name}-{model_date}")

        upstream_key = upstream_key_for_row(row, upstream)
        if upstream_key is None:
            continue

        matched_upstream_keys.add(upstream_key)
        model_info = upstream[upstream_key]
        min_tokens = minimum_tokens(row)
        field_changes: list[FieldChange] = []

        for csv_field, base_key, above_272k_key in PRICE_FIELDS:
            old_text = row.get(csv_field, "").strip()
            old_price = parse_price(old_text)
            new_price = upstream_price(model_info, base_key, above_272k_key, min_tokens)

            # Blank cached-price cells are intentionally used for rows where the
            # project does not track a cached tier. Do not infer a new cached
            # tier from upstream unless the CSV already has a value to correct.
            if csv_field == "Cached Input Price" and old_price is None and new_price is not None:
                continue

            if not price_changed(old_price, new_price):
                continue

            new_text = format_price(new_price, old_text)
            row[csv_field] = new_text
            field_changes.append(FieldChange(csv_field, old_text or "(blank)", new_text or "(blank)"))

        if field_changes:
            changes.append(
                RowChange(
                    model_name=row["Model Name"],
                    model_date=row.get("Model Date", ""),
                    minimum_tokens=row.get("Minimum Tokens", ""),
                    upstream_key=upstream_key,
                    changes=field_changes,
                )
            )

    return changes, matched_upstream_keys, represented_models, rows


def candidate_models(
    upstream: dict[str, dict[str, Any]], matched_upstream_keys: set[str], represented_models: set[str]
) -> list[tuple[str, Decimal | None, Decimal | None, Decimal | None]]:
    candidates: list[tuple[str, Decimal | None, Decimal | None, Decimal | None]] = []
    for key in sorted(upstream):
        if key in matched_upstream_keys or key in represented_models:
            continue
        model_info = upstream[key]
        if model_info.get("litellm_provider") != "openai":
            continue
        input_price = upstream_price(model_info, "input_cost_per_token", "input_cost_per_token", 0)
        output_price = upstream_price(model_info, "output_cost_per_token", "output_cost_per_token", 0)
        if input_price is None and output_price is None:
            continue
        cached_price = upstream_price(
            model_info,
            "cache_read_input_token_cost",
            "cache_read_input_token_cost",
            0,
        )
        candidates.append((key, input_price, cached_price, output_price))
    return candidates


def display_price(value: Decimal | None) -> str:
    if value is None:
        return "(blank)"
    return format(value.normalize(), "f")


def write_summary(
    path: Path,
    changes: list[RowChange],
    added_rows: list[AddedRow],
    candidates: list[tuple[str, Decimal | None, Decimal | None, Decimal | None]],
) -> None:
    lines = [
        "# Pricing Drift Check",
        "",
        f"Source: `{UPSTREAM_URL}`",
        "",
    ]

    if changes:
        lines.extend(["## Changed Rows", ""])
        for change in changes:
            row_label = change.model_name
            if change.model_date:
                row_label += f" ({change.model_date})"
            min_tokens = change.minimum_tokens or "0"
            lines.append(f"### {row_label}, minimum tokens {min_tokens}")
            lines.append("")
            lines.append(f"Upstream key: `{change.upstream_key}`")
            lines.append("")
            for field_change in change.changes:
                lines.append(f"- {field_change.field}: `{field_change.old}` -> `{field_change.new}`")
            lines.append("")
    else:
        lines.extend(["## Changed Rows", "", "No existing row price drift detected.", ""])

    lines.extend(["## Added GPT Models", ""])
    if added_rows:
        lines.append(
            f"Automatically added OpenAI GPT models with major version {AUTO_ADD_GPT_MIN_MAJOR} or newer."
        )
        lines.append("")
        lines.append("| Model | Date | Minimum tokens | Input | Cached input | Output | Upstream key |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for added_row in added_rows:
            row = added_row.row
            lines.append(
                f"| `{added_row.model_name}` | {added_row.model_date or '(blank)'} | "
                f"{added_row.minimum_tokens} | {row['Input Price'] or '(blank)'} | "
                f"{row['Cached Input Price'] or '(blank)'} | {row['Output Price'] or '(blank)'} | "
                f"`{added_row.upstream_key}` |"
            )
        lines.append("")
    else:
        lines.extend(["No new auto-addable GPT models detected.", ""])

    lines.extend(["## Upstream Models Not In CSV", ""])
    if candidates:
        lines.append("These are candidates only. They do not match the conservative GPT auto-add filter.")
        lines.append("")
        lines.append("| Upstream model | Input | Cached input | Output |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key, input_price, cached_price, output_price in candidates:
            lines.append(
                f"| `{key}` | {display_price(input_price)} | "
                f"{display_price(cached_price)} | {display_price(output_price)} |"
            )
    else:
        lines.append("No candidate OpenAI models found.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        upstream = fetch_upstream()
        fieldnames, rows, lineterminator = read_csv(CSV_PATH)
        changes, matched_upstream_keys, represented_models, updated_rows = apply_updates(rows, upstream)
        added_rows = new_gpt_rows(upstream, represented_models, updated_rows)
        if added_rows:
            updated_rows.extend(added_row.row for added_row in added_rows)
            matched_upstream_keys.update(added_row.upstream_key for added_row in added_rows)
            represented_models.update(added_row.model_name for added_row in added_rows)
            represented_models.update(added_row.upstream_key for added_row in added_rows)
        candidates = candidate_models(upstream, matched_upstream_keys, represented_models)
        write_summary(SUMMARY_PATH, changes, added_rows, candidates)

        if changes or added_rows:
            write_csv(CSV_PATH, fieldnames, updated_rows, lineterminator)
            print(f"Pricing drift detected: updated {CSV_PATH} and wrote {SUMMARY_PATH}.")
        else:
            print(f"No pricing drift detected. Wrote {SUMMARY_PATH}.")
        return 0
    except RuntimeError as exc:
        print(f"Pricing check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
