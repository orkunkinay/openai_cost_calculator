import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "gpt_pricing_data.csv"
EXPECTED_COLUMNS = [
    "Model Name",
    "Model Date",
    "Input Price",
    "Cached Input Price",
    "Output Price",
    "Minimum Tokens",
]


def _assert_price(value: str, *, field: str, line_num: int, allow_empty: bool = False) -> None:
    value = value.strip()
    if not value:
        assert allow_empty, f"{field} is required on CSV line {line_num}"
        return

    try:
        price = Decimal(value)
    except InvalidOperation:
        raise AssertionError(f"{field} must be a decimal number on CSV line {line_num}: {value!r}")

    assert price >= 0, f"{field} must be non-negative on CSV line {line_num}: {value!r}"


def _cell(row: dict[str, str | None], field: str, *, line_num: int) -> str:
    value = row[field]
    assert value is not None, f"{field} is missing on CSV line {line_num}"
    return value


def _assert_model_date(value: str, *, line_num: int) -> None:
    value = value.strip()
    if not value:
        return

    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise AssertionError(f"Model Date must be YYYY-MM-DD or empty on CSV line {line_num}: {value!r}")


def _assert_minimum_tokens(value: str, *, line_num: int) -> int:
    value = value.strip()
    assert value, f"Minimum Tokens is required on CSV line {line_num}"
    assert value.isdecimal(), f"Minimum Tokens must be a non-negative integer on CSV line {line_num}: {value!r}"
    return int(value)


def test_pricing_csv_schema_and_value_types():
    with CSV_PATH.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        assert reader.fieldnames == EXPECTED_COLUMNS

        seen_tiers = set()
        minimum_tokens_by_model_date = {}
        row_count = 0

        for row in reader:
            row_count += 1
            assert set(row) == set(EXPECTED_COLUMNS), (
                f"CSV line {reader.line_num} has missing or extra columns: {row!r}"
            )

            model_name = _cell(row, "Model Name", line_num=reader.line_num).strip()
            model_date = _cell(row, "Model Date", line_num=reader.line_num).strip()
            assert model_name, f"Model Name is required on CSV line {reader.line_num}"
            _assert_model_date(model_date, line_num=reader.line_num)
            _assert_price(
                _cell(row, "Input Price", line_num=reader.line_num),
                field="Input Price",
                line_num=reader.line_num,
            )
            _assert_price(
                _cell(row, "Cached Input Price", line_num=reader.line_num),
                field="Cached Input Price",
                line_num=reader.line_num,
                allow_empty=True,
            )
            _assert_price(
                _cell(row, "Output Price", line_num=reader.line_num),
                field="Output Price",
                line_num=reader.line_num,
            )
            minimum_tokens = _assert_minimum_tokens(
                _cell(row, "Minimum Tokens", line_num=reader.line_num),
                line_num=reader.line_num,
            )

            model_date_key = (model_name, model_date)
            tier_key = (*model_date_key, minimum_tokens)
            assert tier_key not in seen_tiers, (
                "Duplicate pricing tier for "
                f"Model Name={model_name!r}, Model Date={model_date!r}, "
                f"Minimum Tokens={minimum_tokens!r}"
            )
            seen_tiers.add(tier_key)
            minimum_tokens_by_model_date.setdefault(model_date_key, set()).add(minimum_tokens)

        assert row_count > 0, "Pricing CSV must contain at least one data row"

    missing_base_tiers = sorted(
        (model_name, model_date)
        for (model_name, model_date), minimum_tokens in minimum_tokens_by_model_date.items()
        if 0 not in minimum_tokens
    )
    assert missing_base_tiers == []
