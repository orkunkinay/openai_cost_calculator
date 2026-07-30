from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pricing.py"
SPEC = importlib.util.spec_from_file_location("occ_pricing_checker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pricing_checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pricing_checker
SPEC.loader.exec_module(pricing_checker)


def _row(model_name: str, input_price: str, cached_input_price: str, output_price: str) -> dict[str, str]:
    return {
        "Model Name": model_name,
        "Model Date": "",
        "Input Price": input_price,
        "Cached Input Price": cached_input_price,
        "Output Price": output_price,
        "Minimum Tokens": "0",
    }


def test_apply_updates_detects_existing_price_changes_without_touching_matching_rows():
    rows = [
        _row("gpt-price-changed", "1.0", "0.2", "2.0"),
        _row("gpt-price-matched", "3.0", "0.6", "6.0"),
    ]
    upstream = {
        "gpt-price-changed": {
            "input_cost_per_token": "0.0000015",
            "cache_read_input_token_cost": "0.0000003",
            "output_cost_per_token": "0.0000025",
        },
        "gpt-price-matched": {
            "input_cost_per_token": "0.000003",
            "cache_read_input_token_cost": "0.0000006",
            "output_cost_per_token": "0.000006",
        },
    }

    changes, matched_keys, _represented_models, updated_rows = pricing_checker.apply_updates(rows, upstream)

    assert matched_keys == {"gpt-price-changed", "gpt-price-matched"}
    assert [change.model_name for change in changes] == ["gpt-price-changed"]
    assert [(change.field, change.old, change.new) for change in changes[0].changes] == [
        ("Input Price", "1.0", "1.5"),
        ("Cached Input Price", "0.2", "0.3"),
        ("Output Price", "2.0", "2.5"),
    ]
    assert updated_rows == [
        _row("gpt-price-changed", "1.5", "0.3", "2.5"),
        _row("gpt-price-matched", "3.0", "0.6", "6.0"),
    ]
