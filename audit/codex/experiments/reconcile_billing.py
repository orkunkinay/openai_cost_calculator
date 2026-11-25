"""Compare an isolated project's daily Costs API result with an expected USD total."""

import argparse
import json
import os
from datetime import date, datetime, time, timezone
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", type=date.fromisoformat, help="Completed UTC billing date")
    parser.add_argument("project_id")
    parser.add_argument("expected_usd", type=Decimal)
    parser.add_argument("--tolerance-usd", type=Decimal, required=True)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_ADMIN_KEY")
    if not key:
        parser.exit(2, "Unverified: OPENAI_ADMIN_KEY is unavailable\n")
    if args.day >= datetime.now(timezone.utc).date():
        parser.error("use a completed UTC day")
    if any(not value.is_finite() or value < 0 for value in
           (args.expected_usd, args.tolerance_usd)):
        parser.error("expected cost and tolerance must be finite and nonnegative")
    start = int(datetime.combine(args.day, time.min, timezone.utc).timestamp())
    query = {
        "start_time": start, "end_time": start + 86400, "bucket_width": "1d",
        "project_ids[]": args.project_id, "group_by[]": ["project_id", "line_item"],
    }
    total = Decimal(0)
    rows = 0
    while True:
        request = Request(
            "https://api.openai.com/v1/organization/costs?" + urlencode(query, doseq=True),
            headers={"Authorization": "Bearer " + key},
        )
        try:
            with urlopen(request, timeout=30) as response:
                page = json.load(response, parse_float=Decimal)
        except HTTPError as error:
            parser.exit(2, f"Unverified: Costs API returned HTTP {error.code}\n")
        for bucket in page["data"]:
            if bucket["start_time"] != start:
                parser.exit(2, "Unverified: unexpected billing bucket\n")
            for row in bucket["results"]:
                if row["project_id"] != args.project_id or row["amount"]["currency"].lower() != "usd":
                    parser.exit(2, "Unverified: unexpected project or currency\n")
                total += Decimal(str(row["amount"]["value"]))
                rows += 1
        if not page["has_more"]:
            break
        next_page = page["next_page"]
        if not next_page or query.get("page") == next_page:
            parser.exit(2, "Unverified: invalid page cursor\n")
        query["page"] = next_page
    error = abs(total - args.expected_usd)
    status = "unverified_empty" if not rows else "match" if error <= args.tolerance_usd else "mismatch"
    print(json.dumps({
        "status": status, "expected_usd": str(args.expected_usd),
        "reported_usd": str(total), "absolute_error_usd": str(error),
        "tolerance_usd": str(args.tolerance_usd), "rows": rows,
    }, indent=2))
    return 0 if status == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
