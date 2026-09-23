"""Human (Markdown) and machine (JSON) reports of a sync run."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Sequence

from ..catalog.io import format_decimal
from .runner import ProviderOutcome

_STATUS_LABEL = {
    "unchanged": "unchanged",
    "updated": "updated",
    "needs_review": "needs review",
    "failed": "FAILED - checked-in data kept",
}


def overall_status(outcomes: Sequence[ProviderOutcome]) -> str:
    statuses = {o.status for o in outcomes}
    for status in ("failed", "needs_review", "updated"):
        if status in statuses:
            return status
    return "unchanged"


def render_markdown(outcomes: Sequence[ProviderOutcome], *, today: date) -> str:
    lines: List[str] = [
        "# Pricing sync report",
        "",
        f"Run date: {today.isoformat()}. Changes listed under *Applied* are included in this "
        "update; everything under *Needs review* was **not** applied and needs a maintainer.",
        "",
        "| Provider | Status | Models fetched | Applied | Needs review | Source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for o in outcomes:
        lines.append(
            f"| {o.provider} | {_STATUS_LABEL[o.status]} | {o.fetched_models} | {len(o.applied)} | "
            f"{len(o.needs_review) + len(o.blocking_issues)} | [{o.source_id}]({o.source_url}) |"
        )
    for o in outcomes:
        if o.status == "unchanged":
            continue
        lines += ["", f"## {o.provider}", ""]
        if o.error:
            lines += [f"**Source failed:** {o.error}", ""]
        if o.applied:
            lines += ["### Applied", ""]
            for decision in o.applied:
                lines.append(f"- **{decision.diff.model_id}** ({decision.diff.kind})")
                if decision.diff.kind == "added" and decision.diff.new is not None:
                    base = decision.diff.new.prices[0].rates
                    rendered = ", ".join(f"{k} {format_decimal(v)}" for k, v in base.items())
                    lines.append(f"  - {rendered}")
                lines += [f"  - {change}" for change in decision.diff.rate_changes]
                lines += [f"  - {change}" for change in decision.diff.metadata_changes]
                lines += [f"  - _{note}_" for note in decision.notes]
            lines.append("")
        if o.needs_review or o.blocking_issues:
            lines += ["### Needs review (not applied)", ""]
            for decision in o.needs_review:
                lines.append(f"- **{decision.diff.model_id}** ({decision.diff.kind})")
                lines += [f"  - reason: {reason}" for reason in decision.reasons]
                lines += [f"  - {change}" for change in decision.diff.rate_changes]
            for issue in o.blocking_issues:
                target = f"**{issue.model_id}**: " if issue.model_id else ""
                lines.append(f"- parser: {target}{issue.message}")
            lines.append("")
        if o.notes:
            lines += ["### Parser notes (handled conservatively)", ""]
            for issue in o.notes:
                target = f"**{issue.model_id}**: " if issue.model_id else ""
                lines.append(f"- {target}{issue.message}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json(outcomes: Sequence[ProviderOutcome], *, today: date) -> str:
    def decision(d: Any) -> Dict[str, Any]:
        return {
            "model": d.diff.model_id,
            "kind": d.diff.kind,
            "applied": d.apply,
            "reasons": d.reasons,
            "rate_changes": [str(c) for c in d.diff.rate_changes],
            "metadata_changes": list(d.diff.metadata_changes),
        }

    payload = {
        "date": today.isoformat(),
        "status": overall_status(outcomes),
        "providers": [
            {
                "provider": o.provider,
                "status": o.status,
                "source": o.source_id,
                "error": o.error,
                "fetched_models": o.fetched_models,
                "written": o.written,
                "decisions": [decision(d) for d in o.decisions],
                "issues": [{"model": i.model_id, "message": i.message, "blocking": i.blocking} for i in o.issues],
            }
            for o in outcomes
        ],
    }
    return json.dumps(payload, indent=2) + "\n"
