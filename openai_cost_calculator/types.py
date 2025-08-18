from __future__ import annotations

from dataclasses import dataclass, asdict, field
from decimal     import Decimal
from typing      import Dict


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Strongly‑typed view of a cost estimate (all values are Decimal)."""
    prompt_cost_uncached: Decimal
    prompt_cost_cached:   Decimal
    completion_cost:      Decimal
    total_cost:           Decimal
    # Optional user defined context (feature, project, user, cost center, etc.)
    metadata:             Dict[str, str] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------
    def as_dict(
        self,
        stringify: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, str | Decimal | Dict[str, str]]:
        """
        Return fields as a plain dict.

        * If ``stringify`` (default) ⇒ 8-dp strings (legacy format for numeric fields)
        * Else ⇒ raw Decimal objects
        * Metadata is excluded by default; set ``include_metadata=True`` to include it.
        """
        payload = asdict(self)
        meta = payload.pop("metadata", {})
        if stringify:
            payload = {k: f"{v:.8f}" for k, v in payload.items()}
        if include_metadata:
            payload["metadata"] = meta
        return payload
