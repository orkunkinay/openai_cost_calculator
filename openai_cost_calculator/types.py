from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal     import Decimal
from typing      import Dict


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Strongly‑typed view of a cost estimate (all values are Decimal)."""
    prompt_cost_uncached: Decimal
    prompt_cost_cached:   Decimal
    completion_cost:      Decimal
    total_cost:           Decimal

    # -- helpers ------------------------------------------------------------
    def as_dict(self, stringify: bool = True) -> Dict[str, str | Decimal]:
        """
        Return the four fields as a plain dict.
        * If ``stringify`` (default) ⇒ 8‑dp strings (legacy format)
        * Else ⇒ raw Decimal objects
        """
        if stringify:
            return {k: f"{v:.8f}" for k, v in asdict(self).items()}
        return asdict(self)
