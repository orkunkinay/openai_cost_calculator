from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, List, Optional

from .types import CostBreakdown

# Maximum length for tag keys/values (hard limit to keep logs tidy)
_MAX_TAG_LENGTH = 128

_LOCK = RLock()
_GLOBAL_TAGS: Dict[str, str] = {}
_SINKS: List[Callable[[CostBreakdown], None]] = []


# ------------------------------- Tag helpers --------------------------------
def _normalize_and_validate(tags: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Normalize to {str:str} and reject keys/values longer than _MAX_TAG_LENGTH.
    None -> {}. Raises ValueError on violation.
    """
    if not tags:
        return {}
    out: Dict[str, str] = {}
    for k, v in tags.items():
        ks, vs = str(k), str(v)
        if len(ks) > _MAX_TAG_LENGTH:
            raise ValueError(f"Tag key too long (> {_MAX_TAG_LENGTH}): {ks!r}")
        if len(vs) > _MAX_TAG_LENGTH:
            raise ValueError(f"Tag value too long (> {_MAX_TAG_LENGTH}) for key {ks!r}")
        out[ks] = vs
    return out


def merge_tags(user_tags: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Merge global tags with user-supplied tags (user overrides).
    Both sides validated & normalized. Never returns None.
    """
    user = _normalize_and_validate(user_tags)
    with _LOCK:
        base = dict(_GLOBAL_TAGS)
    base.update(user)
    return base


# ----------------------------- Global tags API -------------------------------
def set_global_tags(tags: Dict[str, str]) -> None:
    """Replace the global default tags."""
    normalized = _normalize_and_validate(tags)
    with _LOCK:
        global _GLOBAL_TAGS
        _GLOBAL_TAGS = normalized


def get_global_tags() -> Dict[str, str]:
    """Return a shallow copy of current global tags."""
    with _LOCK:
        return dict(_GLOBAL_TAGS)


# ------------------------------- Sink hooks ----------------------------------
def register_cost_sink(sink: Callable[[CostBreakdown], None]) -> None:
    """
    Register a sink function that receives every CostBreakdown produced
    by estimate_cost_typed(). Keep fast; exceptions are swallowed.
    """
    if not callable(sink):
        raise TypeError("sink must be callable")
    with _LOCK:
        _SINKS.append(sink)


def unregister_cost_sink(sink: Callable[[CostBreakdown], None]) -> None:
    with _LOCK:
        try:
            _SINKS.remove(sink)
        except ValueError:
            pass


def clear_cost_sinks() -> None:
    with _LOCK:
        _SINKS.clear()


def emit_to_sinks(cost: CostBreakdown) -> None:
    # Snapshot under lock; call outside to avoid long-held locks
    with _LOCK:
        sinks = list(_SINKS)
    for fn in sinks:
        try:
            fn(cost)
        except Exception:
            # Never let sinks break cost estimation
            pass