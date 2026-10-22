"""F's pass geometry and weighted workload forms, without a planner or I/O."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, TypedDict

OFFLOAD_MIN = 32
WeightedPass = tuple[int, bool, float]


class PassSizing(TypedDict, total=False):
    """Typing view of existing model keys, not a new artifact schema or validator."""

    ubatch: Optional[int]
    ubatch_p: Optional[Mapping[str, Optional[int]]]


def cut(m: PassSizing, p: str, n: int, ub: Optional[int] = None) -> list[int]:
    """Split new tokens into full ubatches and a remainder, using F's truthy fallback."""
    if n <= 0:
        return []
    ub = ub or (m.get("ubatch_p") or {}).get(p) or m.get("ubatch") or n
    return [ub] * (n // ub) + ([n % ub] if n % ub else [])


def served(turns: Iterable[int], reply: Optional[int] = 0) -> list[WeightedPass]:
    """Return (new tokens, cold tier, weight) in the server's modeled turn order.

    The first turn occurs once on a cold tier. A nonzero reply gives each later
    turn two warm forms of half weight: its own tokens, and those tokens plus the
    reply if the rendered history cannot reuse it. No reply keeps each turn once.
    """
    out = []
    for i, n in enumerate(turns):
        if i == 0 or not reply:
            out.append((n, i == 0, 1.0))
        else:
            out += [(n, False, 0.5), (reply + n, False, 0.5)]
    return out


def offloaded(
    m: PassSizing,
    p: str,
    turns: Iterable[int],
    ub: Optional[int] = None,
    reply: Optional[int] = 0,
) -> list[WeightedPass]:
    """Return served passes at or above the existing GPU offload minimum."""
    return [
        (q, first, w)
        for n, first, w in served(turns, reply)
        for q in cut(m, p, n, ub)
        if q >= OFFLOAD_MIN
    ]


def cold_share(ps: Iterable[WeightedPass]) -> float:
    """Return the cold weight fraction, preserving F's two traversals of ps."""
    total = sum(w for _, _, w in ps)
    return sum(w for _, f, w in ps if f) / total if total else 0.0
