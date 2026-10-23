"""F's prefill cost and threshold model, without artifact loading or validation."""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, cast

from inferssd.model.prefill_lane import RING, lane_layer
from inferssd.model.prefill_workload import (
    OFFLOAD_MIN,
    PassSizing,
    cold_share,
    cut,
    offloaded,
    served,
)

LaneCosts = Mapping[str, Optional[float]]


class CoverageModel(TypedDict):
    E: float
    k: float
    union: Sequence[Sequence[float]]


class PassStateCore(TypedDict):
    r: float
    c: float
    d: float


class PassState(PassStateCore, total=False):
    lane: Optional[LaneCosts]


class GraphCosts(TypedDict):
    R: float
    a: float
    d: float


class LaneGeometry(TypedDict, total=False):
    in_place: bool
    route_bytes: float
    ring: float
    ring_locked: bool


PassTables = TypedDict("PassTables", {"pass": Mapping[str, Mapping[str, PassState]]})


class PrefillCore(CoverageModel, PassTables):
    L: float
    B_e: float
    R_d: Optional[float]
    R_io: Optional[float]
    cov_warm: float
    tier_at_warm: float


class PrefillModel(PrefillCore, PassSizing, total=False):
    """A typing view of existing model keys; construction adds no runtime checks."""

    graph: Optional[GraphCosts]
    lane: Optional[LaneGeometry]


def coverage(m: CoverageModel, n: float) -> float:
    """U(n): through the model's points in log-log, past the last on the slope of the last two,
    below the first on uniform routing's shape; at most what uniform routing touches, E (1 -
    (1 - k/E)^n)."""
    if n <= 0:
        return 0.0
    E, k, pts = m["E"], m["k"], m["union"]
    cap: float = E * (1 - (1 - k / E) ** n)
    if not pts:
        return cap
    ns = [x for x, _ in pts]
    i = bisect.bisect_left(ns, n)
    if i < len(ns) and ns[i] == n:
        u = pts[i][1]
    elif i == 0 or len(pts) < 2:
        n0, u0 = pts[0] if i == 0 else pts[-1]
        u = u0 * cap / (E * (1 - (1 - k / E) ** n0))
    else:
        (n0, u0), (n1, u1) = pts[min(i, len(pts) - 1) - 1], pts[min(i, len(pts) - 1)]
        u = u0 * (n / n0) ** (math.log(u1 / u0) / math.log(n1 / n0))
    return min(cap, u)


def pass_seconds(
    m: PrefillModel,
    p: str,
    n: float,
    graph: bool,
    first: bool,
    tier: float,
    k_bytes: float = 1.0,
    k_cpu: float = 1.0,
    k_drive: float = 1.0,
) -> tuple[float, float, float]:
    """(seconds, CPU compute seconds, bytes read) of one pass of n tokens; graph: on the ggml
    graph, else the sink."""
    st = m["pass"][p]["cold" if first else "warm"]
    b = k_bytes * m["L"] * coverage(m, n) * m["B_e"]
    if graph:
        g = cast(GraphCosts, m.get("graph") or {"R": m["R_io"], "a": 0.0, "d": st["d"]})
        return k_drive * b / g["R"] + g["a"] + n * g["d"], 0.0, b
    cov = 0.0 if first else min(1.0, m["cov_warm"] * tier / m["tier_at_warm"])
    read = b * (1 - cov)
    if st.get("lane"):
        # the sink with the GPU lane: every layer alike (the union's mean experts and
        # routes), as the engine places it
        geo: LaneGeometry = m.get("lane") or {}
        U = coverage(m, n) * k_bytes
        routes = n * st["r"] / m["L"] / U if U > 0 else 0.0
        us, cpu_us, _ = lane_layer(
            cast(LaneCosts, st["lane"]),
            n,
            U,
            routes,
            cov,
            m["B_e"],
            geo.get("in_place", False),
            geo.get("route_bytes", 0.0),
            geo.get("ring", RING),
            geo.get("ring_locked", True),
            k_cpu=k_cpu,
            k_drive=k_drive,
        )
        return m["L"] * us / 1e6 + n * st["d"], m["L"] * cpu_us / 1e6, read
    cpu = k_cpu * n * st["r"] * st["c"]
    return max(k_drive * read / m["R_d"] if m["R_d"] else 0.0, cpu) + n * st["d"], cpu, read


def seconds(
    m: PrefillModel,
    p: str,
    n: int,
    tier: float,
    first: bool,
    k_bytes: float = 1.0,
    k_cpu: float = 1.0,
    k_drive: float = 1.0,
    threshold: Optional[float] = None,
    ub: Optional[int] = None,
) -> tuple[float, float, float]:
    """(seconds, CPU compute seconds, bytes read) of N new prompt tokens on placement p: passes of
    the ubatch ub (default the placement's), each on the sink up to the threshold (None:
    every pass) or below the GPU's offload minimum, and on the graph otherwise; first: the
    server's first turn, on a cold tier; k_*: feedback from served turns where there is
    some."""
    tot = cpu = rb = 0.0
    for q in cut(m, p, n, ub):
        graph = threshold is not None and q > threshold and q >= OFFLOAD_MIN
        s, c, b = pass_seconds(m, p, q, graph, first, tier, k_bytes, k_cpu, k_drive)
        tot, cpu, rb = tot + s, cpu + c, rb + b
    return tot, cpu, rb


def break_even(m: PrefillModel, p: str, cold: float = 0.0) -> Optional[float]:
    """Largest pass where the sink is no slower than the graph, weighted by cold and warm tier
    states. Preserve the original doubling/bisection bounds, missing-input None, and
    infinite result when no crossing is found."""
    states = [(w, s == "cold") for w, s in ((cold, "cold"), (1.0 - cold, "warm")) if w > 0]
    sts = [(w, m["pass"][p]["cold" if first else "warm"]) for w, first in states]

    def graph_faster(n: int) -> bool:
        return (
            sum(
                w
                * (
                    pass_seconds(m, p, n, False, first, m["tier_at_warm"])[0]
                    - pass_seconds(m, p, n, True, first, m["tier_at_warm"])[0]
                )
                for w, first in states
            )
            > 0
        )

    g = m.get("graph")
    if any(st.get("lane") for _, st in sts):
        # the sink with the GPU lane: its time is not linear in n (moves follow the
        # routes), so the crossing is found
        # by doubling from the offload minimum, then by bisection; none up to 2^20
        # tokens: the graph is never faster.
        # Only from a measured graph arm: without one the graph's time is a guess, and
        # no threshold is written.
        if not (g and m["B_e"]):
            return None
        hi = OFFLOAD_MIN
        while not graph_faster(hi):
            hi *= 2
            if hi > 1 << 20:
                return math.inf
        lo = hi // 2 if hi > OFFLOAD_MIN else 0
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if graph_faster(mid):
                hi = mid
            else:
                lo = mid
        return lo
    if not (all(st["c"] > 0 for _, st in sts) and m["R_d"] and m["R_io"] and m["B_e"]):
        return None
    R, a = (g["R"], g["a"]) if g else (m["R_io"], 0.0)
    slope = sum(w * (st["r"] * st["c"] + st["d"] - (g["d"] if g else st["d"])) for w, st in sts)
    if slope <= 0:
        return math.inf

    lo, hi = 0, int((m["E"] * m["L"] * m["B_e"] / R + a) / slope) + 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if graph_faster(mid):
            hi = mid
        else:
            lo = mid
    return lo


def threshold(
    m: PrefillModel,
    p: str,
    turns: Optional[Iterable[int]] = None,
    ub: Optional[int] = None,
    reply: Optional[int] = 0,
) -> Optional[float]:
    """Choose one pass threshold for a workload. Candidate sizes and both sides of each size cover
    route choices; equal costs keep the original distance-to-break-even and larger-threshold
    tie break. Replies and tier weights come from the existing workload functions."""
    if not turns:
        return break_even(m, p)
    ps = offloaded(m, p, turns, ub, reply)
    t = break_even(m, p, cold_share(ps))
    if t is None or not ps:
        return t
    # the workload's time changes only where T crosses one of its pass sizes: the break-even
    # and both sides of each
    # size cover every choice; among equal times, the one nearest the break-even
    cand = {t} | {q for q, _, _ in ps} | {q - 1 for q, _, _ in ps}
    forms = served(turns, reply)
    time = {
        T: sum(
            w * seconds(m, p, n, m["tier_at_warm"], first, threshold=T, ub=ub)[0]
            for n, first, w in forms
        )
        for T in cand
    }
    best = min(time.values())
    return min(
        (T for T in cand if time[T] == best), key=lambda T: (0 if T == t else abs(T - t), -T)
    )
