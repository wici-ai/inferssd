"""F's prefill calibration from supplied observations, without artifact loading."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, cast

from ssdash.model.prefill_lane import RING
from ssdash.model.prefill_memory import MemoryModel, MemoryObservation
from ssdash.model.prefill_model import (
    GraphCosts,
    LaneCosts,
    LaneGeometry,
    PassState,
    PrefillModel,
    coverage,
    pass_seconds,
)
from ssdash.model.routing_replay import Routing

GiB = 2**30


class PrefillPointCore(TypedDict):
    n_tokens: float
    unique: float
    unique_per_layer: Sequence[float]
    tier: float
    ssd_gb: float
    prompt_ms: float


class PrefillPoint(PrefillPointCore, total=False):
    cpu_s: Optional[float]
    routes: Optional[float]
    lane: Optional[LaneCosts]
    layers_s: Optional[float]


class PrefillEngine(TypedDict, total=False):
    in_place: bool
    route_bytes: Optional[float]
    ring: Optional[float]
    ring_locked: float
    engine: Optional[str]


class PlacementCore(TypedDict):
    prefill: Sequence[PrefillPoint]
    tier_slots: float


class Placement(PlacementCore, total=False):
    memory: Optional[MemoryObservation]
    prefill_engine: Optional[PrefillEngine]
    prefill_ubatch: Optional[int]


class GraphPass(TypedDict):
    n_tokens: float
    passes: float
    bytes: float
    read_s: float
    prompt_s: float


class GraphObservation(TypedDict, total=False):
    passes: Optional[Sequence[GraphPass]]
    memory: Optional[MemoryObservation]


class InsituCore(TypedDict):
    placements: Mapping[str, Placement]
    reference: str
    n_experts: float
    layers: float


class InsituView(InsituCore, total=False):
    topk: Optional[float]
    union: Optional[Sequence[Sequence[float]]]
    graph: Optional[GraphObservation]
    prefill_ubatch: Optional[int]


class ReadRate(TypedDict):
    gibs: float


class GPUView(TypedDict, total=False):
    h2d_pinned_gbs: Optional[float]
    integrated: bool


class DeviceView(TypedDict, total=False):
    ssd: Optional[Mapping[str, ReadRate]]
    ssd_knee_qd: Optional[int]
    gpu: Optional[GPUView]
    expert_read_bytes: Optional[float]
    bytes_per_role: Optional[Mapping[str, float]]


class CalibratedPass(PassState):
    n: float
    prompt_s: float


class CalibratedGraph(GraphCosts):
    n: list[float]


class CalibratedModel(PrefillModel, MemoryModel, total=False):
    union_scale: Optional[Sequence[float]]


def union_points(segments: Routing) -> list[list[float]]:
    """Mean distinct experts per layer in overlapping power-of-two windows and the complete
    ordered routing sequence.
    """
    seq = [t for s in segments for t in s]
    if not seq:
        return []
    N, L = len(seq), len(seq[0])
    out = []
    for n in sorted({1 << i for i in range(N.bit_length()) if 1 << i <= N} | {N}):
        tot = cnt = 0
        for st in range(0, N - n + 1, max(1, n // 4)):
            win = seq[st : st + n]
            for li in range(L):
                tot += len({e for tok in win for e in tok[li]})
                cnt += 1
        out.append([n, round(tot / cnt, 2)])
    return out


def rest_fit(points: Iterable[Sequence[float]]) -> tuple[float, float]:
    """Fit per-pass and per-token seconds, retaining the original singular and negative-
    coefficient fallbacks.
    """
    sp, sn, sr = (sum(x[i] for x in points) for i in range(3))
    spp, snn, spn = (
        sum(p * p for p, _, _ in points),
        sum(n * n for _, n, _ in points),
        sum(p * n for p, n, _ in points),
    )
    spr, snr = sum(p * r for p, _, r in points), sum(n * r for _, n, r in points)
    det = spp * snn - spn * spn
    a, d = (
        ((spr * snn - snr * spn) / det, (snr * spp - spr * spn) / det)
        if det > 1e-9 * spp * snn
        else (-1.0, 0.0)
    )
    if a < 0:
        return 0.0, sr / sn
    if d < 0:
        return sr / sp, 0.0
    return a, d


def build(ins: InsituView, dev: DeviceView, routing: Routing = ()) -> CalibratedModel:
    """Build a prefill model from recorded bootstrap/device observations and routing; nested
    source observations retain their identity.
    """
    P, ref = ins["placements"], ins["reference"]
    E, L = ins["n_experts"], ins["layers"]
    k = ins.get("topk") or len(routing[0][0][0])
    pts = sorted(P[ref]["prefill"], key=lambda x: x["tier"])
    ssd, knee, gpu = dev.get("ssd") or {}, dev.get("ssd_knee_qd"), dev.get("gpu") or {}
    r_d = ssd[str(knee)]["gibs"] * GiB if str(knee) in ssd else None
    link = (
        cast(float, gpu["h2d_pinned_gbs"]) * 1e9
        if gpu.get("h2d_pinned_gbs") and not gpu.get("integrated")
        else None
    )
    rates = [r for r in (r_d, link) if r]
    b_e = dev.get("expert_read_bytes") or statistics.median(
        x["ssd_gb"] * 1e9 / (x["unique"] - x["tier"]) for x in pts if x["unique"] > x["tier"]
    )
    # the graph's copy target: one role tensor of a layer's experts, the largest (the header's role
    # slices; without
    # them a whole expert's bytes bound it)
    role = E * max((dev.get("bytes_per_role") or {}).values() or [b_e])
    g = ins.get("graph") or {}
    gp = [x for x in g.get("passes") or [] if x["bytes"] and x["read_s"]]
    # U(n): the prefill passes (the sink's distinct experts over the layers they report, the graph's
    # copied bytes per
    # pass); below the smallest, the decode trace scaled to it at its size, when the trace reaches
    # it within one
    # doubling (the spacing of its points), else as it is
    dec = {n: v for n, v in (ins.get("union") or union_points(routing[:1]))}
    prompt = {x["n_tokens"]: x["unique"] / len(x["unique_per_layer"]) for x in pts}
    for x in gp:
        prompt[round(x["n_tokens"] / x["passes"])] = x["bytes"] / (x["passes"] * L * b_e)
    u, scale = dict(dec), None
    if prompt and dec and min(prompt) <= 2 * max(dec):
        n0 = min(prompt)
        scale = [
            n0,
            round(
                prompt[n0]
                / coverage({"E": E, "k": k, "union": sorted(map(list, dec.items()))}, n0),
                4,
            ),
        ]
        u = {n: v * scale[1] for n, v in dec.items() if n < n0}
    u.update(prompt)
    union, top = [], 0.0
    for n in sorted(u):
        top = max(top, u[n])
        union.append([n, round(top, 2)])
    graph: Optional[CalibratedGraph] = None
    if gp:
        a, d = rest_fit(
            [(x["passes"], x["n_tokens"], max(0.0, x["prompt_s"] - x["read_s"])) for x in gp]
        )
        graph = {
            "R": sum(x["bytes"] for x in gp) / sum(x["read_s"] for x in gp),
            "a": a,
            "d": d,
            "n": [x["n_tokens"] for x in gp],
        }

    def state(x: PrefillPoint) -> CalibratedPass:
        # r: the routes a prompt token takes, as the sink counted them: k L, less where a layer
        # computes fewer tokens
        # (Muse's last layer only the output's)
        cpu = x.get("cpu_s")
        c = cpu / cast(float, x["routes"]) if cpu and x.get("routes") else 0.0
        r = cast(float, x["routes"]) / x["n_tokens"] if x.get("routes") else k * L
        read = x["ssd_gb"] * 1e9 / r_d if r_d else 0.0
        out: CalibratedPass = {
            "n": x["n_tokens"],
            "c": c,
            "r": r,
            "prompt_s": x["prompt_ms"] / 1e3,
            "d": max(0.0, (x["prompt_ms"] / 1e3 - max(read, cpu or 0.0)) / x["n_tokens"]),
        }
        if x.get("lane") and x.get("layers_s") is not None:
            # the pass with the GPU lane: its calibrated table, and its time a token outside its
            # layers' sink time
            cast(dict[str, object], out).update(
                lane=x["lane"],
                d=max(0.0, (x["prompt_ms"] / 1e3 - cast(float, x["layers_s"])) / x["n_tokens"]),
            )
        return out

    passes = {}
    for p in P:
        mine = sorted(P[p]["prefill"], key=lambda x: x["tier"])
        passes[p] = {"cold": state(mine[0]), "warm": state(mine[-1])}
    engine = ins["placements"][ref].get("prefill_engine") or {}
    return {
        "E": E,
        "k": k,
        "L": L,
        "B_e": b_e,
        "R_d": r_d,
        "R_io": min(rates) if rates else None,
        "union": union,
        "lane": cast(
            LaneGeometry,
            {
                "in_place": bool(engine.get("in_place")),
                "route_bytes": engine.get("route_bytes") or 0.0,
                "engine": engine.get("engine"),
                "ring": engine.get("ring") or RING,
                "ring_locked": bool(engine.get("ring_locked", 1)),
            },
        )
        if engine
        else None,
        "union_scale": scale,
        "graph": graph,
        "graph_memory": g.get("memory") if gp else None,
        "role_mib": role / 2**20,
        "integrated": bool(gpu.get("integrated")),
        "cov_warm": pts[-1]["tier"] / pts[-1]["unique"],
        "tier_at_warm": P[ref]["tier_slots"],
        "pass": passes,
        "memory": {p: P[p].get("memory") or {} for p in P},
        "ubatch": ins.get("prefill_ubatch"),
        "ubatch_p": {p: P[p].get("prefill_ubatch") or ins.get("prefill_ubatch") for p in P},
    }


def check(m: PrefillModel) -> list[tuple[str, str, float, float, float, Optional[float]]]:
    """Compare each recorded lane prefill with its own and opposite tier-state cost table; return
    seconds in placement/state order.
    """
    rows = []
    for p in m["pass"]:
        for s, other in (("cold", "warm"), ("warm", "cold")):
            st = cast(CalibratedPass, m["pass"][p][s])
            if not st.get("lane") or not st.get("prompt_s"):
                continue
            first = s == "cold"
            own = pass_seconds(m, p, st["n"], False, first, m["tier_at_warm"])[0]
            ot, cross = m["pass"][p][other], None
            if ot.get("lane"):
                alt = {
                    **m,
                    "pass": {
                        **m["pass"],
                        p: {**m["pass"][p], s: dict(st, lane=ot["lane"], d=ot["d"])},
                    },
                }
                cross = pass_seconds(
                    cast(PrefillModel, alt), p, st["n"], False, first, m["tier_at_warm"]
                )[0]
            rows.append((p, s, st["n"], st["prompt_s"], own, cross))
    return rows
