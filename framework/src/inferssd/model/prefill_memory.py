"""F's graph-memory bounds on recorded observations; this module reads no device."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, TypedDict, Union

from inferssd.model.prefill_workload import OFFLOAD_MIN


class MemoryObservation(TypedDict, total=False):
    free_mib: float
    dense_mib: float
    kv_mib: float
    compute_mib: float
    host_compute_mib: float
    n_ubatch: float
    reserves: Sequence[Union[list[float], tuple[float, ...]]]


class MemoryCore(TypedDict):
    role_mib: float
    memory: Mapping[str, MemoryObservation]


class MemoryModel(MemoryCore, total=False):
    graph_memory: Optional[MemoryObservation]
    integrated: bool


class GraphLine(TypedDict):
    role: float
    slope: float
    host: float
    n: float


def graph_line(m: MemoryModel, p: str) -> Optional[GraphLine]:
    """Role-copy MiB plus device/host MiB per token from the largest eligible recorded graph
    reserve; None without one.
    """
    role = m["role_mib"]
    g = m.get("graph_memory")
    for src, mine in ((g, lambda r: True), (m["memory"].get(p), lambda r: r[3])):
        rs = [
            r
            for r in (src or {}).get("reserves") or []
            if mine(r) and r[0] >= OFFLOAD_MIN and r[1] > role
        ]
        if (
            src is g
            and g
            and "reserves" not in g
            and g.get("compute_mib", 0) > role
            and g.get("n_ubatch")
        ):
            # the graph's prefills before reserves were read: its reserve at load is a graph pass's
            rs = [[g["n_ubatch"], g["compute_mib"], g.get("host_compute_mib", 0.0), 1]]
        if rs:
            n, dev, host, _ = max(rs)
            return {"role": role, "slope": (dev - role) / n, "host": host / n, "n": n}
    return None


def graph_pass_max(m: MemoryModel, p: str, limit: int) -> int:
    """Largest graph pass up to limit tokens admitted by recorded device memory, counting host
    buffers on unified memory.
    """
    mem = m.get("graph_memory") or m["memory"].get(p) or {}
    g = graph_line(m, p)
    if not (g and mem.get("free_mib") and mem.get("dense_mib")):
        return 0
    room = mem["free_mib"] - mem["dense_mib"] - mem.get("kv_mib", 0.0) - 1024 - g["role"]
    per_token = g["slope"] + (g["host"] if m.get("integrated") else 0.0)
    return max(0, min(limit, int(room / per_token))) if room > 0 else 0
