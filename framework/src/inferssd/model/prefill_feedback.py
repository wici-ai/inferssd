"""F's prefill feedback on supplied turns and an injected route callback."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Optional, TypedDict, Union, cast

from inferssd.model.prefill_model import PrefillModel, seconds

Route = Callable[[str], tuple[Optional[float], Optional[int]]]


class ServedTurn(TypedDict):
    cpus: str
    n_prompt: int
    tier: float
    first: bool
    cpu_s: Optional[float]
    read_bytes: Optional[float]
    prefill_s: float


class PrefillFeedback(TypedDict):
    pf_cpu: dict[str, float]
    k_bytes: float
    pf_drive: float


def drive_factor(
    time_of: Callable[[float], float], measured: float, lo: float = 1 / 16, hi: float = 16.0
) -> Optional[float]:
    """Find the read-time factor by forty log-space bisections; preserve callback order and
    return None when the observed time cannot identify it.
    """
    if time_of(hi) <= time_of(lo) or measured <= time_of(lo):
        return None
    if measured >= time_of(hi):
        return hi
    for _ in range(40):
        f = math.sqrt(lo * hi)
        lo, hi = (f, hi) if time_of(f) < measured else (lo, f)
    return math.sqrt(lo * hi)


def feedback(
    m: PrefillModel, turns: Sequence[ServedTurn], how: Route, recent: int = 1
) -> PrefillFeedback:
    """Measured/modelled CPU, bytes and recent read-time factors from supplied turns, using the
    injected placement threshold/ubatch callback.
    """

    def model(t: ServedTurn, **kw: float) -> tuple[float, float, float]:
        T, ub = how(t["cpus"])
        return seconds(m, t["cpus"], t["n_prompt"], t["tier"], t["first"], threshold=T, ub=ub, **kw)

    out: PrefillFeedback = {"pf_cpu": {}, "k_bytes": 1.0, "pf_drive": 1.0}
    for p in sorted({t["cpus"] for t in turns}):
        pc = [(t["cpu_s"], model(t)[1]) for t in turns if t["cpus"] == p and t["cpu_s"]]
        if pc and sum(c for _, c in pc) > 0:
            out["pf_cpu"][p] = sum(x for x, _ in pc) / sum(c for _, c in pc)
    rd: Union[list[tuple[float, float]], list[float]] = [
        (t["read_bytes"], model(t)[2]) for t in turns if t["read_bytes"]
    ]
    rd = [x / b for x, b in cast(list[tuple[float, float]], rd) if b > 0]
    if rd:
        out["k_bytes"] = statistics.mean(rd)
    fs = []
    for t in turns[-recent:]:
        kw = {"k_bytes": out["k_bytes"], "k_cpu": out["pf_cpu"].get(t["cpus"], 1.0)}
        f = drive_factor(lambda f: model(t, k_drive=f, **kw)[0], t["prefill_s"])
        if f:
            fs.append(f)
    if fs:
        out["pf_drive"] = statistics.mean(fs)
    return out
