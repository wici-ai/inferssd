"""F's decode response fit and pricing, preserving arithmetic and iteration order."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, TypedDict, Union, cast

from inferssd.contracts import integer, mapping, numeric, sequence

Number = Union[int, float]
LayerSample = tuple[Number, Number, Number]
LayerRow = tuple[int, int]


class PriceInputsCore(TypedDict):
    layer_table: Mapping[str, Sequence[Number]]
    layer_beta: Sequence[Number]
    prefetch_share: Number
    pf_k: int
    t_out_ms: Number


class PriceInputs(PriceInputsCore, total=False):
    ready_given_nx: Optional[Mapping[str, Sequence[Number]]]
    ready_share_all: Optional[Number]


@dataclass(frozen=True)
class LayerResponse:
    """Nested view of a legacy placement, not a new top-level artifact schema.

    Tuple order preserves the original table summation. None versus empty ready
    counts and absent versus explicitly zero/null ready share stay distinguishable.
    """

    layer_table: tuple[tuple[str, LayerSample], ...]
    layer_beta: tuple[Number, ...]
    prefetch_share: Number
    pf_k: int
    t_out_ms: Number
    ready_given_nx: Optional[tuple[tuple[str, tuple[Number, ...]], ...]]
    ready_share_all_present: bool
    ready_share_all: Optional[Number]

    @classmethod
    def from_legacy(cls, data: Mapping[str, object]) -> LayerResponse:
        table = []
        for key, raw in mapping(data["layer_table"], "layer_table").items():
            values = sequence(raw, "layer sample")
            table.append(
                (
                    key,
                    (
                        numeric(values[0], "count"),
                        numeric(values[1], "ms"),
                        numeric(values[2], "cpu_ms"),
                    ),
                )
            )
        ready = data.get("ready_given_nx")
        share = data.get("ready_share_all")
        return cls(
            tuple(table),
            tuple(numeric(value, "beta") for value in sequence(data["layer_beta"], "layer_beta")),
            numeric(data["prefetch_share"], "prefetch_share"),
            integer(data["pf_k"], "pf_k"),
            numeric(data["t_out_ms"], "t_out_ms"),
            None
            if ready is None
            else tuple(
                (key, tuple(numeric(n, "ready count") for n in sequence(value, "ready counts")))
                for key, value in mapping(ready, "ready_given_nx").items()
            ),
            "ready_share_all" in data,
            None if share is None else numeric(share, "ready_share_all"),
        )


@dataclass(frozen=True)
class DecodeEstimate:
    ms_per_token: float
    tok_s: float
    sink_ms: float
    cpu_ms: float

    def to_mapping(self) -> dict[str, float]:
        return {
            "ms_per_token": self.ms_per_token,
            "tok_s": self.tok_s,
            "sink_ms": self.sink_ms,
            "cpu_ms": self.cpu_ms,
        }


def fit_response(x: Sequence[Sequence[Number]], y: Sequence[Number]) -> tuple[float, ...]:
    """Normal equations, column-order elimination and the original scaled pivot rule.

    This deliberately retains zip truncation and the original failure behavior for
    empty/ragged input; it neither normalizes nor exchanges dependent columns.
    """
    n = len(x[0])
    a = [
        [sum(row[i] * row[j] for row in x) for j in range(n)]
        + [sum(row[i] * value for row, value in zip(x, y))]
        for i in range(n)
    ]
    scale = [a[i][i] for i in range(n)]
    for c in range(n):
        if a[c][c] <= 1e-9 * scale[c]:
            a[c] = [0.0] * (n + 1)
            continue
        for r in range(n):
            if r != c:
                factor = a[r][c] / a[c][c]
                a[r] = [u - factor * v for u, v in zip(a[r], a[c])]
    return tuple(a[i][n] / a[i][i] if a[i][i] else 0.0 for i in range(n))


def price_decode(
    rows: Sequence[Sequence[LayerRow]], response: LayerResponse, cpu_scale: Number
) -> DecodeEstimate:
    """Adapt the validated response to the one F Mapping pricing implementation. Uncapped null
    readiness now raises F's native TypeError at exponentiation; the old adapter's bespoke error
    text is not retained."""
    inputs: PriceInputs = {
        "layer_table": dict(response.layer_table),
        "layer_beta": response.layer_beta,
        "prefetch_share": response.prefetch_share,
        "pf_k": response.pf_k,
        "t_out_ms": response.t_out_ms,
        "ready_given_nx": dict(response.ready_given_nx or ()),
    }
    if response.ready_share_all_present:
        inputs["ready_share_all"] = response.ready_share_all
    return DecodeEstimate(**price_mapping(rows, inputs, cpu_scale))


def price_mapping(
    rows: Sequence[Sequence[LayerRow]], ins: PriceInputs, cpu_scale: Number
) -> dict[str, float]:
    """Mean ms per token for rows from simulate(), from the engine's measured per-layer response
    (insitu.py): a layer with nv tier hits and nx other experts has nr of those host-ready
    (prefetched, or in a RAM cache) with the bootstrap's own distribution of nr given nx (any
    mechanism); for an nx the bootstrap saw too rarely, Binomial(min(nx, pf_k), share) when
    readiness never exceeded pf_k a layer (prefetch only), else Binomial(nx, share of all non-
    tier experts). The rest are read from the SSD; a layer's time is the bootstrap's mean for
    its class (linear fit for rare classes), with the CPU compute part scaled for another CPU
    set; cpu_ms is that CPU compute part of the sink time (the rest waits on the drive or the
    GPU)."""
    table, beta = ins["layer_table"], ins["layer_beta"]
    a, k = min(1.0, ins["prefetch_share"]), ins["pf_k"]
    rd = ins.get("ready_given_nx") or {}
    capped = all(j <= k for c in rd.values() for j, x in enumerate(c) if x)
    a_all = ins.get("ready_share_all", a)
    memo: dict[LayerRow, tuple[float, float]] = {}

    seen = [e for e in table.values() if e[0] >= 5]
    cpu_share = sum(e[0] * e[2] for e in seen) / max(1e-9, sum(e[0] * e[1] for e in seen))

    def cls_ms(nv: int, nr: int, nm: int) -> tuple[float, float]:
        # (layer time, its CPU compute part); a rare class: the linear fit, split at the table's CPU
        # share
        e = table.get(f"{nv},{nr},{nm}")
        if e and e[0] >= 5:
            return e[1] + (cpu_scale - 1.0) * e[2], cpu_scale * e[2]
        t = beta[0] + beta[1] * nv + beta[2] * nr + beta[3] * nm + beta[4] * (1 if nm else 0)
        return t, cpu_share * t

    def ready_dist(nx: int) -> list[float]:
        c = rd.get(str(nx))
        if c and sum(c) >= 20:
            return [x / sum(c) for x in c]
        m, q = (min(nx, k), a) if capped else (nx, cast(Number, a_all))
        return [math.comb(m, j) * q**j * (1 - q) ** (m - j) for j in range(m + 1)]

    def layer_ms(nv: int, nx: int) -> tuple[float, float]:
        key = (nv, nx)
        if key not in memo:
            c = [(pj, cls_ms(nv, j, nx - j)) for j, pj in enumerate(ready_dist(nx)) if pj]
            memo[key] = (sum(pj * t for pj, (t, _) in c), sum(pj * u for pj, (_, u) in c))
        return memo[key]

    tot = cpu = 0.0
    for row in rows:
        for nv, nx in row:
            t, u = layer_ms(nv, nx)
            tot += t
            cpu += u
    n = max(1, len(rows))
    sink = tot / n
    ms = ins["t_out_ms"] + sink
    return {"ms_per_token": ms, "tok_s": 1000.0 / ms, "sink_ms": sink, "cpu_ms": cpu / n}
