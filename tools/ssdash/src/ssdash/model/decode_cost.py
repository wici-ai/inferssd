"""F's decode response fit and pricing, preserving arithmetic and iteration order."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

from ssdash.contracts import integer, mapping, numeric, sequence

Number = Union[int, float]
LayerSample = tuple[Number, Number, Number]
LayerRow = tuple[int, int]


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
    rows: Sequence[Sequence[LayerRow]],
    response: LayerResponse,
    cpu_scale: Number,
) -> DecodeEstimate:
    """Price fixed replay rows from measured classes or the same F fallback fit."""
    table, beta = dict(response.layer_table), response.layer_beta
    a, k = min(1.0, response.prefetch_share), response.pf_k
    rd = dict(response.ready_given_nx or ())
    capped = all(j <= k for counts in rd.values() for j, count in enumerate(counts) if count)
    a_all = response.ready_share_all if response.ready_share_all_present else a
    memo: dict[LayerRow, tuple[float, float]] = {}

    seen = [entry for entry in table.values() if entry[0] >= 5]
    cpu_share = sum(entry[0] * entry[2] for entry in seen) / max(
        1e-9, sum(entry[0] * entry[1] for entry in seen)
    )

    def class_ms(nv: int, nr: int, nm: int) -> tuple[float, float]:
        entry = table.get(f"{nv},{nr},{nm}")
        if entry and entry[0] >= 5:
            return entry[1] + (cpu_scale - 1.0) * entry[2], cpu_scale * entry[2]
        t = beta[0] + beta[1] * nv + beta[2] * nr + beta[3] * nm + beta[4] * (1 if nm else 0)
        return t, cpu_share * t

    def ready_dist(nx: int) -> list[float]:
        counts = rd.get(str(nx))
        if counts and sum(counts) >= 20:
            return [count / sum(counts) for count in counts]
        m, q = (min(nx, k), a) if capped else (nx, a_all)
        if q is None:
            # F evaluates no probability expression when range(m + 1) is empty.
            if m < 0:
                return []
            raise TypeError("ready_share_all is null in an uncapped fallback")
        return [math.comb(m, j) * q**j * (1 - q) ** (m - j) for j in range(m + 1)]

    def layer_ms(nv: int, nx: int) -> tuple[float, float]:
        key = (nv, nx)
        if key not in memo:
            choices = [
                (prob, class_ms(nv, j, nx - j)) for j, prob in enumerate(ready_dist(nx)) if prob
            ]
            memo[key] = (
                sum(prob * t for prob, (t, _) in choices),
                sum(prob * u for prob, (_, u) in choices),
            )
        return memo[key]

    total = cpu = 0.0
    for row in rows:
        for nv, nx in row:
            t, u = layer_ms(nv, nx)
            total += t
            cpu += u
    n = max(1, len(rows))
    sink = total / n
    ms = response.t_out_ms + sink
    return DecodeEstimate(ms, 1000.0 / ms, sink, cpu / n)
