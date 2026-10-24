"""F's complete machine model, with explicit lazy artifact and host inputs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Optional, Union, cast

from inferssd.model import _machine_prefill as prefill
from inferssd.model.decode_cost import PriceInputs, price_mapping
from inferssd.model.machine_types import (
    ArtifactReader,
    Feedback,
    Insitu,
    ModelArtifact,
    Placement,
    PlanRow,
    Prediction,
    SavedPlan,
)
from inferssd.model.prefill_calibration import DeviceView as CalibrationDevice
from inferssd.model.prefill_calibration import InsituView as CalibrationInsitu
from inferssd.model.routing_replay import Replay, Routing, Rows
from inferssd.plan import settings
from inferssd.plan.settings_types import Device, HostThreads, KnownWorkload, MachineView

NEUTRAL: Feedback = {
    "k_cpu": {},
    "out0": {},
    "k_drive": 1.0,
    "k_bytes": 1.0,
    "pf_cpu": {},
    "pf_drive": 1.0,
}


def layer_faults(pl: Placement, k: int) -> list[tuple[int, int, int, float]]:
    """Nonpositive measured or fitted layer classes, in (nv, nr, nm) iteration order."""
    table, beta = pl["layer_table"], cast(Sequence[float], pl["layer_beta"])
    out = []
    for nv in range(k + 1):
        for nr in range(k + 1 - nv):
            nm = k - nv - nr
            e = table.get(f"{nv},{nr},{nm}")
            t = (
                e[1]
                if e and e[0] >= 5
                else beta[0] + beta[1] * nv + beta[2] * nr + beta[3] * nm + beta[4] * (nm > 0)
            )
            if t <= 0:
                out.append((nv, nr, nm, round(t, 3)))
    return out


def row_faults(row: PlanRow) -> list[str]:
    """Nonpositive totals/rates or negative prefill costs, including turns, in original order."""
    out = [
        f"{k} {cast(Mapping[str, float], row)[k]}"
        for k in ("total_s", "decode_tps")
        if cast(Mapping[str, float], row)[k] <= 0
    ]
    out += [f"prefill_s {row['prefill_s']}"] if row["prefill_s"] < 0 else []
    for i, t in enumerate(row.get("turns") or []):
        out += [
            f"turn {i} {k} {cast(Mapping[str, float], t)[k]}"
            for k in ("decode_tps",)
            if cast(Mapping[str, float], t)[k] <= 0
        ]
        out += [f"turn {i} prefill_s {t['prefill_s']}"] if t["prefill_s"] < 0 else []
    return out


class Machine:
    """An instance-owned stateful model; artifacts and host fallback are explicit ports."""

    _replays: dict[Hashable, Replay]
    _sinks: dict[tuple[str, int, str, int], tuple[float, float, float]]

    def __init__(
        self,
        artifacts: ArtifactReader,
        pending: Optional[Iterable[str]] = None,
        workload: Optional[KnownWorkload] = None,
        *,
        host_threads: HostThreads,
    ):
        """pending: the settings the model's conf holds (PENDING); workload: the turns' new prompt
        tokens and replies ({'prompts': [...], 'decode': D}). None: those plan.py was given
        (plan.json), and nothing derived for a plan that predates derived settings."""
        self._host_threads = host_threads
        self.ins = cast(Insitu, artifacts.read_json("insitu.json"))
        self.dev = cast(Device, artifacts.read_json("probe/device.json"))
        self.routing = cast(ModelArtifact, artifacts.read_json("model.json"))["routing"]
        self.P, self.ref = self.ins["placements"], self.ins["reference"]
        ref = self.P[self.ref]
        self.budget = int(ref["tier_slots"])
        self.mc = ref["admission_min_count"] if ref["admission_min_count"] is not None else 1.0
        # time outside the sink against the context: the reference bootstrap's slope when its tokens
        # span enough
        # positions to tell (a few hundred do not), else flat until a conversation spans enough
        # (online.fit_slope)
        f = ref["t_out_fit"]
        self.slope = (
            max(0.0, f["b_ms_per_ktok"] / 1e3)
            if f["positions"][1] - f["positions"][0] >= 2000
            else 0.0
        )
        self.PF = prefill.build(
            cast(CalibrationInsitu, self.ins), cast(CalibrationDevice, self.dev), self.routing
        )
        # configurations: the tier at quarters of the budget, each capped by what the placement
        # leaves (a larger
        # prefill ubatch leaves less GPU memory to the tier)
        self.caps = {p: int(m["tier_slots"]) for p, m in self.P.items()}
        tiers = {max(1, int(self.budget * f)) for f in (0.25, 0.5, 0.75, 1.0)}
        self.configs = [
            (t, p) for p in self.P for t in sorted({min(t, self.caps[p]) for t in tiers})
        ]
        self._replays, self._sinks = {}, {}
        if pending is None and artifacts.exists("plan.json"):
            plan = cast(SavedPlan, artifacts.read_json("plan.json"))
            pending = (plan.get("derived") or {}).get("pending")
            workload = workload or plan.get("workload")
        self.pending, self.workload = pending, workload
        self.derived = (
            {p: settings.derive(cast(MachineView, self), p, pending, workload) for p in self.P}
            if pending is not None
            else {}
        )

    def _derived(self, p: str, name: str) -> Optional[Union[int, float, str]]:
        """The value of setting `name` the server of placement p runs with (derived and not
        pending), else None."""
        s = next((s for s in self.derived.get(p, ()) if s.name == name and not s.pending), None)
        return s.value if s else None

    def route(self, p: str) -> Optional[float]:
        """The prefill threshold of placement p (prefill.seconds); None: every pass on the sink."""
        t = self._derived(p, "GGML_KIMI_CPU_PREFILL_MAX_TOKENS")
        return None if t is None or t == settings.EVERY_PASS else cast(float, t)

    def ubatch(self, p: str) -> Optional[int]:
        """The prefill ubatch of placement p when settings.py derives it; None: the placement's."""
        return cast(Optional[int], self._derived(p, "UBATCH"))

    def env(self, tier: int, p: str) -> str:
        """The launcher environment of a configuration, as settings.py writes it."""
        return settings.env(
            settings.configuration(
                cast(MachineView, self), tier, p, host_threads=self._host_threads
            )
            + self.derived.get(p, [])
        )

    def t_out(self, p: str, x: float, fb: Feedback = NEUTRAL) -> float:
        """ms outside the sink per decode token at context position x."""
        if p in fb["out0"]:
            return fb["out0"][p] + self.slope * x
        f = self.P[p]["t_out_fit"]
        return f["mean_ms"] + self.slope * (x - f["mean_position"])

    def price(self, rows: Rows, p: str) -> tuple[float, float]:
        """(sink ms, its CPU compute part) per token of replayed rows on placement p."""
        r = price_mapping(
            rows,
            cast(
                PriceInputs,
                dict(self.P[p], pf_k=self.P[p].get("pf_k", self.ins["pf_k"]), t_out_ms=0.0),
            ),
            1.0,
        )
        return r["sink_ms"], r["cpu_ms"]

    def replay(self, key: Hashable, segs: Routing, tier: int) -> Replay:
        """The routing segments replayed through a tier of `tier` slots, carried on from the last
        call under `key` when segs extends what that call replayed (the turns of a
        conversation), else from the start."""
        r = self._replays.get(key)
        try:
            if r is None or r.tier.cap != tier:
                raise ValueError
            r.advance(segs)
        except ValueError:
            r = self._replays[key] = Replay(tier, self.mc, self.ins["skip"]).advance(segs)
        return r

    def sink(
        self, segs: Routing, tier: int, p: str, key: str = "conversation"
    ) -> tuple[float, float, float]:
        """(sink ms, its CPU part, tier hit share) per decode token with this routing on (tier, p);
        priced once per routing, whatever the turns asked about."""
        r = self.replay((key, tier), segs, tier)
        k = (key, tier, p, len(r.rows))
        if k not in self._sinks:
            self._sinks[k] = self.price(r.rows, p) + (r.stats()["hit_v"],)
        return self._sinks[k]

    def predict(
        self,
        segs: Routing,
        tier: int,
        p: str,
        ctx: float,
        prompts: Iterable[int],
        decode: int,
        w0: int,
        w1: int,
        cold: bool,
        fb: Feedback = NEUTRAL,
        key: str = "conversation",
    ) -> list[Prediction]:
        """Turns of `prompts` new tokens and `decode` replies each, from context position ctx, on
        (tier, p): per turn the prefill seconds and the decode rate of tokens w0..w1. cold: the
        first turn meets an empty tier."""
        sink, cpu, hv = self.sink(segs, tier, p, key)
        ms = fb["k_cpu"].get(p, 1.0) * cpu + fb["k_drive"] * (sink - cpu)
        out: list[Prediction]
        x, out = ctx, []
        for j, n in enumerate(prompts):
            pf = prefill.seconds(
                self.PF,
                p,
                n,
                tier,
                cold and j == 0,
                fb["k_bytes"],
                fb["pf_cpu"].get(p, 1.0),
                fb["pf_drive"],
                self.route(p),
                self.ubatch(p),
            )[0]
            x += n
            out.append(
                {
                    "prefill_s": pf,
                    "decode_tps": 1000.0 / (ms + self.t_out(p, x + (w0 + w1) / 2, fb)),
                    "hit_v": hv,
                }
            )
            x += decode
        return out
