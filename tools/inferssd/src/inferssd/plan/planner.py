"""The complete offline planner on supplied artifacts; no implicit process or probe."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Optional, cast

from inferssd.model.machine import Machine, layer_faults, row_faults
from inferssd.model.machine_types import (
    ArtifactReader,
    Calibration,
    PlanArtifact,
    PlanRow,
    Rejected,
)
from inferssd.plan.planner_types import Emit, PlanArguments
from inferssd.plan.settings_types import HostThreads


def run_plan(
    a: PlanArguments,
    artifacts: ArtifactReader,
    *,
    host_threads: HostThreads,
    clock: Callable[[], float],
    emit: Emit,
    write_json: Callable[[str, object], None],
    factory: type[Machine] = Machine,
) -> PlanArtifact:
    t_start = clock()
    w0, w1 = (int(x) for x in a.window.split("-"))
    prompts = [int(x) for x in a.prompts.split(",")]
    m = factory(
        artifacts,
        pending=a.pending.split(),
        workload={"prompts": prompts, "decode": a.decode, "known": a.known},
        host_threads=host_threads,
    )

    table: list[PlanRow] = []
    for tier, p in m.configs:
        turns = m.predict(
            m.routing, tier, p, 0, prompts, a.decode, w0, w1, cold=True, key="bootstrap"
        )
        table.append(
            {
                "tier": tier,
                "cpus": p,
                "total_s": round(
                    sum(t["prefill_s"] + a.decode / t["decode_tps"] for t in turns), 2
                ),
                "prefill_s": round(sum(t["prefill_s"] for t in turns), 2),
                "decode_tps": round(sum(t["decode_tps"] for t in turns) / len(turns), 3),
                "hit_v": round(turns[0]["hit_v"], 4),
                "turns": [
                    {"prefill_s": round(t["prefill_s"], 2), "decode_tps": round(t["decode_tps"], 3)}
                    for t in turns
                ],
                "env": m.env(tier, p),
            }
        )
    # a configuration priced at a time or a rate at or below zero, or on a placement whose layer
    # table prices a class at
    # or below zero, has an input gone wrong upstream: it is set aside with its reasons, and the
    # plan chooses among the
    # rest; with none left there is no plan
    k = m.ins["topk"]
    bad_layers = {p: layer_faults(m.P[p], k) for p in m.P}
    rejected: list[Rejected] = []
    r: Optional[PlanRow]
    for r in table:
        why = row_faults(r) + (
            [
                f"layer classes priced at or below zero: {bad_layers[r['cpus']][:4]}"
                f"{' ...' if len(bad_layers[r['cpus']]) > 4 else ''}"
            ]
            if bad_layers[r["cpus"]]
            else []
        )
        if why:
            rejected.append(
                {"tier": r["tier"], "cpus": r["cpus"], "total_s": r["total_s"], "why": why}
            )
    table = [
        r
        for r in table
        if not any(x["tier"] == r["tier"] and x["cpus"] == r["cpus"] for x in rejected)
    ]
    for x in rejected:
        emit(
            f"rejected {x['cpus']} tier {x['tier']} (total {x['total_s']} s): {'; '.join(x['why'])}"
        )
    if not table:
        raise SystemExit("plan: every configuration was rejected (above); no plan")
    table.sort(key=lambda r: r["total_s"])
    best = table[0]
    other = next((r for r in table if r["cpus"].split("@")[0] != best["cpus"].split("@")[0]), None)

    # in-sample: each placement at its bootstrap's tier, decode at that run's mean position, against
    # what it measured
    calib: dict[str, Calibration] = {}
    for p, mp in m.P.items():
        sink, _, hv = m.sink(m.routing, min(m.budget, m.caps[p]), p, key="bootstrap")
        calib[p] = {
            "decode_tps_model": round(
                1000.0 / (sink + m.t_out(p, mp["t_out_fit"]["mean_position"])), 3
            ),
            "decode_tps_measured": mp["measured"]["decode_tps_window"],
            "hit_v_model": round(hv, 4),
            "hit_v_measured": round(mp["measured"]["hits"]["v"], 4),
        }

    out: PlanArtifact = {
        "reference": m.ref,
        "budget_tier_slots": m.budget,
        "admission_min_count": m.mc,
        "t_out_slope_ms_per_ktok": round(m.slope * 1e3, 4),
        "prefill_model": m.PF,
        "workload": {"prompts": prompts, "decode": a.decode, "window": a.window, "known": a.known},
        "plan": best,
        "other": other,
        "table": table,
        "rejected": rejected,
        "calibration": calib,
        "derived": {
            "pending": m.pending,
            "plan": [s._asdict() for s in m.derived.get(best["cpus"], [])],
        },
        "plan_seconds": round(clock() - t_start, 1),
    }
    write_json(a.out, out)
    emit(
        json.dumps(
            {
                k: cast(Mapping[str, object], out)[k]
                for k in (
                    "reference",
                    "budget_tier_slots",
                    "admission_min_count",
                    "workload",
                    "calibration",
                    "derived",
                    "plan_seconds",
                )
            },
            indent=1,
        )
    )
    for label, r in (("plan", best), ("other", other)):
        if r:
            emit(
                f"{label:6s} {r['cpus']} tier {r['tier']}: total {r['total_s']:.1f} s, "
                f"prefill {r['prefill_s']:.1f} s, "
                f"decode {r['decode_tps']:.2f} tok/s, "
                f"hit_v {r['hit_v']}; env: {r['env'] or '(engine defaults)'}"
            )
    emit("all configurations:")
    for r in table:
        emit("  ", {k: v for k, v in r.items() if k not in ("turns", "env")})

    return out
