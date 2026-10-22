"""Complete recorded calibration command; ports run at their original F call points."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional, TextIO

from ssdash.model import prefill_calibration as prefill
from ssdash.model.observed_calibration import ArmReader, measure, measure_graph, working_set
from ssdash.telemetry.types import Record


@dataclass(frozen=True)
class CalibrationArguments:
    run: Sequence[str]
    skip: int
    profile: str
    out: str
    graph: Optional[str] = None
    ubatch_probe: Optional[str] = None


def run_calibration(
    a: CalibrationArguments,
    *,
    read_arm: ArmReader,
    open_file: Callable[..., TextIO],
    emit: Callable[..., None],
) -> None:
    """Preserve direct JSON writes and their partial-output/implicit-close behavior.

    Arguments and records are views, not normalized artifacts. In particular the
    first profile is written before the optional ubatch read and graph calculation.
    """
    placements: dict[str, Record] = {}
    for spec in a.run:
        name, run = spec.split("=", 1)
        placements[name] = measure(run, a.skip, read_arm=read_arm)
    first = next(iter(placements.values()))
    union = prefill.union_points(first["routing"])
    routing = [seg for m in placements.values() for seg in m.pop("routing")]
    n_exp = max(e for seg in routing for tok in seg for ex in tok for e in ex) + 1

    prof = {
        "schema": 2,
        "layers": len(routing[0][0]),
        "n_experts": n_exp,
        "topk": len(routing[0][0][0]),
        "tokens": [len(s) for s in routing],
        "working_set_mean": working_set([1, 2, 4, 8, 16, 32, 64, 128, 256], routing),
        "mrc_online": first["mrc"],
        "routing": routing,
    }
    json.dump(prof, open_file(a.profile, "w"), separators=(",", ":"))
    # every placement's decode table at one drive state, the reference's: each bootstrap ran at
    # its own (the drive drifts between runs: 12 then 6 GiB/s on one machine), and the part of a
    # layer's time that waits on the drive scales with its service time per byte, the CPU compute
    # part does not
    gref = placements[next(iter(placements))].get("drive_us_per_mb")
    for name, m in placements.items():
        g = m.get("drive_us_per_mb")
        f = gref / g if gref and g else 1.0
        m["drive_scale"] = round(f, 4)
        if f != 1.0:
            m["layer_table_raw"] = m["layer_table"]
            m["layer_table"] = {
                k: [v[0], round(v[2] + (v[1] - v[2]) * f, 4), v[2]]
                for k, v in m["layer_table"].items()
            }
            seen = [v for v in m["layer_table_raw"].values() if v[0] >= 5]
            cpu_share = sum(v[0] * v[2] for v in seen) / max(1e-9, sum(v[0] * v[1] for v in seen))
            m["layer_beta"] = [
                round(b * (cpu_share + (1 - cpu_share) * f), 4) for b in m["layer_beta"]
            ]
    # larger prefill ubatches: each measured CPU placement again with a smaller tier; everything
    # else of the placement holds, the prefill passes take more tokens each.  The probe
    # (ubprobe.py) and the reference bootstrap (the engine default, as the probe) give the same
    # quantity, the tier after its first decode growth; the difference is what the larger ubatch's
    # buffers take from the tier for decode, and each placement's tier loses it. A probe of the
    # older format (one number, the most the tier held) is that tier.  Neither prices the graph
    # route's workspace, which a graph prefill takes back from the tier on unified memory and
    # decode refills afterwards (design F_g).
    if a.ubatch_probe:
        ref_first = placements[next(iter(placements))].get("tier_slots_first")
        for ub, v in sorted(
            json.load(open_file(a.ubatch_probe)).items(), key=lambda kv: int(kv[0])
        ):
            probe_first, held = (
                (v.get("first"), v.get("held")) if isinstance(v, dict) else (None, v)
            )
            if not (probe_first or held):
                continue
            for name in [n for n in placements if "@" not in n]:
                m = placements[name]
                slots = (
                    (m["tier_slots"] or 0) - (ref_first - probe_first)
                    if probe_first and ref_first
                    else held
                )
                if not slots or slots >= (m["tier_slots"] or 0):
                    continue  # no smaller tier: no cost, or a noisy start
                placements[f"{name}@{ub}"] = dict(
                    m, tier_slots=float(slots), prefill_ubatch=int(ub), derived_from=name
                )
    # the largest prompt chunk one prefill pass takes (the engine's ubatch); a longer prompt is
    # prefilled in passes of it
    ubs = [
        m["prefill_ubatch"]
        for m in placements.values()
        if m.get("prefill_ubatch") and "derived_from" not in m
    ]
    ins = {
        "schema": 3,
        "skip": a.skip,
        "pf_k": first["pf_k"],
        "reference": next(iter(placements)),
        "layers": prof["layers"],
        "prefill_ubatch": max(ubs) if ubs else None,
        "n_experts": n_exp,
        "topk": prof["topk"],
        "union": union,
        "placements": placements,
        "graph": measure_graph(a.graph, read_arm=read_arm) if a.graph else None,
    }
    json.dump(ins, open_file(a.out, "w"), indent=1)
    for name, m in placements.items():
        emit(f"== placement {name}: {m['run']}")
        emit("  measured", json.dumps(m["measured"]))
        emit(
            f"  drive {m.get('drive_us_per_mb')} us/MB read in decode, table scaled "
            f"x{m.get('drive_scale')} to the reference's drive state"
        )
        emit(
            f"  t_out {m['t_out_fit']}  prefetch {m['prefetch_share']} (of all non-tier "
            f"{m['ready_share_all']})  min_count {m['admission_min_count']}  tier "
            f"{m['tier_slots']}"
            f"  threads {m['decode_threads_log']}"
        )
        for p in m["prefill"]:
            emit("  prefill", {k: v for k, v in p.items() if k != "unique_per_layer"})
        if m["graph_read_gbps"]:
            emit(f"  the graph route's expert copy read {m['graph_read_gbps']} GB/s")
        emit(
            f"  memory at load: {m['memory']}; checkpoints after the batch at multiples of "
            f"{m['checkpoint_align']}; "
            f"host-memory ledger {m['ledger']}"
        )
        load = m.get("cache_load")
        emit(
            "  RAM cache at load: "
            + (
                f"{int(load['slots'])} slots, {load['committed_bytes'] / 1e9:.2f} GB paged in in "
                f"{load['us'] / 1e6:.2f} s"
                if load
                else "none (at its first decode token)"
            )
            + (
                "; decode committed {} GB of new slots (first touches) in tokens 1-128 "
                "and {} GB after"
            ).format(*(m.get("cache_new_slot_gb") or [None, None]))
            + (
                f"; the server loaded in {m['load_s']} s"
                if m.get("load_s") is not None
                else "; the server's load time not recorded"
            )
        )
        pf = m["prefetch"] or {}
        emit(
            f"  prefetch depth {m['pf_k']}: {pf.get('layers')} hinted layers, idle window p50 "
            f"{pf.get('window_us_p50')} us, "
            f"novel {pf.get('novel_r')}, novel and routed {pf.get('novel_routed_r')} of "
            f"{pf.get('predicted_r')}; "
            f"other readers {m['drive_other_bytes']} B in decode; RAM cache {m['ram_slots']} slots"
        )
        dp, hr, ho = m.get("decode_place") or {}, m.get("host_register") or {}, m.get("host") or {}
        if dp:
            copy = ho.get("copy_us_per_mb")
            emit(
                f"  decode plan: {dp.get('gpu_per_token')} GPU experts a token from host memory "
                f"(in place {ho.get('in_place')}, "
                f"copy {round(1e3 / copy, 1) if copy else None} GB/s), refused "
                f"{dp.get('refused')}; predicted lanes' time "
                f"error p50 {dp.get('error_p50_pct')}% p90 {dp.get('error_p90_pct')}%; "
                f"page-locked slots {hr.get('registered')} "
                f"(failures {hr.get('failures')}, {hr.get('us_mean')} us each)"
            )
        for key, v in sorted(m["layer_table"].items(), key=lambda kv: -kv[1][0])[:5]:
            emit(f"  layer class {key}: n={v[0]} mean {v[1]:.3f} ms (cpu {v[2]:.3f})")
        need = ("free_mib", "dense_mib", "compute_mib", "n_ubatch")
        if not all(m["memory"].get(k) for k in need):
            emit(
                f"  warning: the server log has no "
                f"{', '.join(k for k in need if not m['memory'].get(k))} at load: "
                "no graph pass is admitted (prefill.graph_pass_max), every pass stays on the sink"
            )
    g = ins["graph"]
    if g:
        emit(f"== the GPU graph's prefill: {g['run']}")
        for x in g["passes"]:
            emit(
                f"  {x['n_tokens']} tokens in {x['passes']} pass(es): {x['bytes'] / 1e9:.1f} GB "
                f"copied in {x['read_s']} s "
                f"({x['bytes'] / 1e9 / x['read_s']:.2f} GB/s), prompt {x['prompt_s']} s"
            )
        emit(f"  {g['why'] or 'memory at load: ' + str(g['memory'])}")
    emit(
        f"model profile: {sum(prof['tokens'])} decode tokens in {len(routing)} segments, "
        f"{prof['layers']} layers x {n_exp} "
        f"experts, top-{prof['topk']}; W(tau) {prof['working_set_mean']}; U(n) of the reference "
        f"{union}"
    )
