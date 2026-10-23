"""F calibration calculations over recorded observations, without probes or validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import cast

from inferssd.model.decode_cost import fit_response as lstsq
from inferssd.telemetry.legacy import hits, median
from inferssd.telemetry.types import ArmResult, Record

ArmReader = Callable[[str, int], ArmResult]


def measure(run: str, skip: int, *, read_arm: ArmReader) -> Record:
    arm = cast(Record, read_arm(run, skip))
    chats = [c for c in arm["client"] if c.get("kind", "chat") == "chat" and "error" not in c]
    # the depth the bootstrap's engine ran (it prints none below depth 1)
    pf = arm["prefetch"] or {}
    pf_k = int(pf.get("topk") or 0)
    # other readers of the drives in decode: what the drives read from the first decode token to
    # the last (the client) less the sink's own reads, its layers' and its prefetch's (the decode
    # of the run is the first request's)
    seen = [c["drive_read_bytes"] for c in chats if c.get("drive_read_bytes") is not None]
    own = sum(b for _, b in arm["lio"].values()) + (pf.get("reader_bytes") or 0.0)
    other = max(0.0, seen[0] - own) if seen and arm["lio"] else None
    groups = [g for g in arm["groups"] if len(g) >= 64]
    window = [k for g in groups for k in g[skip:]]
    h = hits(cast(ArmResult, arm), window)
    pre = cap = nxs = 0
    ready: dict[int, list[int]] = defaultdict(
        lambda: [0] * 17
    )  # nx -> how often nr of them were host-ready (prefetched or RAM cache)
    cls: dict[tuple[int, int, int], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    X: list[list[float]] = []
    Y: list[float] = []
    for k in window:
        for layer, (sel, vh, ch) in arm["trace"][k].items():
            nx = sum(1 for x in vh if not x)
            nr = sum(1 for x, y in zip(vh, ch) if not x and y)
            pre += nr
            cap += min(nx, pf_k)
            nxs += nx
            ready[nx][nr] += 1
            lp = arm["lprof"].get((k[0], k[1], layer))
            if not lp:
                continue
            nv, nm = len(vh) - nx, nx - nr
            c = cls[(nv, nr, nm)]
            c[0] += 1
            c[1] += lp[0] / 1e3
            c[2] += lp[1] / 1e3
            X.append([1.0, nv, nr, nm, 1.0 if nm else 0.0])
            Y.append(lp[0] / 1e3)
    win = [k for k in window if k in arm["prof"]]
    # the drive's state while this bootstrap decoded: its service time per MB read (placement-
    # free: on a Gen4 x4 machine the rest of the sink over the model's went with it to the power
    # 1.2, R^2 0.98, both placements alike)
    io = [arm["lio"][k] for k in win if k in arm.get("lio", {})]
    drive_us_per_mb = (
        sum(x[0] for x in io) / (sum(x[1] for x in io) / 1e6)
        if io and sum(x[1] for x in io)
        else None
    )
    sink = {
        f: sum(arm["prof"][k][i] for k in win) / len(win) / 1e3
        for i, f in enumerate(("total", "wait_io", "compute", "other"))
    }
    # time outside the sink against the context position, and the decode rate of the window
    pts, rates = [], []
    for g, c in zip(groups, chats):
        ts = c.get("tok_s") or []
        for i, k in enumerate(g):
            if (
                i < 16
                or i + 1 >= len(ts)
                or ts[i] is None
                or ts[i + 1] is None
                or k not in arm["prof"]
            ):
                continue
            pts.append((float(k[1]), (ts[i + 1] - ts[i]) * 1e3 - arm["prof"][k][0] / 1e3))
        if len(ts) > skip + 2 and ts[skip] is not None and ts[-1] is not None:
            rates.append((len(ts) - skip - 1) / (ts[-1] - ts[skip]))
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0
    # prefill points: passes in log order matched to the client records that carry a prompt time
    timed = [c for c in arm["client"] if (c.get("timings") or {}).get("prompt_ms")]
    # and each one's sink compute and the graph route's expert copy (bytes, read time), one record
    # per prefill
    prefill = []
    graphs = arm["pf_graphs"] if len(arm["pf_graphs"]) == len(timed) else [None] * len(timed)
    for p, c, g in zip(arm["prefill"], timed, graphs):
        prefill.append(
            {
                "kind": c.get("kind", "chat"),
                "n_tokens": p["n_tokens"],
                "unique": p["unique"],
                "tier": p["tier"],
                "routes": p["routes"],
                "ssd_gb": round(p["ssd_bytes"] / 1e9, 2),
                "prompt_ms": c["timings"]["prompt_ms"],
                "unique_per_layer": p["unique_per_layer"],
                "graph_gb": round(g["graph_bytes"] / 1e9, 3) if g else None,
                "graph_read_s": round(g["graph_read_us"] / 1e6, 3) if g else None,
                "cpu_s": round(g["cpu_s"], 3) if g else None,
            }
        )
        # The pass ran with the GPU lane: its table and layers' sink time.
        if p.get("lane"):
            prefill[-1].update(
                lane=p["lane"], layers_s=round(p["layers_s"], 3), layers_costed=p["layers_costed"]
            )
    rb = sum(g["graph_bytes"] for g in graphs if g)
    ru = sum(g["graph_read_us"] for g in graphs if g)
    routing = []
    for g in groups:
        L = len(arm["trace"][g[0]])
        routing.append(
            [
                [arm["trace"][k][layer][0] for layer in sorted(arm["trace"][k])]
                for k in g
                if len(arm["trace"][k]) == L
            ]
        )
    adm = [x for x in arm["admission"] if x["min_count"] is not None]
    return {
        "run": run,
        "measured": {
            "decode_tps_window": round(sum(rates) / len(rates), 3) if rates else None,
            "sink_ms_per_token": {k: round(v, 2) for k, v in sink.items()},
            "hits": h,
            "tokens_in_window": len(win),
        },
        "t_out_fit": {
            "a_ms": round(my - slope * mx, 3),
            "b_ms_per_ktok": round(slope * 1e3, 4),
            "points": len(pts),
            "positions": [int(min(x for x, _ in pts)), int(max(x for x, _ in pts))],
            "mean_position": round(mx),
            "mean_ms": round(my, 2),
        },
        "prefetch_share": round(pre / cap, 4) if cap else 0.0,
        "ready_share_all": round(pre / nxs, 4) if nxs else 0.0,
        "ready_given_nx": {str(n): c[: n + 1] for n, c in sorted(ready.items())},
        "admission_min_count": median([x["min_count"] for x in adm])
        if adm
        else arm.get("tier_min_count"),
        "tier_slots": arm["tier_slots_max"],
        "tier_slots_first": arm.get("tier_slots_first"),
        "tier_slot_bytes": round(arm["tier_bytes"][0]) if arm["tier_bytes"] else None,
        "tier_base_bytes": arm["tier_bytes"][1] if arm["tier_bytes"] else None,
        "decode_threads_log": arm["threads"] or {},
        "layer_table": {
            f"{a},{b},{c}": [v[0], round(v[1] / v[0], 4), round(v[2] / v[0], 4)]
            for (a, b, c), v in sorted(cls.items())
        },
        "layer_beta": [round(b, 4) for b in lstsq(X, Y)],
        "prefill": prefill,
        "graph_read_gbps": round(rb / ru / 1e3, 4) if ru else None,
        "memory": arm["memory"],
        "checkpoint_align": arm["checkpoint_align"],
        "ledger": arm["ledger"],
        "routing": routing,
        "mrc": arm["mrc"],
        "prefill_ubatch": max((g.get("ubatch") or 0 for g in arm["pf_graphs"]), default=0) or None,
        "drive_us_per_mb": round(drive_us_per_mb, 2) if drive_us_per_mb else None,
        "cache_load": arm["cache_load"],
        "prefill_engine": arm.get("prefill_engine"),
        "cache_new_slot_gb": [round(x / 1e9, 3) for x in arm["new_slot_windows"]],
        "load_s": arm["load_s"],
        "pf_k": pf_k,
        "prefetch": pf or None,
        "drive_other_bytes": other,
        "ram_slots": arm["ram_slots"],
        # the decode plan (CPU or GPU for each non-resident expert) and the page-locking it copies
        # from
        "decode_place": arm.get("decode_place"),
        "host_register": arm.get("host_register"),
        "host": arm.get("host"),
    }


def measure_graph(run: str, *, read_arm: ArmReader) -> Record:
    """Match graph-prefill records by tokens; preserve missing/failed-run reasons."""
    try:
        arm = cast(Record, read_arm(run, 0))
    except OSError as e:
        return {"run": run, "passes": [], "memory": {}, "why": f"no server log ({e})"}
    graphs = [g for g in arm["pf_graphs"] if g["graph_bytes"] and g["graph_read_us"]]
    passes = []
    for c in arm["client"]:
        t = c.get("timings") or {}
        g = (
            next((g for g in graphs if g["n_tokens"] == t.get("prompt_n")), None)
            if t.get("prompt_ms")
            else None
        )
        if g:
            graphs.remove(g)
            passes.append(
                {
                    "n_tokens": g["n_tokens"],
                    "passes": g["graph_passes"] or 1,
                    "bytes": g["graph_bytes"],
                    "read_s": round(g["graph_read_us"] / 1e6, 3),
                    "prompt_s": round(t["prompt_ms"] / 1e3, 3),
                }
            )
    return {
        "run": run,
        "passes": passes,
        "memory": arm["memory"],
        "why": None
        if passes
        else "no graph prefill (the server did not start, or its prefills failed)",
    }


def working_set(taus: Sequence[int], routing: list[list[list[list[int]]]]) -> dict[int, float]:
    out = {}
    for tau in taus:
        tot = cnt = 0
        for seg in routing:
            for st in range(0, len(seg) - tau + 1, max(1, tau // 4)):
                for li in range(len(seg[0])):
                    tot += len({e for tok in seg[st : st + tau] for e in tok[li]})
                    cnt += 1
        if cnt:
            out[tau] = round(tot / cnt, 2)
    return out
