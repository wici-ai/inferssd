"""F's complete settings derivation and ordered receipts, without file or host I/O."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Union, cast

from ssdash.model.prefill_calibration import CalibratedGraph, CalibratedPass
from ssdash.model.prefill_model import LaneCosts
from ssdash.model.routing_replay import Replay
from ssdash.plan import _prefill as prefill
from ssdash.plan import placement
from ssdash.plan.settings_types import (
    HostThreads,
    KnownWorkload,
    Ledger,
    MachineView,
    Placement,
    Prefetch,
    Setting,
    Spill,
    Workload,
)

EVERY_PASS = 4294967295
CONTEXT_STEP = 256
OWN_ENGINE = {"mxfp4/mxfp4", "iq2_xxs/q2_k"}
GB_PER_GIB = 2**30 / 1e9


def configuration(m: MachineView, tier: int, p: str, *, host_threads: HostThreads) -> list[Setting]:
    """The settings of configuration (tier, p) of machine model m: a tier below what the
    placement's bootstrap reached caps the elastic growth and, where the engine starts from
    a fixed base budget (a discrete GPU; on unified memory the launcher starts the tier from
    one block), lowers that base to the tier's bytes; then the CPU placement's, and a larger
    prefill ubatch ('all@16384')."""
    out = []
    cap = m.caps[p]
    if tier < cap:
        out.append(
            Setting(
                "GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS",
                tier,
                "decided",
                f"{tier} of the {cap} tier slots the bootstrap of {p} reached",
            )
        )
        sb, base = m.P[m.ref].get("tier_slot_bytes"), m.P[m.ref].get("tier_base_bytes")
        if sb and base and tier * sb < base and not (m.dev.get("gpu") or {}).get("integrated"):
            out.append(
                Setting(
                    "GGML_KIMI_CPU_SINK_VRAM_BYTES",
                    int(tier * sb),
                    "decided",
                    f"{tier} slots of {sb} bytes, under the base budget of {int(base)} bytes",
                )
            )
    cpus, _, ub = p.partition("@")
    for kv in placement.env(m.dev, cpus, host_threads=host_threads).split():
        k, v = kv.split("=", 1)
        out.append(
            Setting(k, v, "decided", f"CPU placement {cpus} (placement.py on probe/device.json)")
        )
    if ub:
        out.append(
            Setting(
                "UBATCH",
                ub,
                "decided",
                f"prefill passes of {ub} tokens, the tier then {cap} slots (ubatch.json)",
            )
        )
    return out


def prefill_route(
    m: MachineView,
    p: str,
    limit: int,
    bound: str = "the ubatch a server of this machine started with",
    turns: Optional[Sequence[int]] = None,
    reply: int = 0,
) -> list[Setting]:
    """The prefill threshold and the ubatch of placement p (design 3.2), passes of at most
    `limit` tokens: when a graph pass larger than the model's threshold T fits the GPU, T is
    the threshold and the largest graph pass that fits the ubatch (each graph pass reads the
    union of its tokens' experts, so fewer passes read less, and the sink's passes past T
    are compute-bound, the same however a prompt is cut); otherwise every pass is the
    sink's, of up to `limit` tokens (`bound` says why). A configuration with its own ubatch
    ('all@16384') keeps it and takes T when its passes fit the graph. turns: the workload's
    turns, each turn's new prompt tokens, and reply the tokens of each reply
    (prefill.threshold: the least time for them as the server prefills them, one
    conversation on a freshly started server; None: the warm tier of a serving server)."""
    PF, ub = m.PF, p.partition("@")[2]
    ug = prefill.graph_pass_max(PF, p, int(ub) if ub else limit)
    cut = int(ub) if ub else (ug or limit)
    T = prefill.threshold(PF, p, turns, cut, reply)
    g, scale = PF.get("graph"), PF.get("union_scale")
    mem = PF.get("graph_memory") or PF["memory"].get(p) or {}
    gl = prefill.graph_line(PF, p)
    if gl:
        host = f" + {gl['host']:.4f} a token on the host" if PF.get("integrated") else ""
        line = (
            f"the graph's compute {gl['role']:.2f} + {gl['slope']:.4f} a token{host} (its "
            f"largest role copy, the "
            f"slope from its reserve at {gl['n']} tokens) + 1 GiB"
        )
    else:
        line = "the graph's compute not measured: no reserve of a graph pass"
    held = (
        f"({mem.get('free_mib')} MiB free at load, dense {mem.get('dense_mib')}, KV "
        f"{mem.get('kv_mib')}; {line})"
    )
    fits = (
        f"graph passes up to {ug} tokens fit the GPU {held}"
        if ug
        else f"no graph pass fits the GPU {held}"
    )

    def state(s: str) -> str:
        st = cast(CalibratedPass, PF["pass"][p][s])
        if st.get("lane"):
            t = cast(LaneCosts, st["lane"])
            return (
                f"with the GPU lane, the table of its {s} prefill of {st['n']} tokens (CPU "
                f"{t.get('cpu_fixed_us', 0):.0f} + "
                f"{t.get('cpu_route_us', 0):.1f} us a route, GPU "
                f"{t.get('gpu_fixed_us', 0):.0f} + "
                f"{t.get('gpu_route_us', 0):.2f}, link {t.get('link_gbps', 0):.2f} GB/s, drive "
                f"{t.get('drive_gbps', 0):.2f} GB/s), {st['r']:.1f} routes a token and "
                f"{st['d'] * 1e3:.2f} ms a token "
                "outside its layers"
            )
        return (
            f"{st['c'] * 1e6:.1f} us a route, {st['r']:.1f} routes a token and "
            f"{st['d'] * 1e3:.2f} ms a token "
            f"besides ({s} prefill of {st['n']} tokens)"
        )

    if not turns:
        sink = f"the sink's compute {state('warm')}"
    else:
        ps = prefill.offloaded(PF, p, turns, cut, reply)
        nc = sum(w for _, f, w in ps if f)
        nw = sum(w for _, f, w in ps if not f)
        t = prefill.break_even(PF, p, prefill.cold_share(ps))
        again = (
            f", each follow-up also as the {reply}-token reply before it prefilled again with it "
            f"({','.join(str(reply + n) for n in turns[1:])} tokens: the server's cache "
            f"parts inside a reply whose "
            "text gives back other tokens), the two forms weighed 1/2 each"
            if reply and len(turns) > 1
            else ""
        )
        sink = (
            f"the least time for the workload's turns of {','.join(map(str, turns))} new "
            f"tokens, the first on a "
            f"cold tier and the rest on the warm one{again}, and between their sizes the "
            f"break-even over the tier "
            f"states their passes of at least {prefill.OFFLOAD_MIN} tokens meet ({nc:g} "
            f"cold, {nw:g} warm"
            + (
                f"; the break-even alone, {t}, would not give the workload its least time"
                if T != t
                else ""
            )
            + "): the sink's compute "
            + ", ".join(state(s) for s, k in (("cold", nc), ("warm", nw or not ps)) if k)
        )
    graph = (
        f"the graph's copy at {g['R'] / 1e9:.2f} GB/s, {g['a']:.2f} s a pass and "
        f"{g['d'] * 1e3:.2f} ms a token "
        f"besides (its prefills of {', '.join(map(str, cast(CalibratedGraph, g)['n']))} tokens)"
        if g
        else f"provisional, the graph's prefill not measured: its reads at "
        f"{cast(float, PF['R_io']) / 1e9:.2f} GB/s, the sink's "
        "time a token besides"
    )
    if T is None:
        why = "no threshold: the sink's compute or the drive's rate was not measured"
    elif T == math.inf:
        why = f"the graph is never faster: {sink}; {graph}"
    else:
        u = (
            f" (prompt tokens: the decode trace times {scale[1]}, its ratio at the prefill "
            f"of {scale[0]} tokens)"
            if scale and T < scale[0]
            else ""
        )
        why = (
            f"n*={T}: {sink}; U({T})={prefill.coverage(PF, T):.1f} of {PF['E']} experts a "
            f"layer{u}, {PF['B_e']} "
            f"bytes an expert; {graph}"
        )
    out = []
    if T is not None and T < ug and (not ub or ug == int(ub)):
        out.append(Setting("GGML_KIMI_CPU_PREFILL_MAX_TOKENS", T, "derived", f"{why}; {fits}"))
        if not ub:
            out.append(
                Setting(
                    "UBATCH",
                    ug,
                    "derived",
                    f"the largest graph pass, up to {limit} tokens ({bound}): {fits}",
                )
            )
    else:
        out.append(
            Setting(
                "GGML_KIMI_CPU_PREFILL_MAX_TOKENS",
                EVERY_PASS,
                "derived",
                f"every pass on the sink: {why}; {fits}",
            )
        )
        if not ub:
            out.append(
                Setting(
                    "UBATCH",
                    limit,
                    "derived",
                    f"every pass on the sink, of up to {limit} tokens ({bound})",
                )
            )
    return out


def cost_spec(terms: Mapping[str, float]) -> str:
    """The engine's GGML_KIMI_CPU_PREFILL_COST value: name=value items, comma separated."""
    return ",".join(f"{k}={v:.6g}" for k, v in terms.items())


def prefill_carried(m: MachineView, p: str) -> Optional[Setting]:
    """The cost table the bootstrap of placement p calibrated (B6), carried to its servers as
    their start values: the last pass's table of its warm prefill with the GPU lane
    ([kimi_cpu_prefill_cost] calibrated=), in place of the probe's. None without the lane (a
    bootstrap before it, as the stored runs')."""
    pts = sorted((x for x in m.P[p].get("prefill") or [] if x.get("lane")), key=lambda x: x["tier"])
    if not pts:
        return None
    x = pts[-1]
    terms = {
        k: v
        for k, v in cast(LaneCosts, x["lane"]).items()
        if v is not None and (v > 0 or (k == "gpu_token_us" and v == 0))
    }
    return Setting(
        "GGML_KIMI_CPU_PREFILL_COST",
        cost_spec(terms),
        "derived",
        f"the table the bootstrap's warm prefill of {x['n_tokens']} tokens calibrated with "
        f"the GPU lane "
        f"(its last layer's [kimi_cpu_prefill_cost] line, over the pass's "
        f"{x.get('layers_costed')} layers), "
        "carried as the engine's start values",
    )


def prefill_start(m: MachineView) -> list[Setting]:
    """The cost table's start values (B4) from the probe: the CPU's grouped path and the GPU's
    batched engine, each a line of fixed + per-route microseconds an expert timed busy
    beside the other lanes at the model's expert shape and types (test-kimi-vram-group
    --time); one copy from page-locked memory and the pageable copy (memprobe); the drive at
    its knee. The engine starts from them and replaces each with its own measure as it runs.
    Nothing without the lane timing (a probe before it, as the stored runs')."""
    dev = m.dev
    ln, gpu, ssd = dev.get("lanes"), dev.get("gpu") or {}, dev.get("ssd") or {}
    if not ln:
        return []
    terms: dict[str, float]
    terms, why = {}, []
    if "cpu" in ln:
        terms.update(cpu_fixed_us=ln["cpu"]["fixed_us"], cpu_route_us=ln["cpu"]["per_route_us"])
        why.append(
            f"the CPU's grouped path {ln['cpu']['fixed_us']:.1f} + "
            f"{ln['cpu']['per_route_us']:.3f} us a route "
            f"on {ln.get('threads')} threads"
        )
    if ln.get("format", "").lower() in OWN_ENGINE:
        why.append(f"no GPU line: the tier computes {ln['format']} on an engine of its own")
    else:
        if "gpu" in ln:
            terms.update(gpu_fixed_us=ln["gpu"]["fixed_us"], gpu_route_us=ln["gpu"]["per_route_us"])
            why.append(
                f"the batched engine from device memory {ln['gpu']['fixed_us']:.1f} + "
                f"{ln['gpu']['per_route_us']:.3f} us a route"
            )
        if "gpu_host" in ln:
            terms.update(
                gpu_host_fixed_us=ln["gpu_host"]["fixed_us"],
                gpu_host_route_us=ln["gpu_host"]["per_route_us"],
            )
            why.append(
                f"in place from host memory {ln['gpu_host']['fixed_us']:.1f} + "
                f"{ln['gpu_host']['per_route_us']:.3f} us a route"
            )
    link = ln.get("copy_gbps") or gpu.get("h2d_pinned_gbs")
    if link and not gpu.get("integrated"):
        terms["link_gbps"] = link
        why.append(f"a page-locked copy {link:.2f} GB/s")
    # the pageable copy: a discrete GPU's unlocked sources; on unified memory the transport the
    # in-place choice weighs
    # (its read ring stays unregistered there), the probe's copy where memprobe has none
    pageable = gpu.get("h2d_pageable_gbs") or (
        ln.get("copy_gbps") if gpu.get("integrated") else None
    )
    if pageable:
        terms["pageable_gbps"] = pageable
        why.append(f"a pageable copy {pageable:.2f} GB/s")
    knee = dev.get("ssd_knee_qd")
    if str(knee) in ssd:
        terms["drive_gbps"] = ssd[str(knee)]["gibs"] * GB_PER_GIB
        why.append(f"the drive {terms['drive_gbps']:.2f} GB/s at QD {knee}")
    if not terms:
        return []
    return [
        Setting(
            "GGML_KIMI_CPU_PREFILL_COST",
            cost_spec(terms),
            "measured",
            f"the probe's lanes at {ln.get('format')} latent {ln.get('latent')} hidden "
            f"{ln.get('hidden')} "
            f"(test-kimi-vram-group --time, each busy beside the others): {'; '.join(why)}",
        )
    ]


def depth(
    nu: Sequence[float],
    p: Sequence[float],
    need: float,
    cost: Optional[Sequence[float]],
    gain: float = 1.0,
    spill: Spill = 0.0,
) -> tuple[int, int, int, int]:
    """The prefetch depth k* = min(k_pay, k_mem) from the hint's ranks, best first: nu[j] the
    share of hints whose rank-j expert is novel (neither held by the GPU tier nor in the RAM
    cache), p[j] the share of those the next layer selects (and computes on the CPU, so a
    read of it is used); need the novel reads that fill the drive's idle window; cost[j]
    what the cache slot rank j's staging takes would serve a token, each layer (None:
    nothing is taken); gain the demand time a used read saves and spill what a read adds to
    the next layer's demand reads, both in reads of one expert at the prefetch's rate. The
    reads go in rank order: k_win ranks start inside the window (the novel reads before them
    do not fill it; none when there is no window). A read pays when p gain exceeds its
    spill, plus its own transfer when it starts after the window; k_pay is the number of
    leading ranks that pay, 0 (no prefetch) when none does. spill is one value for every
    rank, or one per rank (read_spill: the engine's measure where it read the rank, which
    already holds whatever the read ran past its window, None where it did not; such a rank
    costs the most a measured rank does, plus the transfer past the window). k_mem is the
    largest depth whose every rank saves more reads (nu p) than its slot would have served.
    Returns (k*, k_win, k_mem, k_pay)."""
    top, start, k_win = len(nu), 0.0, 0
    while k_win < top and start < need:
        start += nu[k_win]
        k_win += 1
    if isinstance(spill, (list, tuple)):
        worst = max((x for x in spill if x is not None), default=0.0)
        price = [
            cast(float, spill[j])
            if j < len(spill) and spill[j] is not None
            else worst + (0.0 if j < k_win else 1.0)
            for j in range(top)
        ]
    else:
        price = [spill + (0.0 if j < k_win else 1.0) for j in range(top)]
    k_pay = next((j for j in range(top) if p[j] * gain <= price[j]), top)
    k_mem = top
    if cost is not None:
        k_mem = next((j for j in range(top) if nu[j] * p[j] < cost[j]), top)
    return min(k_pay, k_mem), k_win, k_mem, k_pay


def read_gain(pl: Placement, b_e: float, t_e: float) -> tuple[float, str]:
    """The demand time a used prefetch read saves (us), and how it was found: the layer fit of
    placement pl's bootstrap (insitu.py), a read expert less a host-ready one (the layer
    waits on the drive for the one, not the other); without a fit, the decode reads' drive
    time of an expert of b_e bytes; else one read of t_e us."""
    beta = pl.get("layer_beta")
    if beta and len(beta) >= 4 and beta[3] > beta[2]:
        return (beta[3] - beta[2]) * 1e3, "the layer fit's read expert less a ready one"
    if pl.get("drive_us_per_mb"):
        return cast(
            float, pl["drive_us_per_mb"]
        ) * b_e / 1e6, "the decode reads' drive time an expert"
    return t_e, "one read at the probe's rate (no layer fit)"


def read_spill(
    pf: Prefetch, other_us: float
) -> tuple[Optional[Union[float, list[Optional[float]]]], str]:
    """What a prefetch read adds to the next layer's demand reads (us), from the engine's
    report, and how it was found; None without reads to measure it. The engine's measure
    (spill_us, spill_us_r, started_r): the readers' wall time past a window's close on reads
    of experts the next run does not take, shared out over the hint's ranks by each rank's
    own time past the close, over the rank's reads that started; one value per rank, None
    for a rank the run did not read. A report without it (an engine before the measure): the
    readers' busy time beyond the drive's idle windows (less the other readers' share of
    them), over the reads, one value for every rank; a lower bound, zero whenever the
    readers are busy for less than all windows together."""
    started, own = pf.get("started_r"), pf.get("spill_us_r")
    if pf.get("spill_us") is not None and started and own is not None and sum(started):
        total = sum(own)
        per = [
            (cast(float, pf["spill_us"]) * o / total / n if total else 0.0) if n else None
            for o, n in zip(own, started)
        ]
        return per, (
            f"{cast(float, pf['spill_us']) / 1e6:.2f} s past the windows "
            f"({int(pf.get('spills') or 0)} role reads ran past "
            f"one) over {sum(started)} experts read, an expert's read of each rank "
            f"{','.join('-' if x is None else f'{x:.0f}' for x in per)} us"
        )
    issued, busy = pf.get("issued") or 0, pf.get("reader_busy_us")
    if not issued or busy is None or pf.get("window_us_mean") is None:
        return None, "0 (no reads to measure it)"
    idle = pf["windows"] * max(0.0, cast(float, pf["window_us_mean"]) - other_us)
    spill = max(0.0, busy - idle) / issued
    return spill, (
        f"{spill:.0f} us a read (the readers busy {busy / 1e6:.2f} s, the windows "
        f"{idle / 1e6:.2f} s, "
        f"{int(issued)} reads; a lower bound, the engine not measuring it)"
    )


def rank_costs(m: MachineView, p: str, top: int) -> tuple[Optional[list[float]], str]:
    """What the cache slot each rank of the prefetch takes would serve a token, each layer
    (hits a token of one slot a layer; infinity where no slot is left), and how it was
    found; None without the facts. unified memory depth k stages max(1, k) slots a layer in
    the DRAM the GPU tier would otherwise hold: from the C slots the bootstrap reached at
    its depth b the tier has C + (max(1, b) - max(1, k)) L at depth k. Rank 1 takes no slot,
    rank j the L slots between depths j - 1 and j, priced by the bootstrap's routing
    replayed through the tier at both sizes. a discrete GPU the prefetch lands in the RAM
    cache's own slots, s a layer, in place of its coldest expert: rank j takes the j-th slot
    from the end, H(s - j + 1) - H(s - j) for H the hits of the RAM cache behind the
    replayed tier (sim.RamCache, the engine's policy)."""
    C, L, pf = int(m.caps[p]), cast(int, m.PF["L"]), m.P[p]
    if (m.dev.get("gpu") or {}).get("integrated"):
        b = max(1, int(pf.get("pf_k") or 0))
        cap = [C + (b - max(1, k)) * L for k in range(top + 1)]
        hits = {
            c: m.replay(("bootstrap", c), m.routing, c).stats()["hits_per_token"]
            for c in set(cap)
            if c > 0
        }
        cost = [
            0.0
            if cap[j] == cap[j - 1]
            else math.inf
            if cap[j] <= 0
            else max(0.0, (hits[cap[j - 1]] - hits[cap[j]]) / L)
            for j in range(1, top + 1)
        ]
        return cost, (
            f"the GPU tier of {C} slots at depth {b}: {hits[cap[1]]:.2f} hits a token at depth 1 "
            f"({cap[1]} slots), {hits[cap[top]]:.2f} at depth {top} ({cap[top]})"
            if cap[top] > 0
            else f"the GPU tier of {C} slots at depth {b}"
        )
    s = (pf.get("ram_slots") or 0) // L
    if s < 1:
        return None, "the RAM cache has no slot a layer"
    r = Replay(C, m.mc, m.ins["skip"], behind=True).advance(m.routing)
    H = [r.behind_hits(x) for x in range(s + 1)]
    cost = [
        max(0.0, (H[s - j + 1] - H[s - j]) / L) if j <= s else math.inf for j in range(1, top + 1)
    ]
    return cost, (
        f"the RAM cache of {s} slots a layer behind a {C}-slot tier: "
        f"{', '.join(f'{x:.2f}' for x in reversed(H))} hits a token at {s} down to 0 slots a "
        f"layer"
    )


def prefetch_depth(m: MachineView, p: str) -> list[Setting]:
    """The next-layer prefetch depth of placement p (design 4.1) from its bootstrap: the hint
    statistics per rank and the drive's idle window per hinted layer (the engine's prefetch
    report, at any depth), the other readers' share of that window (the drive's reads in
    decode less the sink's own), an expert's read at the probe's rate at the readers' queue
    depth, the demand time a used read saves (the layer fit) and what a read spills into the
    next layer's (the engine's measure of the reads' time past their windows, read_spill),
    and what the cache slot each rank takes would serve. 0 when no read pays; nothing
    without the statistics."""
    pf, ssd = m.P[p].get("prefetch") or {}, m.dev.get("ssd") or {}
    pr, nr, rr = pf.get("predicted_r"), pf.get("novel_r"), pf.get("novel_routed_r")
    if not (pr and nr and rr and pf.get("windows") and ssd and m.PF.get("B_e")):
        return []
    # the engine's readers each read one role at a time; they print their number when they start
    # (depth >= 1)
    q = int(pf.get("readers") or len(m.dev.get("bytes_per_role") or {}) or 1)
    qd = min((int(x) for x in ssd), key=lambda x: (abs(x - q), x))
    rate = ssd[str(qd)]["gibs"] * prefill.GiB
    t_e = m.PF["B_e"] / rate * 1e6
    other = (m.P[p].get("drive_other_bytes") or 0.0) / rate * 1e6 / pf["layers"]
    window = max(0.0, pf["window_us_p50"] - other)
    nu = [n / t if t else 0.0 for n, t in zip(nr, pr)]
    hit = [r / n if n else 0.0 for r, n in zip(rr, nr)]
    cost, why = rank_costs(m, p, len(nu))
    saved, how = read_gain(m.P[p], m.PF["B_e"], t_e)
    spill, spilled = read_spill(pf, other)
    spill = (
        [None if x is None else x / t_e for x in spill]
        if isinstance(spill, list)
        else (spill or 0.0) / t_e
    )
    k, k_win, k_mem, k_pay = depth(nu, hit, window / t_e, cost, saved / t_e, spill)
    return [
        Setting(
            "GGML_KIMI_CPU_SINK_PREFETCH",
            k,
            "derived",
            f"min(k_pay {k_pay}, k_mem {k_mem}) of top-k {len(nu)}: a read pays when the "
            f"share of it the next "
            f"layer uses times the demand time it saves, {saved / 1e3:.2f} ms ({how}), "
            f"exceeds its spill into "
            f"the next layer's demand reads, {spilled}, and, when it starts after the "
            f"drive's idle window "
            f"(k_win {k_win} start inside it), its own transfer; the drive's idle window "
            f"{pf['window_us_p50']:.0f} us a hinted layer (median of {int(pf['windows'])}) "
            f"less {other:.0f} us "
            f"of other readers, {window / t_e:.2f} expert reads of {t_e:.0f} us "
            f"({m.PF['B_e']} bytes at "
            f"{rate / prefill.GiB:.2f} GiB/s, the probe at QD {qd} for {q} readers); novel "
            f"per rank "
            f"{','.join(f'{x:.3f}' for x in nu)}, of them routed "
            f"{','.join(f'{x:.3f}' for x in hit)}; "
            + (
                f"the slots they take serve {','.join(f'{x:.4f}' for x in cost)} a token ({why})"
                if cost is not None
                else f"no slot cost ({why})"
            )
            + f" (the bootstrap's depth {m.P[p].get('pf_k')})",
        )
    ]


def context(workload: Workload) -> int:
    """The context a conversation of the workload fills: every turn's new prompt tokens and its
    reply, rounded up to a multiple of 256. It bounds what a server can take, so derive()
    writes it only for a known workload: a learned one is the plan's forecast (turns the
    size of the bootstrap text), and a longer conversation would not fit."""
    n = sum(workload["prompts"]) + len(workload["prompts"]) * workload["decode"]
    return -(-n // CONTEXT_STEP) * CONTEXT_STEP


def derive(
    m: MachineView,
    p: str,
    pending: Iterable[str] = (),
    workload: Optional[KnownWorkload] = None,
    ledgers: Iterable[tuple[str, Optional[Ledger]]] = (),
) -> list[Setting]:
    """The derived settings of placement p (the plan's configurations with p share them) for
    the workload (the plan's: turns of new prompt tokens and replies, 'known' when it is the
    conversation to be served); those named in pending are marked so. ledgers: the host-
    memory ledger's records of runs besides the bootstraps (logparse.ledger)."""
    PF, out = m.PF, []
    limit = (
        (PF.get("ubatch_p") or {}).get(p.partition("@")[0])
        or PF.get("ubatch")
        or (PF["memory"].get(p) or {}).get("n_ubatch")
    )
    bound = f"the ubatch a server of this machine started with, {limit}"
    if workload and workload.get("known"):
        C = context(workload)
        out.append(
            Setting(
                "CONTEXT",
                C,
                "derived",
                f"the workload: turns of {','.join(map(str, workload['prompts']))} "
                f"new prompt tokens and {workload['decode']} decode tokens each, up to a "
                f"multiple of "
                f"{CONTEXT_STEP}",
            )
        )
        bound = f"the context, at most {bound}" if limit else "the context"
        limit = min(limit, C) if limit else C
    if limit:  # an engine that reports no prefill ubatch leaves nothing to derive the passes from
        out += prefill_route(
            m,
            p,
            cast(int, limit),
            bound,
            workload["prompts"] if workload else None,
            cast(KnownWorkload, workload or {}).get("decode") or 0,
        )
    carried = prefill_carried(m, p)
    out += [carried] if carried else prefill_start(m)
    out += prefetch_depth(m, p)
    align = m.P[p].get("checkpoint_align")
    if align and workload and len(workload["prompts"]) == 1:
        out.append(
            Setting(
                "CTX_CHECKPOINTS",
                0,
                "derived",
                "the workload is one turn: no later request restores a state, and a "
                "checkpoint would only "
                "take host memory (on unified memory, from the GPU tier) while the request runs",
            )
        )
    elif align:
        out.append(
            Setting(
                "CTX_CHECKPOINTS",
                1,
                "derived",
                f"the memory saves an earlier state after each prompt batch (at multiples of "
                f"{align} "
                "tokens, the bootstrap's log), so no prompt is split for one: the latest "
                "end of a prompt "
                "serves an append-only conversation, the disk prompt cache the rest",
            )
        )
    # on unified memory the host-memory ledger's margin grows by the growth it does not track in
    # decode; a margin a
    # run learned is above the start the launcher gave it, and is carried to the servers after the
    # plan
    runs = [(f"bootstrap {q}", m.P[q].get("ledger")) for q in m.P if "@" not in q] + [
        ("/".join(d.rstrip("/").split("/")[-2:]), x) for d, x in ledgers
    ]
    learned = [
        (r, -(-int(x["margin"]) // 2**20))
        for r, x in runs
        if x and x.get("unified") and x.get("margin_updates")
    ]
    if learned:
        out.append(
            Setting(
                "GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB",
                max(b for _, b in learned),
                "self-tuned",
                f"unified memory: the largest margin the host-memory ledger learned "
                f"({', '.join(f'{r}: {b} MiB' for r, b in learned)})",
            )
        )
    return [s._replace(pending=s.name in pending) for s in out]


def env(settings: Iterable[Setting]) -> str:
    """The settings as a launcher environment ('K=V K=V'): the knobs of a server the framework
    starts; pending values are not passed."""
    return " ".join(f"{s.name}={s.value}" for s in settings if not s.pending)


def render(settings: Iterable[Setting], head: Iterable[str] = ()) -> str:
    """settings.env: the header as comments, then one line per value with its receipt (a
    comment if pending)."""
    lines = [f"# {h}" for h in head]
    lines += [
        f"{'# pending ' if s.pending else ''}{s.name}={s.value}  # source={s.source}; "
        f"input={s.input}"
        for s in settings
    ]
    return "\n".join(lines) + "\n"
