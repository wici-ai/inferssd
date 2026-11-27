# SPDX-License-Identifier: Apache-2.0
"""Pre-inference decode cost model, with explicit placement and routing assumptions.

Synthetic matrix measurements price arithmetic. The MoE reference uses one rank-based
routing prior and whole expert-cache slots for every checkpoint; scenarios vary cache
overlap, I/O overlap and the measured timing spread, not the routing prior. Attention,
graph overhead and cache warm-up remain outside the service-time estimate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from inferssd.machine.characteristics import GPU, Characteristics, ComputeRow
from inferssd.strategy.derive import (
    IN_FLIGHT,
    cpu_affinity,
    drive_rate,
    gpu_reserve_mib,
    host_headroom_mib,
    link_rates,
    placement,
    single_gpu,
    tier_gpus,
)
from inferssd.strategy.inputs import Experts, ModelFacts, RuntimeFacts
from inferssd.strategy.kernels import gpu_expert_refusal
from inferssd.strategy.routing import cache_slot_bytes, hit_at_slots, rank_hit_curve
from inferssd.strategy.weights import DenseWeights

GB = 1e9
ENGINE_RESERVE_BYTES = 512 << 20


@dataclass(frozen=True)
class Estimate:
    tokens_per_second: Optional[float]
    detail: str
    assumptions: tuple[str, ...] = ()
    scenarios: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    scenario_details: tuple[dict[str, object], ...] = ()
    scenario_range: Optional[tuple[float, float]] = None
    reference_scenario: str = "rank_serial"


def drive_iqr(
    c: Characteristics, experts: Experts, key: Optional[str], central_Bps: float
) -> tuple[float, float, int]:
    """Repeated measurements of the same request size/QD used by drive_rate.

    This describes the profiling window only; it cannot predict a later shared-
    storage stall. Do not use measurements from different queues as repetitions.
    """
    entry = next(row for row in c["storage"] if row["drive_key"] == key)
    request = experts.mean_read_bytes / 3
    knee = min(entry["knee"], key=lambda r: (abs(math.log(r["bytes"] / request)), r["bytes"]))
    cap = min(IN_FLIGHT, entry["qd_cap"] or IN_FLIGHT)
    first = knee["qd"] if knee["qd"] <= cap else 1
    points = [
        r
        for r in entry["random_read"]
        if r["bytes"] == knee["bytes"] and first <= r["qd"] <= cap and r["Bps"] > 0
    ]
    point = max(points, key=lambda r: (r["Bps"], -r["qd"]))
    samples = sorted(x for x in point["Bps_runs"] if x > 0)
    if not samples:
        return central_Bps, central_Bps, 0

    def q(fraction: float) -> float:
        pos = (len(samples) - 1) * fraction
        lower = math.floor(pos)
        return samples[lower] + (pos - lower) * (samples[math.ceil(pos)] - samples[lower])

    return min(q(0.25), central_Bps), max(q(0.75), central_Bps), len(samples)


def headroom_bytes(characteristics: Characteristics, runtime: RuntimeFacts, platform: str) -> int:
    mib = host_headroom_mib(characteristics, runtime, platform)
    return ENGINE_RESERVE_BYTES if mib is None else mib << 20


def host_memory_bytes(characteristics: Characteristics, runtime: RuntimeFacts) -> int:
    """Host memory an estimate plans with: the available memory, or the installed memory when
    that is not known, within the memory budget and the installed memory."""
    total = characteristics["memory"]["total_bytes"]
    memory = total if runtime.available_bytes is None else min(runtime.available_bytes, total)
    if runtime.memory_budget_bytes is not None:
        memory = min(memory, runtime.memory_budget_bytes)
    return memory


def matvec_rate(
    rows: list[ComputeRow],
    name: str,
    outputs: int,
    inputs: int,
    timing: str = "median",
    *,
    target_bytes: Optional[int] = None,
) -> Optional[float]:
    """Closest timed shape, retaining measured fixed cost below the smallest profiled matrix."""
    candidates = [
        row
        for row in rows
        if row["format"].upper() == name.upper()
        and any(p["tokens"] == 1 and p.get("weight_Bps", 0) > 0 for p in row["points"])
    ]
    if not candidates:
        return None
    row = min(
        candidates,
        key=lambda r: abs(math.log(r["rows"] / outputs)) + abs(math.log(r["cols"] / inputs)),
    )

    def measured(r: ComputeRow) -> tuple[float, float]:
        point = next(p for p in r["points"] if p["tokens"] == 1 and p.get("weight_Bps", 0) > 0)
        duration = (
            point.get("us_q1", point["us"])
            if timing == "q1"
            else point.get("us_q3", point["us"])
            if timing == "q3"
            else point["us"]
        )
        rate = point["weight_Bps"] * point["us"] / duration if duration > 0 else point["weight_Bps"]
        return duration, rate

    if (
        target_bytes
        and outputs * inputs < min(r["rows"] * r["cols"] for r in candidates)
        and target_bytes < min(r["weight_bytes"] for r in candidates)
    ):
        # Probe graphs time a synchronized call. Extrapolating bandwidth alone shrinks
        # its fixed cost in proportion to the payload. The two smallest measured sizes
        # of this format identify T(B) = a + bB. Pool transposes at the same byte size.
        by_bytes: dict[int, list[float]] = {}
        for r in candidates:
            by_bytes.setdefault(r["weight_bytes"], []).append(measured(r)[0])
        sizes = sorted(by_bytes)
        if len(sizes) >= 2:
            first, second = sizes[:2]
            t_first, t_second = median(by_bytes[first]), median(by_bytes[second])
            slope = (t_second - t_first) / (second - first)
            intercept = t_first - slope * first
            if slope >= 0 and intercept > 0:
                return target_bytes * 1e6 / (intercept + slope * target_bytes)
    # Older profiles lack quartiles; their median remains usable.
    return measured(row)[1]


def model_bits(experts: Experts) -> dict[str, float]:
    """Bits per weight of each ggml type the routed EXPERTS take, from their bytes."""
    return {
        name.upper(): role.bits_per_weight(name)
        for role in experts.roles
        for name, _, _ in role.formats
    }


def format_proxies(rows: list[ComputeRow], wanted: Mapping[str, float]) -> dict[str, str]:
    """Each format of WANTED (with its bits per weight) priced as itself when ROWS measure it at
    one token, else as the measured format nearest it in bits per weight (the higher of two as
    near): a default for a format the probe does not measure, such as IQ4_XS and IQ4_NL, until a
    run calibrates it. A row's bits per weight are its weight bytes over its matrix. Rows that
    measure nothing give no entry."""
    measured: dict[str, float] = {}
    for row in rows:
        if any(p["tokens"] == 1 and p.get("weight_Bps", 0) > 0 for p in row["points"]):
            measured.setdefault(
                row["format"].upper(), 8 * row["weight_bytes"] / (row["rows"] * row["cols"])
            )
    proxies = {}
    for name, want in wanted.items():
        if name in measured:
            proxies[name] = name
        elif measured:
            proxies[name] = min(measured, key=lambda m: (abs(measured[m] - want), -measured[m], m))
    return proxies


def proxy_note(where: str, proxies: dict[str, str]) -> Optional[str]:
    borrowed = sorted((name, used) for name, used in proxies.items() if name.upper() != used)
    if not borrowed:
        return None
    return (
        f"{where} rate of unmeasured "
        + ", ".join(f"{name} taken from {used}" for name, used in borrowed)
        + " (nearest in bits per weight, weight bytes per second); a default, not calibrated."
    )


def cpu_decode_seconds(
    c: Characteristics,
    experts: Experts,
    threads: int,
    timing: str = "median",
    *,
    small_matrix_cost: bool = False,
) -> tuple[Optional[float], str]:
    """One expert, using only one-token points and all of its quantization types."""
    rows = [row for row in c["compute"]["cpu"] if not row.get("cpus")]
    names = model_bits(experts)
    proxies = format_proxies(rows, names)
    missing = sorted(set(names) - set(proxies))
    if missing:
        return None, "no CPU compute measurement of " + ", ".join(missing)
    rates = {}
    for count in sorted({row["threads"] for row in rows}):
        selected = [row for row in rows if row["threads"] == count]
        values = [
            matvec_rate(
                selected,
                proxies[name.upper()],
                role.rows,
                role.cols,
                timing,
                target_bytes=size if small_matrix_cost else None,
            )
            for role in experts.roles
            for name, _, size in role.formats
        ]
        if all(rate is not None for rate in values):
            rates[count] = [rate or 0 for rate in values]
    if not rates:
        return None, "no thread count measured every expert format at one token"
    lower = max((n for n in rates if n <= threads), default=min(rates))
    upper = min((n for n in rates if n >= threads), default=max(rates))
    share = (threads - lower) / (upper - lower) if upper != lower else 0
    total, index = 0.0, 0
    for role in experts.roles:
        for name, layers, _ in role.formats:
            rate = rates[lower][index] + share * (rates[upper][index] - rates[lower][index])
            total += layers / experts.layers * role.slice_bytes(name) / rate
            index += 1
    return total, ""


def shared_gpu_bytes(c: Characteristics, runtime: RuntimeFacts, gpu_work_bytes: int) -> int:
    """Host memory the GPU takes on a unified machine: its working memory and the launch reserve
    for other compute processes, both of which the native ledger counts against host memory."""
    if not c["memory"]["unified"]:
        return 0
    return gpu_work_bytes + (gpu_reserve_mib(tier_gpus(runtime, c)) << 20)


def tier_gpu(
    c: Characteristics, runtime: RuntimeFacts, gpu_work_bytes: int
) -> tuple[Optional[GPU], int, list[ComputeRow]]:
    """The one discrete GPU with the VRAM tier (CUDA, ROCm) the engine puts dense weights on, the
    bytes available after the launch reserve and GPU_WORK_BYTES, and its one-token matrix
    measurements; (None, 0, []) without one."""
    gpu = single_gpu(c, runtime)
    if not gpu or c["memory"]["unified"] or gpu["integrated"]:
        return None, 0, []
    usable = tier_gpus(runtime, c)
    free = usable[0].free_bytes if usable else None
    reserve_bytes = gpu_reserve_mib(usable) << 20
    room = max(0, min(free or 0, gpu["vram_total_bytes"]) - reserve_bytes - gpu_work_bytes)
    return gpu, room, gpu.get("decode_compute", [])


def moved_fraction(
    used: int, hit: float, slots: int, cpu_s: float, gpu_s: float, copy_s: float
) -> float:
    """Expected whole-expert moves per route, for the mean number of hits in a layer."""
    if not copy_s or slots == 0:
        return 0.0

    def best_moves(resident: int) -> int:
        # Ready host experts are copied as one group. Resident GPU work can run
        # during those copies, then the GPU computes the group. Prefer fewer
        # moves on a tie, as the native ready-expert planner does.
        host = used - resident
        limit = min(host, max(0, slots - resident))

        def finish(moved: int) -> float:
            cpu = (host - moved) * cpu_s
            gpu = max(resident * gpu_s, moved * copy_s) + moved * gpu_s
            return max(cpu, gpu)

        return min(range(limit + 1), key=finish)

    hits = used * hit
    lower = math.floor(hits)
    fraction = hits - lower
    moved = best_moves(lower)
    if fraction:
        return (moved + fraction * (best_moves(lower + 1) - moved)) / used
    return moved / used


def decode_estimate(
    characteristics: Characteristics,
    model: ModelFacts,
    dense_read_bytes: int,
    runtime: RuntimeFacts,
    platform: str,
    drive_key: Optional[str],
    *,
    host_work_bytes: int = 0,
    gpu_work_bytes: int = 0,
    observed_hits: Optional[tuple[float, float]] = None,
    graph_weights: Optional[DenseWeights] = None,
    measured_dense_ms: Optional[float] = None,
) -> Estimate:
    """Estimate default placement, optionally reserving known KV/graph working memory.

    Work reserves are additional to the engine's host margin. They must come from
    the intended context/graph allocation, not fitting a measured tokens/s value.
    Zero means unknown and makes cache capacities upper estimates. OBSERVED_HITS, the
    fraction of routes hitting VRAM and the fraction missing both VRAM and the host cache,
    replace the routing prior in the reference, for checking the prior against a measured run.
    MEASURED_DENSE_MS, the dense time of a one-row decode step the engine measured on this
    machine (inferssd.engine.decode_steps), replaces the dense time priced from bytes.
    """
    if min(host_work_bytes, gpu_work_bytes, dense_read_bytes) < 0:
        raise ValueError("memory sizes must be nonnegative")
    if observed_hits is not None and (
        len(observed_hits) != 2
        or not all(0 <= value <= 1 for value in observed_hits)
        or sum(observed_hits) > 1
    ):
        raise ValueError("VRAM hit and two-tier miss fractions each require [0, 1] and sum <= 1")
    experts = model.experts
    if experts is None:
        return Estimate(None, "the model has no routed experts")
    if not experts.used or not experts.routed_bytes:
        return Estimate(None, "the model names no routed experts per token")
    # A visible CUDA/ROCm tier initializes the executors for the declared layer formats,
    # even when the profiler has no timing for that GPU or cannot price its placement.
    devices = tier_gpus(runtime, characteristics)
    if devices:
        refusal = gpu_expert_refusal(experts)
        if refusal:
            return Estimate(None, refusal)
    memory_Bps = max((r["Bps"] for r in characteristics["memory"]["dram_read_Bps"]), default=0)
    try:
        threads, _, _ = placement(characteristics, model, runtime, [])
    except ValueError as error:
        return Estimate(None, str(error))
    drive_gbps, why = drive_rate(characteristics, experts, drive_key)
    drive_reason = why
    available = host_memory_bytes(characteristics, runtime)
    room = max(
        0,
        available
        - headroom_bytes(characteristics, runtime, platform)
        - host_work_bytes
        - shared_gpu_bytes(characteristics, runtime, gpu_work_bytes),
    )
    gpu, gpu_room, rows = tier_gpu(characteristics, runtime, gpu_work_bytes)
    assumptions = [
        "Warmed cache hits come from a shared 1/rank routing assumption, capped at one selection "
        "per expert per token; hits are not observed on this machine.",
        "Equal slots per layer. Distinct host/device coverage is the midpoint of completely "
        "overlapping and disjoint slot counts.",
        "Serial I/O plus CPU/GPU service time; prefetch overlap is a separate scenario.",
        "Hot experts are cached first, after warm-up; routing order and replacement are unknown.",
        "Below the smallest profiled expert matrix, two same-format probe sizes estimate a "
        "fixed call time plus time per weight byte when both terms are nonnegative.",
        "CPU/GPU timing and drive IQR describe profiling samples, not a confidence interval.",
        "Drive IQR covers profiling repetitions only, not future shared-storage stalls.",
        "Attention, non-matmul graph work, lookup I/O and cache warm-up are unpriced.",
        "KV/graph working memory is unreserved unless work-byte arguments are supplied.",
    ]
    reserve = gpu_reserve_mib(devices)
    if reserve:
        assumptions.append(
            f"A busy GPU leaves the launch reserve of {reserve} MiB. "
            "The estimate omits compute contention."
        )
    # Dense quantization is not retained by ModelFacts. Q8_0 is an explicit proxy;
    # never substitute a GPU's peak bandwidth for measured quantized computation.
    dense_rate = matvec_rate(rows, "Q8_0", 4096, 4096)
    gpu_expert_s = 0.0
    gpu_fast_s = 0.0
    gpu_slow_s = 0.0
    missing = []
    names = model_bits(experts)
    cpu_proxies = format_proxies(
        [row for row in characteristics["compute"]["cpu"] if not row.get("cpus")], names
    )
    gpu_proxies = format_proxies(rows, names)
    for role in experts.roles:
        for name, layers, size in role.formats:
            used = gpu_proxies.get(name.upper(), name)
            rate = matvec_rate(rows, used, role.rows, role.cols)
            if rate is None:
                missing.append(name)
            else:
                weight = layers * size / experts.layers
                for_median = (
                    matvec_rate(rows, used, role.rows, role.cols, target_bytes=size) or rate
                )
                fast = matvec_rate(rows, used, role.rows, role.cols, "q1", target_bytes=size)
                slow = matvec_rate(rows, used, role.rows, role.cols, "q3", target_bytes=size)
                gpu_expert_s += weight / for_median
                gpu_fast_s += weight / (fast or for_median)
                gpu_slow_s += weight / (slow or for_median)
    placement_bytes = graph_weights.placement_bytes if graph_weights else dense_read_bytes
    if graph_weights:
        assumptions.extend(graph_weights.graph_assumptions())
    use_gpu = bool(gpu and gpu_room >= placement_bytes and dense_rate and not missing)
    for note in (
        proxy_note("CPU", cpu_proxies),
        proxy_note("GPU", gpu_proxies) if use_gpu else None,
    ):
        if note:
            assumptions.append(note)
    if gpu and not use_gpu:
        assumptions.append(
            "CPU fallback estimate: GPU matvec data or dense placement capacity is missing."
        )
    if use_gpu:
        resident = dense_read_bytes
        cache = room  # dense weights are in VRAM, not subtracted again from RAM
        gpu_cache = max(0, gpu_room - placement_bytes)
        dense_s = dense_read_bytes / (dense_rate or 1)
        assumptions.append(
            "Dense weights on one GPU; dense Q8_0 matvec proxy (actual types unknown)."
        )
    else:
        if graph_weights:
            resident = dense_read_bytes - graph_weights.host_spill_bytes(room)
        else:
            resident = min(dense_read_bytes, room)
        cache = max(0, room - placement_bytes)
        gpu_cache = 0
    routed = experts.routed_bytes
    total_slots = experts.layers * experts.count
    host_slot_bytes, gpu_slot_bytes = cache_slot_bytes(experts)
    host_slots = min(total_slots, cache // host_slot_bytes)
    gpu_slots = min(total_slots, gpu_cache // gpu_slot_bytes)
    if not use_gpu and memory_Bps <= 0:
        return Estimate(None, "no memory read rate in the profile")
    cpu_time, cpu_reason = cpu_decode_seconds(
        characteristics, experts, threads.decode or 1, small_matrix_cost=True
    )
    if cpu_time is None and (
        gpu_slots < total_slots or (observed_hits is not None and observed_hits[0] < 1)
    ):
        return Estimate(None, cpu_reason)
    # For full VRAM residency every route executes on the GPU.
    cpu_expert_s = cpu_time or 0.0
    inclusive_slots = max(host_slots, gpu_slots)
    disjoint_slots = min(total_slots, host_slots + gpu_slots)
    spill = 0 if use_gpu else dense_read_bytes - resident
    if drive_gbps is None and (
        spill or inclusive_slots < total_slots or (observed_hits and observed_hits[1] > 0)
    ):
        return Estimate(None, drive_reason)
    if observed_hits is not None and observed_hits[0] > 0 and not use_gpu:
        return Estimate(
            None, "observed VRAM hits require a GPU placement with measured matvec rates"
        )
    drive_Bps = (drive_gbps or 0.0) * GB
    if not drive_Bps:
        assumptions.append("The dense weights and routed store fit; no drive timing is needed.")
    if not use_gpu:
        dense_s = resident / memory_Bps + (spill / drive_Bps if spill else 0)
    top_k = experts.used
    routes = experts.layers * top_k
    io_all_s = routes * experts.mean_read_bytes / drive_Bps if drive_Bps else 0.0

    pinned = link_rates(characteristics, runtime)[0].get("link_gbps") if use_gpu else None
    # The copy contains the role payloads, without the drive read's page padding.
    copy_expert_s = routed / total_slots / (pinned * GB) if pinned else 0.0
    if use_gpu:
        assumptions.append(
            "Dispatch chooses whole ready experts per layer, including zero moves. Fractional "
            "mean hit counts mix the two adjacent integer counts. Slot capacity bounds moves; "
            "live free slots, page locking and read readiness are unknown."
            if pinned
            else "No pinned H2D measurement; host-sourced experts are priced on the CPU."
        )

    def terms(gpu_hit: float, storage_miss: float) -> tuple[float, float, float, float, float]:
        moved = moved_fraction(top_k, gpu_hit, gpu_slots, cpu_expert_s, gpu_expert_s, copy_expert_s)
        executed = gpu_hit + moved
        return (
            routes * cpu_expert_s * (1 - executed),
            routes * gpu_expert_s * executed,
            io_all_s * storage_miss,
            routes * copy_expert_s * moved,
            executed,
        )

    io_low, io_high, io_samples = (
        drive_iqr(characteristics, experts, drive_key, drive_Bps) if drive_Bps else (0.0, 0.0, 0)
    )
    cpu_fast, _ = cpu_decode_seconds(
        characteristics, experts, threads.decode or 1, "q1", small_matrix_cost=True
    )
    cpu_slow, _ = cpu_decode_seconds(
        characteristics, experts, threads.decode or 1, "q3", small_matrix_cost=True
    )
    if use_gpu:
        dense_fast = dense_read_bytes / (
            matvec_rate(rows, "Q8_0", 4096, 4096, "q1") or dense_rate or 1
        )
        dense_slow = dense_read_bytes / (
            matvec_rate(rows, "Q8_0", 4096, 4096, "q3") or dense_rate or 1
        )
    else:
        dense_fast = resident / memory_Bps + (spill / io_high if spill else 0)
        dense_slow = resident / memory_Bps + (spill / io_low if spill else 0)
    if measured_dense_ms is not None:
        dense_s = dense_fast = dense_slow = measured_dense_ms / 1e3
        assumptions.append(
            "The dense time is the engine's measurement of a one-row decode step on this machine "
            "and engine, with the model's draft head loaded, in place of the time priced from "
            "the dense bytes."
        )
    cases: list[tuple[str, float, float, dict[str, object]]] = []
    hits = rank_hit_curve(experts.count, experts.used)
    gpu_hit = hit_at_slots(hits, gpu_slots / experts.layers)
    shared_slots = (inclusive_slots + disjoint_slots) / 2
    disk_miss = 1 - hit_at_slots(hits, shared_slots / experts.layers)
    for name, slots in (
        ("rank", shared_slots),
        ("rank_inclusive", inclusive_slots),
        ("rank_disjoint", disjoint_slots),
    ):
        cases.append(
            (
                name,
                gpu_hit,
                1 - hit_at_slots(hits, slots / experts.layers),
                {
                    "gpu_slots": gpu_slots,
                    "host_slots": host_slots,
                    "distinct_slots": slots,
                },
            )
        )
    reference = "rank_serial"
    route_detail = "rank-based routing estimate"
    if observed_hits is not None:
        gpu_hit, disk_miss = observed_hits
        cases.append(("observed", gpu_hit, disk_miss, {}))
        reference = "observed_serial"
        route_detail = "supplied cache fractions"
        assumptions.append(
            "Reference VRAM hits and misses of both caches are supplied; "
            "the rank scenarios keep the routing prior."
        )
    scenarios: dict[str, float] = {}
    details: list[dict[str, object]] = []
    lows, highs = [], []
    for name, gh, miss, parameters in cases:
        ct, gt, it, copy_s, executed = terms(gh, miss)
        no_overlap = 1 / (dense_s + ct + gt + copy_s + it)
        # Only half of the smaller I/O/compute service time is hidden in this
        # intermediate scenario; the optimistic endpoint allows full overlap.
        partial = 1 / (dense_s + ct + gt + copy_s + it - 0.5 * min(it, ct + gt + copy_s))
        slow_io = it * drive_Bps / io_low if it else 0.0
        fast_io = it * drive_Bps / io_high if it else 0.0
        slow_cpu = routes * (cpu_slow or cpu_expert_s) * (1 - executed)
        fast_cpu = routes * (cpu_fast or cpu_expert_s) * (1 - executed)
        slow_gpu, fast_gpu = routes * gpu_slow_s * executed, routes * gpu_fast_s * executed
        low = min(no_overlap, 1 / (dense_slow + slow_cpu + slow_gpu + copy_s + slow_io))
        high = max(no_overlap, 1 / (dense_fast + max(fast_cpu, fast_gpu + copy_s, fast_io)))
        scenarios[name + "_serial"] = no_overlap
        details.append(
            {
                "name": name,
                **parameters,
                "gpu_hit_fraction": gh,
                "gpu_execution_fraction": executed,
                "h2d_ms": copy_s * 1e3,
                "disk_miss_fraction": miss,
                "cpu_experts_ms": ct * 1e3,
                "gpu_experts_ms": gt * 1e3,
                "storage_ms": it * 1e3,
                "no_overlap_tps": no_overlap,
                "half_io_overlap_tps": partial,
                "scenario_low_tps": low,
                "scenario_high_tps": high,
            }
        )
        lows.append(low)
        highs.append(high)
    cpu_s, gpu_s, drive_s, copy_s, executed = terms(gpu_hit, disk_miss)
    seconds = dense_s + cpu_s + gpu_s + copy_s + drive_s
    drive_detail = (
        f"{disk_miss:.0%} from the drive at {drive_gbps:.2f} GB/s {drive_s * 1e3:.0f} ms"
        if drive_gbps is not None
        else "fully resident; 0 ms from the drive"
    )
    detail = (
        f"{seconds * 1e3:.0f} ms a token: dense {dense_s * 1e3:.0f} ms"
        + (f" ({spill / GB:.1f} GB of it from the drive)" if spill else "")
        + f", CPU experts {cpu_s * 1e3:.0f} ms, GPU experts {gpu_s * 1e3:.0f} ms, "
        f"H2D {copy_s * 1e3:.0f} ms, {drive_detail} "
        f"(a {cache / GB:.1f} GB cache of {routed / GB:.0f} GB of experts); "
        f"{route_detail}"
    )
    return Estimate(
        1 / seconds,
        detail,
        tuple(assumptions),
        scenarios,
        {
            "dense_ms": dense_s * 1e3,
            "cpu_experts_ms": cpu_s * 1e3,
            "gpu_experts_ms": gpu_s * 1e3,
            "storage_ms": drive_s * 1e3,
            "host_cache_capacity_bytes": float(cache),
            "gpu_cache_capacity_bytes": float(gpu_cache),
            "gpu_hit_fraction": gpu_hit,
            "gpu_execution_fraction": executed,
            "h2d_ms": copy_s * 1e3,
            "storage_miss_fraction": disk_miss,
            "decode_threads": float(threads.decode or 1),
            "memory_budget_bytes": float(available),
            "drive_Bps": drive_Bps,
            "drive_iqr_low_Bps": io_low,
            "drive_iqr_high_Bps": io_high,
            "drive_repetitions": float(io_samples),
            **(graph_weights.graph_components() if graph_weights else {}),
        },
        tuple(details),
        (min(lows), max(highs)),
        reference,
    )


def _dense_drive_rates(
    c: Characteristics, drive_key: Optional[str], disk_bytes: int, *, max_depth: int = 8
) -> Optional[tuple[float, float]]:
    """Measured throughput range for possible chunk sizes at attainable queue depths."""
    # ggml-cuda/ssd/transport.cpp: up to eight chunks of 16 MiB in flight.
    drive = next((d for d in c["storage"] if d["drive_key"] == drive_key), None)
    if drive is None:
        return None
    cap = min(max_depth, drive["qd_cap"] or max_depth)
    by_size: dict[int, tuple[int, float]] = {}
    for row in drive["random_read"]:
        size, depth = row["bytes"], row["qd"]
        if size > 16 << 20 or depth > min(cap, math.ceil(disk_bytes / size)) or row["Bps"] <= 0:
            continue
        if size not in by_size or depth > by_size[size][0]:
            by_size[size] = (depth, row["Bps"])
    if not by_size:
        return None
    speeds = [speed for _, speed in by_size.values()]
    return min(speeds), max(speeds)


def _dense_compute_times(
    rows: list[ComputeRow],
    formats: tuple[tuple[str, int], ...],
    read_bytes: int,
    *,
    device: str = "GPU",
    proxy: str = "Q8_0",
    weights: Optional[DenseWeights] = None,
) -> tuple[Optional[tuple[float, float, float]], list[str]]:
    """Matvec seconds (median, fast, slow), including the non-layer output projection."""
    known_bytes = sum(size for _, size in formats)
    # Catalogs describe all repeating-layer tensors. If a caller excluded additional lookup
    # weights, normalize that mix to the smaller number of bytes actually visited per token.
    scale = min(1.0, read_bytes / known_bytes) if known_bytes else 1.0
    mix = [(name, 4096, 4096, size * scale) for name, size in formats]
    residual = read_bytes - min(read_bytes, known_bytes)
    if residual:
        mix.append(("outside catalogued layers", 4096, 4096, float(residual)))
    if weights is not None:
        mix = [
            (name, rows or 4096, cols or 4096, float(size))
            for name, rows, cols, size in weights.shapes
        ]
    times = [0.0, 0.0, 0.0]
    borrowed = set()
    for name, outputs, inputs, size in mix:
        measured = name
        median = matvec_rate(rows, measured, outputs, inputs)
        if not median:
            measured = proxy
            borrowed.add(name)
            median = matvec_rate(rows, measured, outputs, inputs)
        if not median:
            return None, []
        times[0] += size / median
        for i, timing in enumerate(("q1", "q3"), start=1):
            rate = matvec_rate(rows, measured, outputs, inputs, timing) or median
            times[i] += size / rate
    notes = [
        device + " timings use the profiled matvec nearest each GGUF matrix shape; "
        "nonmatrix weights use a 4096 x 4096 bandwidth proxy."
        if weights is not None
        else device + " matvec timings use the profiled matrix nearest 4096 x 4096."
    ]
    if borrowed:
        notes.append(f"Dense {proxy} matvec proxy for " + ", ".join(sorted(borrowed)) + ".")
    return (times[0], times[1], times[2]), notes


def _dense_host_estimate(
    c: Characteristics,
    read_bytes: int,
    model: Optional[ModelFacts],
    weights: Optional[DenseWeights],
    runtime: RuntimeFacts,
    host_room: int,
    drive_key: Optional[str],
    assumptions: list[str],
) -> Estimate:
    """Conditional CPU execution with a reusable page cache and mmap fault sensitivity."""
    allowed, _ = cpu_affinity(c, runtime)
    dram = max(
        (row["Bps"] for row in c["memory"]["dram_read_Bps"] if row["threads"] <= len(allowed)),
        default=0.0,
    )
    # Use one thread placement consistently across all formats. With no layer format facts
    # the measured bulk DRAM scan remains a useful standalone bandwidth reference.
    rows = [
        row
        for row in c["compute"]["cpu"]
        if 0 < row["threads"] <= len(allowed) and set(row.get("cpus", ())) <= allowed
    ]
    times = None
    if rows and (model or weights or not dram):
        desired = len(allowed)
        pinned: tuple[int, ...] = ()
        if model:
            threads, _, _ = placement(c, model, runtime, [])
            desired = threads.decode or desired
            pinned = threads.decode_cpus or threads.cpus
        eligible = [r for r in rows if tuple(r.get("cpus", ())) == pinned]
        if not eligible and pinned:
            eligible = [r for r in rows if not r.get("cpus")]
        if eligible:
            count = min({r["threads"] for r in eligible}, key=lambda n: (abs(n - desired), n))
            selected = [r for r in eligible if r["threads"] == count]
            # When DRAM is missing, use a real one-token row as the bandwidth proxy for
            # unknown formats, choosing the measured encoding closest to an 8-bit matrix.
            proxy = "Q8_0" if dram else format_proxies(selected, {"Q8_0": 8.5}).get("Q8_0", "Q8_0")
            times, notes = _dense_compute_times(
                selected,
                model.layer_format_bytes if model else (),
                read_bytes,
                device=f"{count}-thread CPU",
                proxy=proxy,
                weights=weights,
            )
            if times:
                assumptions.extend(notes)
    if not dram and not times:
        return Estimate(None, "no usable CPU one-token matvec or RAM read rate in the profile")
    memory_s = read_bytes / dram if dram else 0.0
    median, fast, slow = (max(t, memory_s) for t in (times or (0.0, 0.0, 0.0)))
    resident = min(weights.placement_bytes if weights else read_bytes, host_room)
    disk = weights.host_spill_bytes(host_room) if weights else read_bytes - resident
    disk_s = disk_fast = disk_slow = 0.0
    scenarios = {}
    if disk:
        speeds = _dense_drive_rates(c, drive_key, disk, max_depth=1)
        if not speeds:
            return Estimate(None, "CPU dense paging requires a matching model-drive measurement")
        low, high = speeds
        disk_fast, disk_slow = disk / high, read_bytes / low
        disk_s = disk / math.sqrt(low * high)
        scenarios["dense_host_full_reread"] = 1 / (median + read_bytes / math.sqrt(low * high))
        if weights and weights.graph_store_bytes:
            hot_miss = max(0, read_bytes - host_room)
            scenarios["dense_host_hot_experts"] = 1 / (median + hot_miss / math.sqrt(low * high))
            disk_fast = hot_miss / high
        assumptions.append(
            "CPU paging reference keeps useful resident pages and serializes misses with compute. "
            "One-outstanding-read measurements proxy mmap faults. The full-reread scenario allows "
            "cyclic scans to evict the entire prior token's pages."
        )
    seconds = median + disk_s
    scenarios["dense_host_serial"] = 1 / seconds
    low_tps, high_tps = 1 / (slow + disk_slow), 1 / max(fast, disk_fast)
    if dram:
        detail = f"{read_bytes / GB:.2f} GB of dense weights from RAM at {dram / GB:.0f} GB/s"
    else:
        detail = f"{read_bytes / GB:.2f} GB of dense weights at the measured CPU matvec rate"
    if disk:
        detail += f", including {disk / GB:.2f} GB first paged from disk"
    assumptions.append(
        "Host execution reference. GPU execution and multi-device splits are unpriced."
    )
    return Estimate(
        1 / seconds,
        f"{seconds * 1e3:.1f} ms a token: {detail}",
        tuple(assumptions),
        scenarios,
        {
            "dense_ms": seconds * 1e3,
            "dense_read_bytes": float(read_bytes),
            "host_resident_bytes": float(resident),
            "dense_drive_bytes": float(disk),
            "cpu_compute_ms": median * 1e3,
            "storage_ms": disk_s * 1e3,
            **(weights.graph_components() if weights else {}),
        },
        scenario_range=(low_tps, high_tps) if low_tps != high_tps else None,
        reference_scenario="dense_host_serial",
    )


def dense_decode_estimate(
    characteristics: Characteristics,
    dense_read_bytes: int,
    runtime: RuntimeFacts,
    *,
    model: Optional[ModelFacts] = None,
    weights: Optional[DenseWeights] = None,
    platform: Optional[str] = None,
    drive_key: Optional[str] = None,
    host_work_bytes: int = 0,
    gpu_work_bytes: int = 0,
) -> Estimate:
    """One-stream dense decode with VRAM residency and pipelined H2D and disk transfers."""
    if min(host_work_bytes, gpu_work_bytes) < 0:
        raise ValueError("memory sizes must be nonnegative")
    if dense_read_bytes <= 0:
        return Estimate(None, "the model's dense weight bytes are not known")
    assumptions = [
        "One read of active non-lookup weights per token. Attention, active KV reads, lookup "
        "I/O and graph execution overhead are unpriced.",
    ]
    if weights:
        assumptions.extend(weights.graph_assumptions())
    available = host_memory_bytes(characteristics, runtime)
    platform = platform or characteristics["identity"]["hard"]["platform"]
    host_room = max(
        0,
        available
        - headroom_bytes(characteristics, runtime, platform)
        - host_work_bytes
        - shared_gpu_bytes(characteristics, runtime, gpu_work_bytes),
    )
    gpu, gpu_room, rows = tier_gpu(characteristics, runtime, gpu_work_bytes)
    formats = model.layer_format_bytes if model else ()
    devices = tier_gpus(runtime, characteristics)
    compute, notes = _dense_compute_times(rows, formats, dense_read_bytes, weights=weights)
    if not gpu or not compute or not devices or devices[0].free_bytes is None:
        if devices and len(devices) > 1:
            assumptions.append("Multiple GPUs are visible; the per-device layer split is unknown.")
        elif gpu and devices and devices[0].free_bytes is None:
            assumptions.append("Free VRAM is unknown for dense layer placement.")
        elif gpu and not compute:
            assumptions.append("GPU matvec timing unavailable; run profile --estimate-compute.")
        return _dense_host_estimate(
            characteristics,
            dense_read_bytes,
            model,
            weights,
            runtime,
            host_room,
            drive_key,
            assumptions,
        )
    reserve = gpu_reserve_mib(devices)
    if reserve:
        assumptions.append(f"A busy GPU leaves the launch reserve of {reserve} MiB.")
    # The engine streams the repeating layers' matrices. The catalog groups their bytes by
    # type, so round-off to whole layers and the transport ring are outside this byte model.
    layer_bytes = (
        min(dense_read_bytes, sum(size for _, size in formats)) if formats else dense_read_bytes
    )
    if weights is not None:
        layer_bytes = weights.layer_matrix_bytes
    placement_bytes = weights.placement_bytes if weights else dense_read_bytes
    # The native loader enables bit-reversed layer streaming only for models with n_expert=0.
    # A packed ordinary-graph MoE needs its entire expert tensor to fit the execution device.
    fixed = (
        placement_bytes if weights and weights.graph_store_bytes else dense_read_bytes - layer_bytes
    )
    if gpu_room < fixed:
        assumptions.append("Free VRAM falls below the estimated non-streamable dense weights.")
        return _dense_host_estimate(
            characteristics,
            dense_read_bytes,
            model,
            weights,
            runtime,
            host_room,
            drive_key,
            assumptions,
        )
    assumptions.extend(notes)
    resident = min(gpu_room, placement_bytes)
    streamed = placement_bytes - resident
    pinned = min(streamed, host_room)
    disk = streamed - pinned
    assumptions.append(
        "Byte-level placement after work reserves. Supply GPU/host work bytes "
        "for KV, graph and staging allocations. Whole-layer rounding is omitted."
    )
    link_s = drive_s = drive_slow = drive_fast = 0.0
    if streamed:
        rates, why = link_rates(characteristics, runtime)
        pinned_gbps = rates.get("link_gbps")
        if not pinned_gbps:
            return Estimate(None, "no pinned H2D measurement for dense streaming: " + why)
        link_s = streamed / (pinned_gbps * GB)
    if disk:
        drive_rates = _dense_drive_rates(characteristics, drive_key, disk)
        if not drive_rates:
            return Estimate(None, "no matching model-drive measurement for dense streaming")
        low_bps, high_bps = drive_rates
        drive_fast, drive_slow = disk / high_bps, disk / low_bps
        # The catalog omits the distribution of chunk sizes. The geometric center gives
        # equal multiplicative error against the two measured size endpoints.
        drive_s = math.sqrt(drive_fast * drive_slow)
        assumptions.append(
            "Dense read sizes are unknown. Timed sizes up to 16 MiB and depth 8 span "
            f"{low_bps / GB:.2f}–{high_bps / GB:.2f} GB/s; the reference uses the geometric mean."
        )
    median, fast, slow = compute
    # Every streamed byte crosses the link, including bytes the drive worker puts into its
    # pinned ring. After pipeline fill the limiting service time sets the steady-state rate.
    seconds = max(median, link_s, drive_s)
    serial = median + link_s + drive_s
    detail = (
        f"{seconds * 1e3:.1f} ms a token from one GPU at {dense_read_bytes / median / GB:.0f} GB/s"
    )
    if streamed:
        detail += (
            f" effective matvec; per token {pinned / GB:.2f} GB from pinned RAM and "
            f"{disk / GB:.2f} GB from disk, {streamed / GB:.2f} GB over H2D"
        )
        assumptions.append(
            "Steady streaming overlaps drive, H2D and GPU compute; "
            "dense_serial prices the same work without overlap."
        )
    low, high = 1 / (slow + link_s + drive_slow), 1 / max(fast, link_s, drive_fast)
    return Estimate(
        1 / seconds,
        detail,
        tuple(assumptions),
        {"dense_pipeline": 1 / seconds, "dense_serial": 1 / serial},
        {
            "dense_ms": seconds * 1e3,
            "dense_read_bytes": float(dense_read_bytes),
            "dense_resident_bytes": float(resident),
            "dense_pinned_bytes": float(pinned),
            "dense_drive_bytes": float(disk),
            "gpu_compute_ms": median * 1e3,
            "h2d_ms": link_s * 1e3,
            "storage_ms": drive_s * 1e3,
            **(weights.graph_components() if weights else {}),
        },
        scenario_range=(low, high) if low != high else None,
        reference_scenario="dense_pipeline",
    )
