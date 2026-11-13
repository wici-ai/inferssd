# SPDX-License-Identifier: Apache-2.0
"""Pre-inference decode cost model, with explicit placement and routing assumptions.

Synthetic matrix measurements price arithmetic; cache capacity is NOT a measured
hit rate. The central estimate assumes uniform routing and inclusive host/VRAM
caches. Scenario rates are conditional sensitivity estimates, not confidence
bounds. Attention, graph overhead and cache warm-up are not measured here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from inferssd.machine.characteristics import GPU, Characteristics, ComputeRow
from inferssd.strategy.derive import (
    IN_FLIGHT,
    drive_rate,
    engine_gpu_backends,
    host_headroom_mib,
    placement,
    single_gpu,
)
from inferssd.strategy.inputs import BLOCKS, ModelFacts, RuntimeFacts

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
    reference_scenario: str = "uniform_serial"
    reference_model: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class RoutingPattern:
    name: str
    expert_fraction: float  # hot experts' fraction of the store
    traffic_fraction: float  # their share of routed requests


# Illustrative hypotheses, NOT a learned distribution or a "typical" confidence band.
ROUTING_PATTERNS = (
    RoutingPattern("uniform", 1.0, 1.0),
    RoutingPattern("hot50_traffic75", 0.5, 0.75),
    RoutingPattern("hot20_traffic80", 0.2, 0.80),
    RoutingPattern("hot20_traffic95", 0.2, 0.95),
)


def routing_hit(capacity: float, hot_fraction: float, traffic_fraction: float) -> float:
    """Two uniform zones with hotter experts admitted first; a nonzero tail remains."""
    if not (0 < hot_fraction <= traffic_fraction <= 1):
        raise ValueError("routing fractions require 0 < hot <= traffic <= 1")
    capacity = max(0.0, min(1.0, capacity))
    if capacity <= hot_fraction:
        return traffic_fraction * capacity / hot_fraction
    return traffic_fraction + (1 - traffic_fraction) * (capacity - hot_fraction) / (
        1 - hot_fraction
    )


def drive_iqr(
    c: Characteristics, model: ModelFacts, key: Optional[str], central_Bps: float
) -> tuple[float, float, int]:
    """Repeated measurements of the same request size/QD used by drive_rate.

    This describes the profiling window only; it cannot predict a later shared-
    storage stall. Do not use measurements from different queues as repetitions.
    """
    entry = next(row for row in c["storage"] if row["drive_key"] == key)
    request = model.mean_expert_read_bytes / 3
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


def matvec_rate(
    rows: list[ComputeRow], name: str, outputs: int, inputs: int, timing: str = "median"
) -> Optional[float]:
    """Use the closest measured shape, not the fastest measurement across shapes."""
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
    point = next(p for p in row["points"] if p["tokens"] == 1 and p.get("weight_Bps", 0) > 0)
    duration = (
        point.get("us_q1", point["us"])
        if timing == "q1"
        else point.get("us_q3", point["us"])
        if timing == "q3"
        else point["us"]
    )
    # Older profiles lack quartiles; their median remains usable.
    return point["weight_Bps"] * point["us"] / duration if duration > 0 else point["weight_Bps"]


def bits_per_weight(name: str) -> float:
    block, size = BLOCKS[name.upper()]
    return 8 * size / block


def format_proxies(rows: list[ComputeRow], names: set[str]) -> dict[str, str]:
    """Each of NAMES priced as itself when ROWS measure it at one token, else as the measured
    format nearest it in bits per weight (the higher of two as near): a default for a format the
    probe does not measure, such as IQ4_XS and IQ4_NL, until a run calibrates it. A name with no
    block size, or rows that measure nothing, has no entry."""
    measured = {
        row["format"].upper()
        for row in rows
        if row["format"].upper() in BLOCKS
        and any(p["tokens"] == 1 and p.get("weight_Bps", 0) > 0 for p in row["points"])
    }
    proxies = {}
    for name in names:
        if name.upper() in measured:
            proxies[name] = name.upper()
        elif name.upper() in BLOCKS and measured:
            want = bits_per_weight(name)
            proxies[name] = min(
                measured, key=lambda m: (abs(bits_per_weight(m) - want), -bits_per_weight(m), m)
            )
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
    c: Characteristics, model: ModelFacts, threads: int, timing: str = "median"
) -> tuple[Optional[float], str]:
    """One expert, using only one-token points and all of its quantization types."""
    rows = [row for row in c["compute"]["cpu"] if not row.get("cpus")]
    names = {name.upper() for role in model.roles for name, _ in role.formats}
    proxies = format_proxies(rows, names)
    missing = sorted(names - set(proxies))
    if missing:
        return None, "no CPU compute measurement of " + ", ".join(missing)
    rates = {}
    for count in sorted({row["threads"] for row in rows}):
        selected = [row for row in rows if row["threads"] == count]
        values = [
            matvec_rate(selected, proxies[name.upper()], role.rows, role.cols, timing)
            for role in model.roles
            for name, _ in role.formats
        ]
        if all(rate is not None for rate in values):
            rates[count] = [rate or 0 for rate in values]
    if not rates:
        return None, "no thread count measured every expert format at one token"
    lower = max((n for n in rates if n <= threads), default=min(rates))
    upper = min((n for n in rates if n >= threads), default=max(rates))
    share = (threads - lower) / (upper - lower) if upper != lower else 0
    total, index = 0.0, 0
    for role in model.roles:
        for name, layers in role.formats:
            rate = rates[lower][index] + share * (rates[upper][index] - rates[lower][index])
            total += layers / model.layers * role.slice_bytes(name) / rate
            index += 1
    return total, ""


def cuda_gpu(
    c: Characteristics, runtime: RuntimeFacts, gpu_work_bytes: int
) -> tuple[Optional[GPU], int, list[ComputeRow]]:
    """The one discrete CUDA GPU the engine puts dense weights on, the bytes free on it less
    GPU_WORK_BYTES, and its one-token matrix measurements; (None, 0, []) without one."""
    gpu = single_gpu(c, runtime)
    if (
        not gpu
        or "CUDA" not in engine_gpu_backends(runtime.gpu_backends, c)
        or c["memory"]["unified"]
        or gpu["integrated"]
    ):
        return None, 0, []
    free = runtime.gpus[0].free_bytes if runtime.gpus else None
    room = max(0, min(free or 0, gpu["vram_total_bytes"]) - gpu_work_bytes)
    return gpu, room, gpu.get("decode_compute", [])


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
    routing_patterns: tuple[RoutingPattern, ...] = ROUTING_PATTERNS,
) -> Estimate:
    """Estimate default placement, optionally reserving known KV/graph working memory.

    Work reserves are additional to the engine's host margin. They must come from
    the intended context/graph allocation, not fitting a measured tokens/s value.
    Zero means unknown and makes cache capacities upper estimates.
    """
    if not routing_patterns:
        raise ValueError("at least one routing pattern is required")
    for pattern in routing_patterns:
        routing_hit(0, pattern.expert_fraction, pattern.traffic_fraction)
    if min(host_work_bytes, gpu_work_bytes, dense_read_bytes) < 0:
        raise ValueError("memory sizes must be nonnegative")
    if not model.experts_used or not model.routed_bytes:
        return Estimate(None, "the GGUF header names no routed experts per token")
    memory_Bps = max((r["Bps"] for r in characteristics["memory"]["dram_read_Bps"]), default=0)
    if memory_Bps <= 0:
        return Estimate(None, "no memory read rate in the profile")
    try:
        threads, _, _ = placement(characteristics, model, runtime, [])
    except ValueError as error:
        return Estimate(None, str(error))
    # Decode uses the decode pool; the previous implementation used the prefill pool.
    cpu_expert_s, why = cpu_decode_seconds(characteristics, model, threads.decode or 1)
    if cpu_expert_s is None:
        return Estimate(None, why)
    drive_gbps, why = drive_rate(characteristics, model, drive_key)
    if drive_gbps is None:
        return Estimate(None, why)
    available = runtime.available_bytes
    if available is None:
        available = characteristics["memory"]["total_bytes"]
    if runtime.memory_budget_bytes is not None:
        available = min(available, runtime.memory_budget_bytes)
    available = min(available, characteristics["memory"]["total_bytes"])
    room = max(0, available - headroom_bytes(characteristics, runtime, platform) - host_work_bytes)
    gpu, gpu_room, rows = cuda_gpu(characteristics, runtime, gpu_work_bytes)
    assumptions = [
        "Uniform expert routing; inclusive RAM/VRAM caches, not observed hit rates.",
        "Serial I/O plus CPU/GPU service time; prefetch overlap is a separate scenario.",
        "Hot-set scenarios are hypotheses, not empirical probabilities or typical ranges.",
        "Hot experts are cached first, after warm-up; routing order and replacement are unknown.",
        "CPU/GPU timing and drive IQR describe profiling samples, not a confidence interval.",
        "Drive IQR covers profiling repetitions only, not future shared-storage stalls.",
        "Attention, non-matmul graph work, lookup I/O and cache warm-up are unpriced.",
        "KV/graph working memory is unreserved unless work-byte arguments are supplied.",
    ]
    if runtime.gpus and any(device.processes for device in runtime.gpus):
        assumptions.append("GPU contention is present and is not priced by this estimate.")
    # Dense quantization is not retained by ModelFacts. Q8_0 is an explicit proxy;
    # never substitute peak CUDA bandwidth for measured quantized computation.
    dense_rate = matvec_rate(rows, "Q8_0", 4096, 4096)
    gpu_expert_s = 0.0
    gpu_fast_s = 0.0
    gpu_slow_s = 0.0
    missing = []
    names = {name.upper() for role in model.roles for name, _ in role.formats}
    cpu_proxies = format_proxies(
        [row for row in characteristics["compute"]["cpu"] if not row.get("cpus")], names
    )
    gpu_proxies = format_proxies(rows, names)
    for role in model.roles:
        for name, layers in role.formats:
            used = gpu_proxies.get(name.upper(), name)
            rate = matvec_rate(rows, used, role.rows, role.cols)
            if rate is None:
                missing.append(name)
            else:
                weight = layers * role.slice_bytes(name) / model.layers
                gpu_expert_s += weight / rate
                gpu_fast_s += weight / (matvec_rate(rows, used, role.rows, role.cols, "q1") or rate)
                gpu_slow_s += weight / (matvec_rate(rows, used, role.rows, role.cols, "q3") or rate)
    use_gpu = bool(gpu and gpu_room >= dense_read_bytes and dense_rate and not missing)
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
        gpu_cache = max(0, gpu_room - dense_read_bytes)
        dense_s = dense_read_bytes / (dense_rate or 1)
        assumptions.append(
            "Dense weights on one CUDA GPU; dense Q8_0 matvec proxy (actual types unknown)."
        )
    else:
        resident = min(dense_read_bytes, room)
        cache = room - resident
        gpu_cache = 0
        dense_s = resident / memory_Bps + (dense_read_bytes - resident) / (drive_gbps * GB)
    experts = model.routed_bytes
    ram_fraction = min(1.0, cache / experts)
    gpu_fraction = min(1.0, gpu_cache / experts)
    # A VRAM copy may also occupy RAM: summing the two capacities double-counts it.
    hit = max(ram_fraction, gpu_fraction)
    routes = model.layers * model.experts_used
    cpu_all_s = routes * cpu_expert_s
    gpu_all_s = routes * gpu_expert_s
    io_all_s = routes * model.mean_expert_read_bytes / (drive_gbps * GB)

    def terms(gpu_hit: float, storage_miss: float) -> tuple[float, float, float]:
        return cpu_all_s * (1 - gpu_hit), gpu_all_s * gpu_hit, io_all_s * storage_miss

    cpu_s, gpu_s, drive_s = terms(gpu_fraction, 1 - hit)
    seconds = dense_s + cpu_s + gpu_s + drive_s
    serial = 1 / seconds
    scenarios = {
        "uniform_serial": serial,
        "uniform_perfect_overlap": 1 / (dense_s + max(cpu_s, gpu_s, drive_s)),
    }
    disjoint_miss = 1 - min(1.0, ram_fraction + gpu_fraction)
    scenarios["uniform_disjoint_serial"] = 1 / (dense_s + cpu_s + gpu_s + io_all_s * disjoint_miss)
    io_low, io_high, io_samples = drive_iqr(characteristics, model, drive_key, drive_gbps * GB)
    cpu_fast, _ = cpu_decode_seconds(characteristics, model, threads.decode or 1, "q1")
    cpu_slow, _ = cpu_decode_seconds(characteristics, model, threads.decode or 1, "q3")
    if use_gpu:
        dense_fast = dense_read_bytes / (
            matvec_rate(rows, "Q8_0", 4096, 4096, "q1") or dense_rate or 1
        )
        dense_slow = dense_read_bytes / (
            matvec_rate(rows, "Q8_0", 4096, 4096, "q3") or dense_rate or 1
        )
    else:
        dense_fast = resident / memory_Bps + (dense_read_bytes - resident) / io_high
        dense_slow = resident / memory_Bps + (dense_read_bytes - resident) / io_low
    details: list[dict[str, object]] = []
    lows, highs = [], []
    for pattern in routing_patterns:
        gh = routing_hit(gpu_fraction, pattern.expert_fraction, pattern.traffic_fraction)
        miss = 1 - routing_hit(hit, pattern.expert_fraction, pattern.traffic_fraction)
        ct, gt, it = terms(gh, miss)
        no_overlap = 1 / (dense_s + ct + gt + it)
        # Only half of the smaller I/O/compute service time is hidden in this
        # intermediate scenario; the optimistic endpoint allows full overlap.
        partial = 1 / (dense_s + ct + gt + it - 0.5 * min(it, ct + gt))
        slow_io = it * drive_gbps * GB / io_low
        fast_io = it * drive_gbps * GB / io_high
        slow_cpu = routes * (cpu_slow or cpu_expert_s) * (1 - gh)
        fast_cpu = routes * (cpu_fast or cpu_expert_s) * (1 - gh)
        slow_gpu, fast_gpu = routes * gpu_slow_s * gh, routes * gpu_fast_s * gh
        low = min(no_overlap, 1 / (dense_slow + slow_cpu + slow_gpu + slow_io))
        high = max(no_overlap, 1 / (dense_fast + max(fast_cpu, fast_gpu, fast_io)))
        scenarios[pattern.name + "_serial"] = no_overlap
        details.append(
            {
                "name": pattern.name,
                "hot_expert_fraction": pattern.expert_fraction,
                "hot_traffic_fraction": pattern.traffic_fraction,
                "gpu_hit_fraction": gh,
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
    spill = 0 if use_gpu else dense_read_bytes - resident
    detail = (
        f"{seconds * 1e3:.0f} ms a token: dense {dense_s * 1e3:.0f} ms"
        + (f" ({spill / GB:.1f} GB of it from the drive)" if spill else "")
        + f", CPU experts {cpu_s * 1e3:.0f} ms, GPU experts {gpu_s * 1e3:.0f} ms, "
        f"{1 - hit:.0%} from the drive at {drive_gbps:.2f} GB/s {drive_s * 1e3:.0f} ms "
        f"(a {cache / GB:.1f} GB cache of {experts / GB:.0f} GB of experts); "
        "uniform-routing estimate, not a measured hit rate"
    )
    return Estimate(
        serial,
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
            "uniform_gpu_hit_fraction": gpu_fraction,
            "uniform_storage_miss_fraction": 1 - hit,
            "decode_threads": float(threads.decode or 1),
            "memory_budget_bytes": float(available),
            "drive_Bps": drive_gbps * GB,
            "drive_iqr_low_Bps": io_low,
            "drive_iqr_high_Bps": io_high,
            "drive_repetitions": float(io_samples),
        },
        tuple(details),
        (min(lows), max(highs)),
    )


def dense_decode_estimate(
    characteristics: Characteristics,
    dense_read_bytes: int,
    runtime: RuntimeFacts,
    *,
    gpu_work_bytes: int = 0,
) -> Estimate:
    """A model without routed experts, which runs from memory: a token reads each dense weight once,
    on one CUDA GPU when they fit the memory free on it, else from RAM at the fastest measured read
    rate. A default, as an expert model's dense weights are priced: the Q8_0 matvec measurement
    stands for the weights' own types, and nothing calibrates it against a run yet."""
    if dense_read_bytes <= 0:
        return Estimate(None, "the model's dense weight bytes are not known")
    gpu, gpu_room, rows = cuda_gpu(characteristics, runtime, gpu_work_bytes)
    rate = matvec_rate(rows, "Q8_0", 4096, 4096) if gpu and gpu_room >= dense_read_bytes else None
    assumptions = [
        "Dense model: a token reads each weight once; one-row lookup tables do not count.",
        "Attention, KV reads, non-matmul graph work and kernel launches are unpriced.",
        "A default estimate, not calibrated against a measured run.",
    ]
    if rate:
        seconds = dense_read_bytes / rate
        fast = dense_read_bytes / (matvec_rate(rows, "Q8_0", 4096, 4096, "q1") or rate)
        slow = dense_read_bytes / (matvec_rate(rows, "Q8_0", 4096, 4096, "q3") or rate)
        where = f"one CUDA GPU at {rate / GB:.0f} GB/s"
        assumptions.append(
            "Weights on one CUDA GPU; dense Q8_0 matvec proxy (actual types unknown)."
        )
    else:
        memory_Bps = max((r["Bps"] for r in characteristics["memory"]["dram_read_Bps"]), default=0)
        if memory_Bps <= 0:
            return Estimate(None, "no memory read rate in the profile")
        seconds = fast = slow = dense_read_bytes / memory_Bps
        where = f"RAM at {memory_Bps / GB:.0f} GB/s"
        assumptions.append("Weights read from RAM at the fastest measured read rate.")
    detail = (
        f"{seconds * 1e3:.1f} ms a token: {dense_read_bytes / GB:.2f} GB of dense weights from "
        f"{where}; default estimate, not calibrated"
    )
    return Estimate(
        1 / seconds,
        detail,
        tuple(assumptions),
        {"dense_serial": 1 / seconds},
        {"dense_ms": seconds * 1e3, "dense_read_bytes": float(dense_read_bytes)},
        (),
        (1 / slow, 1 / fast) if slow != fast else None,  # the GPU measurement's quartiles
    )
