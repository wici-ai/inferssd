"""The original pure grouped-prefill lane costs, preserving arithmetic and selection order.

Inputs retain the legacy numeric semantics, including negative values and absent,
null, or zero lane rates. This leaf neither validates artifacts nor performs I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional, cast

RING = 8


def coupled(a: float, b: float, n: float) -> float:
    """kimi_prefill_coupled (kimi-prefill-cost.h): a producer busy for a and a consumer busy for b
    handing the work over through a ring of n buffers."""
    high, low = max(a, b), max(0.0, min(a, b))
    if high <= 0:
        return 0.0
    n = max(1.0, n)
    r = low / high
    if r > 1.0 - 1e-9:
        return high * (n + 1.0) / n
    return cast(float, high * (1.0 - r ** (n + 1.0)) / (1.0 - r**n))


def lane_layer(
    lane: Mapping[str, Optional[float]],
    n: float,
    U: float,
    routes: float,
    cov: float,
    b_e: float,
    in_place: bool = False,
    route_bytes: float = 0.0,
    ring: float = RING,
    ring_locked: bool = True,
    k_cpu: float = 1.0,
    k_drive: float = 1.0,
) -> tuple[float, float, int]:
    """One grouped-prefill layer of n tokens with the GPU lane, as the engine places it
    (kimi_place_layer's streamed form, kimi-prefill-cost.h): U experts touched, `routes` routes
    each (their mean), a share cov of them resident in the tier and the rest read from the drive
    (B_e bytes each) into the ring (`ring` reads; page-locked or not); lane: the table a
    bootstrap pass calibrated (the engine's names: the CPU's line, the batched engine's device
    and in-place lines, the page-locked and pageable copies, the drive, the GPU's time a token
    outside its chains); in_place: the GPU can read host memory in place (the engine's receipt),
    and the layer then reads its moves there when the in-place line is no longer than the device
    line plus a copy. The moves: the read experts, most routes first (all alike here), each off
    the CPU and onto the GPU (its chain after its copy into one of two device slots on the link,
    page-locked DMA or pageable staging, or in place); the GPU lane the pipeline of the
    residents' chains and the moves' copies and chains; the engine keeps the fewest moves within
    one read of the lowest time. Returns (us, the CPU's us, moves)."""
    t: Callable[[str], float] = lambda k: lane.get(k) or 0.0  # noqa: E731
    N = max(0, int(round(U * (1.0 - cov))))
    R = max(0, int(round(U * cov)))
    c = k_cpu * (t("cpu_fixed_us") + t("cpu_route_us") * routes)
    g_dev = t("gpu_fixed_us") + t("gpu_route_us") * routes
    measured_host = t("gpu_host_fixed_us") > 0 or t("gpu_host_route_us") > 0
    g_host = t("gpu_host_fixed_us") + t("gpu_host_route_us") * routes if measured_host else g_dev
    locked_us = 1e3 / t("link_gbps") if t("link_gbps") > 0 else 0.0
    pageable_us = 1e3 / t("pageable_gbps") if t("pageable_gbps") > 0 else 0.0
    mb = b_e / 1e6
    # A read copies from the locked ring at its measured rate (none: no price),
    # or through pageable staging.
    copy_us = locked_us if ring_locked else pageable_us
    chosen = in_place and (
        N == 0
        or g_host <= g_dev + mb * (locked_us if ring_locked and locked_us > 0 else pageable_us)
    )
    priced = chosen or copy_us > 0
    g_mov = g_host if chosen else g_dev
    d = 0.0 if chosen else mb * copy_us
    rows = 0.0 if in_place else route_bytes * routes / 1e6 * (locked_us or pageable_us)
    drive_us_per_mb = k_drive * 1e3 / t("drive_gbps") if t("drive_gbps") > 0 else 0.0
    drive = (
        k_drive * t("drive_fixed_us") + N * b_e / 1e6 * drive_us_per_mb
        if drive_us_per_mb and N
        else 0.0
    )
    fill = b_e / 1e6 * drive_us_per_mb if drive else 0.0
    overhead = n * t("gpu_token_us")
    times, cpus = [], []
    copied, gpu = 0.0, [R * g_dev]
    for m in range((N if priced else 0) + 1):
        if m:
            # Copy m waits for the previous copy and the chain that last used its
            # slot (two device slots).
            i = m - 1
            if d > 0:
                copied = max(copied, gpu[i - 1] if i >= 2 else 0.0) + d
                gpu.append(max(gpu[i], copied) + g_mov)
            else:
                gpu.append(gpu[i] + g_mov)
        cpu = (N - m) * c
        link = (R + m) * rows + m * d
        work = m > 0 or R > 0
        times.append(
            fill + coupled(drive, max(cpu, gpu[m], link), ring) + (overhead if work else 0.0)
        )
        cpus.append(cpu)
    lowest = min(times)
    m = next(i for i, x in enumerate(times) if x <= lowest + fill)
    return times[m], cpus[m], m
