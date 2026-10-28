"""The original placement formatting with an explicitly supplied lazy host fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, cast

from inferssd.plan.settings_types import Device, HostThreads


def default_threads(dev: Device, host_threads: HostThreads) -> Optional[int]:
    if dev.get("default_threads"):
        return int(cast(int, dev["default_threads"]))
    return host_threads()


def groups_of(dev: Device) -> list[list[int]]:
    groups, i = [], 1
    while f"g{i}" in dev["coresets"]:
        groups.append(dev["coresets"][f"g{i}"]["cpus"])
        i += 1
    return groups


def pin_env(dev: Device) -> str:
    groups = groups_of(dev)
    if len(groups) < 2 or len(groups[0]) < 3:
        return ""
    cpus, io = groups[0][:-1], groups[0][-1]
    return (
        f"GGML_KIMI_CPU_SINK_DECODE_THREAD_CHOICE=0 THREADS={len(cpus) + 1} "
        f"GGML_KIMI_CPU_SINK_THREADS={len(cpus) + 1} "
        f"GGML_KIMI_CPU_SINK_COMPUTE_CPUS={','.join(map(str, cpus))} "
        f"GGML_KIMI_CPU_SINK_IO_CPUS={io}"
    )


def split_env(dev: Device, *, host_threads: HostThreads) -> str:
    groups = groups_of(dev)
    if len(groups) < 2 or len(groups[0]) < 3:
        return ""
    t = cast(int, default_threads(dev, host_threads))
    order = [c for g in groups for c in g][:t]
    io = groups[0][-1]
    compute = [c for c in order if c != io]
    if len(compute) != t - 1:
        return ""
    return (
        f"THREADS={t} GGML_KIMI_CPU_SINK_THREADS={t} "
        f"GGML_KIMI_CPU_SINK_COMPUTE_CPUS={','.join(map(str, compute))} "
        f"GGML_KIMI_CPU_SINK_IO_CPUS={io} "
        f"GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS={len(groups[0]) - 1}"
    )


def env(dev: Device, name: str, *, host_threads: HostThreads) -> str:
    """name: a CPU placement, optionally '@N' for the engine's prefill ubatch (UBATCH=N: a
    longer prompt takes fewer passes, each reading the experts its tokens touch, at the cost
    of the prefill buffers' GPU memory)."""
    base, _, ub = name.partition("@")
    e = cast(
        Callable[[Device], str],
        {"pin": pin_env, "split": lambda d: split_env(d, host_threads=host_threads)}.get(
            base, lambda d: ""
        ),
    )(dev)
    return " ".join(x for x in (e, f"UBATCH={ub}" if ub else "") if x)
