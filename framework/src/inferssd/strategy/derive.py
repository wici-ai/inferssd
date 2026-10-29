# SPDX-License-Identifier: Apache-2.0
"""Derive one launch's native settings, thread pools and server arguments.

``derive`` is a pure function of the machine characteristics, the model's
header facts, the runtime facts and the platform. It only chooses values that
keep the computation bit-exact: thread counts and CPU placement, placement cost
priors and the host memory budget. Batch sizes, the prefill route and the VRAM
budget's behaviour stay as the launch has them.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from inferssd.configuration.settings import PLATFORMS, Setting, Snapshot, parse, render, uncomment
from inferssd.launch.types import Threads
from inferssd.machine.characteristics import Characteristics, ComputeRow, dumps
from inferssd.strategy.inputs import ModelFacts, RuntimeFacts

# The engine's GB/s cost terms are 1e9 bytes per second: it prices a copy or a
# read at 1e3 / gbps microseconds per 1e6 bytes.
GB = 1e9
COST_MAX = 1e6  # the registry's bound on every place.costs term
MAX_THREADS = 256  # the launcher's bound on a thread pool
MAX_CPU = 511  # the highest CPU a server CPU mask holds
# Role reads the engine keeps in flight: 8 ring experts of 3 roles each.
IN_FLIGHT = 24
# Context checkpoints stay off to work around a known engine issue: once the RAM
# cache fills host.budget_mib, growing the CPU work buffers fails the request
# with HTTP 500.
SERVER_ARGS = ("--ctx-checkpoints", "0")


@dataclass(frozen=True)
class Choice:
    """A proposed setting: its literal as a settings file takes it, and why."""

    value_text: str
    reason: str


@dataclass(frozen=True)
class Inputs:
    characteristics_sha256: str
    model_paths: tuple[str, ...]
    platform: str


@dataclass(frozen=True)
class Strategy:
    settings: Mapping[str, Choice]
    threads: Threads
    server_args: tuple[str, ...]
    inputs: Inputs
    notes: tuple[str, ...]


def span(cpus: Sequence[int]) -> str:
    """CPU IDs as ascending ranges, such as 0-4,10; "none" for no CPU."""
    runs: list[list[int]] = []
    for cpu in sorted(cpus):
        if runs and runs[-1][1] + 1 == cpu:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs) or "none"


def literal(item: Setting, platform: str) -> str:
    """ITEM's value as render() writes it, without key or receipt."""
    body, _ = uncomment(render(Snapshot((item,), platform)).split("\n")[1])
    return body.partition("=")[2].strip(" \t\r\n")


def _validated(choices: Mapping[str, tuple[str, str]], platform: str) -> dict[str, Choice]:
    """Each choice parsed by the configuration layer and written back in its canonical form."""
    for _, reason in choices.values():
        if "\n" in reason or "\r" in reason:
            raise ValueError("a strategy reason must be one line")
    checked = parse(
        None,
        "strategy",
        overrides=[(key, value) for key, (value, _) in choices.items()],
        platform=platform,
    )
    return {
        key: Choice(literal(checked.at(key), platform), why) for key, (_, why) in choices.items()
    }


def _performance(characteristics: Characteristics) -> set[int]:
    """The fastest cluster's CPUs; every CPU on a uniform machine."""
    clusters = characteristics["cpu"]["clusters"]
    fast = [row for row in clusters if row["kind"] == "performance"]
    return set((fast or clusters)[0]["cpus"]) if clusters else set()


def _measured_counts(
    characteristics: Characteristics, model: ModelFacts, allowed: Sequence[int], notes: list[str]
) -> Threads:
    """Without affinity: the measured thread counts that run the dominant format fastest."""
    name = model.dominant_format.upper()
    rows = [
        row
        for row in characteristics["compute"]["cpu"]
        if row["format"].upper() == name and row["threads"] <= len(allowed)
    ]
    one = [
        (point["weight_Bps"], row["threads"])
        for row in rows
        for point in row["points"]
        if point["tokens"] == 1 and point.get("weight_Bps", 0) > 0
    ]
    tokens = max(
        (p["tokens"] for row in rows for p in row["points"] if p["tokens"] > 1 and p.get("flops")),
        default=0,
    )
    many = [
        (point["flops"], row["threads"])
        for row in rows
        for point in row["points"]
        if point["tokens"] == tokens and point.get("flops", 0) > 0
    ]
    fallback = len(_performance(characteristics) & set(allowed)) or len(allowed)
    fallback = min(fallback, MAX_THREADS)
    # The fastest rate wins; between equal rates, fewer threads.
    decode = max(one, key=lambda item: (item[0], -item[1]))[1] if one else fallback
    batch = max(many, key=lambda item: (item[0], -item[1]))[1] if many else fallback
    if one:
        notes.append(
            f"threads: decode {decode}, the measured count with the most {name} weight bytes/s "
            "at one token"
        )
    else:
        notes.append(f"threads: decode {decode}, the performance cluster ({name} not measured)")
    if many:
        notes.append(
            f"threads: batch {batch}, the measured count with the most {name} FLOP/s "
            f"at {tokens} tokens"
        )
    else:
        notes.append(f"threads: batch {batch}, the performance cluster ({name} not measured)")
    return Threads(min(decode, MAX_THREADS), min(batch, MAX_THREADS), ())


def placement(
    characteristics: Characteristics, model: ModelFacts, runtime: RuntimeFacts, notes: list[str]
) -> tuple[Threads, tuple[int, ...], str]:
    """The thread pools, the I/O CPUs and the reason for both."""
    logical = {row["id"] for row in characteristics["cpu"]["logical"]}
    allowed = sorted(set(runtime.cpus) & logical)
    if not allowed:
        raise ValueError("none of the CPUs this process may use is in the machine characteristics")
    if not characteristics["cpu"]["affinity_supported"]:
        threads = _measured_counts(characteristics, model, allowed, notes)
        return threads, (), "no I/O CPU: threads cannot be pinned on this platform"
    performance = _performance(characteristics)
    fast = [cpu for cpu in allowed if cpu in performance]
    slow = [cpu for cpu in allowed if cpu not in performance]
    shape = f"{len(fast)} performance and {len(slow)} other" if fast and slow else "uniform"
    unsplit = "" if fast and slow else "one CPU cluster"
    if fast and slow and len(fast) < 3:
        unsplit = f"{len(fast)} performance CPUs"
    elif fast and slow:
        io = fast[-1]
        compute = [cpu for cpu in allowed if cpu != io]
        lead = fast[:-1]
        # The engine runs decode on the lowest CPU IDs of the strict mask.
        if compute[: len(lead)] != lead:
            unsplit = f"the performance CPUs {span(lead)} are not the lowest compute CPU IDs"
        elif len(compute) > MAX_THREADS or compute[-1] > MAX_CPU:
            unsplit = f"{len(compute)} compute CPUs up to CPU {compute[-1]} exceed a CPU mask"
        else:
            count = len(compute)
            notes.append(
                f"threads: split placement, {count} pinned threads on CPUs {span(compute)}, "
                f"I/O on CPU {io}"
            )
            return (
                Threads(count, count, tuple(compute)),
                (io,),
                f"split placement: I/O on CPU {io}, the last allowed CPU of the performance "
                f"cluster {span(fast)}; compute on {span(compute)}",
            )
    count = min(MAX_THREADS, max(1, len(allowed) - 1))
    notes.append(
        f"threads: all placement, {count} unpinned threads for the {len(allowed)} allowed CPUs "
        f"{span(allowed)} ({shape}); no split: {unsplit}"
    )
    return (
        Threads(count, count, ()),
        (),
        f"all placement: no I/O CPU, {count} unpinned threads for the {len(allowed)} allowed "
        f"CPUs {span(allowed)}; no split: {unsplit}",
    )


def launch_conflicts(
    threads: Threads, io: Sequence[int], participants: int, platform: str
) -> list[str]:
    """What the launcher would refuse in these thread pools and CPU settings."""
    found = []
    counts = [count for count in (threads.decode, threads.batch) if count is not None]
    if any(type(count) is not int or not 1 <= count <= MAX_THREADS for count in counts):
        found.append(f"thread counts must be integers in 1..{MAX_THREADS}")
    cpus = threads.cpus
    if len(set(cpus)) != len(cpus) or any(not 0 <= cpu <= MAX_CPU for cpu in cpus):
        found.append(f"strict CPUs must be unique CPU IDs in 0..{MAX_CPU}")
    if cpus and platform == "darwin":
        found.append("macOS cannot pin threads to CPUs")
    if cpus and any(count > len(cpus) for count in counts):
        found.append("thread counts exceed the strict CPUs")
    if set(io) & set(cpus):
        found.append("cpu.io_cpus overlaps the strict compute CPUs")
    decode = threads.decode or (len(cpus) if cpus else None)
    if decode is not None and participants > decode:
        found.append("cpu.decode_participants exceeds the decode thread count")
    return found


def size(nbytes: float) -> str:
    """A byte count for a receipt: whole MiB where it is one."""
    if nbytes >= 1 << 20 and nbytes % (1 << 20) == 0:
        return f"{int(nbytes) >> 20} MiB"
    return f"{nbytes:.0f} bytes"


# Per role: weight bytes/s at one token, and FLOP/s by token count.
Rates = list[tuple[float, dict[int, float]]]


def _rates(row: ComputeRow) -> tuple[Optional[float], dict[int, float]]:
    """A row's weight bytes/s at one token and its FLOP/s by token count, positive values only."""
    one = next(
        (p["weight_Bps"] for p in row["points"] if p["tokens"] == 1 and p.get("weight_Bps", 0) > 0),
        None,
    )
    many = {p["tokens"]: p.get("flops", 0) for p in row["points"] if p["tokens"] > 1}
    return one, {tokens: flops for tokens, flops in many.items() if flops > 0}


def cpu_line(
    characteristics: Characteristics, model: ModelFacts, threads: int
) -> tuple[Optional[tuple[float, float]], str]:
    """The CPU's cost line of one expert in a grouped prefill pass, and how it was built.

    The engine prices an expert routed to ``r`` tokens of a pass at
    ``cpu_fixed_us + cpu_route_us * r`` microseconds of wall time: its gate,
    up and down products over those r tokens on the pass's THREADS (the batch
    pool); before decode has measured its own experts, the line at one route
    also prices a decode expert. At one token each role streams its weights,
    its slice bytes over the measured weight bytes/s; at n tokens it computes
    2 * rows * cols * n FLOP at the measured FLOP/s, n the largest token count
    measured. The line runs through both points and keeps both terms
    nonnegative as the engine's own fit does. Rates are interpolated linearly
    between the measured thread counts around THREADS, or taken from the
    nearest one outside them.
    """
    names = [role.format.upper() for role in model.roles]
    rows = {
        (row["format"].upper(), row["threads"]): row for row in characteristics["compute"]["cpu"]
    }
    missing = sorted(set(names) - {name for name, _ in rows})
    if missing:
        return None, "no CPU compute measurement of " + ", ".join(missing)
    measured: dict[int, Rates] = {}
    for count in sorted({count for _, count in rows}):
        roles = [_rates(rows[name, count]) for name in names if (name, count) in rows]
        if len(roles) == len(names) and all(one is not None for one, _ in roles):
            measured[count] = [(one or 0.0, many) for one, many in roles]

    def shared(counts: Sequence[int]) -> set[int]:
        """Token counts above one measured for every role at every count of COUNTS."""
        return set.intersection(*(set(many) for count in counts for _, many in measured[count]))

    measured = {count: rates for count, rates in measured.items() if shared([count])}
    if not measured:
        return None, "no thread count measured every expert format at one and more tokens"
    lower = max((count for count in measured if count <= threads), default=None)
    upper = min((count for count in measured if count >= threads), default=None)
    pair = sorted({count for count in (lower, upper) if count is not None})
    if not shared(pair):
        pair = [min(pair, key=lambda count: (abs(count - threads), count))]
    tokens = max(shared(pair))
    share = (threads - pair[0]) / (pair[1] - pair[0]) if len(pair) == 2 else 0.0

    def rate(role: int, at: int) -> float:
        values = [
            measured[count][role][0] if at == 1 else measured[count][role][1][at] for count in pair
        ]
        return values[0] + share * (values[-1] - values[0])

    one_us = sum(role.slice_bytes() / rate(i, 1) for i, role in enumerate(model.roles)) * 1e6
    many_us = 1e6 * sum(
        2.0 * role.rows * role.cols * tokens / rate(i, tokens) for i, role in enumerate(model.roles)
    )
    route = (many_us - one_us) / (tokens - 1)
    fixed = one_us - route
    if fixed < 0:
        fixed, route = 0.0, (one_us + tokens * many_us) / (1 + tokens * tokens)
    elif route < 0:
        fixed, route = (one_us + many_us) / 2, 0.0
    source = (
        f"the {pair[0]}-thread measurement"
        if len(pair) == 1
        else f"the {pair[0]}- and {pair[1]}-thread measurements"
    )
    why = (
        f"one {'/'.join(names)} expert, latent {model.up.cols} hidden {model.up.rows}, "
        f"on {threads} threads ({source}): {one_us:.1f} us at 1 token, "
        f"{many_us:.1f} us at {tokens} tokens"
    )
    if max(fixed, route) > COST_MAX:
        return None, why + ", beyond the cost term range"
    return (fixed, route), why


def drive_rate(
    characteristics: Characteristics, model: ModelFacts, key: Optional[str]
) -> tuple[Optional[float], str]:
    """The model drive's read rate in GB/s for the requests the engine issues.

    The engine reads each role of an expert as its own request, so the size
    looked up is the largest expert's mean role read; the nearest measured size
    is taken on a log scale. The engine keeps up to IN_FLIGHT of those reads
    queued, so the rate is the best median between the size's knee and that
    depth (or the reader's own cap).
    """
    if key is None:
        return None, "the drive holding the model is not known"
    entry = next((row for row in characteristics["storage"] if row["drive_key"] == key), None)
    if entry is None:
        return None, f"no measurement of the drive {key}; run inferssd profile --model <model>"
    name = key if entry["model"] == "unknown" else entry["model"]
    request = model.expert_read_bytes / 3
    knee = min(entry["knee"], key=lambda row: (abs(math.log(row["bytes"] / request)), row["bytes"]))
    deepest = min(IN_FLIGHT, entry["qd_cap"] or IN_FLIGHT)
    points = [
        row
        for row in entry["random_read"]
        if row["bytes"] == knee["bytes"] and knee["qd"] <= row["qd"] <= deepest and row["Bps"] > 0
    ]
    if not points:
        return None, f"no read rate of {size(knee['bytes'])} reads on {name} up to QD {deepest}"
    # The fastest median wins; between equal rates, the shallower queue.
    best = max(points, key=lambda row: (row["Bps"], -row["qd"]))
    rate = best["Bps"] / GB
    if rate > COST_MAX:
        return None, f"a read rate of {rate} GB/s is beyond the cost term range"
    return rate, (
        f"the drive {name} reads {rate:.2f} GB/s with {size(knee['bytes'])} reads at QD "
        f"{best['qd']}, the best from its knee QD {knee['qd']} up to QD {deepest} "
        f"(a role read is {size(request)} on average)"
    )


def link_rates(
    characteristics: Characteristics, runtime: RuntimeFacts
) -> tuple[dict[str, float], str]:
    """Pinned and pageable host-to-device rates in GB/s at the largest measured copy."""
    if runtime.gpus is None:
        return {}, "the CUDA GPUs could not be read now"
    present = runtime.gpus
    if len(present) != 1:
        if not present:
            return {}, "no CUDA GPU is usable now"
        return {}, f"{len(present)} CUDA GPUs are usable and the engine drives one of them"
    gpu = next((row for row in characteristics["gpus"] if row["uuid"] == present[0].uuid), None)
    if gpu is None:
        return {}, f"the GPU {present[0].uuid} was not measured"
    if gpu["integrated"]:
        return {}, f"{gpu['name']} shares host memory: no copy over a link"
    terms, parts = {}, []
    copies = (
        ("link_gbps", "pinned", gpu["h2d"]["pinned"]),
        ("pageable_gbps", "pageable", gpu["h2d"]["pageable"]),
    )
    for term, kind, rows in copies:
        if not rows:
            continue
        largest = max(rows, key=lambda row: row["bytes"])
        rate = largest["Bps"] / GB
        if 0 < rate <= COST_MAX:
            terms[term] = rate
            parts.append(f"{kind} {rate:.2f} GB/s at {size(largest['bytes'])}")
    return terms, f"{gpu['name']} host-to-device " + (", ".join(parts) or "not measured")


def _costs(
    characteristics: Characteristics,
    model: ModelFacts,
    runtime: RuntimeFacts,
    threads: int,
    key: Optional[str],
    notes: list[str],
) -> Optional[tuple[str, str]]:
    terms: dict[str, float] = {}
    reasons: list[str] = []
    line, why = cpu_line(characteristics, model, threads)
    if line is not None:
        terms["cpu_fixed_us"], terms["cpu_route_us"] = line
        reasons.append(f"cpu {line[0]:.1f} + {line[1]:.2f} us a route for {why}")
    notes.append(
        "place.costs: cpu "
        + (f"{line[0]:.1f} + {line[1]:.2f} us a route for " if line else "omitted, ")
        + why
    )
    link, why = link_rates(characteristics, runtime)
    terms.update(link)
    if link:
        reasons.append(why)
    notes.append("place.costs: link " + ("" if link else "omitted, ") + why)
    rate, why = drive_rate(characteristics, model, key)
    if rate is not None:
        terms["drive_gbps"] = rate
        reasons.append(why)
    notes.append("place.costs: drive " + ("" if rate is not None else "omitted, ") + why)
    if not terms:
        return None
    value = ",".join(f"{term}={number:.17g}" for term, number in sorted(terms.items()))
    return value, "; ".join(reasons)


def derive(
    characteristics: Characteristics,
    model: ModelFacts,
    runtime: RuntimeFacts,
    platform: str,
    *,
    model_drive_key: Optional[str],
) -> Strategy:
    """The strategy for one launch of MODEL on this machine; see the module docstring."""
    if platform not in PLATFORMS:
        raise ValueError("unknown platform " + platform)
    text = dumps(characteristics)
    if characteristics["identity"]["hard"]["platform"] != platform:
        raise ValueError("the machine characteristics describe another platform")
    notes: list[str] = []
    choices: dict[str, tuple[str, str]] = {}
    threads, io, why = placement(characteristics, model, runtime, notes)
    choices["cpu.io_cpus"] = (span(io) if io else "", why)
    choices["cpu.decode_participants"] = ("0", "the engine measures its decode participants")
    conflicts = launch_conflicts(threads, io, 0, platform)
    if conflicts or threads.batch is None:
        raise AssertionError("derived thread pools the launcher refuses: " + "; ".join(conflicts))
    costs = _costs(characteristics, model, runtime, threads.batch, model_drive_key, notes)
    if costs is not None:
        choices["place.costs"] = costs
    if platform == "linux":
        listed = {
            gpu["uuid"]
            for gpu in characteristics["identity"]["hard"]["gpus"]
            if gpu["backend"] == "CUDA"
        }
        if runtime.gpus is None:
            notes.append("vram.budget: default; the CUDA GPUs could not be read now")
        elif not runtime.gpus:
            extra = f", although the machine has {len(listed)}" if listed else ""
            choices["vram.budget"] = ("0", "no CUDA GPU is usable now" + extra)
            notes.append("vram.budget: 0, no CUDA GPU is usable now" + extra)
        else:
            known = [gpu for gpu in runtime.gpus if gpu.uuid in listed]
            shared = {
                row["uuid"]: row["name"] for row in characteristics["gpus"] if row["integrated"]
            }
            if known and all(gpu.uuid in shared for gpu in known):
                # On a GPU that shares host memory the automatic VRAM tier grows into
                # the host's memory and starves prompt processing, so it stays off.
                names = ", ".join(sorted({shared[gpu.uuid] for gpu in known}))
                reason = f"{names} shares host memory with the CPU"
                choices["vram.budget"] = ("0", reason)
                notes.append("vram.budget: 0, " + reason)
            elif known:
                notes.append("vram.budget: default auto for " + ", ".join(g.uuid for g in known))
            else:
                notes.append(
                    "vram.budget: default auto; the usable GPU is not in the characteristics, "
                    "run inferssd profile again"
                )
            for gpu in runtime.gpus:
                if gpu.processes:
                    notes.append(
                        f"vram.budget: {gpu.uuid} runs {gpu.processes} compute process(es); "
                        "the launch refuses a GPU in use"
                    )
    if platform in ("darwin", "windows") and runtime.memory_budget_bytes:
        mib = runtime.memory_budget_bytes >> 20
        if mib:
            choices["host.budget_mib"] = (
                str(mib),
                f"the memory budget of {runtime.memory_budget_bytes} bytes",
            )
            notes.append(f"host.budget_mib: {mib} from the memory budget")
    notes.append("server: --ctx-checkpoints 0")
    return Strategy(
        settings=_validated(choices, platform),
        threads=threads,
        server_args=SERVER_ARGS,
        inputs=Inputs(hashlib.sha256(text.encode()).hexdigest(), model.paths, platform),
        notes=tuple(notes),
    )
