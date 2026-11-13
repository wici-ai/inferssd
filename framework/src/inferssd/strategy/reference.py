# SPDX-License-Identifier: Apache-2.0
"""Offline decode priors. Loading or evaluating one never runs inference.

The bundled costs are hardware-specific. Routing and residual parameters stay
frozen; only cache-capacity interpolation and the measured drive rate are inputs.
An unmatched model, device or memory budget retains the analytical estimate.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files
from typing import Optional, TypedDict, cast

from inferssd.machine.characteristics import Characteristics
from inferssd.strategy.derive import drive_rate, engine_gpu_backends, single_gpu
from inferssd.strategy.estimate import Estimate
from inferssd.strategy.inputs import ModelFacts, RuntimeFacts, model_facts_from


class Hardware(TypedDict):
    platform: str
    arch: str
    cpu_brand: str
    allowed_cpus: int
    gpu_name: str
    minimum_gpu_free_bytes: int


class Route(TypedDict):
    gpu_hit: float
    read_bytes: list[float]


class Costs(TypedDict):
    dense_ms: float
    cpu_all_ms: float
    gpu_all_ms: float


class Parameters(TypedDict):
    compute: Costs
    residual_ms: list[float]
    wait_ms_per_mib: float
    drive_Bps: float
    routing_samples: list[Route]


@dataclass(frozen=True)
class Reference:
    identifier: str
    label: str
    quantization: str
    facts: ModelFacts
    dense_read_bytes: int
    hardware: Hardware
    memory_gib: tuple[int, int]
    workload: dict[str, object]
    provenance: dict[str, object]
    parameters: Parameters


@lru_cache(maxsize=1)
def references() -> tuple[Reference, ...]:
    """Read installed package data, never a user's run history or a remote registry."""
    data = json.loads(files("inferssd.strategy").joinpath("decode-references.json").read_text())
    if data["schema"] != 1:
        raise ValueError("unsupported decode reference schema")
    return tuple(
        Reference(
            row["id"],
            row["label"],
            row["quantization"],
            model_facts_from(row["facts"]),
            row["dense_read_bytes"],
            row["hardware"],
            tuple(row["memory_gib"]),
            row["workload"],
            row["provenance"],
            row["parameters"],
        )
        for row in data["models"]
    )


def match(model: ModelFacts, dense_read_bytes: int) -> Optional[Reference]:
    """Match full expert geometry/type counts and dense bytes, never a filename."""
    return next(
        (
            r
            for r in references()
            if replace(r.facts, paths=()) == replace(model, paths=())
            and r.dense_read_bytes == dense_read_bytes
        ),
        None,
    )


def _domain(
    r: Reference,
    c: Characteristics,
    runtime: RuntimeFacts,
    platform: str,
    host_work_bytes: int,
    gpu_work_bytes: int,
) -> tuple[Optional[float], str]:
    hardware = r.hardware
    hard = c["identity"]["hard"]
    if platform != hardware["platform"] or hard["arch"] != hardware["arch"]:
        return None, "platform differs from the calibrated hardware"
    if hard["cpu_brand"] != hardware["cpu_brand"] or len(runtime.cpus) < hardware["allowed_cpus"]:
        return None, "CPU or available CPU count differs from the calibrated hardware"
    gpu = single_gpu(c, runtime)
    if (
        not gpu
        or gpu["name"] != hardware["gpu_name"]
        or gpu["integrated"]
        or c["memory"]["unified"]
        or "CUDA" not in engine_gpu_backends(runtime.gpu_backends, c)
    ):
        return None, "requires the calibrated single discrete CUDA GPU"
    if not runtime.gpus or runtime.gpus[0].processes:
        return None, "GPU is occupied or its current state is unavailable"
    if (runtime.gpus[0].free_bytes or 0) < hardware["minimum_gpu_free_bytes"]:
        return None, "less free GPU memory than the calibrated placement"
    if host_work_bytes != r.workload["host_work_bytes"] or gpu_work_bytes not in (
        0,
        r.workload["gpu_work_bytes"],
    ):
        return None, "working-memory reserve differs from the reference workload"
    available = min(
        c["memory"]["total_bytes"],
        runtime.available_bytes
        if runtime.available_bytes is not None
        else c["memory"]["total_bytes"],
    )
    if runtime.memory_budget_bytes is not None:
        available = min(available, runtime.memory_budget_bytes)
    budget = available / (1 << 30)
    if not r.memory_gib[0] <= budget <= r.memory_gib[1]:
        return None, "host memory is outside the calibrated capacity range"
    return budget, ""


def apply_reference(
    baseline: Estimate,
    c: Characteristics,
    model: ModelFacts,
    dense_read_bytes: int,
    runtime: RuntimeFacts,
    platform: str,
    drive_key: Optional[str],
    *,
    host_work_bytes: int = 0,
    gpu_work_bytes: int = 0,
) -> Estimate:
    """Use the trained model automatically where its operating assumptions apply."""
    r = match(model, dense_read_bytes)
    if r is None:
        return baseline
    budget, reason = _domain(r, c, runtime, platform, host_work_bytes, gpu_work_bytes)
    rate, why = drive_rate(c, model, drive_key)
    if budget is None or rate is None or rate <= 0:
        return replace(
            baseline,
            assumptions=baseline.assumptions
            + ("Pretrained reference not applied: " + (reason or why or "no drive rate") + ".",),
        )
    params = r.parameters
    fraction = (budget - r.memory_gib[0]) / (r.memory_gib[1] - r.memory_gib[0])
    # Existing hardware-only probe, not a generated token, prices the I/O service.
    io_scale = params["drive_Bps"] / (rate * 1e9)
    times, slow, fast = [], [], []
    cpu_costs, gpu_costs, reads, waits = [], [], [], []
    details: list[dict[str, object]] = []
    compute = params["compute"]
    for i, route in enumerate(params["routing_samples"]):
        read = route["read_bytes"][0] * (1 - fraction) + route["read_bytes"][1] * fraction
        cpu = compute["cpu_all_ms"] * (1 - route["gpu_hit"])
        gpu = compute["gpu_all_ms"] * route["gpu_hit"]
        wait = read / (1 << 20) * params["wait_ms_per_mib"] * io_scale
        base = compute["dense_ms"] + cpu + gpu + wait
        time = base + params["residual_ms"][1]
        times.append(time)
        slow.append(base + params["residual_ms"][2])
        fast.append(base + params["residual_ms"][0])
        cpu_costs.append(cpu)
        gpu_costs.append(gpu)
        reads.append(read)
        waits.append(wait)
        details.append(
            {
                "name": f"reference_route_{i + 1}",
                "gpu_hit_fraction": route["gpu_hit"],
                "read_bytes_per_token": read,
                "decode_tokens_per_second": 1000 / time,
            }
        )
    if min(times + slow + fast) <= 0:
        raise ValueError("nonpositive pretrained decode service time")
    central = 1000 / statistics.median(times)
    bounds = (1000 / max(slow), 1000 / min(fast))
    assumptions = (
        "Offline training parameters; no prompt, short run or user calibration is required.",
        "Training at 32 GiB only; the larger-capacity read prior uses routing replay.",
        "Fresh-server reference workload: about 512 input tokens, 256 outputs; "
        "decode counts the 255 intervals after the first output, excluding TTFT.",
        f"Reference context {r.workload['context']}, prefetch {r.workload['prefetch_depth']}, "
        f"decode threads {r.workload['decode_threads']}, GPU work reserve "
        f"{cast(int, r.workload['gpu_work_bytes']) >> 20} MiB; estimates do not change launches.",
        "Synthetic compute and empirical non-I/O residual retain their calibrated hardware costs; "
        "exposed I/O wait is scaled inversely with the profiled drive rate.",
        "Interior memory capacities interpolate read volume between reference endpoints; "
        "only endpoint capacities were independently validated.",
        "Workload/service scenario range, not a confidence interval or a guarantee against "
        "new routing patterns, engine changes or shared-storage stalls.",
    )
    scenarios = dict(baseline.scenarios, pretrained_reference=central)
    return Estimate(
        central,
        f"offline model reference for {r.label} {r.quantization}; "
        f"{statistics.median(times):.0f} ms a token at {budget:g} GiB; no model inference",
        assumptions,
        scenarios,
        {
            "dense_ms": compute["dense_ms"],
            "cpu_experts_ms": statistics.median(cpu_costs),
            "gpu_experts_ms": statistics.median(gpu_costs),
            "exposed_io_wait_ms": statistics.median(waits),
            "non_io_residual_ms": params["residual_ms"][1],
            "read_bytes_per_token": statistics.median(reads),
            "memory_budget_bytes": budget * (1 << 30),
            "drive_Bps": rate * 1e9,
            "reference_drive_Bps": params["drive_Bps"],
        },
        tuple(details),
        bounds,
        "pretrained_reference",
        {
            "id": r.identifier,
            "workload": r.workload,
            "provenance": r.provenance,
            "hardware": dict(r.hardware),
            "capacity_interpolated": budget not in r.memory_gib,
            "analytical_reference_tokens_per_second": baseline.tokens_per_second,
        },
    )
