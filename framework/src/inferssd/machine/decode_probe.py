# SPDX-License-Identifier: Apache-2.0
"""Optional synthetic CUDA matvec measurements. No model is loaded or executed."""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from inferssd.machine.characteristics import Characteristics
from inferssd.machine.probes import ProbeError, Probes, compute_rows
from inferssd.strategy.inputs import RuntimeFacts


def measure_decode_compute(
    value: Characteristics,
    probes: Probes,
    runtime: RuntimeFacts,
    progress: Callable[[str], None],
    force: bool = False,
) -> Characteristics:
    result = copy.copy(value)
    result["gpus"] = copy.deepcopy(value["gpus"])
    changed = False
    states = {gpu.uuid: gpu for gpu in runtime.gpus or ()}
    cuda = {gpu["uuid"] for gpu in value["identity"]["hard"]["gpus"] if gpu["backend"] == "CUDA"}
    binary = Path(dict(probes.paths).get("cpu", ""))
    digest = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else "unknown"
    for gpu in result["gpus"]:
        cached = (
            gpu.get("decode_compute")
            and digest != "unknown"
            and gpu.get("decode_compute_probe_sha256") == digest
        )
        if gpu["uuid"] not in cuda or (cached and not force):
            continue
        state = states.get(gpu["uuid"])
        if state is None or state.processes:
            raise ProbeError("synthetic CUDA profiling requires an idle, visible GPU")
        if state.free_bytes is None or state.free_bytes < 1 << 30:
            raise ProbeError("synthetic CUDA profiling requires at least 1 GiB free VRAM")
        progress(f"measuring synthetic decode matrices on {gpu['name']} (no model inference)")
        env = dict(probes.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
        isolated = Probes(probes.paths, probes.runner, env)
        started = time.monotonic()
        rows = []
        # General shape plus both DeepSeek expert orientations. Rotated weights keep
        # the working set above the GPU L2 cache; each point reports median and IQR.
        for outputs, inputs in ((4096, 4096), (2304, 5120), (5120, 2304)):
            report = isolated.run(
                "cpu",
                [
                    "--backend",
                    "CUDA0",
                    "--threads",
                    "1",
                    "--tokens",
                    "1",
                    "--rows",
                    str(outputs),
                    "--cols",
                    str(inputs),
                    "--formats",
                    "Q8_0,Q4_K,Q5_K,Q6_K,Q2_K,MXFP4,F16",
                    "--min-seconds",
                    "0.25",
                ],
            )
            if report.get("backend") != "CUDA0":
                raise ProbeError("probe did not confirm the requested CUDA backend")
            rows.extend(compute_rows(report))
        if not rows:
            raise ProbeError("no synthetic CUDA matrix measurements returned")
        changed = True
        gpu["decode_compute"] = rows
        gpu["decode_compute_probe_sha256"] = digest
        gpu["decode_compute_measurement"] = {
            "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": time.monotonic() - started,
            "method": "inferssd-cpuprobe/2 CUDA matvec; rotated 256 MiB; tokens=1",
            "status": "ok",
            "detail": "Synthetic weights; shapes 4096x4096,2304x5120,5120x2304; "
            "no model inference; probe build: " + digest,
        }
    return result if changed else value
