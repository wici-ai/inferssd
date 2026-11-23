# SPDX-License-Identifier: Apache-2.0
"""Transfer one resident reference run's unpriced decode time across models on a machine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import TypedDict, cast

from inferssd.machine.characteristics import Characteristics
from inferssd.strategy.estimate import Estimate
from inferssd.strategy.inputs import RuntimeFacts

SCHEMA = "inferssd-decode-calibration/1"


class Calibration(TypedDict):
    schema: str
    machine: str
    reference_layers: int
    reference_tokens: int
    measured_ms_per_token: float
    estimated_ms_per_token: float
    residual_ms_per_layer: float


def machine_key(machine: Characteristics, runtime: RuntimeFacts) -> str:
    """Bind a calibration to the machine, driver and visible accelerator devices."""
    devices = None if runtime.gpus is None else [(gpu.uuid, gpu.backend) for gpu in runtime.gpus]
    facts = [machine["identity"]["hard"], machine["identity"]["soft"]["gpu_driver"], devices]
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()


def _positive(value: object) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def fit(timing: object, estimate: Estimate, layers: int, key: str) -> Calibration:
    if isinstance(timing, dict):
        timing = timing.get("timings", timing)
    if not isinstance(timing, dict):
        raise ValueError("expected a server response with predicted_n and predicted_ms")
    count, duration = timing.get("predicted_n"), timing.get("predicted_ms")
    if type(count) is not int or count < 2 or not _positive(duration):
        raise ValueError(
            "calibration requires at least two decode tokens and positive predicted_ms"
        )
    if not estimate.tokens_per_second or layers < 1:
        raise ValueError("calibration requires an estimable reference model with known layers")
    if estimate.components.get("storage_ms", 0) > 0:
        raise ValueError("use a warmed RAM/VRAM-resident reference with zero estimated disk reads")
    predicted = 1000 / estimate.tokens_per_second
    # server-context starts the generation clock at token one. N emissions span N-1 intervals.
    observed = cast(float, duration) / (count - 1)
    return Calibration(
        schema=SCHEMA,
        machine=key,
        reference_layers=layers,
        reference_tokens=count,
        measured_ms_per_token=observed,
        estimated_ms_per_token=predicted,
        residual_ms_per_layer=max(0.0, observed - predicted) / layers,
    )


def load(data: object, key: str) -> Calibration:
    if isinstance(data, dict):
        data = data.get("calibration", data)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("expected a calibration from profile --calibrate-from")
    if data.get("machine") != key:
        raise ValueError("calibration belongs to a different machine, driver or visible GPU set")
    for name in ("reference_layers", "reference_tokens"):
        if type(data.get(name)) is not int or data[name] < (2 if name == "reference_tokens" else 1):
            raise ValueError(f"invalid calibration field {name}")
    for name in ("estimated_ms_per_token", "measured_ms_per_token"):
        if not _positive(data.get(name)):
            raise ValueError(f"invalid calibration field {name}")
    residual = data.get("residual_ms_per_layer")
    expected = max(0.0, data["measured_ms_per_token"] - data["estimated_ms_per_token"])
    if (
        not isinstance(residual, (float, int))
        or isinstance(residual, bool)
        or not math.isclose(residual, expected / data["reference_layers"])
    ):
        raise ValueError("invalid calibration residual_ms_per_layer")
    return cast(Calibration, data)


def apply(estimate: Estimate, calibration: Calibration, layers: int) -> Estimate:
    if estimate.tokens_per_second is None or layers == 0:
        return replace(
            estimate, assumptions=(*estimate.assumptions, "Calibration unavailable here.")
        )
    residual = calibration["residual_ms_per_layer"] * layers

    def adjust(tps: float) -> float:
        return 1000 / (1000 / tps + residual)

    details = [
        {k: adjust(v) if k.endswith("_tps") and isinstance(v, float) else v for k, v in row.items()}
        for row in estimate.scenario_details
    ]
    bounds = estimate.scenario_range
    return replace(
        estimate,
        tokens_per_second=adjust(estimate.tokens_per_second),
        detail=f"{estimate.detail}; add {residual:.1f} ms/token from the resident reference",
        assumptions=(
            *estimate.assumptions,
            "The reference's unpriced time scales with layer count.",
        ),
        components={
            **estimate.components,
            "uncalibrated_ms": 1000 / estimate.tokens_per_second,
            "calibration_residual_ms": residual,
        },
        scenarios={name: adjust(rate) for name, rate in estimate.scenarios.items()},
        scenario_details=tuple(details),
        scenario_range=(adjust(bounds[0]), adjust(bounds[1])) if bounds else None,
    )
