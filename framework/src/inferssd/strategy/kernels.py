# SPDX-License-Identifier: Apache-2.0
"""The routed-expert formats the engine's CPU expert executor computes.

The executor multiplies each expert weight row against an input row quantized to the weight type's
vec_dot_type (engine/ggml/src/ggml-cpu/ggml-cpu.c, type_traits_cpu; the checks are in
engine/ggml/src/ggml-cpu/ssd/cpu-sink.cpp). A test compares VEC_DOT_TYPE with that table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

# Each weight type with a CPU vec_dot, and the type its input rows are quantized to.
VEC_DOT_TYPE = {
    "F32": "F32",
    "F16": "F16",
    "BF16": "BF16",
    "Q1_0": "Q8_0",
    "Q2_0": "Q8_0",
    "Q4_0": "Q8_0",
    "Q4_1": "Q8_1",
    "Q5_0": "Q8_0",
    "Q5_1": "Q8_1",
    "Q8_0": "Q8_0",
    "MXFP4": "Q8_0",
    "NVFP4": "Q8_0",
    "IQ4_NL": "Q8_0",
    "Q2_K": "Q8_K",
    "Q3_K": "Q8_K",
    "Q4_K": "Q8_K",
    "Q5_K": "Q8_K",
    "Q6_K": "Q8_K",
    "IQ1_S": "Q8_K",
    "IQ1_M": "Q8_K",
    "IQ2_XXS": "Q8_K",
    "IQ2_XS": "Q8_K",
    "IQ2_S": "Q8_K",
    "IQ3_XXS": "Q8_K",
    "IQ3_S": "Q8_K",
    "IQ4_XS": "Q8_K",
    "TQ1_0": "Q8_K",
    "TQ2_0": "Q8_K",
}
# Values per block of the input types that have a from_float quantizer.
INPUT_BLOCK = {"F16": 1, "BF16": 1, "Q8_0": 32, "Q8_1": 32, "Q8_K": 256}
# Weights whose own block is larger than the input block. Each row must fit both.
WEIGHT_BLOCK = {"Q1_0": 128, "Q2_0": 64, "NVFP4": 64}


def expert_format_refusal(layout: Mapping[str, Any]) -> Optional[str]:
    """Why the CPU expert executor rejects a layer's routed-expert format, or None.

    LAYOUT is the engine's exported expert layout: per-layer format indices, the formats' role
    types, and the input width of up and gate (latent) and of down (hidden)."""
    used = {index for _, index in layout["layers"]}
    for index, roles in enumerate(layout["formats"]):
        if index not in used:
            continue
        inputs = {}
        for role in ("up", "gate", "down"):
            weight = str(roles[role]["type"]).upper()
            input_type = VEC_DOT_TYPE.get(weight)
            columns = layout["hidden"] if role == "down" else layout["latent"]
            if (
                input_type not in INPUT_BLOCK
                or columns % INPUT_BLOCK[input_type]
                or columns % WEIGHT_BLOCK.get(weight, 1)
            ):
                return f"the CPU expert executor cannot compute {weight} experts"
            inputs[role] = input_type
        if inputs["up"] != inputs["gate"]:
            return (
                "the CPU expert executor multiplies up and gate against one quantized input, but "
                f"up={roles['up']['type'].upper()} and gate={roles['gate']['type'].upper()} need "
                "different ones"
            )
    return None
