# SPDX-License-Identifier: Apache-2.0
"""The routed-expert formats the engine's CPU expert executor computes.

The executor multiplies each expert weight row against an input row quantized to the weight type's
vec_dot_type (engine/ggml/src/ggml-cpu/ggml-cpu.c, type_traits_cpu; the checks are in
engine/ggml/src/ggml-cpu/ssd/cpu-sink.cpp). A test compares VEC_DOT_TYPE with that table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any, Optional

from inferssd.strategy.inputs import Experts, needs_layer_formats

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


# The CUDA and ROCm expert tiers compile the same ggml_cuda_moe_stream_mmvq_supports switch.
GPU_EXPERT_TYPES = frozenset(VEC_DOT_TYPE) - {"F32", "TQ1_0", "TQ2_0"}


def gpu_expert_refusal(experts: Experts) -> Optional[str]:
    """Why the active VRAM tier cannot create a decode executor for these routed layers."""
    formats: Iterable[tuple[str, str, str]] = experts.layer_formats
    ambiguous = False
    known_companions = bool(experts.layer_formats)
    if not experts.layer_formats:
        # Older recordings only counted layers per type. All role combinations are safe
        # if every type has a kernel and up/down cannot form the one retired pairing.
        roles = [
            {name: count for name, count, _ in role.formats}
            for role in (experts.up, experts.gate, experts.down)
        ]
        ambiguous = needs_layer_formats(experts.layers, roles[0], roles[2])
        known_companions = all(len(role) == 1 for role in roles)
        formats = product(roles[0], roles[1], roles[2])

    latent, hidden = experts.up.cols, experts.down.cols
    for up, gate, down in formats:
        if not ambiguous and up == "IQ2_XXS" and down == "Q2_K":
            return "IQ2_XXS/Q2_K expert format is retired"
        if not (0 < latent <= 16384 and 0 < hidden <= 16384):
            return "unsupported generic VRAM expert geometry"
        for weight, width in ((up, latent), (gate, latent), (down, hidden)):
            block = WEIGHT_BLOCK.get(weight, INPUT_BLOCK.get(VEC_DOT_TYPE.get(weight, ""), 1))
            if width % block:
                return "unsupported generic VRAM expert geometry"
        missing = [weight for weight in (up, gate, down) if weight not in GPU_EXPERT_TYPES]
        if missing:
            # Marginal counts in older mixed receipts cannot identify the other two weights.
            names = (up, gate, down) if known_companions else tuple(dict.fromkeys(missing))
            native = "/".join(weight.lower().replace("_k", "_K") for weight in names)
            return f"GPU expert layer: no CUDA matrix-vector kernel for {native} experts"
    if ambiguous:
        return (
            "the recorded role counts cannot tell whether a layer uses the retired "
            "IQ2_XXS/Q2_K expert format; supply the GGUF or exported engine facts"
        )
    return None
