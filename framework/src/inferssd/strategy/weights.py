# SPDX-License-Identifier: Apache-2.0
"""Whole-tensor traffic for one decode token from the engine's GGUF tensor roles."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from inferssd.strategy.inputs import ROLES, ModelFacts, routed_tensor


def _lookup_names(model: Mapping[str, Any]) -> set[str]:
    """Weight roles using GET_ROWS (llama-arch.cpp), and encoder-only blocks at decode."""
    names = {t["name"] for t in model["tensors"]}
    has_decoder = any(n.startswith("dec.blk.") for n in names)
    excluded = {
        "per_layer_token_embd.weight",
        "position_embd.weight",
        "token_types.weight",
        "markov_w1.weight",
    }
    # A token table doubles as the output projection when the GGUF has no explicit head.
    if "output.weight" in names:
        excluded.add("token_embd.weight")
    for name in names:
        if has_decoder and name.startswith("enc.blk."):
            excluded.add(name)
        elif name.startswith(("blk.", "enc.blk.", "dec.blk.")) and name.endswith(
            (
                ".engram_embd.weight",
                ".ffn_gate_tid2eid.weight",
                ".attn_rel_b.weight",
                ".attn_compressor_ape.weight",
                ".nextn.embed_tokens.weight",
                ".nextn.enorm.weight",
            )
        ):
            excluded.add(name)
    return excluded


_GRAPH_EXPERT = re.compile(
    r"(?:dec\.)?blk\.\d+\.ffn_(?:gate_up|gate|up|down)_(exps|chexps)\."
    r"(weight|bias|scale|input_scale)$"
)


@dataclass(frozen=True)
class DenseWeights:
    # (format, output rows, input columns, total bytes with this shape); vectors use zero rows.
    shapes: tuple[tuple[str, int, int, int], ...]
    layer_matrix_bytes: int
    graph_store_bytes: int = 0
    graph_active_bytes: int = 0
    graph_unverified_bytes: int = 0

    @property
    def active_bytes(self) -> int:
        return sum(size for _, _, _, size in self.shapes)

    @property
    def placement_bytes(self) -> int:
        return self.active_bytes + self.graph_store_bytes - self.graph_active_bytes

    def host_spill_bytes(self, room: int) -> int:
        """Uncached traffic with dense tensors first, then uniform coverage of packed experts."""
        shared = self.active_bytes - self.graph_active_bytes
        if not self.graph_store_bytes:
            return max(0, shared - room)
        free = max(0, room - shared)
        missing = max(0, self.graph_store_bytes - free)
        return (
            max(0, shared - room)
            + (self.graph_active_bytes * missing + self.graph_store_bytes - 1)
            // self.graph_store_bytes
        )

    def graph_components(self) -> dict[str, float]:
        if not self.graph_store_bytes:
            return {}
        return {
            "graph_expert_store_bytes": float(self.graph_store_bytes),
            "graph_expert_active_bytes": float(self.graph_active_bytes),
            "graph_expert_unverified_bytes": float(self.graph_unverified_bytes),
            "dense_placement_bytes": float(self.placement_bytes),
        }

    def graph_assumptions(self) -> list[str]:
        if not self.graph_store_bytes:
            return []
        notes = [
            "Ordinary-graph packed experts reserve every slice for placement. A token reads "
            "the selected slices where the GGUF expert count, top-k and tensor shape agree. "
            "CPU paging assumes uniform coverage of the ordinary-graph store."
        ]
        if self.graph_unverified_bytes:
            notes.append(
                "Missing or inconsistent expert metadata/shape: affected packed tensors "
                "are priced as full scans."
            )
        return notes


def dense_weights(model: Mapping[str, Any]) -> DenseWeights:
    """Active shapes and placements of the tensors outside the native SSD expert store.

    Native homes require a repeating layer's matrix used in MUL_MAT. The exported header
    provides its dimensions and name, so two-dimensional repeating weights are a proxy
    for this device-dependent runtime decision. Unknown tensor roles keep their full cost.
    """
    sizes: Counter[tuple[str, int, int]] = Counter()
    layer_matrix_bytes = 0
    excluded = _lookup_names(model)
    layout = model.get("experts")
    if layout:
        excluded.update(
            routed_tensor(layer, role) for layer, _ in layout["layers"] for role in ROLES
        )
    prefix = model.get("architecture", "") + "."
    metadata = model.get("metadata", {})
    count = metadata.get(prefix + "expert_count")
    top_k = metadata.get(prefix + "expert_used_count")
    graph_store = graph_active = graph_unverified = 0
    for tensor in model["tensors"]:
        name, shape, size = tensor["name"], tensor["shape"], tensor["bytes"]
        if name in excluded or not size:
            continue
        matrix = len(shape) >= 2 and shape[1] > 1 and all(d == 1 for d in shape[2:])
        rows, cols = (shape[1], shape[0]) if matrix else (0, 0)
        match = _GRAPH_EXPERT.fullmatch(name)
        if match:
            family, suffix = match.groups()
            # MUL_MAT_ID weights are [input, output, experts], ADD_ID biases [output, experts].
            # Per-expert quantization scales contain one entry per expert.
            rank = {"weight": 3, "bias": 2, "scale": 1, "input_scale": 1}[suffix]
            n, k = count, top_k
            if family == "chexps":
                group = metadata.get(prefix + "experts_per_group")
                if type(group) is int and group > 0 and type(n) is int and n % group == 0:
                    n = n // group
                    k = min(k, n) if type(k) is int else None
                else:
                    n = None
            graph_store += size
            if (
                type(n) is int
                and type(k) is int
                and 1 <= k <= n
                and len(shape) == rank
                and shape[-1] == n
                and size % n == 0
            ):
                size = size // n * k
                rows, cols = (shape[1], shape[0]) if suffix == "weight" else (0, 0)
            else:
                graph_unverified += size
            graph_active += size
        elif matrix and name.startswith(("blk.", "dec.blk.", "enc.blk.")):
            layer_matrix_bytes += size
        sizes[tensor["type"].upper(), rows, cols] += size
    return DenseWeights(
        tuple((*key, size) for key, size in sorted(sizes.items())),
        layer_matrix_bytes,
        graph_store,
        graph_active,
        graph_unverified,
    )


def dense_read_bytes(model: Mapping[str, Any], facts: ModelFacts) -> int:
    """Bytes visited outside the SSD expert store in one token."""
    weights = dense_weights(model)
    if weights.placement_bytes > facts.dense_bytes:
        raise ValueError("active tensor footprint exceeds the model facts' non-SSD weights")
    return weights.active_bytes
