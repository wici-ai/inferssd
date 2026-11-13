# SPDX-License-Identifier: Apache-2.0
"""Model facts agree with individual expert extents across alignment patterns."""

from __future__ import annotations

from collections import defaultdict

import pytest
from test_strategy_inputs import (
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    deepseek_like,
    moe,
    muse_like,
    write_gguf,
)

from inferssd.source import header
from inferssd.strategy.inputs import facts_from_headers


def check_extents(headers):
    """Enumerate physical reads independently of the strategy's aggregation."""
    sizes, reads = defaultdict(int), defaultdict(int)
    for item in headers:
        for tensor in item.tensors:
            if "_exps.weight" not in tensor.name:
                continue
            layer = int(tensor.name.split(".")[1])
            size = tensor.nbytes // tensor.shape[2]
            for index in range(tensor.shape[2]):
                key = layer, index
                sizes[key] += size
                start = tensor.offset + index * size
                reads[key] += ((start % 4096 + size + 4095) // 4096) * 4096
    facts = facts_from_headers(headers)
    assert facts.expert_bytes == max(sizes.values())
    assert facts.expert_read_bytes == max(reads.values())
    assert facts.layers == len({layer for layer, _ in sizes})
    assert facts.routed_bytes == sum(sizes.values())


@pytest.mark.parametrize("layout", [deepseek_like, muse_like])
def test_aligned_model_layouts(tmp_path, layout):
    path = write_gguf(tmp_path / "model.gguf", *layout())
    check_extents([header.read(str(path))])


@pytest.mark.parametrize("experts", [1, 2, 5, 127, 129, 384])
@pytest.mark.parametrize("alignment", [32, 4096])
def test_mixed_formats_and_short_expert_sets(tmp_path, experts, alignment):
    tensors, kv = moe(
        "mixed",
        layers=3,
        experts=experts,
        used=1,
        latent=256,
        hidden=256,
        gate=[Q3_K, Q4_K, Q5_K],
        up=[Q4_K, Q5_K, Q6_K],
        down=[Q5_K, Q6_K, Q3_K],
        first=7,
    )
    # Different shard headers give the same layer's roles different starting
    # offsets. Layer 7 crosses the shard boundary instead of being kept together.
    paths = [tmp_path / "first.gguf", tmp_path / "second.gguf"]
    write_gguf(paths[0], tensors[:3], kv, alignment=alignment)
    write_gguf(paths[1], tensors[3:], {}, alignment=alignment)
    check_extents([header.read(str(path)) for path in paths])
