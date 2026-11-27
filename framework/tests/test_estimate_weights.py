# SPDX-License-Identifier: Apache-2.0
"""Derive active weights from the engine's tensor roles for arbitrary local checkpoints."""

from inferssd.strategy.inputs import facts_from_engine, model_facts_from, model_facts_json
from inferssd.strategy.weights import dense_read_bytes, dense_weights


def model(*names):
    return {
        "architecture": "unknown-generator",
        "files": ["model.gguf"],
        "layers": 1,
        "tensors": [
            dict(name=name, bytes=size, shape=[size // 2], type="F16") for name, size in names
        ],
        "experts": None,
    }


def test_input_tables_skip_full_scans_and_tied_output_counts_once():
    raw = model(
        ("token_embd.weight", 4096),
        ("per_layer_token_embd.weight", 8192),
        ("blk.0.engram_embd.weight", 16384),
        ("blk.0.ple_key.weight", 1024),
        ("position_embd.weight", 128),
        ("token_types.weight", 64),
        ("blk.0.unknown.weight", 512),
    )
    assert dense_read_bytes(raw, facts_from_engine(raw)) == 4096 + 1024 + 512
    raw["tensors"].append(dict(name="output.weight", bytes=2048, type="F16", shape=[32, 32]))
    assert dense_read_bytes(raw, facts_from_engine(raw)) == 2048 + 1024 + 512


def test_encoder_decoder_skips_encoder_at_decode_and_other_layer_layouts_are_accepted():
    raw = model(("enc.blk.0.attn_q.weight", 1024), ("dec.blk.0.attn_q.weight", 32))
    facts = facts_from_engine(raw)
    assert facts.layer_format_bytes == (("F16", 32),)
    assert dense_read_bytes(raw, facts) == 32
    raw = model(("alternative.layer.0.unknown.weight", 1024))
    facts = facts_from_engine(raw)
    assert model_facts_from(model_facts_json(facts)) == facts
    assert dense_read_bytes(raw, facts) == 1024


def test_ordinary_graph_experts_price_selected_slices_and_reserve_the_complete_store():
    raw = model(("output.weight", 2048))
    arch = raw["architecture"]
    raw["metadata"] = {arch + ".expert_count": 8, arch + ".expert_used_count": 2}
    raw["tensors"].extend(
        [
            {
                "name": "blk.0.ffn_gate_up_exps.weight",
                "shape": [32, 64, 8],
                "bytes": 32768,
                "type": "F16",
            },
            {
                "name": "blk.0.ffn_down_exps.weight",
                "shape": [32, 32, 8],
                "bytes": 16384,
                "type": "F16",
            },
            {
                "name": "blk.1.ffn_up_exps.weight",
                "shape": [32, 32, 8],
                "bytes": 16384,
                "type": "F16",
            },
            {
                "name": "blk.1.ffn_down_exps.weight",
                "shape": [32, 32, 8],
                "bytes": 16384,
                "type": "F16",
            },
            {"name": "blk.0.ffn_down_exps.bias", "shape": [32, 8], "bytes": 512, "type": "F16"},
        ]
    )
    weights = dense_weights(raw)
    assert weights.graph_store_bytes == 82432
    assert weights.graph_active_bytes == 20608 and weights.graph_unverified_bytes == 0
    assert weights.placement_bytes == 84480 and weights.active_bytes == 22656
    assert weights.host_spill_bytes(2048 + 41216) == 10304
    assert dense_read_bytes(raw, facts_from_engine(raw)) == weights.active_bytes
    assert weights.layer_matrix_bytes == 0
    raw["metadata"].pop(arch + ".expert_used_count")
    unknown = dense_weights(raw)
    assert unknown.graph_unverified_bytes == weights.graph_store_bytes
    assert unknown.active_bytes == weights.placement_bytes
    raw["metadata"][arch + ".expert_used_count"] = 2
    raw["tensors"][1]["shape"][-1] = 4
    assert dense_weights(raw).graph_unverified_bytes == 32768


def test_native_ssd_triples_and_ordinary_graph_experts_can_coexist_without_double_counting():
    import json

    from test_strategy_inputs import ENGINE_FIXTURE

    raw = json.loads(ENGINE_FIXTURE.read_text())
    base = dense_weights(raw)
    assert base.graph_store_bytes == 0
    assert base.active_bytes == facts_from_engine(raw).dense_bytes
    template = raw["tensors"][-1]
    # Native layout accounts for the three roles in blocks 0 and 1, while block 2 is fused.
    raw["tensors"].append(
        {**template, "name": "blk.2.ffn_gate_up_exps.weight", "shape": [64, 64, 4], "bytes": 32768}
    )
    weights = dense_weights(raw)
    assert weights.graph_store_bytes == 32768 and weights.graph_active_bytes == 16384
    assert weights.active_bytes == base.active_bytes + 16384
    assert weights.placement_bytes == base.active_bytes + 32768
