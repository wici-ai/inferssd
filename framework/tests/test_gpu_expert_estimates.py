# SPDX-License-Identifier: Apache-2.0
"""GPU estimates follow the native CUDA/ROCm expert executor's per-layer formats."""

import re
from dataclasses import replace

import pytest
from test_decode_profiling import setup
from test_strategy_inputs import ENGINE_FIXTURE, Q4_K, Q8_0, TYPES, engine_facts, moe

from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import facts_from_engine, model_facts_from, model_facts_json
from inferssd.strategy.kernels import GPU_EXPERT_TYPES

Q2_K, IQ2_XXS, TQ2_0 = 10, 16, 35


def model(monkeypatch, up, gate, down):
    for kind, fmt in (
        (Q2_K, ("q2_K", 256, 84)),
        (IQ2_XXS, ("iq2_xxs", 256, 66)),
        (TQ2_0, ("tq2_0", 256, 66)),
    ):
        monkeypatch.setitem(TYPES, kind, fmt)
    tensors, kv = moe(
        "qwen35moe",
        layers=2,
        experts=8,
        used=4,
        latent=512,
        hidden=256,
        up=up,
        gate=gate,
        down=down,
    )
    return facts_from_engine(engine_facts(tensors, kv))


def estimate(c, rt, facts):
    return decode_estimate(c, facts, facts.dense_bytes, rt, "linux", c["storage"][0]["drive_key"])


@pytest.mark.parametrize(
    ("up", "gate", "down", "reason"),
    [
        (TQ2_0, TQ2_0, TQ2_0, "no CUDA matrix-vector kernel"),
        ([Q4_K, IQ2_XXS], [Q4_K, IQ2_XXS], [Q4_K, Q2_K], "IQ2_XXS/Q2_K expert format is retired"),
        ([Q4_K, IQ2_XXS], [Q4_K, IQ2_XXS], [Q2_K, Q4_K], None),
        (Q4_K, IQ2_XXS, Q2_K, None),  # The retired pairing concerns up, not gate.
        (Q8_0, Q8_0, Q8_0, None),
    ],
)
def test_the_gpu_launch_checks_actual_formats_before_pricing(monkeypatch, up, gate, down, reason):
    c, rt = setup()
    facts = model(monkeypatch, up, gate, down)
    actual = estimate(c, rt, facts)
    assert estimate(c, rt, model_facts_from(model_facts_json(facts))) == actual
    if reason:
        assert actual.tokens_per_second is None and reason in actual.detail and not actual.scenarios
    else:
        assert actual.tokens_per_second > 0 and actual.components["gpu_experts_ms"] > 0
    cpu = estimate(c, replace(rt, gpu_backends=()), facts)
    assert cpu.tokens_per_second > 0 and cpu.components["gpu_experts_ms"] == 0


@pytest.mark.parametrize("device", ["untimed", "unified", "multiple", "rocm"])
def test_an_active_gpu_tier_checks_formats_even_without_a_single_measured_device(
    monkeypatch, device
):
    c, rt = setup()
    if device == "untimed":
        c["gpus"][0].pop("decode_compute")
    if device == "unified":
        c["memory"]["unified"] = True
    if device == "multiple":
        rt = replace(rt, gpus=(*rt.gpus, replace(rt.gpus[0], uuid="other")))
    if device == "rocm":
        rt = replace(rt, gpu_backends=("ROCm",), gpus=(replace(rt.gpus[0], backend="ROCm"),))
    result = estimate(c, rt, model(monkeypatch, TQ2_0, TQ2_0, TQ2_0))
    assert result.tokens_per_second is None and "no CUDA matrix-vector kernel" in result.detail


def test_gpu_expert_types_match_the_native_mmvq_switch():
    source = (ENGINE_FIXTURE.parents[4] / "engine/ggml/src/ggml-cuda/mmvq.cu").read_text()
    function = source.split('extern "C" bool ggml_cuda_moe_stream_mmvq_supports(', 1)[1]
    cases = function.split("return true;", 1)[0]
    assert GPU_EXPERT_TYPES == set(re.findall(r"case GGML_TYPE_(\w+):", cases))


def test_old_mixed_receipts_distinguish_missing_pairing_from_a_known_refusal(monkeypatch):
    c, rt = setup()
    for down, detail in (([Q2_K, Q4_K], "recorded role counts"), (Q2_K, "format is retired")):
        facts = model(monkeypatch, [IQ2_XXS, Q4_K], [IQ2_XXS, Q4_K], down)
        recorded = model_facts_json(facts)
        assert ("layer_formats" in recorded["experts"]) == isinstance(down, list)
        recorded["experts"].pop("layer_formats", None)
        legacy = model_facts_from(recorded)
        gpu = estimate(c, rt, legacy)
        assert gpu.tokens_per_second is None and detail in gpu.detail
        assert estimate(c, replace(rt, gpus=()), legacy).tokens_per_second > 0


def test_recorded_expert_types_accept_ggml_spelling(monkeypatch):
    c, rt = setup()
    recorded = model_facts_json(model(monkeypatch, [IQ2_XXS, Q4_K], [IQ2_XXS, Q4_K], [Q2_K, Q4_K]))
    for fmt in recorded["experts"]["layer_formats"]:
        fmt.update({role: weight.lower() for role, weight in fmt.items()})
    assert "format is retired" in estimate(c, rt, model_facts_from(recorded)).detail


def test_ordinary_records_keep_their_previous_fields(monkeypatch):
    for facts in (
        model(monkeypatch, Q8_0, Q8_0, Q8_0),
        model(monkeypatch, [Q4_K, Q8_0], [Q4_K, Q8_0], Q4_K),
    ):
        recorded = model_facts_json(facts)
        assert "layer_formats" not in recorded["experts"]
        assert model_facts_from(recorded) == facts
