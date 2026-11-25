# SPDX-License-Identifier: Apache-2.0
"""GPU estimates follow the native CUDA/ROCm expert executor's per-layer formats."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from test_decode_profiling import setup
from test_machine_command import execute, machine, resource_descriptor
from test_strategy_inputs import ENGINE_FIXTURE, Q4_K, Q8_0, TYPES, dense, engine_facts, moe

from inferssd.machine.probes import Completed
from inferssd.machine.store import save
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


@pytest.mark.parametrize("device", ["untimed", "unified", "rocm"])
def test_an_active_gpu_tier_checks_formats_even_without_a_single_measured_device(
    monkeypatch, device
):
    c, rt = setup()
    if device == "untimed":
        c["gpus"][0].pop("decode_compute")
    if device == "unified":
        c["memory"]["unified"] = True
    if device == "rocm":
        rt = replace(rt, gpu_backends=("ROCm",), gpus=(replace(rt.gpus[0], backend="ROCm"),))
    result = estimate(c, rt, model(monkeypatch, TQ2_0, TQ2_0, TQ2_0))
    assert result.tokens_per_second is None and "no CUDA matrix-vector kernel" in result.detail


@pytest.mark.parametrize("target", ["gguf", "facts"])
@pytest.mark.parametrize("routed", [True, False], ids=["routed", "dense"])
def test_profile_uses_default_device_visibility_of_the_selected_server(tmp_path, target, routed):
    host, files, build = machine(tmp_path)
    descriptor = resource_descriptor(tmp_path, build)
    resources = json.loads(descriptor.read_text())
    resources["capabilities"]["gpu_backends"] = ["ROCm", "Vulkan"]
    descriptor.write_text(json.dumps(resources))
    Path(resources["binary"]).touch()
    profile, runtime = setup()
    # The host's utilities see only one card. The selected driver can enumerate both.
    runtime = replace(runtime, gpus=(replace(runtime.gpus[0], backend="ROCm"),))
    save(profile, tmp_path / "machine.json")
    if routed:
        tensors, kv = moe(
            "deepseek41",
            layers=2,
            experts=8,
            used=4,
            latent=512,
            hidden=256,
            up=Q8_0,
            gate=Q8_0,
            down=Q8_0,
        )
        kv["deepseek41.swiglu_clamp_exp"] = [7.0, 7.0]
    else:
        tensors, kv = dense("qwen35")
    raw = engine_facts(tensors, kv, path=files[0])
    exported = tmp_path / "model.json"
    exported.write_text(json.dumps(raw))
    args = [
        "--show",
        "--output",
        str(tmp_path / "machine.json"),
        "--resources",
        str(descriptor),
        "--json",
        "--estimate-" + target,
        files[0] if target == "gguf" else str(exported),
    ]
    # Vulkan aliases and accelerators are not additional CUDA/ROCm tiers. A legacy
    # binary that cannot list devices falls back to live runtime visibility.
    devices = "Available devices:\n  Vulkan0: integrated\n  BLAS: accelerator\n"
    one = devices + "  ROCm0: Radeon AI PRO R9700\n"
    for listing, refused in (
        (Completed(0, one + "  ROCm1: Radeon AI PRO R9700\n"), True),
        (Completed(0, one), False),
        (Completed(0, devices), False),
        (Completed(1, "", "unknown argument"), False),
    ):

        def server(argv, timeout_s, environ):
            assert argv[0] == resources["binary"]
            if argv[1:] == ["--list-devices"]:
                return listing
            assert argv[1:] == ["--ssd-model-facts", files[0]]
            return Completed(0, json.dumps(raw))

        code, out, _ = execute(args, host, server, tmp_path, runtime=runtime)
        assert code == 0
        (row,) = json.loads(out[0])["estimates"]
        if refused:
            assert row["detail"] == "one GPU expert executor per context is supported"
            assert row["decode_tokens_per_second"] is None and not row["scenarios"]
        else:
            assert row["decode_tokens_per_second"] > 0

        if refused:
            for settings in (
                {"LLAMA_ARG_DEVICE": "ROCm1"},
                {"LLAMA_ARG_SPLIT_MODE": "none"},
                {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "1"},
                {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "-1"},
                {"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_DEVICE": "ROCm1,ROCm0"},
            ):
                code, out, _ = execute(args, host, server, tmp_path, settings, runtime=runtime)
                assert code == 0
                assert json.loads(out[0])["estimates"][0]["decode_tokens_per_second"] > 0


def test_device_selection_uses_native_ordinals_and_propagates_driver_errors():
    from inferssd.machine.command import DeviceSelectionError, engine_tier_devices

    _, runtime = setup()
    first = runtime.gpus[0]
    second = replace(first, uuid="other", free_bytes=12345, processes=2)
    runtime = replace(runtime, gpus=(first, second))
    listing = Completed(0, "Available devices:\n  CUDA0: first\n  CUDA1: second\n")

    def devices(environ, response=listing):
        return engine_tier_devices(Path("server"), lambda *args: response, environ, runtime)

    assert devices({"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "1"}) == (second,)
    assert devices({"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_DEVICE": "CUDA1,CUDA0"}) == (second,)
    assert devices({"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "-1"}) == ()
    for backend in ("MTL0", "Vulkan0"):
        assert (
            devices(
                {"LLAMA_ARG_SPLIT_MODE": "none"},
                Completed(0, f"Available devices:\n  {backend}: GPU\n"),
            )
            == ()
        )
    with pytest.raises(DeviceSelectionError, match="invalid value for main_gpu: 2"):
        devices({"LLAMA_ARG_SPLIT_MODE": "none", "LLAMA_ARG_MAIN_GPU": "2"})
    with pytest.raises(DeviceSelectionError, match="invalid device: CUDA19"):
        error = (
            'error while handling environment variable "LLAMA_ARG_DEVICE": invalid device: CUDA19'
        )
        devices({"LLAMA_ARG_DEVICE": "CUDA19"}, Completed(1, "", error))


def test_an_unreadable_gpu_inventory_leaves_cpu_only_estimates_available(monkeypatch):
    c, rt = setup()
    facts = model(monkeypatch, Q8_0, Q8_0, Q8_0)
    rt = replace(rt, gpus=None)
    result = estimate(c, rt, facts)
    assert result.tokens_per_second is None and "cannot determine which GPU" in result.detail
    assert estimate(c, replace(rt, gpu_backends=()), facts).tokens_per_second > 0


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


@pytest.mark.parametrize(
    ("first", "second", "reason"),
    [
        (
            (Q8_0,) * 3,
            (TQ2_0,) * 3,
            "GPU expert layer: no CUDA matrix-vector kernel for tq2_0/tq2_0/tq2_0 experts",
        ),
        (
            (TQ2_0,) * 3,
            (IQ2_XXS, IQ2_XXS, Q2_K),
            "GPU expert layer: no CUDA matrix-vector kernel for tq2_0/tq2_0/tq2_0 experts",
        ),
        ((IQ2_XXS, IQ2_XXS, Q2_K), (TQ2_0,) * 3, "IQ2_XXS/Q2_K expert format is retired"),
    ],
)
def test_report_the_first_native_failure_for_real_layer_combinations(
    monkeypatch, first, second, reason
):
    c, rt = setup()
    facts = model(monkeypatch, *([a, b] for a, b in zip(first, second)))
    assert estimate(c, rt, facts).detail == reason
    recorded = model_facts_json(facts)
    assert estimate(c, rt, model_facts_from(recorded)).detail == reason
    del recorded["experts"]["layer_formats"]
    legacy = estimate(c, rt, model_facts_from(recorded))
    assert legacy.tokens_per_second is None and "first failing" in legacy.detail
    assert "q8_0/q8_0/tq2_0" not in legacy.detail


def test_one_possible_missing_kernel_signature_needs_no_additional_record_field(monkeypatch):
    c, rt = setup()
    facts = model(monkeypatch, Q8_0, Q8_0, [Q4_K, TQ2_0])
    recorded = model_facts_json(facts)
    assert "layer_formats" not in recorded["experts"]
    assert estimate(c, rt, model_facts_from(recorded)).detail == (
        "GPU expert layer: no CUDA matrix-vector kernel for q8_0/q8_0/tq2_0 experts"
    )
