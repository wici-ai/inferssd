# SPDX-License-Identifier: Apache-2.0
"""Invariants of pre-inference estimation: budgets, devices, and unknown locality."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from inferssd.app import catalog
from inferssd.machine.decode_probe import measure_decode_compute
from inferssd.machine.probes import Completed, ProbeError, Probes
from inferssd.machine.store import load, save
from inferssd.strategy.derive import cpu_line, placement
from inferssd.strategy.estimate import (
    decode_estimate,
    dense_decode_estimate,
    format_proxies,
    matvec_rate,
)
from inferssd.strategy.inputs import GPUState, RuntimeFacts

GIB = 1 << 30
BASE = load(Path(__file__).parent / "fixtures/machine/linux-x86-rtx5090.json")
MODEL = catalog.MODELS[0]


def synthetic_row(name="MXFP4", rate=500e9, rows=4096, cols=4096):
    return dict(
        format=name,
        threads=1,
        rows=rows,
        cols=cols,
        weight_bytes=16 << 20,
        points=[dict(tokens=1, us=40.0, us_q1=39.0, us_q3=41.0, weight_Bps=rate)],
    )


def setup():
    c = copy.deepcopy(BASE)
    c["memory"]["total_bytes"] = 256 * GIB
    c["memory"]["unified"] = False
    gpu = c["gpus"][0]
    gpu["integrated"] = False
    gpu["decode_compute"] = [synthetic_row("Q8_0"), synthetic_row()]
    rt = RuntimeFacts(
        tuple(r["id"] for r in c["cpu"]["logical"]),
        (GPUState(gpu["uuid"], 30 * GIB, 0),),
        200 * GIB,
    )
    return c, rt


def estimate(c, rt, **kw):
    return decode_estimate(
        c, MODEL.facts, MODEL.dense_read_bytes, rt, "linux", c["storage"][0]["drive_key"], **kw
    )


def test_host_budget_and_gpu_placement_are_distinct():
    c, rt = setup()
    full = estimate(c, rt)
    limited = estimate(c, replace(rt, memory_budget_bytes=32 * GIB))
    assert limited.components["host_cache_capacity_bytes"] < 32 * GIB
    assert (
        limited.components["gpu_cache_capacity_bytes"]
        == full.components["gpu_cache_capacity_bytes"]
    )
    assert limited.tokens_per_second < full.tokens_per_second
    empty = estimate(c, replace(rt, memory_budget_bytes=0))
    assert empty.components["host_cache_capacity_bytes"] == 0


def test_decode_pool_not_prefill_pool_prices_cpu_work():
    c, rt = setup()
    rt = replace(rt, gpus=())
    threads, _, _ = placement(c, MODEL.facts, rt, [])
    line, _ = cpu_line(c, MODEL.facts, threads.decode)
    expected = MODEL.facts.layers * MODEL.facts.experts_used * sum(line) / 1000
    assert estimate(c, rt).components["cpu_experts_ms"] == pytest.approx(expected)


def test_inclusive_caches_are_not_added_and_reserves_reduce_capacity():
    c, rt = setup()
    value = estimate(c, rt)
    comp = value.components
    coverage = max(comp["host_cache_capacity_bytes"], comp["gpu_cache_capacity_bytes"])
    assert comp["uniform_storage_miss_fraction"] == pytest.approx(
        1 - coverage / MODEL.facts.routed_bytes
    )
    reserved = estimate(c, rt, host_work_bytes=2 * GIB, gpu_work_bytes=4 * GIB)
    assert (
        reserved.components["host_cache_capacity_bytes"]
        == comp["host_cache_capacity_bytes"] - 2 * GIB
    )
    assert (
        reserved.components["gpu_cache_capacity_bytes"]
        == comp["gpu_cache_capacity_bytes"] - 4 * GIB
    )


@pytest.mark.parametrize("mode", ["unified", "invisible", "backend-disabled", "missing-compute"])
def test_gpu_must_be_supported_and_measured(mode):
    c, rt = setup()
    if mode == "unified":
        c["memory"]["unified"] = True
    elif mode == "invisible":
        rt = replace(rt, gpus=())
    elif mode == "backend-disabled":
        rt = replace(rt, gpu_backends=())
    else:
        del c["gpus"][0]["decode_compute"]
    assert estimate(c, rt).components["gpu_cache_capacity_bytes"] == 0


def test_scenarios_are_conditional_not_a_fitted_scalar():
    c, rt = setup()
    value = estimate(c, rt)
    assert value.tokens_per_second == value.scenarios["uniform_serial"]
    assert value.scenarios["uniform_perfect_overlap"] >= value.tokens_per_second
    assert value.scenarios["hot20_traffic95_serial"] >= value.scenarios["hot50_traffic75_serial"]
    assert any("not observed" in s for s in value.assumptions)


def test_shape_choice_is_nearest_not_fastest():
    rows = [synthetic_row(rate=900e9), synthetic_row(rate=100e9, rows=2304, cols=5120)]
    assert matvec_rate(rows, "MXFP4", 2304, 5120) == 100e9
    assert matvec_rate(rows, "Q5_K", 2304, 5120) is None


def test_probe_uses_uuid_and_synthetic_input_and_profile_roundtrips(tmp_path):
    c, rt = setup()
    del c["gpus"][0]["decode_compute"]
    calls = []

    def runner(argv, timeout, env):
        calls.append((argv, env))
        assert "--backend" in argv and "CUDA0" in argv
        assert "--model" not in argv
        assert env["CUDA_VISIBLE_DEVICES"] == rt.gpus[0].uuid
        payload = dict(
            schema="inferssd-cpuprobe/2",
            backend="CUDA0",
            ggml_features=[],
            dram=[],
            compute=[synthetic_row()],
        )
        return Completed(0, json.dumps(payload))

    binary = tmp_path / "synthetic-probe"
    binary.write_bytes(b"probe-v1")
    probes = Probes((("cpu", str(binary)),), runner)
    result = measure_decode_compute(c, probes, rt, lambda _: None)
    assert len(calls) == 3
    assert "decode_compute" not in c["gpus"][0]
    assert result["gpus"][0]["decode_compute_measurement"]["status"] == "ok"
    path = tmp_path / "machine.json"
    save(result, path)
    assert load(path) == result
    measure_decode_compute(result, probes, rt, lambda _: None)
    assert len(calls) == 3  # Cached synthetic rows are reused.
    binary.write_bytes(b"probe-v2")
    measure_decode_compute(result, probes, rt, lambda _: None)
    assert len(calls) == 6  # A rebuilt probe invalidates cached CUDA measurements.


def test_busy_gpu_is_not_profiled():
    c, rt = setup()
    del c["gpus"][0]["decode_compute"]
    rt = replace(rt, gpus=(replace(rt.gpus[0], processes=1),))
    with pytest.raises(ProbeError, match="idle"):
        measure_decode_compute(c, Probes(()), rt, lambda _: None)


def test_cpu_decode_does_not_need_prefill_measurements_and_prices_mixed_types():
    from inferssd.strategy.estimate import cpu_decode_seconds

    c, _ = setup()
    rows = [synthetic_row("MXFP4", rate=100e9), synthetic_row("Q8_0", rate=50e9)]
    c["compute"]["cpu"] = rows
    facts = MODEL.facts
    mixed_gate = replace(
        facts.gate, formats=(("MXFP4", facts.layers // 2), ("Q8_0", facts.layers // 2))
    )
    mixed = replace(facts, gate=mixed_gate)
    result, why = cpu_decode_seconds(c, mixed, 1)
    expected = (facts.up.slice_bytes() + facts.down.slice_bytes()) / 100e9
    expected += 0.5 * mixed_gate.slice_bytes("MXFP4") / 100e9
    expected += 0.5 * mixed_gate.slice_bytes("Q8_0") / 50e9
    assert result == pytest.approx(expected)
    assert not why


def test_routing_tail_survives_when_hot_set_fits():
    from inferssd.strategy.estimate import routing_hit

    assert routing_hit(0.2, 0.2, 0.8) == pytest.approx(0.8)
    assert routing_hit(0.6, 0.2, 0.8) == pytest.approx(0.9)
    assert routing_hit(1.0, 0.2, 0.8) == 1
    assert routing_hit(0.0, 0.2, 0.8) == 0
    assert routing_hit(0.6, 1.0, 1.0) == 0.6
    with pytest.raises(ValueError):
        routing_hit(0.6, 0.8, 0.2)


def test_compute_variability_widens_scenarios_without_changing_reference():
    c, rt = setup()
    before = estimate(c, rt)
    for row in c["gpus"][0]["decode_compute"]:
        row["points"][0]["us_q1"] = row["points"][0]["us"] / 2
        row["points"][0]["us_q3"] = row["points"][0]["us"] * 2
    after = estimate(c, rt)
    assert after.tokens_per_second == before.tokens_per_second
    assert after.scenario_range[0] < before.scenario_range[0]
    assert after.scenario_range[1] > before.scenario_range[1]
    for row in after.scenario_details:
        assert row["scenario_low_tps"] <= row["no_overlap_tps"] <= row["scenario_high_tps"]


def test_routing_patterns_may_be_supplied_but_cannot_be_empty():
    from inferssd.strategy.estimate import RoutingPattern

    c, rt = setup()
    one = estimate(c, rt, routing_patterns=(RoutingPattern("hypothesis", 0.3, 0.7),))
    assert len(one.scenario_details) == 1
    assert one.scenario_details[0]["name"] == "hypothesis"
    with pytest.raises(ValueError):
        estimate(c, rt, routing_patterns=())


def test_an_unmeasured_format_takes_the_nearest_measured_one_in_bits_per_weight():
    rows = [synthetic_row("Q4_K"), synthetic_row("MXFP4"), synthetic_row("Q8_0")]
    assert format_proxies(rows, {"IQ4_XS", "IQ4_NL", "Q8_0"}) == {
        "IQ4_XS": "MXFP4",  # 4.25 bits a weight, as MXFP4
        "IQ4_NL": "Q4_K",  # 4.5, as Q4_K
        "Q8_0": "Q8_0",
    }
    assert format_proxies([], {"IQ4_XS"}) == {}


def test_flash_next_is_priced_with_the_formats_it_borrows_named():
    c, rt = setup()
    flash = next(model for model in catalog.MODELS if model.quantization == "IQ4_XS")
    value = decode_estimate(
        c, flash.facts, flash.dense_read_bytes, rt, "linux", c["storage"][0]["drive_key"]
    )
    assert value.tokens_per_second is not None and value.tokens_per_second > 0
    gpu = [a for a in value.assumptions if a.startswith("GPU rate of unmeasured ")]
    assert (
        len(gpu) == 1
        and "IQ4_XS taken from MXFP4" in gpu[0]
        and "IQ4_NL taken from MXFP4" in gpu[0]
    )


def test_a_ram_only_model_reads_its_dense_weights_once_a_token():
    c, rt = setup()
    gemma = next(model for model in catalog.MODELS if model.facts is None)
    on_gpu = dense_decode_estimate(c, gemma.dense_read_bytes, rt)
    assert on_gpu.tokens_per_second == pytest.approx(500e9 / gemma.dense_read_bytes)
    low, high = on_gpu.scenario_range
    assert low < on_gpu.tokens_per_second < high
    assert "from one CUDA GPU at 500 GB/s" in on_gpu.detail
    in_ram = dense_decode_estimate(c, gemma.dense_read_bytes, replace(rt, gpus=()))
    best = max(row["Bps"] for row in c["memory"]["dram_read_Bps"])
    assert in_ram.tokens_per_second == pytest.approx(best / gemma.dense_read_bytes)
    assert dense_decode_estimate(c, 0, rt).tokens_per_second is None
