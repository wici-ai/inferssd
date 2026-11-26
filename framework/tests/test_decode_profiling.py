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
    moved_fraction,
)
from inferssd.strategy.inputs import GPUState, RuntimeFacts
from inferssd.strategy.weights import dense_weights

GIB = 1 << 30
BASE = load(Path(__file__).parent / "fixtures/machine/linux-x86-rtx5090.json")
MODEL = catalog.MODELS[0]


# Bits a weight of the formats the rows measure, as ggml stores them.
BITS = {"MXFP4": 4.25, "Q4_K": 4.5, "Q8_0": 8.5}


def synthetic_row(name="MXFP4", rate=500e9, rows=4096, cols=4096):
    return dict(
        format=name,
        threads=1,
        rows=rows,
        cols=cols,
        weight_bytes=int(rows * cols * BITS[name] / 8),
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


def test_a_host_budget_bounds_the_available_memory():
    c, rt = setup()
    total = c["memory"]["total_bytes"]
    # A budget never adds to the available memory, as in a launch; the installed memory bounds
    # both, and stands in for an unknown reading.
    busy = replace(rt, available_bytes=4 * GIB)
    assert estimate(c, replace(busy, memory_budget_bytes=32 * GIB)) == estimate(c, busy)
    for available in (None, 2 * total):
        planned = estimate(c, replace(rt, available_bytes=available)).components
        assert planned["memory_budget_bytes"] == total
    # A dense decoder plans with the same amount.
    host = replace(rt, gpus=(), available_bytes=6 * GIB)
    dense = dense_decode_estimate(c, 4 * GIB, replace(host, memory_budget_bytes=32 * GIB))
    assert dense == dense_decode_estimate(c, 4 * GIB, host)
    assert dense.components["host_resident_bytes"] == 4 * GIB
    bounded = replace(host, memory_budget_bytes=4 * GIB)
    assert dense_decode_estimate(c, 4 * GIB, bounded).tokens_per_second is None


def test_decode_pool_not_prefill_pool_prices_cpu_work():
    c, rt = setup()
    rt = replace(rt, gpus=())
    threads, _, _ = placement(c, MODEL.facts, rt, [])
    line, _ = cpu_line(c, MODEL.facts.experts, threads.decode)
    expected = MODEL.facts.experts.layers * MODEL.facts.experts.used * sum(line) / 1000
    assert estimate(c, rt).components["cpu_experts_ms"] == pytest.approx(expected)


def test_cache_coverage_lies_between_inclusive_and_disjoint_and_reserves_reduce_capacity():
    c, rt = setup()
    value = estimate(c, rt)
    comp = value.components
    misses = {row["name"]: row["disk_miss_fraction"] for row in value.scenario_details}
    assert misses["rank_disjoint"] <= comp["storage_miss_fraction"] <= misses["rank_inclusive"]
    assert misses["rank"] == comp["storage_miss_fraction"]
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


def test_fully_gpu_resident_uses_only_gpu_timings():
    from inferssd.strategy.routing import cache_slot_bytes

    c, rt = setup()
    experts = MODEL.facts.experts
    _, stride = cache_slot_bytes(experts)
    required = MODEL.dense_read_bytes + stride * experts.layers * experts.count
    c["gpus"][0]["vram_total_bytes"] = required
    full = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=required),))
    expected = estimate(c, full)
    c["compute"]["cpu"] = []
    c["memory"]["dram_read_Bps"] = []
    c["storage"] = []

    def run(runtime, **kwargs):
        return decode_estimate(
            c, MODEL.facts, MODEL.dense_read_bytes, runtime, "linux", None, **kwargs
        )

    resident = run(full)
    assert resident.tokens_per_second == expected.tokens_per_second
    assert resident.components["gpu_hit_fraction"] == 1
    assert resident.components["cpu_experts_ms"] == resident.components["storage_ms"] == 0
    for result in (
        run(replace(full, gpus=(replace(full.gpus[0], free_bytes=required - 1),))),
        run(full, observed_hits=(0.99, 0)),
    ):
        assert result.tokens_per_second is None
        assert "CPU compute measurement" in result.detail


@pytest.mark.parametrize("miss", [0, 1])
def test_host_sourced_experts_can_run_on_gpu_without_a_vram_hit(miss):
    c, rt = setup()
    value = estimate(c, rt, observed_hits=(0, miss))
    terms = value.components
    assert terms["gpu_hit_fraction"] == 0 < terms["gpu_execution_fraction"] < 1
    assert terms["gpu_experts_ms"] > 0 and terms["h2d_ms"] > 0
    assert (terms["storage_ms"] > 0) == bool(miss)
    assert value.tokens_per_second == pytest.approx(
        1000
        / sum(
            terms[key]
            for key in ("dense_ms", "cpu_experts_ms", "gpu_experts_ms", "storage_ms", "h2d_ms")
        )
    )
    c["gpus"][0]["h2d"]["pinned"] = []
    fallback = estimate(c, rt, observed_hits=(0, miss)).components
    assert fallback["gpu_execution_fraction"] == fallback["h2d_ms"] == 0


@pytest.mark.parametrize(("link_gbps", "moves"), [(10, 0), (100, 3)])
def test_observed_ram_hits_choose_whole_experts_per_layer(link_gbps, moves):
    c, rt = setup()
    for row in c["gpus"][0]["h2d"]["pinned"]:
        row["Bps"] = link_gbps * 1e9
    terms = estimate(c, rt, observed_hits=(0, 0)).components
    experts = MODEL.facts.experts
    assert experts.used == 6
    assert terms["gpu_hit_fraction"] == terms["storage_ms"] == 0
    assert terms["gpu_execution_fraction"] == moves / experts.used
    payload = experts.routed_bytes / (experts.layers * experts.count)
    assert terms["h2d_ms"] == pytest.approx(experts.layers * moves * payload / (link_gbps * 1e6))


def test_dispatch_respects_slots_and_preserves_mean_hits():
    # One slot: empty layers can move one expert; a layer with a resident cannot.
    assert moved_fraction(6, 0, 1, 1.0, 0.1, 0.2) == 1 / 6
    assert moved_fraction(6, 1 / 6, 1, 1.0, 0.1, 0.2) == 0
    assert moved_fraction(6, 0.5 / 6, 1, 1.0, 0.1, 0.2) == 0.5 / 6
    # Moving the one host expert would tie the CPU time.
    assert moved_fraction(1, 0, 1, 3.0, 1.0, 2.0) == 0


def test_scenarios_vary_cache_overlap_and_timing_not_the_routing_prior():
    c, rt = setup()
    value = estimate(c, rt)
    assert value.tokens_per_second == value.scenarios["rank_serial"]
    assert set(value.scenarios) == {"rank_serial", "rank_inclusive_serial", "rank_disjoint_serial"}
    for row in value.scenario_details:
        assert row["no_overlap_tps"] <= row["half_io_overlap_tps"] <= row["scenario_high_tps"]
    assert any("not observed" in s for s in value.assumptions)


def test_shape_choice_is_nearest_not_fastest():
    rows = [synthetic_row(rate=900e9), synthetic_row(rate=100e9, rows=2304, cols=5120)]
    assert matvec_rate(rows, "MXFP4", 2304, 5120) == 100e9
    assert matvec_rate(rows, "Q5_K", 2304, 5120) is None


def test_small_expert_preserves_only_an_identifiable_fixed_call_cost():
    # Physical microbench with 10 us fixed overhead and a 1 GB/s transfer term.
    def timed(n):
        row = synthetic_row(rows=n, cols=n)
        size = row["weight_bytes"]
        us = 10 + size / 1000
        row["points"][0].update(us=us, weight_Bps=size * 1e6 / us)
        return row

    small, large = timed(256), timed(512)
    target = small["weight_bytes"] // 4
    rate = matvec_rate([small, large], "MXFP4", 128, 128, target_bytes=target)
    assert target / rate * 1e6 == pytest.approx(10 + target / 1000)
    assert (
        matvec_rate([small, large], "MXFP4", 256, 256, target_bytes=small["weight_bytes"])
        == (small["points"][0]["weight_Bps"])
    )
    # A lone size, or a larger matrix timed faster, cannot identify a nonnegative line.
    large["points"][0]["us"] = small["points"][0]["us"] / 2
    for rows in ([small], [small, large]):
        assert (
            matvec_rate(rows, "MXFP4", 128, 128, target_bytes=target)
            == small["points"][0]["weight_Bps"]
        )


def test_observed_fractions_replace_only_the_reference_routing():
    c, rt = setup()
    prior = estimate(c, rt)
    observed = estimate(c, rt, observed_hits=(0.3, 0.2))
    assert observed.reference_scenario == "observed_serial"
    assert observed.tokens_per_second == observed.scenarios["observed_serial"]
    assert all(observed.scenarios[k] == v for k, v in prior.scenarios.items())
    assert observed.components["gpu_hit_fraction"] == 0.3
    assert observed.components["storage_miss_fraction"] == 0.2
    assert estimate(c, rt, observed_hits=(0, 0)).components["storage_ms"] == 0
    assert estimate(c, replace(rt, gpus=()), observed_hits=(0.1, 0.2)).tokens_per_second is None
    for invalid in ((-1, 0), (0, 1.1), (0.8, 0.3), (0, float("nan")), (1,)):
        with pytest.raises(ValueError, match="fractions"):
            estimate(c, rt, observed_hits=invalid)


def test_probe_uses_uuid_and_synthetic_input_and_profile_roundtrips(tmp_path):
    c, rt = setup()
    del c["gpus"][0]["decode_compute"]
    calls = []

    def runner(argv, timeout, env):
        calls.append((argv, env))
        assert "--backend" in argv and "CUDA0" in argv
        assert "Q4_0" in argv[argv.index("--formats") + 1].split(",")
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
    result["gpus"][0]["decode_compute_measurement"]["method"] = "previous formats without Q4_0"
    result = measure_decode_compute(result, probes, rt, lambda _: None)
    measure_decode_compute(result, probes, rt, lambda _: None)
    assert len(calls) == 6  # An older measurement plan is upgraded once.
    binary.write_bytes(b"probe-v2")
    measure_decode_compute(result, probes, rt, lambda _: None)
    assert len(calls) == 9  # A rebuilt probe invalidates cached CUDA measurements.


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
    facts = MODEL.facts.experts
    q8_0 = facts.gate.rows * facts.gate.cols // 32 * 34
    mixed_gate = replace(
        facts.gate,
        formats=(
            ("MXFP4", facts.layers // 2, facts.gate.slice_bytes("MXFP4")),
            ("Q8_0", facts.layers // 2, q8_0),
        ),
    )
    mixed = replace(facts, gate=mixed_gate)
    result, why = cpu_decode_seconds(c, mixed, 1)
    expected = (facts.up.slice_bytes() + facts.down.slice_bytes()) / 100e9
    expected += 0.5 * mixed_gate.slice_bytes("MXFP4") / 100e9
    expected += 0.5 * mixed_gate.slice_bytes("Q8_0") / 50e9
    assert result == pytest.approx(expected)
    assert not why


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


def test_an_unmeasured_format_takes_the_nearest_measured_one_in_bits_per_weight():
    rows = [synthetic_row("Q4_K"), synthetic_row("MXFP4"), synthetic_row("Q8_0")]
    # Each wanted format with its bits a weight; a row's are its weight bytes over its matrix.
    assert format_proxies(rows, {"IQ4_XS": 4.25, "IQ4_NL": 4.5, "Q8_0": 8.5}) == {
        "IQ4_XS": "MXFP4",  # 4.25 bits a weight, as MXFP4
        "IQ4_NL": "Q4_K",  # 4.5, as Q4_K
        "Q8_0": "Q8_0",
    }
    assert format_proxies([], {"IQ4_XS": 4.25}) == {}


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


def test_a_dense_model_reads_its_dense_weights_once_a_token():
    c, rt = setup()
    gemma = next(model for model in catalog.MODELS if model.facts.experts is None)
    on_gpu = dense_decode_estimate(c, gemma.dense_read_bytes, rt)
    assert on_gpu.tokens_per_second == pytest.approx(500e9 / gemma.dense_read_bytes)
    low, high = on_gpu.scenario_range
    assert low < on_gpu.tokens_per_second < high
    assert "from one GPU at 500 GB/s" in on_gpu.detail
    in_ram = dense_decode_estimate(c, gemma.dense_read_bytes, replace(rt, gpus=()))
    best = max(row["Bps"] for row in c["memory"]["dram_read_Bps"])
    assert in_ram.tokens_per_second == pytest.approx(best / gemma.dense_read_bytes)
    assert dense_decode_estimate(c, 0, rt).tokens_per_second is None


def test_dense_gpu_streaming_charges_link_for_every_nonresident_byte_and_disk_only_for_spill():
    c, rt = setup()
    rt = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=3 * GIB),), memory_budget_bytes=3 * GIB)
    drive = c["storage"][0]
    for row in drive["random_read"]:
        # Only attainable queues enter the forecast. At QD8 the measured request sizes
        # span 5–20 GB/s, so the size-agnostic central rate is 10 GB/s.
        row["Bps"] = {1 << 20: 5e9, 4 << 20: 12e9, 16 << 20: 20e9}[row["bytes"]]
        if row["qd"] != 8:
            row["Bps"] = 100e9
    for row in c["gpus"][0]["h2d"]["pinned"]:
        row["Bps"] = 30e9

    def estimate(key, budget):
        return dense_decode_estimate(
            c,
            8 * GIB,
            replace(rt, memory_budget_bytes=budget),
            drive_key=key,
            host_work_bytes=GIB // 2,
            gpu_work_bytes=GIB,
        )

    # 3 GiB free VRAM less 1 GiB working memory, 3 GiB RAM less 512 MiB each for
    # the engine and the caller. The other four GiB must pass through both disk and H2D.
    result = estimate(drive["drive_key"], 3 * GIB)
    assert result.components["dense_resident_bytes"] == 2 * GIB
    assert result.components["dense_pinned_bytes"] == 2 * GIB
    assert result.components["dense_drive_bytes"] == 4 * GIB
    assert result.components["h2d_ms"] == pytest.approx(1e3 * 6 * GIB / 30e9)
    assert result.tokens_per_second == pytest.approx(10e9 / (4 * GIB))
    assert result.reference_scenario == "dense_pipeline"
    assert result.scenarios["dense_serial"] < result.tokens_per_second
    assert estimate("unknown", 3 * GIB).tokens_per_second is None
    from_ram = estimate("unknown", 16 * GIB)
    assert from_ram.tokens_per_second == pytest.approx(30e9 / (6 * GIB))
    assert from_ram.components["dense_drive_bytes"] == 0


def test_dense_layer_formats_use_their_own_matvec_measurements():
    c, rt = setup()
    gemma = next(m for m in catalog.MODELS if m.facts.experts is None)
    row = synthetic_row("Q4_K", rate=100e9)
    row["format"] = "Q4_0"
    c["gpus"][0]["decode_compute"].append(row)
    result = dense_decode_estimate(c, gemma.dense_read_bytes, rt, model=gemma.facts)
    q4 = dict(gemma.facts.layer_format_bytes)["Q4_0"]
    assert result.tokens_per_second == pytest.approx(
        1 / (q4 / 100e9 + (gemma.dense_read_bytes - q4) / 500e9)
    )
    assert any(
        "Q8_0 matvec proxy" in note and "outside catalogued layers" in note
        for note in result.assumptions
    )


def test_local_dense_facts_price_output_head_shape_and_exclude_lookup_from_streamable_weights():
    c, rt = setup()
    c["gpus"][0]["decode_compute"] = [
        synthetic_row("Q4_K", 100e9, 1024, 2048),
        synthetic_row("Q4_K", 150e9, 4096, 4096),
        synthetic_row("Q8_0", 200e9, 8192, 1024),
        synthetic_row("Q8_0", 80e9, 4096, 4096),
    ]
    # Large gathered table and explicit output head with a different GGUF type.
    weights = dense_weights(
        {
            "tensors": [
                {
                    "name": "blk.0.engram_embd.weight",
                    "type": "I8",
                    "shape": [1024, 300_000],
                    "bytes": 300_000_000,
                },
                {
                    "name": "blk.0.ffn_up.weight",
                    "type": "Q4_K",
                    "shape": [2048, 1024],
                    "bytes": 1024**2,
                },
                {
                    "name": "output.weight",
                    "type": "Q8_0",
                    "shape": [1024, 8192],
                    "bytes": 8 * 1024**2,
                },
            ]
        }
    )
    total = 9 * 1024**2
    result = dense_decode_estimate(c, total, rt, weights=weights)
    assert result.tokens_per_second == pytest.approx(1 / (1024**2 / 100e9 + 8 * 1024**2 / 200e9))
    limited = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=7 * 1024**2),))
    assert (
        dense_decode_estimate(c, total, limited, weights=weights).reference_scenario
        == "dense_host_serial"
    )


def test_dense_placement_distinguishes_unknown_free_vram_and_the_busy_gpu_reserve():
    c, rt = setup()
    gpu = replace(rt.gpus[0], free_bytes=4 * GIB)
    busy = replace(rt, gpus=(replace(gpu, processes=1),))
    result = dense_decode_estimate(c, 4 * GIB, busy)
    assert result.components["dense_pinned_bytes"] == GIB
    unknown = dense_decode_estimate(c, 4 * GIB, replace(rt, gpus=(replace(gpu, free_bytes=None),)))
    assert unknown.tokens_per_second > 0 and unknown.reference_scenario == "dense_host_serial"
    assert any("Free VRAM is unknown" in note for note in unknown.assumptions)


def test_dense_cpu_paging_prices_minimum_spill_and_cache_thrashing():
    c, rt = setup()
    rt = replace(rt, gpus=(), memory_budget_bytes=2 * GIB)
    drive = c["storage"][0]
    for row in drive["random_read"]:
        row["Bps"] = 1e9 if row["qd"] == 1 else 20e9
    result = dense_decode_estimate(c, 4 * GIB, rt, drive_key=drive["drive_key"])
    assert result.components["host_resident_bytes"] == 1.5 * GIB
    assert result.components["dense_drive_bytes"] == 2.5 * GIB
    assert result.components["storage_ms"] == pytest.approx(1000 * 2.5 * GIB / 1e9)
    assert result.scenarios["dense_host_full_reread"] < result.tokens_per_second
    assert result.scenario_range[0] <= result.tokens_per_second <= result.scenario_range[1]
    resident = dense_decode_estimate(c, 4 * GIB, replace(rt, memory_budget_bytes=6 * GIB))
    assert resident.tokens_per_second > result.tokens_per_second
    assert dense_decode_estimate(c, 4 * GIB, rt).tokens_per_second is None


def test_graph_experts_cannot_fit_in_vram_or_ram_using_only_top_k_bytes():
    c, rt = setup()
    weights = dense_weights(
        {
            "architecture": "test",
            "metadata": {"test.expert_count": 64, "test.expert_used_count": 1},
            "tensors": [
                {
                    "name": "blk.0.ffn_gate_up_exps.weight",
                    "shape": [4096, 4096, 64],
                    "type": "Q8_0",
                    "bytes": GIB,
                },
            ],
        }
    )
    # A token reads just 16 MiB. The native graph still requires a 1 GiB expert tensor.
    small = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=GIB // 2),), memory_budget_bytes=GIB)
    with_disk = dense_decode_estimate(
        c, weights.active_bytes, small, weights=weights, drive_key=c["storage"][0]["drive_key"]
    )
    assert with_disk.reference_scenario == "dense_host_serial"
    assert with_disk.components["dense_placement_bytes"] == GIB
    assert with_disk.components["host_resident_bytes"] == GIB // 2
    assert with_disk.components["dense_drive_bytes"] == GIB // 128
    assert with_disk.scenarios["dense_host_hot_experts"] > with_disk.tokens_per_second
    assert (
        dense_decode_estimate(c, weights.active_bytes, small, weights=weights).tokens_per_second
        is None
    )
    fits = dense_decode_estimate(c, weights.active_bytes, rt, weights=weights)
    assert fits.reference_scenario == "dense_pipeline"
    assert fits.components["dense_resident_bytes"] == GIB
    assert fits.components["gpu_compute_ms"] == pytest.approx(1000 * (GIB // 64) / 500e9)


def test_mixed_ssd_and_graph_moe_reserves_the_graph_store_before_allocating_ssd_slots():
    c, rt = setup()
    weights = dense_weights(
        {
            "architecture": "mixed",
            "metadata": {"mixed.expert_count": 16, "mixed.expert_used_count": 2},
            "tensors": [
                {
                    "name": "blk.1.ffn_up_exps.weight",
                    "shape": [4096, 4096, 16],
                    "type": "Q8_0",
                    "bytes": 16 * GIB,
                }
            ],
        }
    )
    key = c["storage"][0]["drive_key"]
    partial = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=4 * GIB),))
    value = decode_estimate(
        c, MODEL.facts, weights.active_bytes, partial, "linux", key, graph_weights=weights
    )
    assert value.components["gpu_cache_capacity_bytes"] == 0
    assert value.components["graph_expert_active_bytes"] == 2 * GIB
    assert value.components["host_cache_capacity_bytes"] == rt.available_bytes - GIB // 2 - 16 * GIB
    value = decode_estimate(
        c, MODEL.facts, weights.active_bytes, rt, "linux", key, graph_weights=weights
    )
    assert value.components["gpu_cache_capacity_bytes"] == 30 * GIB - 16 * GIB


def test_dense_host_can_use_a_measured_cpu_matvec_without_dram():
    c, rt = setup()
    rt = replace(rt, gpus=())
    c["memory"]["dram_read_Bps"] = []
    c["compute"]["cpu"] = [synthetic_row("Q4_K", rate=15e9)]
    result = dense_decode_estimate(c, GIB, rt)
    assert result.tokens_per_second == pytest.approx(15e9 / GIB)
    assert result.reference_scenario == "dense_host_serial"
    assert any("Q4_K matvec proxy" in note for note in result.assumptions)
    c["compute"]["cpu"].clear()
    assert dense_decode_estimate(c, GIB, rt).tokens_per_second is None
