# SPDX-License-Identifier: Apache-2.0
"""Decode speed estimates of the supported models from a machine profile."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from inferssd.app import catalog, hub
from inferssd.machine.characteristics import validate
from inferssd.machine.store import load
from inferssd.strategy.derive import placement
from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import RuntimeFacts

MAC = load(Path(__file__).parent / "fixtures/machine/macos-m5.json")  # M5, 16 GiB, Apple SSD
DRIVE = MAC["storage"][0]["drive_key"]
GIB = 1 << 30


def runtime(available):
    return RuntimeFacts(tuple(row["id"] for row in MAC["cpu"]["logical"]), (), available)


def estimate(model, characteristics=MAC, available=9 * GIB, drive=DRIVE):
    return decode_estimate(
        characteristics, model.facts, model.dense_read_bytes, runtime(available), "darwin", drive
    )


def ssd_models():
    """The supported models with an expert store; a RAM-only model has no expert facts."""
    return [model for model in catalog.MODELS if model.facts is not None]


def unmeasured(model):
    """MODEL's expert formats that the Mac profile has no CPU compute measurement of."""
    measured = {row["format"].upper() for row in MAC["compute"]["cpu"] if not row.get("cpus")}
    return sorted({role.format.upper() for role in model.facts.roles} - measured)


def test_the_catalog_is_the_download_list_and_its_facts_add_up():
    assert hub.MODELS == {model.repo: model.quantization for model in catalog.MODELS}
    for model in catalog.MODELS:
        facts = model.facts
        if facts is None:  # RAM-only: no expert store; a token reads its dense weights
            assert 0 < model.lookup_bytes < model.weight_bytes
            assert model.dense_read_bytes == model.weight_bytes - model.lookup_bytes
            continue
        assert model.weight_bytes == 0
        assert 0 < model.dense_read_bytes < facts.dense_bytes
        # Each type's total is exactly its roles' layers of that type, over every expert.
        per_type: Counter[str] = Counter()
        for role in facts.roles:
            assert sum(layers for _, layers in role.formats) == facts.layers
            for name, layers in role.formats:
                per_type[name] += layers * facts.experts * role.slice_bytes(name)
        assert dict(facts.format_bytes) == dict(per_type)
        assert facts.expert_bytes <= facts.expert_read_bytes
        assert facts.mean_expert_read_bytes <= facts.expert_read_bytes


def test_every_ssd_model_is_priced_on_a_mac_and_names_the_formats_it_borrows():
    # The CPU probe measures a fixed set of ggml types. A model whose experts use another,
    # such as Flash Next's IQ4_XS and IQ4_NL, is priced at the measured rate of the format
    # nearest in bits per weight, a default its assumptions name.
    for model in ssd_models():
        value = estimate(model)
        assert value.tokens_per_second is not None and 0.05 < value.tokens_per_second < 100
        assert " ms a token: dense " in value.detail
        borrowed = [a for a in value.assumptions if a.startswith("CPU rate of unmeasured ")]
        missing = unmeasured(model)
        if missing:
            assert len(borrowed) == 1 and all(
                f"{name} taken from " in borrowed[0] for name in missing
            )
        else:
            assert not borrowed


def test_more_memory_and_a_faster_drive_are_faster():
    faster = copy.deepcopy(MAC)
    for row in faster["storage"][0]["random_read"]:
        row["Bps"] *= 2
    for model in ssd_models():
        base = estimate(model).tokens_per_second
        assert estimate(model, available=40 * GIB).tokens_per_second > base
        assert estimate(model, characteristics=faster).tokens_per_second > base


def test_no_estimate_without_the_models_drive():
    value = estimate(catalog.MODELS[0], drive=None)
    assert value.tokens_per_second is None
    assert value.detail == "the drive holding the model is not known"


def test_qwen_estimate_uses_a_measured_rate_below_an_unreachable_knee():
    c = load(Path(__file__).parent / "fixtures/machine/windows-x64-igpu.json")
    storage = c["storage"][0]
    storage["qd_cap"] = None
    storage["knee"] = [{"bytes": 1 << 20, "qd": 32}]
    template = next(row for row in storage["random_read"] if row["bytes"] == 1 << 20)
    storage["random_read"] = [
        dict(template, qd=qd, Bps=rate) for qd, rate in [(8, 6e9), (16, 7e9), (32, 9e9)]
    ]
    model = catalog.MODELS[2]
    facts = RuntimeFacts(tuple(row["id"] for row in c["cpu"]["logical"]), (), 9 * GIB)
    value = decode_estimate(
        c, model.facts, model.dense_read_bytes, facts, "windows", storage["drive_key"]
    )
    assert value.tokens_per_second is not None and value.tokens_per_second > 0
    assert "from the drive at 7.00 GB/s" in value.detail
    storage["knee"][0]["qd"] = 8
    assert value == decode_estimate(
        c, model.facts, model.dense_read_bytes, facts, "windows", storage["drive_key"]
    )


def test_dense_weights_that_do_not_fit_are_read_from_the_drive():
    muse = catalog.MODELS[1]  # 21 GB of dense weights, more than the 16 GiB Mac has
    spilled = estimate(muse, available=9 * GIB)
    larger = copy.deepcopy(MAC)
    larger["memory"]["total_bytes"] = 64 * GIB
    fits = estimate(muse, characteristics=larger, available=64 * GIB)
    assert "GB of it from the drive" in spilled.detail and "of it from the drive" not in fits.detail
    assert spilled.tokens_per_second < fits.tokens_per_second


def test_no_available_memory_is_none_rather_than_unknown():
    model = catalog.MODELS[0]
    assert estimate(model, available=0).tokens_per_second < estimate(model).tokens_per_second
    assert "(a 0.0 GB cache of" in estimate(model, available=0).detail


def test_a_profile_of_other_cpus_gives_a_reason_not_an_error():
    value = decode_estimate(
        MAC, catalog.MODELS[0].facts, 1, RuntimeFacts((99,), (), 9 * GIB), "darwin", DRIVE
    )
    assert value.tokens_per_second is None and "none of the CPUs" in value.detail


def pinned_profile(model):
    """A recorded hybrid CPU profile with a faster synthetic performance-cluster rate."""
    c = load(Path(__file__).parent / "fixtures/machine/windows-x64-igpu.json")
    # Isolate compute pricing: the expert store fits in RAM and dense bytes are zero.
    c["memory"]["total_bytes"] = 1024 * GIB
    for row in c["compute"]["cpu"]:
        if row.get("cpus") == [0, 1, 2, 3] and row["format"] == model.dominant_format:
            point = next(p for p in row["points"] if p["tokens"] == 1)
            point["weight_Bps"] = 200e9
            point["us"] = row["weight_bytes"] / point["weight_Bps"] * 1e6
            point["us_q1"], point["us_q3"] = point["us"] / 2, point["us"] * 2
    facts = RuntimeFacts(tuple(range(8)), (), c["memory"]["total_bytes"])
    return c, facts


def pinned_estimate(c, model, facts):
    return decode_estimate(validate(c), model, 0, facts, "windows", c["storage"][0]["drive_key"])


@pytest.mark.parametrize("timing", ["median", "q1", "q3"])
def test_decode_estimate_prices_the_selected_pinned_pool(timing):
    model = catalog.MODELS[0].facts
    c, facts = pinned_profile(model)
    threads, _, _ = placement(c, model, facts, [])
    assert threads.decode == 4 and threads.decode_cpus == (0, 1, 2, 3)

    estimate = pinned_estimate(c, model, facts)
    seconds = model.layers * model.experts_used * sum(r.slice_bytes() for r in model.roles) / 200e9
    if timing == "median":
        assert estimate.components["cpu_experts_ms"] == pytest.approx(seconds * 1e3)
        assert estimate.tokens_per_second == pytest.approx(1 / seconds)
    elif timing == "q1":
        assert estimate.scenario_range[1] == pytest.approx(2 / seconds)
    else:
        assert estimate.scenario_range[0] == pytest.approx(0.5 / seconds)


def test_decode_estimate_matches_the_cpu_ids_of_a_selected_cluster_prefix():
    model = catalog.MODELS[0].facts
    c, facts = pinned_profile(model)
    template = next(
        row
        for row in c["compute"]["cpu"]
        if row.get("cpus") == [0, 1, 2, 3] and row["format"] == model.dominant_format
    )
    for cpus, rate in [([4, 5], 900e9), ([0, 1], 300e9)]:
        row = copy.deepcopy(template)
        row["cpus"], row["threads"] = cpus, len(cpus)
        point = row["points"][0]
        point["weight_Bps"] = rate
        point["us"] = point["us_q1"] = point["us_q3"] = row["weight_bytes"] / rate * 1e6
        c["compute"]["cpu"].append(row)
    threads, _, _ = placement(c, model, facts, [])
    assert threads.decode == 2 and threads.decode_cpus == (0, 1)

    estimate = pinned_estimate(c, model, facts)
    seconds = model.layers * model.experts_used * sum(r.slice_bytes() for r in model.roles) / 300e9
    assert estimate.components["cpu_experts_ms"] == pytest.approx(seconds * 1e3)


def test_format_proxy_notes_use_the_selected_cpu_pool():
    model = catalog.MODELS[1].facts  # Q4_K gate/up and Q5_K down.
    c, facts = pinned_profile(model)
    c["compute"]["cpu"] = [
        row
        for row in c["compute"]["cpu"]
        if not (row.get("cpus") == [0, 1, 2, 3] and row["format"] == "Q5_K")
    ]
    threads, _, _ = placement(c, model, facts, [])
    assert threads.decode_cpus == (0, 1, 2, 3)
    estimate = pinned_estimate(c, model, facts)
    assert any("Q5_K taken from Q4_K" in note for note in estimate.assumptions)


def test_unpinned_estimate_ignores_faster_bound_measurements():
    model = catalog.MODELS[0].facts
    c, facts = pinned_profile(model)
    c["cpu"]["affinity_supported"] = False
    threads, _, _ = placement(c, model, facts, [])
    assert threads.decode == 8 and not threads.decode_cpus
    row = next(
        row
        for row in c["compute"]["cpu"]
        if not row.get("cpus") and row["threads"] == 8 and row["format"] == model.dominant_format
    )
    rate = next(point["weight_Bps"] for point in row["points"] if point["tokens"] == 1)
    seconds = model.layers * model.experts_used * sum(r.slice_bytes() for r in model.roles) / rate
    estimate = pinned_estimate(c, model, facts)
    assert estimate.components["cpu_experts_ms"] == pytest.approx(seconds * 1e3)
