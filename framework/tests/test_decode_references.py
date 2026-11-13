# SPDX-License-Identifier: Apache-2.0
"""Offline model integration: frozen predictions, domain guards, and no generation."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from inferssd.app import catalog
from inferssd.machine import command
from inferssd.machine import reference as local_reference
from inferssd.machine.store import load
from inferssd.strategy import reference
from inferssd.strategy.estimate import Estimate
from inferssd.strategy.inputs import GPUState, RuntimeFacts

GIB = 1 << 30
BASE = load(Path(__file__).parent / "fixtures/machine/linux-x86-rtx5090.json")


def setup(r):
    c = copy.deepcopy(BASE)
    h = r.hardware
    c["identity"]["hard"]["cpu_brand"] = h["cpu_brand"]
    c["identity"]["hard"]["arch"] = h["arch"]
    c["memory"]["total_bytes"] = 256 * GIB
    c["memory"]["unified"] = False
    c["gpus"][0]["name"] = h["gpu_name"]
    c["gpus"][0]["vram_total_bytes"] = 32 * GIB
    c["gpus"][0]["integrated"] = False
    rt = RuntimeFacts(
        tuple(range(h["allowed_cpus"])),
        (GPUState(c["gpus"][0]["uuid"], h["minimum_gpu_free_bytes"], 0),),
        128 * GIB,
        32 * GIB,
        ("CUDA",),
    )
    return c, rt


def estimate(r, monkeypatch, budget=32, rate_scale=1, **changes):
    c, rt = setup(r)
    rt = replace(rt, memory_budget_bytes=budget * GIB, **changes)
    monkeypatch.setattr(
        reference, "drive_rate", lambda *_: (r.parameters["drive_Bps"] * rate_scale / 1e9, "")
    )
    return reference.apply_reference(
        Estimate(0.5, "analytical"), c, r.facts, r.dense_read_bytes, rt, "linux", "drive"
    )


@pytest.mark.parametrize(
    "index,expected",
    [
        (0, (1.0588677912567543, 1.622898991380254)),
        (1, (39.85827945053922, 39.85827945053922)),
    ],
)
def test_pretrained_parameters_reproduce_accepted_full_decode(index, expected, monkeypatch):
    r = reference.references()[index]
    for budget, value in zip((32, 96), expected):
        result = estimate(r, monkeypatch, budget)
        assert result.tokens_per_second == pytest.approx(value, rel=1e-8)
        assert result.reference_scenario == "pretrained_reference"
        assert result.scenario_range[0] <= value <= result.scenario_range[1]
        assert result.reference_model["provenance"]["calibration_memory_gib"] == 32


def test_memory_replay_and_drive_rate_are_inputs_not_a_constant_tps(monkeypatch):
    r = reference.references()[0]
    before = copy.deepcopy(r.parameters)
    small, middle, large = [estimate(r, monkeypatch, b) for b in (32, 64, 96)]
    assert small.tokens_per_second < middle.tokens_per_second < large.tokens_per_second
    assert middle.reference_model["capacity_interpolated"]
    assert not small.reference_model["capacity_interpolated"]
    slow = estimate(r, monkeypatch, 32, rate_scale=0.5)
    assert slow.tokens_per_second < small.tokens_per_second
    assert slow.components["exposed_io_wait_ms"] == pytest.approx(
        2 * small.components["exposed_io_wait_ms"]
    )
    assert r.parameters == before


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "cpus",
        "gpu",
        "busy",
        "free",
        "unified",
        "backend",
        "small_ram",
        "large_ram",
        "reserve",
    ],
)
def test_incompatible_hardware_and_configuration_keep_analytical_estimate(case, monkeypatch):
    r = reference.references()[0]
    c, rt = setup(r)
    options = {}
    if case == "cpu":
        c["identity"]["hard"]["cpu_brand"] = "another CPU"
    if case == "cpus":
        rt = replace(rt, cpus=(0, 1))
    if case == "gpu":
        c["gpus"][0]["name"] = "another GPU"
    if case == "busy":
        rt = replace(rt, gpus=(replace(rt.gpus[0], processes=1),))
    if case == "free":
        rt = replace(rt, gpus=(replace(rt.gpus[0], free_bytes=GIB),))
    if case == "unified":
        c["memory"]["unified"] = True
    if case == "backend":
        rt = replace(rt, gpu_backends=("Metal",))
    if case == "small_ram":
        rt = replace(rt, memory_budget_bytes=16 * GIB)
    if case == "large_ram":
        rt = replace(rt, memory_budget_bytes=128 * GIB)
    if case == "reserve":
        options["gpu_work_bytes"] = 4 * GIB
    monkeypatch.setattr(reference, "drive_rate", lambda *_: (10.0, ""))
    result = reference.apply_reference(
        Estimate(0.5, "analytical"), c, r.facts, r.dense_read_bytes, rt, "linux", "drive", **options
    )
    assert result.tokens_per_second == 0.5
    assert result.reference_model is None
    assert "not applied" in result.assumptions[-1]


def test_other_quantization_does_not_inherit_muse_calibration():
    r = reference.references()[0]
    assert reference.match(r.facts, r.dense_read_bytes) is r
    assert (
        reference.match(replace(r.facts, dense_bytes=r.facts.dense_bytes + 1), r.dense_read_bytes)
        is None
    )
    muse = next(m for m in catalog.MODELS if "Muse" in m.repo)
    assert reference.match(muse.facts, muse.dense_read_bytes) is None


def test_local_variant_uses_headers_not_filename_and_is_not_duplicated(tmp_path, monkeypatch):
    r = reference.references()[0]
    paths = [tmp_path / "renamed.gguf", tmp_path / "renamed-duplicate.gguf"]
    for p in paths:
        p.touch()
    lookup_bytes = r.facts.dense_bytes - r.dense_read_bytes
    tensors = [SimpleNamespace(nbytes=lookup_bytes, name="token_embd.weight")]
    monkeypatch.setattr(
        local_reference, "read_model", lambda p: [SimpleNamespace(path=p, tensors=tensors)]
    )
    monkeypatch.setattr(local_reference, "facts_from_headers", lambda _: r.facts)
    models = local_reference.estimate_models(paths)
    assert len(models) == len(catalog.MODELS) + 1
    assert models[-1].facts == r.facts
    assert models[-1].quantization == "Q4_K_S"
    assert models[-1].revision == ""


def test_profile_reports_pretrained_result_without_loading_an_engine(monkeypatch):
    r = reference.references()[1]
    c, rt = setup(r)
    monkeypatch.setattr(reference, "drive_rate", lambda *_: (r.parameters["drive_Bps"] / 1e9, ""))
    monkeypatch.setattr(command, "decode_estimate", lambda *_a, **_k: Estimate(0.5, "analytical"))

    def forbidden(*_a, **_k):
        raise AssertionError("prof reference estimation must not run a process or generate tokens")

    monkeypatch.setattr(command, "run_process", forbidden)
    monkeypatch.setattr(command, "measure_decode_compute", forbidden)
    model = next(m for m in catalog.MODELS if m.facts == r.facts or "Qwen3.6" in m.repo)
    rows = command.estimates(c, "drive", "linux", lambda: rt, models=(model,))
    report = command.report(c, Path("machine.json"), "drive", rows)
    item = report["estimates"][0]
    assert item["reference_scenario"] == "pretrained_reference"
    assert item["decode_tokens_per_second"] == pytest.approx(39.85827945053922)
    assert "pretrained reference" in command.estimate_lines(rows)[0]
    assert "uniform serial reference" not in command.estimate_lines(rows)[0]
    assert item["reference_model"]["analytical_reference_tokens_per_second"] == 0.5
