# SPDX-License-Identifier: Apache-2.0
"""General routing estimates, physical capacity bounds and sample-independent predictions."""

from dataclasses import replace
from pathlib import Path

import pytest

from inferssd.app import catalog
from inferssd.machine.store import load
from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import RuntimeFacts
from inferssd.strategy.routing import cache_slot_bytes, hit_at_slots, rank_hit_curve


@pytest.mark.parametrize(("count", "used"), [(1, 1), (4, 4), (16, 8), (384, 6)])
def test_each_token_routes_to_distinct_experts(count, used):
    hits = rank_hit_curve(count, used)
    masses = [b - a for a, b in zip(hits, hits[1:])]
    assert hits[0] == 0 and hits[-1] == 1
    assert sum(masses) == pytest.approx(1)
    assert all(0 < m <= 1 / used + 1e-12 for m in masses)
    assert all(a >= b - 1e-12 for a, b in zip(masses, masses[1:]))
    assert hit_at_slots(hits, 0.5) == pytest.approx(masses[0] / 2)
    if count == used:
        assert all(m == pytest.approx(1 / used) for m in masses)


def test_estimate_is_invariant_to_checkpoint_identity():
    model = catalog.MODELS[0]
    profile = load(Path(__file__).parent / "fixtures/machine/macos-m5.json")
    runtime = RuntimeFacts(tuple(r["id"] for r in profile["cpu"]["logical"]), (), 12 << 30)
    drive_key = profile["storage"][0]["drive_key"]
    values = [
        decode_estimate(profile, facts, model.dense_read_bytes, runtime, "darwin", drive_key)
        for facts in (
            model.facts,
            replace(model.facts, paths=("unseen.gguf",), architecture="unseen"),
        )
    ]
    before, renamed = values
    assert before == renamed
    assert before.reference_scenario == "rank_serial"
    assert before.tokens_per_second == before.scenarios["rank_serial"]
    assert (
        before.scenarios["rank_inclusive_serial"]
        <= before.tokens_per_second
        <= before.scenarios["rank_disjoint_serial"]
    )
    assert before.tokens_per_second == pytest.approx(
        1000
        / sum(
            before.components[k]
            for k in ("dense_ms", "cpu_experts_ms", "gpu_experts_ms", "storage_ms")
        )
    )


def test_full_host_residency_requires_no_drive_benchmark():
    model = catalog.MODELS[0]
    profile = load(Path(__file__).parent / "fixtures/machine/macos-m5.json")
    experts = model.facts.experts
    host_slot, _ = cache_slot_bytes(experts)
    full = model.dense_read_bytes + experts.layers * experts.count * host_slot + (512 << 20)
    profile["memory"]["total_bytes"] = full
    runtime = RuntimeFacts(tuple(r["id"] for r in profile["cpu"]["logical"]), (), full)
    result = decode_estimate(profile, model.facts, model.dense_read_bytes, runtime, "darwin", None)
    assert result.tokens_per_second > 0
    assert result.components["storage_miss_fraction"] == result.components["storage_ms"] == 0
    # One fewer host slot requires a real drive rate.
    missing = decode_estimate(
        profile,
        model.facts,
        model.dense_read_bytes,
        replace(runtime, available_bytes=full - host_slot),
        "darwin",
        None,
    )
    assert missing.tokens_per_second is None
