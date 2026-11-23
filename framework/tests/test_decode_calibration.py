# SPDX-License-Identifier: Apache-2.0
"""A resident reference transfers per-layer time; traces can refine a workload's routing."""

import json

import pytest
from test_machine_command import execute, machine
from test_machine_probes import FakeRunner, replies
from test_strategy_inputs import F32, engine_facts

from inferssd.strategy.calibration import apply, fit, load
from inferssd.strategy.estimate import Estimate
from inferssd.telemetry.routing import last_request_hits


def test_calibration_transfers_to_another_model_via_cli(tmp_path):
    host, files, build = machine(tmp_path)
    execute(["--model", files[0], "--build", str(build)], host, FakeRunner(replies()), tmp_path)
    timings = tmp_path / "response.json"
    timings.write_text(json.dumps({"timings": {"predicted_n": 32, "predicted_ms": 320}}))
    arguments = ["--show", "--json", "--estimate-facts"]
    raw = tmp_path / "facts.json"
    tensors = [("blk.0.attn_q.weight", [4096, 4096], F32)]
    raw.write_text(json.dumps(engine_facts(tensors, {}, path="reference.gguf")))
    _, lines, _ = execute(
        [*arguments, str(raw), "--calibrate-from", str(timings)], host, FakeRunner({}), tmp_path
    )
    report = json.loads(lines[0])
    assert not report["calibration_applied"]
    sample = tmp_path / "calibration.json"
    sample.write_text(lines[0])
    tensors.append(("blk.1.attn_q.weight", [1024, 1024], F32))
    raw.write_text(json.dumps(engine_facts(tensors, {}, path="other.gguf")))
    _, lines, _ = execute(
        [*arguments, str(raw), "--calibration", str(sample)], host, FakeRunner({}), tmp_path
    )
    result = json.loads(lines[0])
    assert result["calibration_applied"]
    target = result["estimates"][0]
    components = target["components"]
    assert components["calibration_residual_ms"] == pytest.approx(
        2 * report["calibration"]["residual_ms_per_layer"]
    )
    assert 1000 / target["decode_tokens_per_second"] == pytest.approx(
        components["uncalibrated_ms"] + components["calibration_residual_ms"]
    )
    assert target["scenarios"][target["reference_scenario"]] == target["decode_tokens_per_second"]


def test_calibration_guards_and_zero_residual():
    resident = Estimate(50, "raw", scenarios={"uniform_serial": 50})
    timing = {"predicted_n": 10, "predicted_ms": 100}
    sample = fit(timing, resident, 4, "machine A")
    assert (
        fit({"predicted_n": 2, "predicted_ms": 42}, resident, 2, "A")["residual_ms_per_layer"] == 11
    )
    assert sample["residual_ms_per_layer"] == 0
    assert apply(resident, load(sample, "machine A"), 48).tokens_per_second == 50
    with pytest.raises(ValueError, match="different machine"):
        load(sample, "machine B")
    with pytest.raises(ValueError, match="residual"):
        load(dict(sample, residual_ms_per_layer=-1), "machine A")
    with pytest.raises(ValueError, match="disk"):
        fit(timing, Estimate(50, "spilling", components={"storage_ms": 12}), 4, "machine A")
    with pytest.raises(ValueError, match="two decode tokens"):
        fit({"predicted_n": True}, resident, 4, "machine A")


def test_last_request_uses_only_complete_tokens_and_effective_hits():
    events = [{"ev": "ssd.controller.begin", "schema": 1}]

    def layer(epoch, token, layer, gpu, host):
        return {
            "ev": "ssd.place.layer",
            "success": True,
            "decode": True,
            "has_token": True,
            "epoch": epoch,
            "token": token,
            "layer": layer,
            "wall_us": 1,
            "wait_io_us": 0,
            "compute_us": 1,
            "selected_experts": [0, 2],
            "vram_hit": gpu,
            "cache_hit": host,
        }

    for token in (3, 40):
        events.append({"ev": "ssd.prefill.graph", "requested_tokens": token, "configured_max": 512})
        for index in (0, 1):
            events.append(layer(1, token, index, [token == 40, False], [True, index == 1]))
    events.append(layer(1, 41, 0, [False, False], [False, False]))
    lines = [json.dumps(dict(event, v=1, seq=i)) for i, event in enumerate(events)]
    assert last_request_hits(lines, {0, 1}, 3, 2) == (0.5, 0.25)
    with pytest.raises(ValueError, match="topology"):
        last_request_hits(lines, {0, 1}, 2, 2)
    with pytest.raises(ValueError, match="no complete"):
        last_request_hits(lines[:3], {0, 1}, 3, 2)
    next_prefill = dict(events[1], v=1, seq=len(lines))
    with pytest.raises(ValueError, match="no complete"):
        last_request_hits([*lines, json.dumps(next_prefill)], {0, 1}, 3, 2)
