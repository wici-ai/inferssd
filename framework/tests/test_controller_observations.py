"""Real controller decisions remain equal across legacy import and native JSONL."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_loop_model import model_files, reply, trace

from inferssd.engine.runtime import LocalObservations
from inferssd.loop import model
from inferssd.loop.records import ObservationSession
from inferssd.model.machine import Machine
from inferssd.plan.recorded import DirectoryArtifacts
from inferssd.telemetry.controller import NativeObservationSession


def events(turns):
    values = [{"ev": "ssd.controller.begin", "schema": 1}]
    for prompt, tokens in turns:
        values += [
            {"ev": "ssd.prefill.graph", "requested_tokens": prompt, "configured_max": 128},
            {"ev": "ssd.place.layer", "decode": False, "success": True, "compute_us": 12000},
            {
                "ev": "ssd.prefill.copy",
                "bytes": 1048576,
                "read_us": 500,
                "passes": 1,
                "tokens": prompt,
            },
        ]
        for i in range(tokens):
            for layer in range(2):
                values.append(
                    {
                        "ev": "ssd.place.layer",
                        "epoch": 1,
                        "token": prompt + i,
                        "layer": layer,
                        "decode": True,
                        "success": True,
                        "has_token": True,
                        "selected_experts": [i % 7, i % 7 + 1],
                        "vram_hit": [True, False],
                        "cache_hit": [False, True],
                        "wall_us": 1500 + i,
                        "compute_us": 500,
                        "wait_io_us": 500,
                    }
                )
    return [dict(v=1, seq=i, **value) for i, value in enumerate(values)]


def write_events(directory: Path, values):
    data = "".join(json.dumps(v) + "\n" for v in values)
    (directory / "native-events.jsonl").write_text(data)
    return data


def machine(directory):
    directory.mkdir()
    for name, value in model_files().items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return Machine(DirectoryArtifacts(directory), host_threads=lambda: 8)


def test_observer_served_feedback_and_decision_parity(tmp_path):
    m = machine(tmp_path / "model")
    arm = tmp_path / "arm"
    arm.mkdir()
    turns = [(128, 20), (64, 20)]  # a second request reuses token positions in the same epoch
    (arm / "server.log").write_text("".join(trace(prompt=p, tokens=n) for p, n in turns))
    tier, cpus = m.configs[0]
    (arm / "turns.jsonl").write_text(
        "".join(
            json.dumps(dict(reply(p, n), turn=i, tier=tier, cpus=cpus)) + "\n"
            for i, (p, n) in enumerate(turns)
        )
    )
    data = write_events(arm, events(turns))
    legacy = ObservationSession(files=LocalObservations())
    native = NativeObservationSession(files=LocalObservations())
    old = legacy.read_arm_cached(str(arm), 0)
    new = native.read_arm_cached(str(arm), 0)
    for key in new:
        assert new[key] == old[key]
    old_served = model.served(m, [str(arm)], 1, 20, session=legacy)
    new_served = model.served(m, [str(arm)], 1, 20, session=native)
    assert old_served == new_served
    old_turns, old_segs, _ = old_served
    new_turns, new_segs, _ = new_served
    assert model.feedback(m, old_turns) == model.feedback(m, new_turns)
    assert model.decide(
        m, old_turns, old_segs, (tier, cpus), [128], 20, 1, 20, 2.0
    ) == model.decide(m, new_turns, new_segs, (tier, cpus), [128], 20, 1, 20, 2.0)
    # Removing/changing human logs cannot affect production observations.
    (arm / "server.log").unlink()
    assert native.read_arm_cached(str(arm), 0) == new
    # Partial JSONL waits until the next tail; the complete contract is retained.
    with (arm / "native-events.jsonl").open("a") as stream:
        stream.write('{"v":1,"seq":')
    assert native.read_arm_cached(str(arm), 0) == new
    (arm / "native-events.jsonl").write_text(data[:20])
    with pytest.raises(ValueError, match="truncated"):
        native.read_arm_cached(str(arm), 0)


@pytest.mark.parametrize(
    "mutation", ["version", "gap", "missing-route", "failed-layer", "contract"]
)
def test_invalid_native_observations_fail_closed(tmp_path, mutation):
    values = copy.deepcopy(events([(128, 1)]))
    if mutation == "version":
        values[0]["v"] = 2
    elif mutation == "gap":
        values[-1]["seq"] += 1
    elif mutation == "missing-route":
        del values[-1]["selected_experts"]
    elif mutation == "failed-layer":
        values[-1]["success"] = False
    else:
        values[0]["schema"] = 2
    write_events(tmp_path, values)
    with pytest.raises(ValueError):
        NativeObservationSession(files=LocalObservations()).read_arm_cached(str(tmp_path), 0)
