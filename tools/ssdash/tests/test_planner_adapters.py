from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_offline_planner import DATA, observe_candidate, observe_f, runtime_key

from ssdash import cli
from ssdash.model.decode_cost import LayerResponse, price_decode, price_mapping
from ssdash.plan.recorded import DirectoryArtifacts, explicit_host_threads, write_json


def files_for(tmp_path, case):
    for name, value in case["files"].items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.mark.parametrize("name", ["complete-discrete", "complete-unified", "lazy-host-explicit"])
def test_cli_real_artifacts(tmp_path, capsys, name):
    case = next(c for c in DATA["cases"] if c["name"] == name)
    files_for(tmp_path, case)
    ticks = iter(case.get("clock", [100.0, 101.25]))
    out = tmp_path / "plan.json"
    with patch.object(cli, "time", SimpleNamespace(time=lambda: next(ticks))):
        assert (
            cli.main(["plan", str(tmp_path), *case["argv"], "--host-threads", "8", "-o", str(out)])
            == 0
        )
    expected = case["expected"][runtime_key()]
    assert out.read_text() == expected["writes"]["plan.json"]
    assert capsys.readouterr().out == expected["stdout"]


def test_cli_no_implicit_host(tmp_path):
    case = next(c for c in DATA["cases"] if c["name"] == "lazy-host-explicit")
    files_for(tmp_path, case)
    output = tmp_path / "chosen.json"
    with patch("os.cpu_count", side_effect=AssertionError("must not inspect current host")):
        with pytest.raises(ValueError, match="explicit --host-threads"):
            cli.main(["plan", str(tmp_path), *case["argv"], "-o", str(output)])
    assert not output.exists()


def test_cli_all_needs_no_host(tmp_path, capsys):
    case = next(c for c in DATA["cases"] if c["name"] == "single-placement")
    case = copy.deepcopy(case)
    case["files"]["probe/device.json"].pop("default_threads")
    files_for(tmp_path, case)
    with patch("os.cpu_count", side_effect=AssertionError("must not inspect current host")):
        assert (
            cli.main(["plan", str(tmp_path), *case["argv"], "-o", str(tmp_path / "plan.json")]) == 0
        )
    assert (tmp_path / "plan.json").exists()


def test_import_has_no_artifact_host_or_process_effect():
    script = """
import builtins, os, pathlib, socket, subprocess, sys
before = sys.path[:]
def denied(*args, **kwargs):
    raise AssertionError('import attempted external work')
builtins.open = denied
pathlib.Path.open = denied
os.open = denied
os.cpu_count = denied
socket.socket = denied
subprocess.Popen = denied
import ssdash.cli
import ssdash.model.machine
import ssdash.plan.planner
assert sys.path == before
assert 'machine' not in sys.modules and 'prefill' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_recorded_reader_is_lazy_and_closes(tmp_path):
    reader = DirectoryArtifacts(tmp_path / "absent")
    assert not reader.exists("insitu.json")
    with pytest.raises(FileNotFoundError):
        reader.read_json("insitu.json")
    stream = io.StringIO("{")
    with patch("builtins.open", return_value=stream):
        with pytest.raises(json.JSONDecodeError):
            reader.read_json("insitu.json")
    assert stream.closed


def test_writer_failure_closes_without_atomic_policy():
    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("write fault")

    stream = Broken()
    with patch("builtins.open", return_value=stream):
        with pytest.raises(OSError, match="write fault"):
            write_json("chosen.json", {"value": 1})
    assert stream.closed


@pytest.mark.parametrize("value", [0, -1, 8])
def test_explicit_host_values_are_not_normalized(value):
    assert explicit_host_threads(value)() == value


def test_shared_mapping_core_null_failure():
    data = {
        "layer_table": {},
        "layer_beta": [1, 0, 0, 1, 0],
        "prefetch_share": 0.5,
        "pf_k": 0,
        "t_out_ms": 1.0,
        "ready_given_nx": {"1": [0, 1]},
        "ready_share_all": None,
    }
    response = LayerResponse.from_legacy(data)
    with pytest.raises(TypeError) as raw:
        price_mapping([[(0, 2)]], data, 1.0)
    with pytest.raises(TypeError) as adapted:
        price_decode([[(0, 2)]], response, 1.0)
    assert str(adapted.value) == str(raw.value)
    assert "null in an uncapped fallback" not in str(raw.value)
    assert price_decode([[(0, -1)]], response, 1.0).to_mapping() == price_mapping(
        [[(0, -1)]], data, 1.0
    )


@pytest.mark.parametrize("pending", [[], ["UBATCH"], ["UBATCH", "UBATCH"]])
def test_later_replay_queries_and_feedback(pending):
    # Additional direct whole-F observations; no new implementation computes expected.
    case = copy.deepcopy(next(c for c in DATA["cases"] if c["name"] == "machine-no-plan"))
    case["pending"] = pending
    case["actions"] += [
        {"method": "replay", "routing": True, "kwargs": {"key": "later", "tier": 1}},
        {"method": "replay", "routing": True, "kwargs": {"key": "later", "tier": 4}},
        {
            "method": "predict",
            "routing": True,
            "kwargs": {
                "tier": 2,
                "p": "split",
                "ctx": 0,
                "prompts": [17, 32],
                "decode": 8,
                "w0": 1,
                "w1": 8,
                "cold": False,
                "fb": {
                    "k_cpu": {"split": 1.25},
                    "k_drive": 0.8,
                    "k_bytes": 1.2,
                    "pf_cpu": {"split": 1.1},
                    "pf_drive": 1.3,
                    "out0": {"split": 2.5},
                },
            },
        },
    ]
    assert observe_candidate(case) == observe_f(case)


@pytest.mark.parametrize("case_name", ["complete-discrete", "reject-all"])
def test_stdout_failure_keeps_original_publication_order(case_name):
    case = next(c for c in DATA["cases"] if c["name"] == case_name)
    with patch("builtins.print", side_effect=OSError("stdout failed")):
        original = observe_f(case)
    with patch("builtins.print", side_effect=OSError("stdout failed")):
        candidate = observe_candidate(case)
    assert candidate == original
    assert candidate["outcome"] == ["OSError", "stdout failed"]
    assert bool(candidate["writes"]) == (case_name == "complete-discrete")


def test_pending_iterator_and_input_aliases():
    from test_offline_planner import F_MACHINE, Transcript, exact

    from ssdash.model.machine import Machine

    case = next(c for c in DATA["cases"] if c["name"] == "machine-no-plan")
    original_io, candidate_io = Transcript(case), Transcript(case)
    a, b = iter(["UBATCH", "CONTEXT"]), iter(["UBATCH", "CONTEXT"])
    with (
        patch.object(F_MACHINE, "open", original_io.open, create=True),
        patch.object(
            F_MACHINE, "json", SimpleNamespace(load=lambda f: original_io.read_json(f.name))
        ),
    ):
        original = F_MACHINE.Machine("/case/run", pending=a)
    candidate = Machine(candidate_io, pending=b, host_threads=candidate_io.host)
    assert original.pending is a and candidate.pending is b
    assert list(a) == list(b) == []
    assert exact(original.derived) == exact(candidate.derived)
    for machine in (original, candidate):
        assert machine.P is machine.ins["placements"]
        assert machine.PF["memory"]["all"] is machine.P["all"]["memory"]
    assert original_io.events == candidate_io.events
    assert original_io.reads == candidate_io.reads


@pytest.mark.parametrize("in_place,locked", [(False, True), (True, True), (True, False)])
def test_complete_calibrated_lane_callgraph(in_place, locked):
    case = copy.deepcopy(next(c for c in DATA["cases"] if c["name"] == "complete-discrete"))
    lane = {
        "cpu_fixed_us": 200.0,
        "cpu_route_us": 8.0,
        "gpu_fixed_us": 80.0,
        "gpu_route_us": 1.0,
        "gpu_host_fixed_us": 60.0,
        "gpu_host_route_us": 0.5,
        "link_gbps": 4.0,
        "pageable_gbps": 2.0,
        "drive_gbps": 1.25,
        "drive_fixed_us": 2.0,
        "gpu_token_us": 1.5,
    }
    for p in case["files"]["insitu.json"]["placements"].values():
        p["prefill_engine"] = {
            "in_place": in_place,
            "route_bytes": 512,
            "ring": 2,
            "ring_locked": locked,
            "engine": "q4_K/q5_K",
        }
        for point in p["prefill"]:
            point.update(lane=lane, layers_s=0.01)
    case["files"]["probe/device.json"]["gpu"]["integrated"] = in_place
    original = observe_f(case)
    assert original["outcome"] == ["return"]
    assert json.loads(original["writes"]["plan.json"])["prefill_model"]["lane"] is not None
    assert observe_candidate(case) == original
