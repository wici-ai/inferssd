"""Native stop mode preserves dry-run and rejects before the ownership spawn port."""

from __future__ import annotations

import copy

import pytest
import test_launch_linux as linux_fixture
from observe_launch_refusal_F import CASES, observe
from test_engine_runtime import policy

from ssdash.engine import runtime
from ssdash.launch.linux import LinuxState, resolve_linux
from ssdash.launch.types import Finished, LaunchInput, Reply


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_native_refusal_matches_original_shell_boundary(case):
    old = observe(case)
    state = LinuxState(LaunchInput("fixture", ()), {}, execute=case["dry_run"] != "1")
    try:
        state.refuse(case["message"])
        code, continued = 0, "continued\n"
    except Finished as stopped:
        code, continued = stopped.code, ""
    assert code == old["code"] and continued == old["stdout"]
    assert "".join(e.text for e in state.outputs) == old["stderr"]


REFUSALS = [
    ("port-busy", {}, "Port 18181 is already in use", "port_busy"),
    ("dense-no-uring", {}, "DENSE_HOMES", "has_uring"),
    ("busy", {}, "The GPU already runs", "gpu_processes"),
    ("base", {"cgroup_v2": False}, "cgroup v2 is required", "cgroup_v2"),
    ("insufficient-memory", {}, "Only ", "memory_kib"),
    ("base", {"euid": 501, "user_manager": False}, "No systemd user manager", "user_manager"),
    ("base", {"euid": 501, "memory_controller": False}, "not delegated", "memory_controller"),
]


def collect(record, monkeypatch, overrides=None, execute=False):
    facts_seen = {}

    def resolving(inputs, facts):
        selected = {
            key: Reply((overrides or {}).get(key.name, val.value), val.stdout, val.stderr, val.code)
            for key, val in facts.items()
        }
        facts_seen.update(selected)
        return resolve_linux(inputs, selected, execute=execute)

    monkeypatch.setattr(linux_fixture, "resolve_linux", resolving)
    result, _ = linux_fixture.replay(record, record["runs"][0])
    return result, facts_seen


@pytest.mark.parametrize("name,overrides,message,last", REFUSALS, ids=[x[2] for x in REFUSALS])
def test_all_host_refusals_stop_before_owner(tmp_path, monkeypatch, name, overrides, message, last):
    record = copy.deepcopy(next(r for r in linux_fixture.GOLDENS if r["case"]["name"] == name))
    result, facts = collect(record, monkeypatch, overrides, execute=True)
    assert result.code == 1 and result.needed is None
    assert message in result.outputs[-1].text
    assert not result.outputs[-1].text.startswith("dry run:")
    assert result.observations[-1].name == last
    emitted = []
    monkeypatch.setattr(runtime, "LiveArtifacts", lambda *a, **kw: lambda fact: facts[fact])
    monkeypatch.setattr(
        runtime, "OwnedProcess", lambda *a, **kw: pytest.fail("refused before spawn")
    )
    value = runtime.Runtime(
        LaunchInput("@repo", tuple(record["env"].items())),
        policy(tmp_path, "linux"),
        host=object(),
        emit=emitted.append,
    )
    directory = tmp_path / "instance"
    with pytest.raises(RuntimeError, match="preparation failed"):
        value.engine(
            str(directory),
            int(next(f.arguments[0] for f in facts if f.name == "port_busy")),
            0,
            "2:2",
            "",
            [],
        )
    assert (directory / ".ssdash-owner").exists()
    assert (directory / "server.log").read_bytes() == b""
    assert not (directory / "runtime.json").exists()
    assert not (directory / "ownership.json").exists()
    assert message in emitted[-1].text


@pytest.mark.parametrize("record", linux_fixture.GOLDENS, ids=lambda r: r["case"]["name"])
def test_other_source_settings_and_profile_errors_still_stop(monkeypatch, record):
    dry, _ = collect(record, monkeypatch)
    native, _ = collect(record, monkeypatch, execute=True)
    refusals = [e for e in dry.outputs if e.text.startswith("dry run: the launch would stop here:")]
    if refusals:
        assert native.code == 1
        first = dry.outputs.index(refusals[0])
        assert native.outputs[:-1] == dry.outputs[:first]
        assert native.outputs[-1].text == refusals[0].text.removeprefix(
            "dry run: the launch would stop here: "
        )
    else:
        assert native == dry


@pytest.mark.parametrize("failure", ["missing-model", "artifact"])
def test_darwin_error_closes_before_owner(tmp_path, monkeypatch, failure):
    emitted = []
    env = () if failure == "missing-model" else (("SSD_LLAMA_MODEL", "/fixture/model.gguf"),)

    def observed(fact):
        if fact.name == "platform_system":
            return Reply("Darwin")
        if fact.name == "platform_machine":
            return Reply("arm64")
        if fact.name == "resolved_path":
            return Reply(fact.arguments[0])
        if fact.name in ("is_file", "is_executable"):
            return Reply(True)
        if fact.name == "mkdir":
            return Reply()
        if fact.name in ("physical_memory", "disk_free"):
            return Reply(str(16 << 30))
        if fact.name == "mac_alias":
            raise OSError("fixture alias failure")
        raise AssertionError(fact)

    monkeypatch.setattr(runtime, "LiveArtifacts", lambda *a, **kw: observed)
    monkeypatch.setattr(runtime, "OwnedProcess", lambda *a, **kw: pytest.fail("no owner"))
    value = runtime.Runtime(
        LaunchInput("/fixture", env), policy(tmp_path), host=object(), emit=emitted.append
    )
    out = tmp_path / "instance"
    with pytest.raises(RuntimeError, match="preparation failed"):
        value.engine(str(out), 8080, 0, "static", "", [])
    assert emitted[-1].channel == "stderr"
    assert (out / ".ssdash-owner").exists() and (out / "server.log").read_bytes() == b""
    assert not (out / "ownership.json").exists()
