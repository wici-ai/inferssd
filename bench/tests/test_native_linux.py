import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_linux as linux


@pytest.mark.parametrize("limits,minimum", [( (4096, 8192), 4096), ((0, 0), 0),
                                            ((linux.resource.RLIM_INFINITY, linux.resource.RLIM_INFINITY), 1 << 40)])
def test_memlock_finite_zero_and_unlimited_are_explicit(monkeypatch, limits, minimum):
    monkeypatch.setattr(linux.resource, "getrlimit", lambda kind: limits if kind == linux.resource.RLIMIT_MEMLOCK else None)
    receipt = linux.memlock_limits(minimum)
    assert (receipt["soft"], receipt["hard"]) == limits
    rendered = ["unlimited" if value == linux.resource.RLIM_INFINITY else str(value) for value in limits]
    linux.check_server_memlock("Max locked memory    " + "    ".join(rendered) + "    bytes\n", receipt)


def test_insufficient_soft_memlock_cannot_use_larger_hard_limit(monkeypatch):
    monkeypatch.setattr(linux.resource, "getrlimit", lambda _kind: (4096, linux.resource.RLIM_INFINITY))
    with pytest.raises(ValueError, match="soft limit"):
        linux.memlock_limits(4097)


@pytest.mark.parametrize("row", ["Max locked memory 2048 8192 bytes\n", "Max locked memory 4096 16384 bytes\n",
                                  "Max locked memory 4096 8192 kB\n", "missing"])
def test_server_memlock_must_prove_exact_inheritance(row):
    with pytest.raises(ValueError):
        linux.check_server_memlock(row, {"soft": 4096, "hard": 8192, "minimum_bytes": 4096})


def cgroup_fixture(tmp_path):
    mount, proc = tmp_path / "cgroup", tmp_path / "proc"
    scope = mount / "example.scope"
    scope.mkdir(parents=True)
    (proc / "123").mkdir(parents=True)
    (proc / "123/cgroup").write_text("0::/example.scope\n")
    for path, limit, swap in ((mount, "max", "max"), (scope, str(4 << 30), "0")):
        for key, value in {"memory.max": limit, "memory.swap.max": swap, "memory.peak": "10",
                           "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", "cgroup.procs": "123\n"}.items():
            (path / key).write_text(value)
    return mount, proc, scope


def test_own_finite_limit_and_all_ancestor_oom(tmp_path):
    mount, proc, scope = cgroup_fixture(tmp_path)
    initial = linux.cgroup_snapshot(123, 4 << 30, mount, proc)
    linux.unchanged_oom(initial, copy.deepcopy(initial))
    (mount / "memory.events").write_text("oom 1\noom_kill 0\n")
    with pytest.raises(ValueError, match="OOM"):
        linux.unchanged_oom(initial, linux.cgroup_snapshot(123, 4 << 30, mount, proc))
    (scope / "memory.swap.max").write_text("max")
    with pytest.raises(ValueError, match="zero swap"):
        linux.cgroup_snapshot(123, 4 << 30, mount, proc)


def test_lower_ancestor_limit_is_not_requested_budget(tmp_path):
    mount, proc, _ = cgroup_fixture(tmp_path)
    (mount / "memory.max").write_text(str(3 << 30))
    with pytest.raises(ValueError):
        linux.cgroup_snapshot(123, 4 << 30, mount, proc)


def test_scope_descendants_and_missing_membership_fail_closed(tmp_path):
    _, _, scope = cgroup_fixture(tmp_path)
    child = scope / "nested"
    child.mkdir()
    (child / "cgroup.procs").write_text("456\n")
    assert linux.scope_pids(scope) == {123, 456}
    (scope / "cgroup.procs").unlink()
    with pytest.raises(FileNotFoundError):
        linux.scope_pids(scope)


@pytest.mark.parametrize("status,expected", [("ActiveState=active\nControlGroup=\n", False),
                                            ("ActiveState=inactive\nControlGroup=\n", True),
                                            ("LoadState=not-found\n", True), ("", False)])
def test_unknown_or_active_scope_is_never_clean(monkeypatch, status, expected):
    monkeypatch.setattr(linux, "systemctl", lambda *_args: {"returncode": 0, "stdout": status, "stderr": ""})
    assert linux.scope_state("owned.scope")["gone"] is expected


def test_process_identity_uses_start_ticks_and_exe(monkeypatch):
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/123/stat":
            return "123 (name with spaces) " + " ".join(["S"] + ["0"] * 18 + ["777"] + ["0"] * 5)
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return "boot"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: Path("/reviewed/server"))
    identity = json.loads(linux.LinuxRuntime({}, Path(".")).process_identity(123))
    assert identity == {"pid": 123, "start_ticks": "777", "boot_id": "boot", "exe": "/reviewed/server"}


@pytest.mark.parametrize("fault", [None, "foreign", "inode", "missing", "changed"])
def test_linux_maps_bind_actual_inode_bytes_and_complete_set(tmp_path, monkeypatch, fault):
    binary, library, foreign = [tmp_path / x for x in ("server", "libggml-base.so", "libkimi-foreign.so")]
    for path in (binary, library, foreign):
        path.write_bytes(path.name.encode())
    identity = linux.common.file_identity
    specs = [{"path": str(p), "sha256": identity(p, True)["sha256"]} for p in (binary, library)]
    profile = {"binary": specs[0], "images": specs[1:]}
    maps = []
    for path in (binary, library, foreign) if fault == "foreign" else (binary, library):
        if fault == "missing" and path == library:
            continue
        info = path.stat()
        inode = info.st_ino + (1 if fault == "inode" else 0)
        maps.append(f"0-1 r-xp 0 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {inode} {path}")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: "\n".join(maps) if str(p) == "/proc/123/maps" else original(p, *a, **kw))
    monkeypatch.setattr(linux.common, "file_identity", lambda p, h=False: identity(binary if str(p) == "/proc/123/exe" else p, h))
    if fault == "changed":
        library.write_bytes(b"changed")
    if fault:
        with pytest.raises(ValueError):
            linux.mapped_images(123, profile, tmp_path / "receipt")
    else:
        assert linux.mapped_images(123, profile, tmp_path / "receipt") == specs[1:]


def test_scope_uncertainty_retains_lease_and_invalidates_result(tmp_path, monkeypatch):
    # The platform operations are mocked; this tests the actual controller's
    # ordering, not a claim of native systemd support on the test host.
    receipt_path = tmp_path / "wrapper.json"
    environment = {"SSDASH_RUN_DIR": str(tmp_path), "SSDASH_WRAPPER_RECEIPT": str(receipt_path),
                   "SSDASH_REQUEST_TIMEOUT": "10"}
    profile = {"resources": {"memory_budget_bytes": 4 << 30}, "load_timeout_s": 10}
    events = []
    class Lease:
        path, peer = "mock-socket", {"pid": 321}
        def __init__(self, fd, unit):
            events.append(("lease", fd))
        def close(self):
            events.append("lease-close")
    class Process:
        pid, returncode = 123, 0
        def __init__(self, *_args, **_kwargs):
            receipt_path.write_text(json.dumps({"status": "passed", "server": {"owned_processes_gone": True}, "evidence": []}))
        def poll(self):
            return 0
    states = iter((False, True))
    def state(_unit):
        gone = next(states)
        events.append(("scope-gone", gone))
        return {"gone": gone}
    monkeypatch.setattr(linux, "sys", SimpleNamespace(platform="linux", executable=sys.executable))
    monkeypatch.setattr(linux, "LeaseServer", Lease)
    monkeypatch.setattr(linux.subprocess, "Popen", Process)
    monkeypatch.setattr(linux, "scope_state", state)
    monkeypatch.setattr(linux, "systemctl", lambda *_args: {"returncode": 0})
    monkeypatch.setattr(linux.common, "group_exists", lambda _process: False)
    monkeypatch.setattr(linux.time, "sleep", lambda _seconds: None)
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda p, *a, **kw: tmp_path / "lock" if str(p).startswith("/proc/self/fd/") else original(p, *a, **kw))
    assert linux.run_scope(profile, environment, 9) == 1
    assert events == [("lease", 9), ("scope-gone", False), ("scope-gone", True), "lease-close"]
    result = json.loads(receipt_path.read_text())
    assert result["status"] == "failed" and result["server"]["resource_scope_gone"]
    assert (tmp_path / "quarantine.json").exists()
