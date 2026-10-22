import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_linux as linux


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
