import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_common as common
import native_macos as mac


def test_mac_counters_keep_units_and_soft_boundary_explicit():
    vm = mac.parse_vm("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                      "Pages free: 2.\nPages inactive: 3.\nPages occupied by compressor: 4.\nPageouts: 5.\n")
    assert vm["page_size"] == 16384 and vm["pages"]["Pageouts"] == 5
    assert mac.parse_swap("total = 1024.00M used = 1.25G free = 0.00M") == 1342177280
    with pytest.raises(ValueError):
        mac.parse_vm("no counters")
    with pytest.raises(ValueError):
        mac.parse_swap("unknown")


@pytest.mark.parametrize("key,value", [("rss_bytes", 101), ("reclaimable_estimate_bytes", 9),
                                       ("compressor_bytes", 21), ("swap_used_bytes", 11)])
def test_soft_thresholds_fail_without_claiming_a_hard_cap(key, value):
    limits = {"memory_budget_bytes": 100, "host_reserve_bytes": 10,
              "max_compressor_growth_bytes": 10, "max_swap_growth_bytes": 0}
    baseline = {"rss_bytes": 1, "reclaimable_estimate_bytes": 100, "compressor_bytes": 10, "swap_used_bytes": 10}
    mac.enforce_sample(baseline, baseline, limits)
    with pytest.raises(RuntimeError):
        mac.enforce_sample(dict(baseline, **{key: value}), baseline, limits)


def test_actual_images_must_match_complete_reviewed_set(tmp_path, monkeypatch):
    binary, library = tmp_path / "server", tmp_path / "libggml.dylib"
    binary.touch()
    library.touch()
    profile = {"binary": {"path": str(binary), "sha256": "a" * 64},
               "images": [{"path": str(library), "sha256": "b" * 64}]}
    runtime = mac.MacRuntime(profile, tmp_path)
    expected = [profile["binary"], *profile["images"]]
    monkeypatch.setattr(mac, "snapshot_images", lambda *_a, **_kw: {"images": expected})
    assert runtime.images(123, "test") == profile["images"]
    expected.append({"path": "/foreign/libkimi-other.dylib", "sha256": "c" * 64})
    with pytest.raises(ValueError, match="complete Mac image set"):
        runtime.images(123, "test")


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin native lsof/ps contract; no model")
def test_native_mac_pid_listener_and_same_lock_fd_with_harmless_child(tmp_path):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    with (tmp_path / "device.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',int(__import__('sys').argv[1]))); s.listen(); time.sleep(30)"
        child = subprocess.Popen([sys.executable, "-c", code, str(port)], pass_fds=(lock.fileno(),), start_new_session=True)
        try:
            runtime = mac.MacRuntime({}, tmp_path)
            until = time.monotonic() + 5
            while True:
                try:
                    runtime.confirm_port(child.pid, port)
                    break
                except RuntimeError:
                    if time.monotonic() > until:
                        raise
                    time.sleep(.05)
            runtime.confirm_child(child.pid, lock.fileno())
            assert json.loads(runtime.process_identity(child.pid))["pid"] == child.pid
            with pytest.raises((ValueError, RuntimeError)):
                runtime.confirm_port(os.getpid(), port)
            assert common.stop_owned(child, 1)["group_gone"]
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
