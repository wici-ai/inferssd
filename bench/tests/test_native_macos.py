import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import signal
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


def wait_for(predicate, seconds=8):
    until = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > until:
            raise AssertionError("harmless guardian fixture deadline")
        time.sleep(.02)


def lock_available(path):
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def guardian_fixture(tmp_path, mode):
    worker = tmp_path / "worker.py"
    worker.write_text('''import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import native_common as common
from native_lease import Lease
p=argparse.ArgumentParser()
for arg in ("socket", "lock-dev", "lock-ino", "mode", "out"): p.add_argument("--"+arg,required=True)
a=p.parse_args(); out=Path(a.out)
lease=Lease(a.socket,expected_identity=(int(a.lock_dev),int(a.lock_ino)))
child=None; failed=False
try:
 if a.mode != "unknown":
  code="import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,lambda *_:(Path("+repr(str(out / "stopping"))+ ").touch(),time.sleep(.6),exit(0))); time.sleep(60)"
  child=subprocess.Popen([sys.executable,"-c",code],start_new_session=True,pass_fds=(lease.fd,))
  time.sleep(.15)
 (out/"active.json").write_text(json.dumps({"worker":os.getpid(),"child":child.pid if child else None}))
 start=time.monotonic()
 while a.mode != "normal" or time.monotonic()-start < .15:
  lease.check(); time.sleep(.02)
except InterruptedError:
 failed=True
finally:
 gone=True
 if child is not None: gone=common.stop_owned(child,2)["group_gone"]
 common.save(out/"receipt.json",{"status":"failed" if failed else "passed","server":{"owned_processes_gone":gone,"resource_scope_gone":True},"evidence":[]})
 lease.close()
sys.exit(1 if failed else 0)
''')
    bench = str(Path(__file__).resolve().parents[1])
    code = '''import fcntl,os,sys
from pathlib import Path
import native_macos as mac
out=Path(sys.argv[1]); lock=(out/"device.lock").open("a+")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,SSDASH_RUN_DIR=str(out),SSDASH_WRAPPER_RECEIPT=str(out/"receipt.json"))
sys.exit(mac.guard_worker([sys.executable,sys.argv[2],"--out",str(out),"--mode",sys.argv[3]],env,lock.fileno(),20))
'''
    env = dict(os.environ, PYTHONPATH=bench)
    return subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(worker), mode],
                            env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def test_guardian_normal_cleanup_releases_same_lease(tmp_path):
    controller = guardian_fixture(tmp_path, "normal")
    try:
        assert controller.wait(timeout=8) == 0, controller.stderr.read().decode()
        assert lock_available(tmp_path / "device.lock")
        assert mac.cleanup_confirmed(tmp_path / "receipt.json")
        receipt = json.loads((tmp_path / "receipt.json").read_text())
        assert receipt["status"] == "passed"
        assert json.loads((tmp_path / "guardian-cleanup.json").read_text())["worker_group_gone"]
    finally:
        if controller.poll() is None:
            controller.kill(); controller.wait()


def test_controller_sigkill_worker_cleans_child_before_lock_release(tmp_path):
    controller = guardian_fixture(tmp_path, "disconnect")
    identities = None
    verified = False
    try:
        wait_for(lambda: (tmp_path / "active.json").exists())
        identities = json.loads((tmp_path / "active.json").read_text())
        controller.kill(); controller.wait()
        wait_for(lambda: (tmp_path / "stopping").exists())
        assert not lock_available(tmp_path / "device.lock")
        wait_for(lambda: lock_available(tmp_path / "device.lock"))
        assert mac.cleanup_confirmed(tmp_path / "receipt.json")
        assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"
        with pytest.raises(ProcessLookupError):
            os.kill(identities["child"], 0)
        verified = True
    finally:
        if controller.poll() is None:
            controller.kill(); controller.wait()
        if identities and not verified:
            for pid in identities.values():
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def test_worker_death_quarantines_controller_and_retains_lock(tmp_path):
    controller = guardian_fixture(tmp_path, "unknown")
    worker = None
    try:
        wait_for(lambda: (tmp_path / "active.json").exists())
        active = json.loads((tmp_path / "active.json").read_text())
        assert active["child"] is None  # This negative fixture starts no child.
        worker = active["worker"]
        os.kill(worker, signal.SIGKILL)
        wait_for(lambda: (tmp_path / "guardian-quarantine.json").exists())
        assert controller.poll() is None and not lock_available(tmp_path / "device.lock")
        # Fixture simulates a separately verified empty state after the unknown
        # state has already made this arm fail; this cannot turn it into a pass.
        common.save(tmp_path / "receipt.json", {"status": "passed", "server": {
                    "owned_processes_gone": True, "resource_scope_gone": True}, "evidence": []})
        assert controller.wait(timeout=5) == 1
        assert lock_available(tmp_path / "device.lock")
        assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"
    finally:
        if worker:
            try:
                os.kill(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if controller.poll() is None:
            controller.kill(); controller.wait()
