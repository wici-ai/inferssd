import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_common as common
import native_wrapper
import run_final
from test_stream_chat import fixture


def spec(path, model=False):
    result = common.file_identity(path, True)
    return result if model else {k: result[k] for k in ("path", "sha256")}


@pytest.fixture
def profile(tmp_path):
    binary, library, shard, alias, template = [tmp_path / n for n in ("server", "libggml.so", "model.gguf", "alias.tsv", "template.jinja")]
    for path in (binary, library, shard, template):
        path.write_bytes(b"fixture bytes\n")
    alias.write_text(f"source_path\toffset\tnbytes\n{shard}\t0\t4\n")
    return {"schema_version": 1, "platform": "linux", "revision": "a" * 40,
            "binary": spec(binary), "images": [spec(library)], "shards": [spec(shard, True)],
            "template": spec(template), "alias": spec(alias), "args": ["-t", "2"],
            "environment": {"GGML_MOE_EXPERT_GGUF_ALIAS_TSV": str(alias)},
            "resources": {"memory_budget_bytes": 4 << 30, "host_reserve_bytes": 3 << 30, "minimum_cpus": 2,
                          "minimum_memlock_bytes": 0},
            "effective_settings": {"context_tokens": 8192, "source_mode": "direct-gguf",
                                   "h2d_limit_bytes_per_second": None, "expert_cache_budget_bytes": 1 << 30,
                                   "vram_budget_bytes": 0, "unified_memory_budget_bytes": 0, "gpu_reserve_bytes": 0},
            "load_timeout_s": 600, "cleanup_timeout_s": 2}


def test_verified_model_stat_receipt_and_alias(profile):
    common.validate_profile(profile)
    value = common.verify_inputs(profile)
    assert value["alias_entries"] == 1
    path = Path(profile["shards"][0]["path"])
    path.write_bytes(b"changed bytes!\n")
    with pytest.raises(ValueError, match="identity differs"):
        common.verify_inputs(profile)


def test_nonregular_never_blocks(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        common.file_identity(path, True)


@pytest.mark.parametrize("change", [
    lambda p: p["args"].extend(["--parallel", "2"]),
    lambda p: p["args"].append("--cache-ram=1"),
    lambda p: p["environment"].update(H2D_LIMIT="1"),
    lambda p: p["environment"].update(LLAMA_ARG_MODEL="other"),
    lambda p: p["environment"].update(GGML_MOE_EXPERT_PACK="pack"),
    lambda p: p["environment"].update(SSD_LLAMA_MAC_EXPERT_PACK="pack"),
    lambda p: p["environment"].update(GGML_MOE_EXPERT_PACK_OVERLAY="pack"),
    lambda p: p["environment"].update(DYLD_LIBRARY_PATH="other"),
    lambda p: p["resources"].update(memory_budget_bytes=True),
    lambda p: p["resources"].pop("minimum_memlock_bytes"),
    lambda p: p["effective_settings"].update(h2d_limit_bytes_per_second=0),
    lambda p: p["images"].append(copy.deepcopy(p["images"][0])),
    lambda p: p["shards"].append(copy.deepcopy(p["shards"][0])),
])
def test_reject_unreviewable_configuration(profile, change):
    change(profile)
    with pytest.raises(ValueError):
        common.validate_profile(profile)


def test_partial_shard_set_rejected(profile):
    profile["shards"][0]["path"] = "/models/example-00001-of-00002.gguf"
    with pytest.raises(ValueError, match="all shards"):
        common.validate_profile(profile)


def test_explicit_argv_and_clean_environment(profile, monkeypatch):
    monkeypatch.setenv("GGML_MOE_EXPERT_PACK", "old")
    monkeypatch.setenv("H2D_RATE", "old")
    monkeypatch.setenv("LD_PRELOAD", "old")
    env = common.clean_environment(profile)
    assert "GGML_MOE_EXPERT_PACK" not in env and "H2D_RATE" not in env and "LD_PRELOAD" not in env
    args = common.server_argv(profile, 12345)
    assert args.count("--no-warmup") == 1
    assert args[args.index("--cache-ram") + 1] == "0"
    assert args[args.index("-np") + 1] == "1"


def test_completion_task_count():
    native_wrapper.confirm_first_request("health", "processing task, is_child = 0")
    for before, after in (("processing task, is_child = 0", ""), ("", ""),
                          ("", "processing task, is_child = 0\nprocessing task, is_child = 0")):
        with pytest.raises(ValueError):
            native_wrapper.confirm_first_request(before, after)


def test_permission_unknown_group_is_never_clean(monkeypatch):
    class Process:
        pid = 123
        def poll(self):
            return None
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("owned child", timeout)
    def denied(*_args):
        raise PermissionError("group state unknown")
    monkeypatch.setattr(os, "killpg", denied)
    assert common.group_exists(Process())
    with pytest.raises(subprocess.TimeoutExpired):
        common.stop_owned(Process(), 0)


def test_same_inherited_lock_stays_held_through_child(tmp_path):
    path = tmp_path / "device.lock"
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = common.inherited_lock({"INFERSSD_LOCK_FD": str(lock.fileno())})
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                   pass_fds=(fd,), start_new_session=True)
        try:
            lock.close()
            with path.open("a+") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = common.stop_owned(process, .1)
            assert result["group_gone"]
            with path.open("a+") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def test_successful_leader_with_stubborn_child_is_not_success(tmp_path):
    marker = tmp_path / "child"
    code = "import os,signal,time,sys; p=os.fork(); " + "\nif p == 0:\n signal.signal(signal.SIGTERM, signal.SIG_IGN)\n open(sys.argv[1],'w').write(str(os.getpid()))\n time.sleep(20)\n"
    process = subprocess.Popen([sys.executable, "-c", code, str(marker)], start_new_session=True)
    try:
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        result = common.stop_owned(process, .1)
        assert result["orphaned_descendants"] and result["forced_kill"]
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_failed_profile_does_not_run_model(profile, tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    run = subprocess.run([sys.executable, str(Path(native_wrapper.__file__)), "--profile", str(path)], capture_output=True)
    assert run.returncode != 0
    assert b"invoke through" in run.stderr


class MockRuntime:
    """Only protocol/lifecycle tests use this; production cannot select it."""
    def __init__(self, profile, out, fail_cleanup=False):
        self.profile, self.out, self.fail_cleanup = profile, out, fail_cleanup
    def prepare(self):
        pass
    def sample(self, _pid):
        pass
    def process_identity(self, pid):
        return str(pid)
    def confirm_child(self, _pid, _fd):
        pass
    def confirm_port(self, _pid, _port):
        pass
    def images(self, _pid, _name):
        return self.profile["images"]
    def host_receipt(self):
        return {"hardware": "mock hardware", "os": "mock OS", "memory_budget_bytes": 4 << 30,
                "resource_checks_passed": True, "effective_settings": common.effective_settings(self.profile),
                "os_page_cache_treatment": "none; OS cache state uncontrolled"}
    def finish(self):
        if self.fail_cleanup:
            raise RuntimeError("mock cleanup unknown")
        return True
    def evidence_paths(self):
        return []


def fake_server(tmp_path, profile, mode="normal"):
    records = fixture(256)
    records[-1]["event"]["tokens_evaluated"] = 4
    records[-1]["event"]["timings"].update(predicted_ms=1000, prompt_n=4)
    fixture_path = tmp_path / "events.json"
    fixture_path.write_text(json.dumps(records))
    script = tmp_path / "fake_server.py"
    script.write_text('''import argparse,json,signal,time,sys
from http.server import BaseHTTPRequestHandler, HTTPServer
p=argparse.ArgumentParser(); p.add_argument('--port',type=int); a,_=p.parse_known_args()
rows=json.load(open(sys.argv[1])); mode=sys.argv[2]; stopping=False
def stop(*_):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*_): pass
 def do_GET(self):
  self.send_response(200); self.end_headers()
  self.wfile.write(json.dumps({'status':'ok','total_slots':1,'default_generation_settings':{'n_ctx':8192}}).encode())
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  self.send_response(200); self.end_headers()
  if self.path=='/apply-template':
   self.wfile.write(json.dumps({'prompt':'test','tokens':[1,2,3,4]}).encode()); return
  print('processing task, is_child = 0',file=sys.stderr,flush=True)
  if mode=='partial': return
  if mode=='timeout':
   while not stopping: time.sleep(.01)
   return
  for row in rows:
   self.wfile.write(b'data: '+json.dumps(row['event']).encode()+b'\\n\\n'); self.wfile.flush(); time.sleep(.0001)
s=HTTPServer(('127.0.0.1',a.port),Handler); s.timeout=.05
while not stopping: s.handle_request()
s.server_close()
''')
    profile["binary"] = spec(Path(sys.executable))
    profile["args"] = [str(script), str(fixture_path), mode]
    return profile


def arm_environment(out, profile, fd, tier="512", timeout="5"):
    return {"INFERSSD_RUN_DIR": str(out), "INFERSSD_RESULT_PATH": str(out / "result.json"),
            "INFERSSD_WRAPPER_RECEIPT": str(out / "wrapper.json"), "INFERSSD_TIER": tier,
            "INFERSSD_REQUEST_TIMEOUT": timeout, "INFERSSD_DEVICE": "mock", "INFERSSD_MODEL": "mock",
            "INFERSSD_REVISION": profile["revision"], "INFERSSD_OUTPUT_TOKENS": "256", "INFERSSD_LOCK_FD": str(fd)}


def test_mock_two_tiers_use_fresh_owned_servers_one_completion_each(profile, tmp_path):
    profile = fake_server(tmp_path, profile)
    common.validate_profile(profile)
    rows = []
    with (tmp_path / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for tier in ("512", "4096"):
            out = tmp_path / tier
            out.mkdir()
            env = arm_environment(out, profile, lock.fileno(), tier)
            assert native_wrapper.run_native(profile, env, MockRuntime(profile, out), lock.fileno()) == 0
            config = {"device": "mock", "model": "mock", "revision": profile["revision"]}
            receipt = run_final.validate_receipt(out / "wrapper.json", config, tier)
            run_final.read_result(out / "result.json", tier, 5)
            rows.append(receipt)
            assert receipt["server"]["completion_requests"] == 1
        assert rows[0]["server"]["pid"] != rows[1]["server"]["pid"]
        assert run_final.fixed_identity(rows[0]) == run_final.fixed_identity(rows[1])


@pytest.mark.parametrize("mode,fail_cleanup", [("partial", False), ("timeout", False), ("normal", True)])
def test_client_failure_timeout_and_unknown_cleanup_preserve_failed_receipt(profile, tmp_path, mode, fail_cleanup):
    profile = fake_server(tmp_path, profile, mode)
    out = tmp_path / "arm"
    out.mkdir()
    with (tmp_path / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = arm_environment(out, profile, lock.fileno(), timeout="1")
        assert native_wrapper.run_native(profile, env, MockRuntime(profile, out, fail_cleanup), lock.fileno()) == 1
    receipt = json.loads((out / "wrapper.json").read_text())
    assert receipt["status"] == "failed" and receipt["server"]["owned_processes_gone"]
    assert receipt["server"]["resource_scope_gone"] is (not fail_cleanup)


def test_profile_hash_binds_settings_beyond_the_summary(profile):
    before = common.effective_settings(profile)
    profile["args"] = ["-t", "4"]
    assert before != common.effective_settings(profile)


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM])
def test_interruption_during_request_cleans_both_owned_processes(profile, tmp_path, signum):
    profile = fake_server(tmp_path, profile, "timeout")
    out = tmp_path / "arm"
    out.mkdir()
    cancel = threading.Event()
    def interrupt():
        while not cancel.wait(.01):
            log = out / "server.log"
            if log.exists() and "processing task" in log.read_text():
                os.kill(os.getpid(), signum)
                return
    thread = threading.Thread(target=interrupt)
    with (tmp_path / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        thread.start()
        try:
            env = arm_environment(out, profile, lock.fileno(), timeout="3")
            assert native_wrapper.run_native(profile, env, MockRuntime(profile, out), lock.fileno()) == 1
        finally:
            cancel.set()
            thread.join()
    receipt = json.loads((out / "wrapper.json").read_text())
    assert receipt["signals"] == [signum]
    assert receipt["server"]["owned_processes_gone"] and receipt["server"]["resource_scope_gone"]


def test_controller_disconnect_is_not_a_successful_guard_check():
    class Disconnected:
        def check(self):
            raise InterruptedError("controller connection lost")
    guard = native_wrapper.Guard(None, [], time.monotonic() + 10, Disconnected())
    with pytest.raises(InterruptedError, match="controller connection lost"):
        guard.check()


def test_ordered_linux_alias_list_is_verified_against_all_shards(profile, tmp_path):
    second = tmp_path / "dense.tsv"
    second.write_text(Path(profile["alias"]["path"]).read_text())
    profile["aliases"] = [profile.pop("alias"), spec(second)]
    profile["environment"]["GGML_MOE_EXPERT_GGUF_ALIAS_TSV"] = ":".join(x["path"] for x in profile["aliases"])
    common.validate_profile(profile)
    assert common.verify_inputs(profile)["alias_entries"] == 2
    profile["platform"] = "macos"
    with pytest.raises(ValueError, match="exactly one alias"):
        common.validate_profile(profile)


def test_prelaunch_failure_confirms_empty_arm_without_faking_success(profile, tmp_path):
    path = tmp_path / "profile.json"
    profile["args"] = ["--parallel", "2"]
    path.write_text(json.dumps(profile))
    out = tmp_path / "arm"
    out.mkdir()
    env = dict(os.environ, **arm_environment(out, profile, 9))
    result = subprocess.run([sys.executable, str(Path(native_wrapper.__file__)), "--profile", str(path)], env=env, capture_output=True)
    assert result.returncode != 0
    receipt = json.loads((out / "wrapper.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["server"] == {"owned_processes_gone": True, "resource_scope_gone": True}
