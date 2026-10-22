import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_common as common
import native_wrapper


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
            "resources": {"memory_budget_bytes": 4 << 30, "host_reserve_bytes": 3 << 30, "minimum_cpus": 2},
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
    lambda p: p["environment"].update(DYLD_LIBRARY_PATH="other"),
    lambda p: p["resources"].update(memory_budget_bytes=True),
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


def test_same_inherited_lock_stays_held_through_child(tmp_path):
    path = tmp_path / "device.lock"
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = common.inherited_lock({"SSDASH_LOCK_FD": str(lock.fileno())})
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
