import argparse
import fcntl
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_chat
import run_final
import stream_chat
from test_stream_chat import fixture


def records(count=256, input_count=4200):
    rows = fixture(count)
    rows[-1]["event"]["tokens_evaluated"] = input_count
    rows[-1]["event"]["timings"].update(prompt_n=input_count, predicted_ms=1000)
    return rows


@pytest.mark.parametrize("context,slots,success", [(8192, 1, True), (4096, 1, False), (8192, 2, False)])
def test_formal_http_accepts_actual_4k_count_and_checks_capacity(tmp_path, context, slots, success):
    received = []
    ids = [7] * 4200  # The nominal 4K prompt can exceed 4096 after templating.

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            assert self.path == "/props"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"total_slots": slots,
                "default_generation_settings": {"n_ctx": context}}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            if self.path == "/apply-template":
                data = json.dumps({"prompt": "rendered", "tokens": ids}).encode()
            else:
                data = b"".join(b"data: " + json.dumps(row["event"]).encode() + b"\n\n" for row in records())
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        args = argparse.Namespace(url=f"http://127.0.0.1:{server.server_port}", tier="4096",
                                  timeout=5, output=tmp_path / "result.json")
        assert final_chat.run(args) == (0 if success else 1)
        result = json.loads(args.output.read_text())
        assert result["status"] == ("passed" if success else "failed")
        if success:
            assert received[1][1]["prompt"] == ids
            assert received[1][1]["n_predict"] == 256
            assert result["input_tokens"] == 4200
            assert run_final.read_result(args.output, "4096", 5) == result
            assert len(result["tokens"]) == 256
        else:
            assert [endpoint for endpoint, _ in received] == ["/apply-template"]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("timeout", [0, .5, 7201, float("inf"), float("nan"), True])
def test_formal_deadline_is_explicit_and_bounded(timeout):
    with pytest.raises(ValueError):
        final_chat.bounded_timeout(timeout)


def test_short_client_cap_remains_unchanged(tmp_path):
    args = argparse.Namespace(input_tokens=4200, output_tokens=256, output=tmp_path / "never.json")
    with pytest.raises(ValueError, match="short protocol"):
        stream_chat.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), 1e-320, True])
def test_native_timing_is_required_and_not_inferred(value):
    result = stream_chat.summarize(records(), 0, 257_000_000_000, [7] * 4200, 256)
    result["timings"]["predicted_ms"] = value
    with pytest.raises(ValueError):
        final_chat.validate_timings(result)


def test_fixed_prompt_hash_and_window_denominators(tmp_path, monkeypatch):
    assert [stream_chat.digest(final_chat.prompt_path(t).read_bytes()) for t in run_final.ORDER] == [
        final_chat.PROMPTS[t][1] for t in run_final.ORDER]
    monkeypatch.setitem(final_chat.PROMPTS, "512", ("prompt-512.txt", "0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        final_chat.prompt_path("512")
    rows = records()
    rows[-1]["monotonic_ns"] = 999_000_000_000  # Stop delay must not affect decode windows.
    summary = stream_chat.summarize(rows, 0, 999_000_000_000, [7] * 4200, 256)
    assert summary["metrics"]["decode_first128_tps"] == 1
    assert summary["metrics"]["decode_second128_tps"] == 1
    assert summary["metrics"]["decode_tps"] == 1
    for count in (127, 128, 255):
        with pytest.raises(ValueError, match="requested completion"):
            stream_chat.summarize(records(count), 0, 999_000_000_000, [7] * 4200, 256)


def write_artifacts(directory, tier, config):
    directory.mkdir()
    ids = [7] * (600 if tier == "512" else 4200)
    raw_prompt = final_chat.prompt_path(tier).read_bytes()
    rows = records(input_count=len(ids))
    result = stream_chat.summarize(rows, 0, 257_000_000_000, ids, 256)
    final_chat.validate_timings(result)
    result.update(status="passed", prompt_ids=ids, prompt_sha256=stream_chat.digest(raw_prompt))
    protocol = {"protocol": final_chat.PROTOCOL, "tier": tier,
        "prompt_sha256": result["prompt_sha256"], "output_tokens": 256,
        "timeout_s": config["request_timeout_s"], "prompt_ids": ids, "input_tokens": len(ids),
        "server_context_tokens": 8192, "server_props": {"total_slots": 1,
            "default_generation_settings": {"n_ctx": 8192}},
        "chat_request": {"messages": [{"role": "user", "content": raw_prompt.decode()}],
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"},
        "request": {"prompt": ids, "n_predict": 256, "stream": True,
            "temperature": 0, "top_k": 1, "samplers": ["top_k"], "seed": 0,
            "n_probs": 5, "post_sampling_probs": False, "ignore_eos": True,
            "cache_prompt": False, "return_tokens": True, "timings_per_token": True}}
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "result.protocol.json").write_text(json.dumps(protocol))
    (directory / "result.events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "result.events.sse").write_bytes(b"".join(
        b"data: " + json.dumps(row["event"]).encode() + b"\n\n" for row in rows))
    (directory / "host.log").write_text("Synthetic evidence; no device or model was used.\n")
    receipt = {"schema_version": 1, "status": "passed", "protocol": final_chat.PROTOCOL,
        "tier": tier, "device": config["device"], "model": config["model"], "revision": config["revision"],
        "server": {"pid": int(tier), "start_identity": tier, "argv": ["synthetic-server"],
            "fresh_process": True, "completion_requests": 1, "returncode": 0,
            "owned_processes_gone": True, "resource_scope_gone": True},
        "build": {"revision": config["revision"], "binary_sha256": "a" * 64,
            "actual_loaded_libraries_verified": True,
            "loaded_libraries": [{"path": "synthetic-lib", "sha256": "b" * 64}]},
        "model_identity": {"verified": True, "shards": [{"path": "synthetic.gguf", "bytes": 1, "sha256": "c" * 64}]},
        "host": {"hardware": "synthetic", "os": "synthetic", "memory_budget_bytes": 1024,
            "resource_checks_passed": True, "effective_settings": {"source_mode": "direct-gguf",
                "h2d_limit_bytes_per_second": None, "expert_cache_budget_bytes": 0,
                "vram_budget_bytes": 0, "unified_memory_budget_bytes": 0, "gpu_reserve_bytes": 0},
            "os_page_cache_treatment": "not applicable to synthetic fixture"},
        "evidence": [{"path": "host.log", "sha256": stream_chat.digest((directory / "host.log").read_bytes())}]}
    (directory / "wrapper.json").write_text(json.dumps(receipt))


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "wrapper.py"
    script.write_text('''import os, pathlib, shutil, signal, sys, time
out = pathlib.Path(os.environ['SSDASH_RUN_DIR'])
source = pathlib.Path(sys.argv[1]) / os.environ['SSDASH_TIER']
for child in source.iterdir(): shutil.copyfile(child, out / child.name)
if sys.argv[2] == 'wait':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    (out / 'ready').touch()
    time.sleep(30)
if sys.argv[2] == 'fail': sys.exit(7)
''')
    config = {"device": "synthetic", "model": "synthetic", "revision": "a" * 40,
              "argv": [sys.executable, str(script), str(tmp_path / "fixtures"), "ok"],
              "request_timeout_s": 1, "arm_timeout_s": 2}
    (tmp_path / "fixtures").mkdir()
    for tier in run_final.ORDER:
        write_artifacts(tmp_path / "fixtures" / tier, tier, config)
    return tmp_path, config


def command(root, config):
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return [sys.executable, str(Path(run_final.__file__)), "--config", str(path),
            "--output", str(root / "run"), "--lock", str(root / "device.lock"), "--stop", str(root / "STOP")]


def test_fixed_once_only_order_and_raw_rows(setup):
    root, config = setup
    subprocess.run(command(root, config), check=True)
    result = json.loads((root / "run/summary.json").read_text())
    assert result["complete"] and result["samples_per_tier"] == 1
    assert [row["tier"] for row in result["rows"]] == ["512", "4096"]
    assert [row["result"]["input_tokens"] for row in result["rows"]] == [600, 4200]
    assert "mean" not in json.dumps(result) and "B_over_A" not in json.dumps(result)
    assert all(row["owned_group_cleanup_confirmed"] for row in result["rows"])
    assert subprocess.run(command(root, config), capture_output=True).returncode == 2


@pytest.mark.parametrize("change", [
    lambda r: r["server"].update(completion_requests=2),
    lambda r: r["server"].update(owned_processes_gone=False),
    lambda r: r["server"].update(resource_scope_gone=False),
    lambda r: r["server"].update(returncode=-9),
    lambda r: r["build"].update(actual_loaded_libraries_verified=False),
    lambda r: r["build"].update(loaded_libraries=[]),
    lambda r: r["model_identity"]["shards"][0].update(sha256="x"),
    lambda r: r["host"].update(resource_checks_passed=False),
    lambda r: r["host"].update(os_page_cache_treatment=""),
    lambda r: r["host"]["effective_settings"].update(source_mode="legacy-pack"),
    lambda r: r["host"]["effective_settings"].update(h2d_limit_bytes_per_second=123),
    lambda r: r["host"]["effective_settings"].update(expert_cache_budget_bytes=-1),
    lambda r: r["evidence"][0].update(sha256="0" * 64),
    lambda r: r["evidence"][0].update(path="../outside"),
    lambda r: r.update(server=None),
])
def test_wrapper_receipt_is_validated_not_just_present(setup, change):
    root, config = setup
    path = root / "fixtures/512/wrapper.json"
    receipt = json.loads(path.read_text())
    change(receipt)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        run_final.validate_receipt(path, config, "512")


@pytest.mark.parametrize("file,change", [
    ("result.json", lambda r: r["metrics"].update(decode_tps=9)),
    ("result.json", lambda r: r.update(status="failed")),
    ("result.protocol.json", lambda r: r["request"].update(cache_prompt=True)),
    ("result.protocol.json", lambda r: r["request"].update(n_predict=255)),
    ("result.protocol.json", lambda r: r["server_props"].update(total_slots=2)),
])
def test_result_protocol_and_raw_evidence_are_revalidated(setup, file, change):
    root, config = setup
    path = root / "fixtures/512" / file
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run_final.read_result(root / "fixtures/512/result.json", "512", 1)


def test_raw_sse_truncation_is_not_a_success(setup):
    root, _ = setup
    path = root / "fixtures/512/result.events.sse"
    path.write_bytes(path.read_bytes().rsplit(b"data:", 1)[0])
    with pytest.raises(ValueError, match="raw SSE"):
        run_final.read_result(path.parent / "result.json", "512", 1)


@pytest.mark.parametrize("kind", ["result", "protocol", "props", "settings", "timings"])
def test_nonobject_inputs_keep_failed_attempt_and_summary(setup, kind):
    root, config = setup
    directory = root / "fixtures/512"
    if kind in ("result", "protocol"):
        path = directory / ("result.json" if kind == "result" else "result.protocol.json")
        path.write_text("[]")
    elif kind in ("props", "settings"):
        path = directory / "result.protocol.json"
        protocol = json.loads(path.read_text())
        if kind == "props":
            protocol["server_props"] = []
        else:
            protocol["server_props"]["default_generation_settings"] = []
        path.write_text(json.dumps(protocol))
    else:
        rows = records(input_count=600)
        rows[-1]["event"]["timings"] = []
        (directory / "result.events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "result.events.sse").write_bytes(b"".join(
            b"data: " + json.dumps(row["event"]).encode() + b"\n\n" for row in rows))
    assert subprocess.run(command(root, config)).returncode == 1
    report = json.loads((root / "run/summary.json").read_text())
    attempts = [json.loads(line) for line in (root / "run/attempts.jsonl").read_text().splitlines()]
    assert [r["status"] for r in report["rows"]] == ["failed", "not_run"]
    assert len(attempts) == 2 and not report["complete"]
    assert "ValueError" in attempts[0]["error"]
    assert attempts[0]["owned_group_cleanup_confirmed"] and attempts[0]["wrapper_cleanup_confirmed"]


@pytest.mark.parametrize("kind", ["settings", "same-server", "missing-cleanup"])
def test_changed_settings_reused_server_and_missing_cleanup_fail(setup, kind):
    root, config = setup
    path = root / "fixtures/4096/wrapper.json"
    receipt = json.loads(path.read_text())
    if kind == "settings":
        receipt["host"]["effective_settings"]["expert_cache_budget_bytes"] = 1024
    elif kind == "same-server":
        receipt["server"].update(pid=512, start_identity="512")
    else:
        receipt["server"]["resource_scope_gone"] = False
    path.write_text(json.dumps(receipt))
    assert subprocess.run(command(root, config)).returncode == 1
    report = json.loads((root / "run/summary.json").read_text())
    assert [r["status"] for r in report["rows"]] == ["passed", "failed"]
    assert not report["complete"]
    assert (root / "STOP").exists() == (kind == "missing-cleanup")


def test_failure_preserved_and_second_tier_not_retried(setup):
    root, config = setup
    config["argv"][-1] = "fail"
    assert subprocess.run(command(root, config)).returncode == 1
    rows = json.loads((root / "run/summary.json").read_text())["rows"]
    assert [row["status"] for row in rows] == ["failed", "not_run"]
    assert not (root / "run/4096").exists()


def test_existing_lock_and_stop_preserve_inode(setup):
    root, config = setup
    path = root / "device.lock"
    path.write_text("existing owner\n")
    inode = path.stat().st_ino
    with path.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert subprocess.run(command(root, config), capture_output=True).returncode == 2
    assert not (root / "run").exists()
    (root / "STOP").touch()
    assert subprocess.run(command(root, config), capture_output=True).returncode == 2
    assert path.stat().st_ino == inode and path.read_text() == "existing owner\n"


@pytest.mark.parametrize("stop", ["STOP", "HUP", "deadline"])
def test_active_stop_signal_and_deadline_cleanup(setup, stop):
    root, config = setup
    config["argv"][-1] = "wait"
    process = subprocess.Popen(command(root, config))
    try:
        ready = root / "run/512/ready"
        until = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert ready.exists()
        if stop == "STOP":
            (root / "STOP").touch()
        elif stop == "HUP":
            process.send_signal(signal.SIGHUP)
        assert process.wait(timeout=8) == 1
        report = json.loads((root / "run/summary.json").read_text())
        assert [r["status"] for r in report["rows"]] == ["failed", "not_run"]
        assert report["rows"][0]["owned_group_cleanup_confirmed"]
        with (root / "device.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)
