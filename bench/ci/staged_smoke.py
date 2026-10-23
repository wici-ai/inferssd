#!/usr/bin/env python3
"""One operational native CI smoke through the real isolated stage interface.

This deliberately records no benchmark verdict. One short Muse completion
unblocks integration while the other model downloads; final acceptance remains
the separately pinned base/candidate performance protocol.
"""

import argparse
import os
import subprocess
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import load, save
from ci.stage_session import StageSession
from ci.staged_chat import NativeHTTP


def collect_log(session, stage, output):
    for stream in ("stdout", "stderr"):
        offset = 0
        with (output / (stream + ".log")).open("xb") as target:
            while True:
                data, offset, eof = session.output(stage, stream, offset)
                target.write(data)
                if offset > 64 * 1024 * 1024:
                    raise ValueError("native smoke log bound exceeded")
                if eof or not data:
                    break



def host_memory(output, name):
    if sys.platform != "darwin":
        return
    values = {}
    for key, argv in (("vm_stat", ["/usr/bin/vm_stat"]),
                      ("swap", ["/usr/sbin/sysctl", "vm.swapusage"]),
                      ("memory_bytes", ["/usr/sbin/sysctl", "-n", "hw.memsize"])):
        value = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
        values[key] = value.stdout
    save(output / name, dict(platform="darwin", monotonic_ns=time.monotonic_ns(), **values))


def smoke(session, config, output, *, stage_id="candidate-smoke"):
    """The manifest independently fixes source/models/argv admission and lease."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, purpose="integration-smoke", model=config["model"],
                  revision=config["revision"], status="failed", benchmark_samples=0,
                  performance_acceptance=False, output_tokens=8)
    stage = None
    try:
        stage = session.start(stage_id, "server", config["argv"],
                              config["environment"], config["cwd"], config["timeout_s"])
        save(output / "started.json", stage.__dict__)
        deadline = time.monotonic() + config["timeout_s"]
        http = NativeHTTP(session, stage, config["port"])
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("native smoke startup deadline")
            state = session.call(stage, "stage.poll")
            if state.get("running") is not True:
                raise RuntimeError("native server exited before readiness")
            probe = session.call(stage, "stage.probe", port=config["port"])
            if probe.get("ready", True):
                health = http.request("GET", "/health", None, 2)
                if health["status"] == 200:
                    break
            time.sleep(0.2)
        before = session.call(stage, "stage.probe", port=config["port"])
        save(output / "runtime-before.json", before)
        host_memory(output, "host-memory-before-request.json")
        chat = {"messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
        templated = http.json("POST", "/apply-template", chat, 10)
        ids = templated.get("tokens")
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("native template did not return authoritative input tokens")
        props = http.json("GET", "/props", None, 10)
        if props.get("total_slots") != 1 or props.get("default_generation_settings", {}).get("n_ctx", 0) < len(ids) + 8:
            raise ValueError("native server slot/context differs from smoke contract")
        body = dict(prompt=ids, n_predict=8, stream=False, temperature=0, top_k=1,
                    samplers=["top_k"], seed=0, ignore_eos=True, cache_prompt=False,
                    return_tokens=True)
        save(output / "request.json", body)
        response = http.json("POST", "/completion", body, config["request_timeout_s"])
        save(output / "response.json", response)
        if response.get("tokens_predicted") != 8 or response.get("truncated") is not False or response.get("stop_type") != "limit":
            raise ValueError("native completion did not finish the eight-token smoke")
        after = session.call(stage, "stage.probe", port=config["port"])
        save(output / "runtime-after.json", after)
        host_memory(output, "host-memory-after-request.json")
        for key in ("worker_pid", "worker_uid", "birth_identity", "scope_identity"):
            if before.get(key) != after.get(key) or before.get(key) != getattr(stage, key):
                raise ValueError("native runtime identity changed during smoke")
        result.update(status="passed", input_tokens=len(ids), completion_requests=1)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if stage is not None:
            try:
                stopped = session.call(stage, "stage.stop", timeout=50)
                save(output / "stopped.json", stopped)
                collect_log(session, stage, output)
                if stopped.get("cleanup_confirmed") is not True or stopped.get("returncode") != 0:
                    raise RuntimeError("native smoke did not stop normally and cleanly")
                result["cleanup_confirmed"] = True
                host_memory(output, "host-memory-after-cleanup.json")
            except Exception as error:
                result.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        save(output / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with StageSession(args.socket, args.job_id, controller_uid=os.getuid()) as session:
        result = smoke(session, load(args.config), args.output)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
