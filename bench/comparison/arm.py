#!/usr/bin/env python3
"""One controlled Linux release-comparison arm using the fixed 512/256 protocol.

This is an independent comparison, never an assertion that CI or FINAL passed.
Source/build/launch specifications are reviewed and frozen before invoking it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.error
import urllib.request

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import final_chat
import native_common as common
import native_linux
import run_final
from inferssd.engine.owned import OwnedProcess
from inferssd.launch.types import Resolution
from inferssd.platform.policy import Policy
from inferssd.platform.run_directory import reserve
from inferssd.platform.worker import cleanup_confirmed


def inputs(config):
    observed = {}
    for name in ("binary", "template", "alias", "source_receipt"):
        spec = config[name]
        actual = common.file_identity(spec["path"], True)
        if actual["sha256"] != spec["sha256"]:
            raise ValueError("frozen input changed: " + name)
        observed[name] = actual
    observed["images"] = [common.verify_file(spec) for spec in config["images"]]
    build = json.loads(Path(config["source_receipt"]["path"]).read_text())
    if build["source"] != config["source"]:
        raise ValueError("source identity differs from its pinned build receipt")
    built = {row["resolved_path"]: row["sha256"] for row in build["binaries"]}
    for spec in (config["binary"], *config["images"]):
        path = Path(spec["path"]).resolve(strict=True)
        if str(path).startswith(("/usr/lib/", "/lib/")):
            continue
        if built.get(str(path)) != spec["sha256"]:
            raise ValueError("project ELF is not bound to the exact-source build receipt")
    observed["shards"] = []
    for spec in config["shards"]:
        actual = common.file_identity(spec["path"])
        if any(actual[key] != spec[key] for key in ("path", *common.FILE_FIELDS)):
            raise ValueError("GGUF file identity changed")
        observed["shards"].append(actual)
    sizes = {row["path"]: row["bytes"] for row in observed["shards"]}
    rows = 0
    expected_tensors = {row["name"]: row for row in config["expert_tensors"]}
    seen = set()
    with Path(config["alias"]["path"]).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            path = str(Path(row["source_path"]).resolve(strict=True))
            start, size = int(row["offset"]), int(row["nbytes"])
            if path not in sizes or start < 0 or size <= 0 or start + size > sizes[path]:
                raise ValueError("expert alias range escaped the same GGUF set")
            tensor = expected_tensors[row["tensor"]]
            index = int(row["expert_idx"])
            count = tensor["shape"][2]
            if (not 0 <= index < count or tensor["nbytes"] % count or size != tensor["nbytes"] // count
                    or path != tensor["path"] or start != tensor["offset"] + index * size
                    or (row["tensor"], index) in seen):
                raise ValueError("expert alias differs from the actual GGUF tensor table")
            seen.add((row["tensor"], index))
            rows += 1
    if rows != sum(row["shape"][2] for row in expected_tensors.values()):
        raise ValueError("expert alias does not cover the complete GGUF expert set")
    if not rows:
        raise ValueError("empty GGUF alias")
    observed["alias_rows"] = rows
    if "typed_settings" in config:
        observed["typed_settings"] = common.verify_file(config["typed_settings"])
    return observed


def snapshot(process, config, output, name):
    pid = process.state()["pid"]
    if type(pid) is not int or process.poll() is not None:
        raise ValueError("owned live server is required")
    raw = Path(f"/proc/{pid}/stat").read_text()
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    environment = dict(item.decode().split("=", 1) for item in
                       Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in item)
    if cmdline != config["argv"] or any(environment.get(k) != v for k, v in config["environment"].items()):
        raise ValueError("actual server launch differs from the frozen specification")
    forbidden = {k: v for k, v in environment.items() if k.startswith(("GGML_", "LLAMA_", "LD_", "H2D_"))}
    if forbidden != {k: v for k, v in config["environment"].items() if k.startswith(("GGML_", "LLAMA_", "LD_", "H2D_"))}:
        raise ValueError("unreviewed native launch environment")
    libraries = native_linux.mapped_images(pid, config, output / (name + "-images"))
    limits = Path(f"/proc/{pid}/limits").read_text()
    native_linux.check_server_memlock(limits, native_linux.memlock_limits(config["policy"]["minimum_memlock_bytes"]))
    value = {"pid": pid, "start_ticks": raw[raw.rfind(")") + 2:].split()[19],
             "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "cmdline": cmdline, "environment": environment, "limits": limits,
             "cpus": sorted(os.sched_getaffinity(pid)), "libraries": libraries,
             "host_meminfo": Path("/proc/meminfo").read_text(),
             "process_io": Path(f"/proc/{pid}/io").read_text(),
             "cgroup": native_linux.cgroup_snapshot(pid, config["policy"]["memory_budget_bytes"])}
    if value["cpus"] != config["effective_contract"]["cpu_affinity"]:
        raise ValueError("actual CPU affinity differs from the reviewed resource contract")
    common.save(output / (name + ".json"), value)
    return value


def run(config, output, lock_fd=None):
    if config.get("purpose") != "release-comparison" or config.get("tier") != "512":
        raise ValueError("this helper only supports the independent 512 comparison")
    if config.get("output_tokens") != 256 or config.get("h2d_limit_bytes_per_second") is not None:
        raise ValueError("256 outputs and unrestricted H2D are required")
    if config.get("source_mode") != "direct-gguf" or any(config["environment"].get(k) for k in common.LEGACY_SOURCES):
        raise ValueError("only direct GGUF is accepted")
    if any(k.startswith("H2D_") for k in config["environment"]):
        raise ValueError("H2D throttling environment is not accepted")
    expected_options = {"-m": config["shards"][0]["path"], "--chat-template-file": config["template"]["path"],
                        "--host": "127.0.0.1", "--port": str(config["port"]), "-np": "1",
                        "--reasoning": "off", "--fit": "off", "--cache-ram": "0"}
    argv = config["argv"]
    if argv[0] != config["binary"]["path"] or "--no-warmup" not in argv:
        raise ValueError("reviewed binary and no warmup are required")
    for option, expected in expected_options.items():
        if argv.count(option) != 1 or argv[argv.index(option) + 1] != expected:
            raise ValueError("frozen launch option differs: " + option)
    if "typed_settings" in config:
        if argv.count("--ssd-settings") != 1 or argv[argv.index("--ssd-settings") + 1] != config["typed_settings"]["path"]:
            raise ValueError("actual typed settings file differs")
        if Path(config["telemetry_path"]) != output / "native-events.jsonl" or Path(config["telemetry_path"]).exists():
            raise ValueError("native telemetry must use the fresh arm output")
        if argv.count("--ssd-telemetry") != 1 or argv[argv.index("--ssd-telemetry") + 1] != config["telemetry_path"]:
            raise ValueError("actual telemetry destination differs")
    elif config["environment"].get("GGML_MOE_EXPERT_GGUF_ALIAS_TSV") != config["alias"]["path"]:
        raise ValueError("legacy source must be the exact reviewed alias")
    policy = Policy.read(config["policy"])
    final_chat.bounded_timeout(config["request_timeout_s"])
    final_chat.bounded_timeout(config["load_timeout_s"])
    if policy.runtime_seconds <= config["request_timeout_s"] + config["load_timeout_s"]:
        raise ValueError("owned deadline must also cover cleanup")
    if policy.platform != "linux":
        raise ValueError("Linux comparison only")
    with socket.socket() as check:
        check.bind(("127.0.0.1", config["port"]))
    reservation = reserve(str(output))
    env = config["environment"]
    resolution = Resolution(tuple(config["argv"]), tuple(env.items()), tuple(env.items()),
                            tuple(config["scope"]), (), (), None, 0)
    process = None
    result = {"purpose": "release-comparison", "status": "failed", "performance_acceptance": False,
              "source": config["source"], "completion_requests": 0, "full_model_sha256_verified": False}
    def interrupted(number, _frame):
        raise InterruptedError(f"controller signal {number}")
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        common.save(output / "arm-config.json", config)
        before_inputs = inputs(config)
        common.save(output / "inputs-before.json", before_inputs)
        process = OwnedProcess(resolution, str(output), policy, reservation=reservation,
                               inherited_lock_fd=lock_fd)
        deadline = time.monotonic() + config["load_timeout_s"]
        while True:
            if process.poll() is not None:
                raise RuntimeError("owned server exited before readiness")
            if time.monotonic() >= deadline or Path(policy.stop_path).exists():
                raise TimeoutError("server readiness deadline or STOP")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config['port']}/health", timeout=1) as response:
                    if response.status == 200:
                        process.confirm_listener(config["port"])
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(.1)
        before = snapshot(process, config, output, "runtime-before")
        marker = "processing task, is_child = "
        if marker in (output / "server.log").read_text():
            raise ValueError("completion occurred before the first comparison request")
        result["completion_requests"] = 1
        common.save(output / "request-started.json", {"completion_requests": 1, "monotonic_ns": time.monotonic_ns()})
        code = final_chat.run(argparse.Namespace(url=f"http://127.0.0.1:{config['port']}", tier="512",
                                               output=output / "result.json", timeout=config["request_timeout_s"]))
        if code:
            raise RuntimeError("the single completion failed; preserve its original artifacts")
        measured = run_final.read_result(output / "result.json", "512", config["request_timeout_s"])
        after = snapshot(process, config, output, "runtime-after")
        stable = ("pid", "start_ticks", "boot_id", "cmdline", "environment", "limits", "cpus", "libraries")
        if any(before[k] != after[k] for k in stable):
            raise ValueError("server identity/configuration changed across the request")
        native_linux.unchanged_oom(before["cgroup"], after["cgroup"])
        if before_inputs != inputs(config):
            raise ValueError("frozen build/model/input identity changed across the request")
        if (output / "server.log").read_text().count(marker) != 1:
            raise ValueError("native log does not confirm exactly one completion")
        if "typed_settings" in config:
            from inferssd.telemetry.jsonl import effective_settings, read_events
            with Path(config["telemetry_path"]).open() as stream:
                actual_settings = effective_settings(read_events(stream))
            if actual_settings["values"] != common.typed_snapshot(config).document()["values"]:
                raise ValueError("native effective settings differ from the frozen typed input")
        common.save(output / "inputs-after.json", inputs(config))
        result.update(status="passed", result=measured)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is None:
            result["cleanup_confirmed"] = not (output / "ownership.json").exists()
        else:
            try:
                process.stop()
                result["cleanup_confirmed"] = True
            except BaseException as error:
                result.update(status="failed", cleanup_confirmed=cleanup_confirmed(output / "ownership.json"),
                              cleanup_error=f"{type(error).__name__}: {error}")
        common.save(output / "arm.json", result)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(json.loads(args.config.read_text()), args.output.resolve())
    print(json.dumps({k: v for k, v in value.items() if k != "result"}, sort_keys=True))
    raise SystemExit(0 if value["status"] == "passed" else 1)
