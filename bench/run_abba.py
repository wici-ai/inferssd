#!/usr/bin/env python3
"""Run one explicit short-request ABBA sequence under a cooperating device lock."""
import argparse
import fcntl
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import time
import uuid

ORDER = ("A", "B", "B", "A")


def load_config(path):
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("protocol"), dict):
        raise ValueError("config needs a protocol object")
    protocol = config["protocol"]
    prompt_hash = protocol.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
        raise ValueError("protocol needs the fixed prompt SHA-256")
    for field, maximum in (("input_tokens", 1024), ("output_tokens", 512)):
        value = protocol[field]
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f"{field} must be 1..{maximum}; this driver is for short requests")
    if not isinstance(config.get("arms"), dict) or set(config["arms"]) != {"A", "B"}:
        raise ValueError("exactly arms A and B are required")
    for arm in config["arms"].values():
        if not isinstance(arm, dict) or not isinstance(arm.get("argv"), list) or not arm["argv"] or not all(
                isinstance(word, str) and word for word in arm["argv"]):
            raise ValueError("each argv must be a nonempty string array")
        if not isinstance(arm.get("revision"), str) or not arm["revision"]:
            raise ValueError("each arm needs a declared revision")
    return config


def read_result(path, protocol):
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    for field in ("input_tokens", "output_tokens"):
        if type(result.get(field)) is not int or result[field] != protocol[field]:
            raise ValueError(f"result {field} differs from the fixed protocol")
    if result.get("prompt_sha256") != protocol["prompt_sha256"]:
        raise ValueError("result prompt SHA-256 differs from the fixed protocol")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("result must contain named numeric metrics")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip() or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("metrics must have string keys and finite nonnegative values")
    return result


def terminate_owned_group(process):
    # start_new_session makes this PID the ID of the group created for this arm.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # The group leader may exit before a child that ignores TERM. Keep checking
    # the whole owned group before releasing the device lock.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def owned_group_exists(process):
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(config, output, lock_path, stop_path, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("arm timeout must be finite and positive")
    # Compatible with `exec 9>device.lock; flock -n 9`. Never unlink or truncate
    # the shared inode; only this process's descriptor is released on exit.
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("device lock is held; no arm was started") from error
        if stop_path.exists():
            raise RuntimeError("STOP exists; no arm was started")
        output.mkdir(parents=True, exist_ok=False)
        run_id = str(uuid.uuid4())
        manifest = {"schema_version": 1, "run_id": run_id, "order": ORDER,
                    "config": config, "owner_pid": os.getpid(), "started_unix_s": time.time()}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        interrupted = []
        handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda signum, _frame: interrupted.append(signum))
        rows = []
        abort_reason = None
        metric_names = None
        try:
            with (output / "arms.jsonl").open("x") as ledger:
                for index, label in enumerate(ORDER):
                    row = {"schema_version": 1, "run_id": run_id, "index": index + 1,
                           "arm": label, "revision": config["arms"][label]["revision"]}
                    if stop_path.exists() or interrupted:
                        abort_reason = "STOP or termination requested"
                    if abort_reason:
                        row.update(status="not_run", error=abort_reason)
                    else:
                        arm_dir = output / f"{index + 1:02d}-{label}"
                        arm_dir.mkdir()
                        result_path = arm_dir / "result.json"
                        env = dict(os.environ, SSDASH_ARM=label, SSDASH_RUN_ID=run_id,
                                   SSDASH_RUN_DIR=str(arm_dir), SSDASH_RESULT_PATH=str(result_path))
                        start = time.monotonic()
                        process = None
                        try:
                            with (arm_dir / "stdout.log").open("wb") as log:
                                process = subprocess.Popen(config["arms"][label]["argv"], env=env,
                                                           stdout=log, stderr=subprocess.STDOUT,
                                                           start_new_session=True)
                                while process.poll() is None:
                                    if stop_path.exists() or interrupted:
                                        raise InterruptedError("STOP or termination requested")
                                    if time.monotonic() - start > timeout:
                                        raise TimeoutError("arm timeout")
                                    time.sleep(0.1)
                                row["returncode"] = process.returncode
                                if process.returncode:
                                    raise RuntimeError(f"arm exited {process.returncode}")
                                if owned_group_exists(process):
                                    raise RuntimeError("arm exited while descendants remain in its process group")
                            row["result"] = read_result(result_path, config["protocol"])
                            names = set(row["result"]["metrics"])
                            if metric_names is not None and names != metric_names:
                                raise ValueError("metric names differ between arms")
                            metric_names = names
                            row["status"] = "passed"
                        except (OSError, ValueError, KeyError, RuntimeError) as error:
                            row.update(status="failed", error=str(error))
                            abort_reason = str(error)
                        finally:
                            if process is not None:
                                if process.poll() is None or abort_reason is not None:
                                    terminate_owned_group(process)
                                row["returncode"] = process.returncode
                            row["elapsed_s"] = time.monotonic() - start
                            row["result_file"] = str(result_path.relative_to(output))
                    rows.append(row)
                    ledger.write(json.dumps(row, allow_nan=False) + "\n")
                    ledger.flush()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        complete = (len(rows) == 4 and all(row["status"] == "passed" for row in rows)
                    and not interrupted and not stop_path.exists())
        summary = {"schema_version": 1, "run_id": run_id, "complete": complete,
                   "expected_arms": 4, "recorded_arms": len(rows), "order": ORDER,
                   "passed_arms": sum(row["status"] == "passed" for row in rows)}
        if complete:
            comparisons = {}
            for metric in sorted(metric_names):
                values = {label: [row["result"]["metrics"][metric] for row in rows if row["arm"] == label]
                          for label in ("A", "B")}
                means = {label: sum(value / 2 for value in values[label]) for label in ("A", "B")}
                ratio = means["B"] / means["A"] if means["A"] else None
                comparisons[metric] = {"raw": values, "mean": means,
                                       "B_over_A": ratio if ratio is None or math.isfinite(ratio) else None}
            summary["metrics"] = comparisons
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--arm-timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(load_config(args.config), args.output.resolve(), args.lock, args.stop, args.arm_timeout)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
