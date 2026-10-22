#!/usr/bin/env python3
"""One fresh native server and one final_chat request under run_final's lock."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import final_chat
import native_common as common

HERE = Path(__file__).resolve().parent


def inputs(environment, profile):
    required = ("SSDASH_RUN_DIR", "SSDASH_RESULT_PATH", "SSDASH_WRAPPER_RECEIPT", "SSDASH_TIER",
                "SSDASH_REQUEST_TIMEOUT", "SSDASH_DEVICE", "SSDASH_MODEL", "SSDASH_REVISION", "SSDASH_OUTPUT_TOKENS")
    if any(not environment.get(key) for key in required):
        raise ValueError("invoke through bench/run_final.py with its full wrapper environment")
    if environment["SSDASH_REVISION"] != profile["revision"] or environment["SSDASH_OUTPUT_TOKENS"] != "256":
        raise ValueError("profile differs from final source/output contract")
    tier = environment["SSDASH_TIER"]
    final_chat.prompt_path(tier)
    final_chat.bounded_timeout(float(environment["SSDASH_REQUEST_TIMEOUT"]))
    out = Path(environment["SSDASH_RUN_DIR"]).resolve(strict=True)
    for key in ("SSDASH_RESULT_PATH", "SSDASH_WRAPPER_RECEIPT"):
        path = Path(environment[key]).resolve()
        if path.parent != out or path.exists():
            raise ValueError("fresh result/receipt paths must be directly in the arm directory")
    return out


class Guard:
    def __init__(self, runtime, signals, deadline, lease=None):
        self.runtime, self.signals, self.deadline, self.lease = runtime, signals, deadline, lease
        self.last_sample = 0.0

    def check(self, pid=None):
        if self.signals:
            raise InterruptedError("wrapper termination requested")
        if self.lease is not None:
            self.lease.check()
        if time.monotonic() > self.deadline:
            raise TimeoutError("native wrapper stage deadline expired")
        if time.monotonic() - self.last_sample >= 1:
            self.runtime.sample(pid)
            self.last_sample = time.monotonic()


def health(server, port, guard):
    while True:
        guard.check(server.pid)
        if server.poll() is not None:
            raise RuntimeError("owned server exited during initialization")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(.05)


def confirm_first_request(log_before, log_after):
    # This is the production server's task-launch log, not a TCP connection count.
    marker = "processing task, is_child = "
    if marker in log_before or log_after.count(marker) != 1:
        raise ValueError("exactly one server completion task, with no warm-up completion, is required")


def run_native(profile, environment, runtime, lock_fd, lease=None):
    out = inputs(environment, profile)
    receipt_path = Path(environment["SSDASH_WRAPPER_RECEIPT"])
    receipt = {"schema_version": 1, "status": "failed", "protocol": final_chat.PROTOCOL,
               "tier": environment["SSDASH_TIER"], "device": environment["SSDASH_DEVICE"],
               "model": environment["SSDASH_MODEL"], "revision": profile["revision"],
               "server": {"owned_processes_gone": False, "resource_scope_gone": False}, "evidence": []}
    server = client = None
    signals = []
    previous = {s: signal.signal(s, lambda n, _f: signals.append(n)) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    cleanup_errors = []
    evidence = []
    try:
        common.save(out / "native-profile.json", profile)
        evidence.append(out / "native-profile.json")
        before = common.verify_inputs(profile)
        common.save(out / "inputs-before.json", before)
        evidence.append(out / "inputs-before.json")
        runtime.prepare()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        argv = common.server_argv(profile, port)
        env = common.clean_environment(profile)
        common.save(out / "launch.json", {"argv": argv, "environment": profile["environment"],
                                          "library_path": str(Path(profile["binary"]["path"]).resolve().parent)})
        evidence.append(out / "launch.json")
        guard = Guard(runtime, signals, time.monotonic() + profile["load_timeout_s"], lease)
        guard.check()
        with (out / "server.log").open("xb") as log:
            # No signals are masked. Responsibility is published before the next check.
            server = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=(lock_fd,))
        start_identity = runtime.process_identity(server.pid)
        receipt["server"].update(pid=server.pid, start_identity=start_identity, argv=argv, fresh_process=True,
                                 completion_requests=0)
        common.save(out / "owner.json", {"wrapper_pid": os.getpid(), "server": receipt["server"]})
        evidence += [out / "owner.json", out / "server.log"]
        runtime.confirm_child(server.pid, lock_fd)
        health(server, port, guard)
        runtime.confirm_port(server.pid, port)
        loaded_before = runtime.images(server.pid, "images-before")
        before_log = (out / "server.log").read_text()
        if "processing task, is_child = " in before_log:
            raise ValueError("server completed or began a request before the formal client")
        common.save(out / "before-request.json", {"process_identity": start_identity, "completion_requests": 0})
        evidence.append(out / "before-request.json")
        client_argv = [sys.executable, str(HERE / "final_chat.py"), "--url", f"http://127.0.0.1:{port}",
                       "--tier", environment["SSDASH_TIER"], "--output", environment["SSDASH_RESULT_PATH"],
                       "--timeout", environment["SSDASH_REQUEST_TIMEOUT"]]
        guard.deadline = time.monotonic() + float(environment["SSDASH_REQUEST_TIMEOUT"]) + 5
        with (out / "client.log").open("xb") as log:
            client = subprocess.Popen(client_argv, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=(lock_fd,))
        while client.poll() is None:
            guard.check(server.pid)
            if server.poll() is not None:
                raise RuntimeError("server exited during the completion")
            time.sleep(.05)
        if client.returncode or common.group_exists(client):
            raise RuntimeError("formal client failed or left descendants")
        guard.check(server.pid)
        if runtime.process_identity(server.pid) != start_identity:
            raise ValueError("server process identity changed")
        loaded_after = runtime.images(server.pid, "images-after")
        if loaded_before != loaded_after:
            raise ValueError("actual project image set or identity changed during the request")
        after = common.verify_inputs(profile)
        if before != after:
            raise ValueError("build, model or input identity changed")
        common.save(out / "inputs-after.json", after)
        evidence += [out / "inputs-after.json", out / "client.log"]
        confirm_first_request(before_log, (out / "server.log").read_text())
        receipt["server"]["completion_requests"] = 1
        receipt.update(build={"revision": profile["revision"], "binary_sha256": profile["binary"]["sha256"],
                              "actual_loaded_libraries_verified": True, "loaded_libraries": loaded_after},
                       model_identity={"verified": True, "shards": [{k: x[k] for k in ("path", "bytes", "sha256")}
                                                                   for x in before["shards"]]},
                       host=runtime.host_receipt(), status="passed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        # Each owned resource gets its cleanup attempt even if another fails.
        for name, process in (("client", client), ("server", server)):
            if process is None:
                continue
            try:
                cleanup = common.stop_owned(process, profile["cleanup_timeout_s"])
                common.save(out / f"{name}-cleanup.json", cleanup)
                evidence.append(out / f"{name}-cleanup.json")
                if not cleanup["group_gone"] or cleanup["forced_kill"] or cleanup["orphaned_descendants"]:
                    raise RuntimeError(f"{name} cleanup needed force or found remaining descendants")
                if name == "server":
                    receipt["server"]["returncode"] = cleanup["returncode"]
                    if cleanup["returncode"] != 0 or not cleanup["was_running"]:
                        raise RuntimeError("server did not stop normally from its owned live state")
                elif cleanup["returncode"] != 0:
                    raise RuntimeError("client did not succeed")
            except BaseException as error:
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        try:
            receipt["server"]["owned_processes_gone"] = all(p is None or not common.group_exists(p) for p in (client, server))
            receipt["server"]["resource_scope_gone"] = runtime.finish()
        except BaseException as error:
            cleanup_errors.append(f"native scope: {type(error).__name__}: {error}")
        if cleanup_errors or signals:
            receipt["status"] = "failed"
        receipt.update(cleanup_errors=cleanup_errors, signals=signals)
        evidence += runtime.evidence_paths()
        for path in sorted(set(evidence)):
            if path.is_file():
                receipt["evidence"].append({"path": str(path.relative_to(out)),
                                             "sha256": common.file_identity(path, True)["sha256"]})
        common.save(receipt_path, receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if receipt["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    try:
        profile = common.validate_profile(json.loads(args.profile.read_text()))
        inputs(os.environ, profile)
        lock_fd = common.inherited_lock(os.environ)
        if profile["platform"] == "linux":
            from native_linux import run_scope
            return run_scope(profile, os.environ.copy(), lock_fd)
        from native_macos import MacRuntime
        return run_native(profile, os.environ, MacRuntime(profile, Path(os.environ["SSDASH_RUN_DIR"])), lock_fd)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"native wrapper: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
