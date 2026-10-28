"""Internal explicit guardian/worker entry; importing the module has no effects.

The controller passes its already-held lock to a guardian. A scoped worker gets
that same open-file-description over SCM_RIGHTS. Unknown cleanup retains it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from inferssd.platform.guardian import guard_worker
from inferssd.platform.lease import Lease
from inferssd.platform.policy import Policy
from inferssd.platform.process import group_exists, stop_owned
from inferssd.platform.resources import Resources, scope_pids


def save(path: Path, value: object) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


class Effects:
    save = staticmethod(save)
    group_exists = staticmethod(group_exists)

    @staticmethod
    def file_identity(path: Path, hash_file: bool) -> Mapping[str, Any]:
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def systemctl(unit: str, user: bool, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", *(["--user"] if user else []), *arguments, unit],
        capture_output=True,
        text=True,
        timeout=10,
    )


def scope_state(unit: str, user: bool) -> dict[str, Any]:
    value = systemctl(
        unit,
        user,
        "show",
        "--property=LoadState",
        "--property=ActiveState",
        "--property=ControlGroup",
    )
    fields = dict(row.split("=", 1) for row in value.stdout.splitlines() if "=" in row)
    group = fields.get("ControlGroup", "")
    pids: list[int] = []
    if group:
        path = Path("/sys/fs/cgroup") / group.lstrip("/")
        if path.exists():
            pids = sorted(scope_pids(path))
    gone = fields.get("LoadState") == "not-found" or (
        value.returncode == 0 and fields.get("ActiveState") in ("inactive", "failed") and not pids
    )
    return {
        "gone": gone and not pids,
        "fields": fields,
        "pids": pids,
        "returncode": value.returncode,
        "stdout": value.stdout,
        "stderr": value.stderr,
    }


def cleanup_confirmed(path: Path) -> bool:
    try:
        server = json.loads(path.read_text())["server"]
        return (
            server.get("owned_processes_gone") is True and server.get("resource_scope_gone") is True
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_guardian(args: argparse.Namespace, specification: dict[str, Any], policy: Policy) -> int:
    out = Path(specification["directory"])
    receipt = out / "ownership.json"
    connection = socket.socket(fileno=args.controller_fd)
    connection.setblocking(False)
    environment = dict(os.environ, INFERSSD_RUN_DIR=str(out), INFERSSD_WRAPPER_RECEIPT=str(receipt))
    prefix: list[str] = specification["scope"]
    unit: Optional[str] = specification["unit"]
    user = "--user" in prefix
    command = [
        *prefix,
        sys.executable,
        "-I",
        "-m",
        "inferssd.platform.worker",
        "--worker",
        "--spec",
        args.spec,
    ]
    confirmed = False

    def interrupted() -> bool:
        try:
            message = connection.recv(1, socket.MSG_PEEK)
        except BlockingIOError:
            return False
        return message == b"" or bool(message)

    def cleanup(path: Path) -> bool:
        nonlocal confirmed
        if unit is None:
            confirmed = cleanup_confirmed(path)
            return confirmed
        # Only the fresh, prechecked unit from this immutable launch request.
        # Stopping it does not select processes by port, command name or GPU use.
        result = systemctl(unit, user, "stop")
        state = scope_state(unit, user)
        save(
            out / "scope-cleanup.json",
            {"unit": unit, "stop_returncode": result.returncode, "state": state},
        )
        if state["gone"]:
            value = (
                json.loads(path.read_text())
                if path.exists()
                else {"status": "failed", "server": {}, "evidence": []}
            )
            value["server"].update(owned_processes_gone=True, resource_scope_gone=True)
            save(path, value)
            confirmed = True
            return True
        return False

    def request_stop(pid: int) -> None:
        if unit is None:
            os.kill(pid, signal.SIGTERM)
        else:
            value = systemctl(unit, user, "kill", "--signal=SIGTERM", "--kill-whom=all")
            if value.returncode:
                # Only the original owned launcher; a missing/unknown unit will
                # still be checked before the guardian releases its lock.
                os.kill(pid, signal.SIGTERM)

    if unit is not None:
        state = scope_state(unit, user)
        if state["fields"].get("LoadState") != "not-found":
            raise RuntimeError("fresh resource scope name is already present or unknown")
    try:
        return guard_worker(
            command,
            environment,
            args.lock_fd,
            policy.runtime_seconds,
            effects=Effects(),
            cleanup_confirmed=cleanup,
            lease_unit=unit,
            interrupted=interrupted,
            request_stop=request_stop,
        )
    finally:
        # Keep the inherited original fd even when receipt I/O itself fails.
        # A guardian may be killed only after independently proving cleanup.
        if not confirmed:
            while True:
                try:
                    if cleanup(receipt):
                        break
                except BaseException:
                    pass
                time.sleep(1)
        connection.close()
        os.close(args.lock_fd)


def run_worker(args: argparse.Namespace, specification: dict[str, Any], policy: Policy) -> int:
    out = Path(specification["directory"])
    receipt = out / "ownership.json"
    lease = Lease(args.socket, expected_identity=(args.lock_dev, args.lock_ino))
    assert lease.fd is not None
    runtime = Resources(policy)
    child: Optional[subprocess.Popen[bytes]] = None
    signals: list[int] = []
    handlers = {
        number: signal.signal(number, lambda n, _f: signals.append(n))
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    failure: Optional[str] = None
    clean = False
    try:
        server_environment = dict(os.environ)
        overrides = json.loads(server_environment.pop("INFERSSD_INTERNAL_SERVER_ENV"))
        if not isinstance(overrides, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in overrides.items()
        ):
            raise ValueError("invalid private server environment transport")
        server_environment.update(overrides)
        for name in ("INFERSSD_RUN_DIR", "INFERSSD_WRAPPER_RECEIPT"):
            server_environment.pop(name, None)
        if Path(policy.stop_path).exists() or (out / "STOP").exists():
            raise InterruptedError("STOP exists before native launch")
        save(out / "resources-before.json", runtime.prepare())
        with (out / "server.log").open("ab") as log:
            child = subprocess.Popen(
                specification["argv"],
                env=server_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(lease.fd,),
            )
        save(out / "server-state.json", {"pid": child.pid, "returncode": None})
        while child.poll() is None:
            lease.check()
            if signals or Path(policy.stop_path).exists() or (out / "STOP").exists():
                break
            with (out / "memory-samples.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(dict(runtime.sample(child.pid), monotonic=time.monotonic())) + "\n"
                )
            time.sleep(0.5)
        if child.poll() is not None and not signals and not (out / "STOP").exists():
            if specification.get("command_mode") is not True:
                failure = "server exited before a requested stop"
            elif child.returncode != 0:
                failure = f"command exited {child.returncode}"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        while True:
            try:
                cleanup = stop_owned(child, policy.cleanup_seconds) if child else None
                clean = cleanup is None or cleanup["group_gone"]
                if (
                    specification.get("command_mode") is True
                    and cleanup
                    and (cleanup["forced_kill"] or cleanup["orphaned_descendants"])
                ):
                    failure = failure or "command left descendants or required forced cleanup"
                if runtime.scope is not None:
                    # A server that escaped its process group remains inside this
                    # scope. Do not release the last lease while it is present.
                    clean = clean and scope_pids(runtime.scope) == {os.getpid()}
                if clean:
                    break
            except BaseException as error:
                failure = failure or f"cleanup: {type(error).__name__}: {error}"
            try:
                save(
                    out / "worker-quarantine.json",
                    {"pid": os.getpid(), "retaining_lock": True, "reason": failure},
                )
            except OSError:
                pass
            time.sleep(1)
        try:
            if runtime.baseline is not None:
                save(out / "resources-after.json", runtime.sample(None))
        except BaseException as error:
            failure = failure or f"final resource sample: {type(error).__name__}: {error}"
        save(
            out / "server-state.json",
            {"pid": child.pid if child else None, "returncode": child.returncode if child else 1},
        )
        save(
            receipt,
            {
                "status": "failed" if failure else "passed",
                "error": failure,
                "server": {
                    "owned_processes_gone": clean,
                    "resource_scope_gone": policy.platform == "darwin",
                },
                "evidence": [],
            },
        )
        lease.close()
        for number, previous in handlers.items():
            signal.signal(number, previous)
    return 1 if failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--guardian", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--spec", required=True)
    for name in ("controller-fd", "lock-fd", "lock-dev", "lock-ino"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--socket")
    args = parser.parse_args()
    specification = json.loads(Path(args.spec).read_text())
    policy = Policy.read(specification["policy"])
    return (
        run_guardian(args, specification, policy)
        if args.guardian
        else run_worker(args, specification, policy)
    )


if __name__ == "__main__":
    raise SystemExit(main())
