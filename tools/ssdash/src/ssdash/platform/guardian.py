"""Shared guardian algorithm; resource policy and receipts are caller-owned ports.

The default worker path preserves the standalone wrapper lifecycle. Unknown
cleanup keeps the original lock alive. No timeout authorizes abandoning a child.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Protocol, TypedDict

from .lease import LeaseServer
from .process import Process

CORE_API_VERSION = 1


class _SpawnExtras(TypedDict, total=False):
    pass_fds: tuple[int, ...]


class GuardianEffects(Protocol):
    def save(self, path: Path, value: object) -> None: ...
    def group_exists(self, process: Process) -> bool: ...
    def file_identity(self, path: Path, hash_file: bool) -> Mapping[str, Any]: ...


def guard_worker(
    argv: Sequence[str],
    environment: Mapping[str, str],
    lock_fd: int,
    deadline_s: float,
    *,
    effects: GuardianEffects,
    cleanup_confirmed: Callable[[Path], bool],
    pass_fds: tuple[int, ...] = (),
    lease_unit: Optional[str] = None,
) -> int:
    """Keep a second owner outside the worker's session until cleanup is proven.

    A killed controller disconnects the lease; the worker remains alive to clean
    its server/client groups. A killed worker leaves the controller holding the
    lease in quarantine. Never force-kill that cleanup guardian to meet a timeout.
    """
    common = effects
    spawn_extras: _SpawnExtras = {"pass_fds": pass_fds} if pass_fds else {}
    out, receipt_path = (
        Path(environment["SSDASH_RUN_DIR"]),
        Path(environment["SSDASH_WRAPPER_RECEIPT"]),
    )
    lease = LeaseServer(lock_fd) if lease_unit is None else LeaseServer(lock_fd, lease_unit)
    identity = os.fstat(lock_fd)
    command = [
        *argv,
        "--socket",
        lease.path,
        "--lock-dev",
        str(identity.st_dev),
        "--lock-ino",
        str(identity.st_ino),
    ]
    common.save(out / "guardian-launch.json", {"argv": command})
    signals: list[int] = []
    worker: Optional[subprocess.Popen[bytes]] = None
    failure: Optional[str] = None
    asked_stop = False
    handlers = {
        s: signal.signal(s, lambda n, _f: signals.append(n))
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    deadline = time.monotonic() + deadline_s
    try:
        with (out / "guardian.log").open("xb") as log:
            worker = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                **spawn_extras,
            )
        lease.expected_pid = worker.pid
        while worker.poll() is None:
            try:
                lease.poll()
            except BaseException as error:
                failure = f"lease: {type(error).__name__}: {error}"
            if failure or signals or time.monotonic() > deadline:
                failure = failure or "guardian interrupted or timed out"
                if not asked_stop:
                    os.kill(worker.pid, signal.SIGTERM)
                    asked_stop = True
                common.save(
                    out / "guardian-quarantine.json",
                    {
                        "pid": os.getpid(),
                        "worker_pid": worker.pid,
                        "retaining_lock": True,
                        "reason": failure,
                    },
                )
            time.sleep(0.05)
        if lease.peer is None or worker.returncode != 0:
            failure = failure or "worker failed or did not receive the inherited lease"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        # No child was created on a spawn failure; otherwise unknown cleanup
        # retains the same original flock, including after worker SIGKILL.
        while worker is not None and (
            common.group_exists(worker) or not cleanup_confirmed(receipt_path)
        ):
            failure = failure or "worker cleanup uncertain; retaining inherited lock"
            common.save(
                out / "guardian-quarantine.json",
                {
                    "pid": os.getpid(),
                    "worker_pid": worker.pid,
                    "retaining_lock": True,
                    "reason": failure,
                },
            )
            time.sleep(1)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
        else:
            receipt = {
                "status": "failed",
                "server": {"owned_processes_gone": True, "resource_scope_gone": True},
                "evidence": [],
            }
        if failure or signals:
            receipt.update(status="failed", guardian_error=failure, guardian_signals=signals)
        common.save(
            out / "guardian-cleanup.json",
            {
                "worker": lease.peer,
                "worker_returncode": worker.returncode if worker else None,
                "worker_group_gone": worker is None or not common.group_exists(worker),
                "cleanup_receipt_confirmed": worker is None or cleanup_confirmed(receipt_path),
            },
        )
        for path in (
            out / "guardian-launch.json",
            out / "guardian.log",
            out / "guardian-cleanup.json",
        ):
            if path.exists():
                receipt.setdefault("evidence", []).append(
                    {"path": path.name, "sha256": common.file_identity(path, True)["sha256"]}
                )
        common.save(receipt_path, receipt)
        lease.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if receipt["status"] == "passed" else 1
