"""Controller-side handle for an explicit native guardian and resource worker."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import stat
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from inferssd.launch.types import Resolution
from inferssd.platform.policy import Policy
from inferssd.platform.resources import Resources
from inferssd.platform.run_directory import Reservation
from inferssd.platform.worker import cleanup_confirmed, save


def scope_command(resolution: Resolution, policy: Policy) -> tuple[list[str], Optional[str]]:
    if policy.platform == "darwin":
        if resolution.scope:
            raise ValueError("Darwin launch must not carry a Linux scope")
        return [], None
    prefix = list(resolution.scope)
    if prefix[:2] not in (["systemd-run", "--scope"], ["systemd-run", "--user"]):
        raise ValueError("native Linux launch requires the resolved systemd scope")
    memory = [value.split("=", 1)[1] for value in prefix if value.startswith("MemoryMax=")]
    if memory != [str(policy.memory_budget_bytes)] or "MemorySwapMax=0" not in prefix:
        raise ValueError("resolved scope must match the explicit byte budget and zero swap")
    units = [i for i, value in enumerate(prefix) if value.startswith("--unit=")]
    if len(units) != 1:
        raise ValueError("resolved scope must contain exactly one unit")
    unit = "inferssd-" + uuid.uuid4().hex + ".scope"
    prefix[units[0]] = "--unit=" + unit
    index = prefix.index("taskset") if "taskset" in prefix else len(prefix)
    prefix[index:index] = ["-p", "Delegate=yes"]
    return prefix, unit


def worker_environment(environment: dict[str, str]) -> dict[str, str]:
    # Engine loaders/knobs belong to the server, not the Python lock owner.
    # Carry them privately in the process environment; never persist values in
    # runtime.json or command arguments. A worker restores them only for Popen.
    prefixes = (
        "GGML_",
        "LLAMA_",
        "SSD_LLAMA_",
        "LD_",
        "DYLD_",
        "CUDA_",
        "OMP_",
        "KMP_",
        "OPENBLAS_",
    )
    moved = {
        key: value
        for key, value in environment.items()
        if key.startswith(prefixes) or key in ("PYTHONPATH", "PYTHONHOME")
    }
    control = {key: value for key, value in environment.items() if key not in moved}
    control["INFERSSD_INTERNAL_SERVER_ENV"] = json.dumps(moved)
    return control


class OwnedProcess:
    def __init__(
        self,
        resolution: Resolution,
        directory: str,
        policy: Policy,
        *,
        reservation: Reservation,
        inherited_lock_fd: Optional[int] = None,
        command_mode: bool = False,
    ) -> None:
        if resolution.code != 0 or resolution.needed is not None or not resolution.argv:
            raise ValueError("native launch requires a complete successful resolution")
        reservation.verify()
        if Path(directory).resolve() != reservation.directory:
            raise ValueError("execution directory differs from reservation")
        self.directory, self.policy = reservation.directory, policy
        self.connection: Optional[socket.socket] = None
        self.guardian: Optional[subprocess.Popen[bytes]] = None
        self._lock: Optional[int] = None
        if Path(policy.stop_path).exists() or (self.directory / "STOP").exists():
            raise InterruptedError("STOP exists before native launch")
        scope, unit = scope_command(resolution, policy)
        specification = {
            "argv": list(resolution.argv),
            "directory": str(self.directory.resolve()),
            "scope": scope,
            "unit": unit,
            "policy": asdict(policy),
            "command_mode": command_mode,
        }
        path = self.directory / "runtime.json"
        child_end: Optional[socket.socket] = None
        try:
            if inherited_lock_fd is None:
                self._lock = os.open(policy.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            else:
                actual, expected = os.fstat(inherited_lock_fd), os.stat(policy.lock_path)
                if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                    expected.st_dev,
                    expected.st_ino,
                ):
                    raise ValueError("inherited lease differs from the existing device lock inode")
                # dup retains the same flock open-file-description. Closing this
                # owner's copy does not release the caller's batch lease.
                self._lock = os.dup(inherited_lock_fd)
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reservation.verify()
            save(path, specification)
            self.connection, child_end = socket.socketpair()
            with (self.directory / "owner.log").open("xb") as log:
                self.guardian = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-m",
                        "inferssd.platform.worker",
                        "--guardian",
                        "--spec",
                        str(path),
                        "--controller-fd",
                        str(child_end.fileno()),
                        "--lock-fd",
                        str(self._lock),
                    ],
                    env=worker_environment(dict(resolution.effective_environment)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    pass_fds=(child_end.fileno(), self._lock),
                )
        except BaseException:
            if self.connection is not None:
                self.connection.close()
            if self._lock is not None:
                os.close(self._lock)
                self._lock = None
            raise
        finally:
            if child_end is not None:
                child_end.close()

    def state(self) -> dict[str, Any]:
        path = self.directory / "server-state.json"
        return json.loads(path.read_text()) if path.exists() else {"pid": None, "returncode": None}

    def poll(self) -> Optional[int]:
        assert self.guardian is not None
        ended = self.guardian.poll()
        if ended is not None:
            return ended
        receipt = self.directory / "ownership.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("status") == "failed":
            return 1
        value = self.state()["returncode"]
        return None if value is None else int(value)

    def confirm_listener(self, port: int) -> None:
        pid = self.state()["pid"]
        if not isinstance(pid, int):
            raise RuntimeError("owned server PID has not been published")
        Resources(self.policy).confirm_listener(pid, port)

    def stop(self) -> None:
        assert self.guardian is not None
        (self.directory / "STOP").touch()
        try:
            ended = self.guardian.wait(timeout=self.policy.cleanup_seconds + 15)
        except subprocess.TimeoutExpired as error:
            # Do not kill a cleanup owner to satisfy the caller's timeout.
            raise RuntimeError(
                "cleanup is unconfirmed; guardian retains the device lock"
            ) from error
        if not cleanup_confirmed(self.directory / "ownership.json"):
            raise RuntimeError("native cleanup receipt is missing or unconfirmed")
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None
        receipt = json.loads((self.directory / "ownership.json").read_text())
        if ended != 0 or receipt.get("status") != "passed":
            reason = receipt.get("error") or f"guardian exited with {ended}"
            raise RuntimeError(f"owned native execution failed: {reason}; see {self.directory}")
