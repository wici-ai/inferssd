"""Owned process-group cleanup; unknown membership never means success."""

from __future__ import annotations

import os
import signal
import time
from typing import Optional, Protocol, TypedDict

CORE_API_VERSION = 1


class Process(Protocol):
    @property
    def pid(self) -> int: ...
    @property
    def returncode(self) -> Optional[int]: ...
    def poll(self) -> Optional[int]: ...
    def wait(self, timeout: Optional[float] = None) -> int: ...


class Cleanup(TypedDict):
    returncode: Optional[int]
    group_gone: bool
    forced_kill: bool
    orphaned_descendants: bool
    was_running: bool


def group_exists(process: Process) -> bool:
    process.poll()
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin may briefly report EPERM while an owned group exits, before
        # waitpid reports completion. Unknown is still present, never clean.
        return True


def stop_owned(process: Process, timeout: float) -> Cleanup:
    was_running = process.poll() is None
    orphan = not was_running and group_exists(process)
    if group_exists(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    until = time.monotonic() + timeout
    while group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    forced = group_exists(process)
    if forced:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    process.wait(timeout=5)
    until = time.monotonic() + 2
    while group_exists(process) and time.monotonic() < until:
        time.sleep(0.05)
    return {
        "returncode": process.returncode,
        "group_gone": not group_exists(process),
        "forced_kill": forced,
        "orphaned_descendants": orphan,
        "was_running": was_running,
    }
