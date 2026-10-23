"""Original guardian state transitions, with no real child, signals, sockets or files."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).parent / "fixtures/oracle/runtime_native_macos.py"


class EndObservation(BaseException):
    pass


def observe(case):
    events, files, calls = [], {}, {}
    signals, leases = {}, []
    receipt_name = "/case/receipt.json"

    def event(name, *args):
        events.append([name, list(args)])
        calls[name] = calls.get(name, 0) + 1
        if case.get("fault") == [name, calls[name]]:
            raise OSError("synthetic " + name + " failure")

    # The original uses path.name for receipt labels, not string conversion.
    class PathView:
        def __init__(self, name):
            self.full = str(name)

        @property
        def name(self):
            return self.full.rsplit("/", 1)[-1]

        def __str__(self):
            return self.full

        def __truediv__(self, part):
            return PathView(self.full + "/" + part)

        def exists(self):
            event("exists", self.full)
            return self.full in files

        def read_text(self):
            event("read", self.full)
            return json.dumps(files[self.full])

        def open(self, mode):
            event("open", self.full, mode)
            files[self.full] = ""
            return self

        def __enter__(self):
            event("enter", self.full)
            return self

        def __exit__(self, *args):
            event("exit", self.full, args[0].__name__ if args[0] else None)

    def save(path, value):
        event("save", str(path), copy.deepcopy(value))
        files[str(path)] = copy.deepcopy(value)

    def identity(path, do_hash):
        event("identity", str(path), do_hash)
        return {"sha256": "a" * 64}

    def cleanup(path):
        event("cleanup_confirmed", str(path))
        return not case.get("unknown")

    def group_exists(process):
        event("group_exists", process.pid)
        return bool(case.get("group_unknown"))

    class Lease:
        def __init__(self, fd):
            event("lease", fd)
            self.path, self.peer, self.expected_pid = "/case/socket", None, None
            self.closed = False
            leases.append(self)

        def poll(self):
            event("lease_poll", self.expected_pid)
            self.peer = {"pid": 12, "uid": 501, "gid": 20, "cgroup": None}

        def close(self):
            event("lease_close")
            self.closed = True

    class Process:
        pid, returncode = 12, None

        def poll(self):
            event("poll")
            if calls["poll"] >= case.get("polls", 2):
                self.returncode = case.get("returncode", 0)
                if case.get("receipt", True):
                    files[receipt_name] = {
                        "status": "passed" if self.returncode == 0 else "failed",
                        "server": {"owned_processes_gone": True, "resource_scope_gone": True},
                        "evidence": [],
                    }
            return self.returncode

    def spawn(command, **kwargs):
        event("spawn", command, {k: str(v) if k == "stdout" else v for k, v in kwargs.items()})
        return Process()

    def set_signal(number, handler):
        event("signal_handler", number, "callable" if callable(handler) else handler)
        old = signals.get(number, "old")
        signals[number] = handler
        return old

    def clock():
        event("clock")
        return 100.0 + (100 if case.get("timeout") and calls["clock"] > 1 else 0)

    def sleep(seconds):
        event("sleep", seconds)
        if case.get("signal") and seconds == 0.05:
            signals[15](15, None)
        if seconds == 1:
            raise EndObservation("bounded observer stopped while original guardian retains lock")

    def kill(pid, number):
        event("kill", pid, number)

    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "guard_worker"]
    namespace = dict(
        Path=PathView,
        LeaseServer=Lease,
        json=json,
        os=SimpleNamespace(
            fstat=lambda fd: SimpleNamespace(st_dev=3, st_ino=5), getpid=lambda: 10, kill=kill
        ),
        signal=SimpleNamespace(SIGINT=2, SIGTERM=15, SIGHUP=1, signal=set_signal),
        time=SimpleNamespace(monotonic=clock, sleep=sleep),
        subprocess=SimpleNamespace(Popen=spawn, STDOUT=-2),
        common=SimpleNamespace(save=save, group_exists=group_exists, file_identity=identity),
        cleanup_confirmed=cleanup,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    result, error = None, None
    try:
        result = namespace["guard_worker"](
            ["python", "worker.py"],
            {"SSDASH_RUN_DIR": "/case", "SSDASH_WRAPPER_RECEIPT": receipt_name},
            7,
            30,
        )
    except BaseException as exc:
        error = [type(exc).__name__, str(exc)]
    return copy.deepcopy(
        {
            "name": case["name"],
            "events": events,
            "result": result,
            "error": error,
            "files": files,
            "lease_closed": [v.closed for v in leases],
        }
    )


CASES = [
    {"name": "normal"},
    {"name": "spawn-failure", "fault": ["spawn", 1]},
    {"name": "lease-failure", "fault": ["lease_poll", 1]},
    {"name": "worker-stop", "returncode": 1},
    {"name": "worker-controller-disconnect", "returncode": 1, "polls": 3},
    {"name": "signal-stop", "signal": True, "polls": 3},
    {"name": "timeout", "timeout": True},
    {"name": "unknown-receipt", "unknown": True},
    {"name": "group-unknown", "group_unknown": True},
    {"name": "no-lease", "polls": 1},
]


if __name__ == "__main__":
    print(json.dumps({"cases": [observe(c) for c in CASES]}, indent=2))
