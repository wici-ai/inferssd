"""Observed shared-core algorithms at the reviewed standalone-wrapper baseline."""

from __future__ import annotations

import array
import ast
import copy
import json
import signal
import socket
from pathlib import Path
from types import SimpleNamespace

ORACLE = Path(__file__).parent / "fixtures/oracle"


def group_case(case):
    events, count, now = [], [0], [0.0]

    def killpg(pid, value):
        events.append(["killpg", pid, int(value)])
        if value == 0:
            items = case["probes"]
            answer = items[min(count[0], len(items) - 1)]
            count[0] += 1
            if answer == "absent":
                raise ProcessLookupError()
            if answer == "unknown":
                raise PermissionError()

    def clock():
        events.append(["monotonic"])
        now[0] += 0.25
        return now[0]

    def sleep(value):
        events.append(["sleep", value])

    class Process:
        pid = 100
        returncode = case.get("returncode")

        def poll(self):
            events.append(["poll"])
            return self.returncode

        def wait(self, *, timeout):
            events.append(["wait", timeout])
            if case.get("wait_error"):
                raise TimeoutError("fixture wait timeout")
            self.returncode = 0

    nodes = [
        n
        for n in ast.parse((ORACLE / "runtime_native_common.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in ("group_exists", "stop_owned")
    ]
    namespace = dict(
        os=SimpleNamespace(killpg=killpg),
        signal=signal,
        time=SimpleNamespace(monotonic=clock, sleep=sleep),
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "runtime_native_common.py", "exec"),
        namespace,
    )
    try:
        result, error = namespace["stop_owned"](Process(), 0.5), None
    except BaseException as exc:
        result, error = None, [type(exc).__name__, str(exc)]
    return {"name": case["name"], "events": events, "result": result, "error": error}


def lease_case(case):
    events = []

    class Connection:
        def settimeout(self, value):
            events.append(["timeout", value])

        def connect(self, path):
            events.append(["connect", path])

        def recvmsg(self, length, space):
            assert space == socket.CMSG_SPACE(array.array("i").itemsize)
            events.append(["recvmsg", length, "CMSG_SPACE(one_int)"])
            descriptors = case.get("descriptors", [11])
            return (
                case.get("message", "L").encode(),
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", descriptors).tobytes())],
                socket.MSG_CTRUNC if case.get("truncated") else 0,
                None,
            )

        def setblocking(self, value):
            events.append(["blocking", value])

        def recv(self, size, flags):
            assert flags == socket.MSG_PEEK
            events.append(["recv", size, "MSG_PEEK"])
            reply = case.get("check", "alive")
            if reply == "alive":
                raise BlockingIOError()
            return reply.encode()

        def close(self):
            events.append(["connection_close"])

    def file_stat(fd):
        events.append(["fstat", fd])
        return SimpleNamespace(st_dev=3, st_ino=case.get("inode", 5))

    def close(fd):
        events.append(["close", fd])

    def flock(fd, flags):
        events.append(["flock", fd, flags])
        if case.get("flock_error"):
            raise BlockingIOError("fixture locked")

    nodes = [
        n
        for n in ast.parse((ORACLE / "runtime_native_lease.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Lease"
    ]
    namespace = dict(
        array=array,
        socket=SimpleNamespace(
            socket=lambda *args: Connection(),
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
            CMSG_SPACE=socket.CMSG_SPACE,
            SOL_SOCKET=socket.SOL_SOCKET,
            SCM_RIGHTS=socket.SCM_RIGHTS,
            MSG_CTRUNC=socket.MSG_CTRUNC,
            MSG_PEEK=socket.MSG_PEEK,
        ),
        os=SimpleNamespace(fstat=file_stat, close=close),
        fcntl=SimpleNamespace(flock=flock, LOCK_EX=2, LOCK_NB=4),
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "runtime_native_lease.py", "exec"),
        namespace,
    )
    lease, error = None, None
    try:
        lease = namespace["Lease"]("/fixture/lease", expected_identity=(3, 5))
        lease.check()
    except BaseException as exc:
        error = [type(exc).__name__, str(exc)]
    finally:
        if lease is not None:
            lease.close()
    return copy.deepcopy({"name": case["name"], "events": events, "error": error})


GROUP_CASES = [
    {"name": "absent", "probes": ["absent"], "returncode": 0},
    {"name": "term", "probes": ["present", "present", "absent"]},
    {"name": "forced", "probes": ["present"] * 6 + ["absent"]},
    {"name": "orphan", "probes": ["present", "present", "absent"], "returncode": 0},
    {"name": "permission-unknown", "probes": ["unknown"]},
    {"name": "wait-error", "probes": ["absent"], "wait_error": True},
]
LEASE_CASES = [
    {"name": "live"},
    {"name": "disconnect", "check": ""},
    {"name": "unexpected-message", "check": "X"},
    {"name": "wrong-inode", "inode": 6},
    {"name": "flock-error", "flock_error": True},
    {"name": "truncated", "truncated": True},
    {"name": "no-fd", "descriptors": []},
    {"name": "two-fds", "descriptors": [11, 12]},
    {"name": "wrong-message", "message": "X"},
]


def generate():
    return {
        "groups": [group_case(c) for c in GROUP_CASES],
        "leases": [lease_case(c) for c in LEASE_CASES],
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
