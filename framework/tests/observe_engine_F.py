"""Original Engine/post observations with synthetic process, file and HTTP effects.

No observer imports the candidate package or starts a process, thread or socket.
The returned transcript is copied at command return, before fixture collection.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import signal
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ORACLE = Path(__file__).parent / "fixtures/oracle/controller_f.py"


def scalar(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (tuple, list)):
        return [scalar(x) for x in value]
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    return value


class Effects:
    def __init__(self, case):
        self.case, self.events, self.streams = case, [], []
        self.env = {"LAUNCH": "/fixture/launch.sh", **case.get("env", {})}
        for key in case.get("unset", []):
            self.env.pop(key, None)
        self.files = {"/case/inst/server.log": b"synthetic server log\n"}
        self.files["/proc/meminfo"] = case.get("meminfo", "MemAvailable: 524288 kB\n")
        self.counts, self.now, self.polls, self.healths, self.connects = {}, 100.0, 0, 0, 0
        self.process = Process(self)
        self.thread = None

    def event(self, name, *args):
        self.events.append([name, scalar(args)])
        self.counts[name] = self.counts.get(name, 0) + 1
        fault = self.case.get("fault")
        if fault and fault[0] == name and self.counts[name] == fault[1]:
            kind = {"ValueError": ValueError, "RuntimeError": RuntimeError}.get(
                fault[2] if len(fault) > 2 else "", OSError
            )
            raise kind("synthetic " + name + " failure")

    def open(self, path, mode="r", compresslevel=None):
        path = str(path)
        self.event("gzip_open" if compresslevel else "open", path, mode, compresslevel)
        raw = self.files.get(path, b"" if "b" in mode else "")
        if "w" in mode:
            raw = b"" if "b" in mode else ""
        if "b" in mode and isinstance(raw, str):
            raw = raw.encode()
        if "b" not in mode and isinstance(raw, bytes):
            raw = raw.decode()
        base = io.BytesIO if "b" in mode else io.StringIO
        owner = self

        class Stream(base):
            def __enter__(self):
                owner.event("enter", path)
                return self

            def __exit__(self, *args):
                owner.event("exit", path, args[0].__name__ if args[0] else None)
                self.close()

            def read(self, *args):
                owner.event("read", path, args)
                return super().read(*args)

            def write(self, value):
                owner.event("write", path, value)
                return super().write(value)

            def close(self):
                if not self.closed:
                    owner.event("close", path)
                    owner.files[path] = self.getvalue()
                super().close()

        stream = Stream(raw)
        stream.fixture_path = path
        self.streams.append(stream)
        if "a" in mode:
            stream.seek(0, 2)
        return stream

    def makedirs(self, path, *, exist_ok):
        self.event("makedirs", path, exist_ok)

    def remove(self, path):
        self.event("remove", path)
        del self.files[path]

    def clock(self):
        self.event("clock")
        self.now += 0.25
        return self.now

    def sleep(self, seconds):
        self.event("sleep", seconds)
        self.now += seconds
        if seconds == 0.5:
            self.process.returncode = 0

    def spawn(self, argv, **kwargs):
        self.event(
            "spawn",
            argv,
            kwargs["stdout"].fixture_path,
            kwargs["stderr"],
            kwargs["stdin"],
            list(kwargs["env"].items()),
        )
        return self.process

    def make_thread(self, *, target, daemon):
        self.event("thread", target.__name__, daemon)
        owner = self

        class Thread:
            def start(self):
                owner.event("thread_start")

        self.thread = Thread()
        return self.thread

    def urlopen(self, request, *, timeout):
        if isinstance(request, str):
            self.event("health", request, timeout)
            values = self.case.get("health", ['{"status":"ok"}'])
            reply = values[min(self.healths, len(values) - 1)]
            self.healths += 1
        else:
            self.event("post", request.full_url, request.data, request.header_items(), timeout)
            reply = self.case.get("reply", '{"content":"答"}')
        if reply in ("OSError", "ValueError"):
            raise {"OSError": OSError, "ValueError": ValueError}[reply]("synthetic HTTP failure")
        owner = self

        class Response:
            def __enter__(self):
                owner.event("http_enter")
                return self

            def read(self):
                owner.event("http_read")
                return reply.encode()

            def __exit__(self, *args):
                owner.event("http_exit", args[0].__name__ if args[0] else None)

        return Response()

    def socket(self):
        self.event("socket")
        owner = self

        class Socket:
            def __enter__(self):
                owner.event("socket_enter")
                return self

            def connect_ex(self, address):
                owner.event("connect_ex", address)
                values = owner.case.get("port", [1])
                result = values[min(owner.connects, len(values) - 1)]
                owner.connects += 1
                return result

            def __exit__(self, *args):
                owner.event("socket_exit", args[0].__name__ if args[0] else None)

        return Socket()

    def run(self, argv, **kwargs):
        self.event("run", argv, kwargs)

    def copy(self, source, target):
        self.event("copyfileobj", source.fixture_path, target.fixture_path)
        target.write(source.read())


class Process:
    def __init__(self, owner):
        self.owner, self.returncode = owner, None
        self.pid = 101

    def poll(self):
        self.owner.event("poll")
        values = self.owner.case.get("poll")
        if values:
            self.returncode = values[min(self.owner.polls, len(values) - 1)]
            self.owner.polls += 1
        return self.returncode

    def kill(self):
        self.owner.event("kill")
        self.returncode = -9

    def send_signal(self, value):
        self.owner.event("signal", int(value))

    def wait(self, timeout=None):
        self.owner.event("wait", timeout)
        if timeout == 60 and self.owner.case.get("wait_timeout"):
            raise subprocess.TimeoutExpired("synthetic", 60)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def api(effects):
    tree = ast.parse(ORACLE.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("post", "Engine")
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TRACE_ENV" for target in node.targets
        )
    ]
    namespace = dict(
        os=SimpleNamespace(environ=effects.env, makedirs=effects.makedirs, remove=effects.remove),
        open=effects.open,
        time=SimpleNamespace(time=effects.clock, sleep=effects.sleep),
        subprocess=SimpleNamespace(
            Popen=effects.spawn,
            run=effects.run,
            STDOUT=-2,
            DEVNULL=-3,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
        threading=SimpleNamespace(Thread=effects.make_thread),
        urllib=SimpleNamespace(
            request=SimpleNamespace(Request=urllib.request.Request, urlopen=effects.urlopen)
        ),
        signal=signal,
        socket=SimpleNamespace(socket=effects.socket),
        json=json,
        gzip=SimpleNamespace(open=effects.open),
        shutil=SimpleNamespace(copyfileobj=effects.copy),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ORACLE), "exec"), namespace)
    return SimpleNamespace(**namespace)


CASES = [
    {"name": "ready"},
    {"name": "retry-health", "health": ["OSError", '{"status":"loading"}', '"ok"']},
    {"name": "process-dies-loading", "health": ["OSError"], "poll": [3]},
    {"name": "health-exhausted", "health": ["OSError"]},
    {"name": "health-unexpected-error", "health": ["ValueError"]},
    {"name": "health-substring", "health": ['{"message":"ok"}']},
    {"name": "duplicate-knobs", "env": {"TRACE_ENV": "X=1 X=2"}, "envs": "X=3 Y=a=b"},
    {"name": "extra-env", "env": {"TRACE_ENV": "X=1", "KNOBS_VIA": "extra_env"}, "envs": "Y=2"},
    {"name": "empty-trace", "env": {"TRACE_ENV": ""}, "envs": ""},
    {"name": "bad-knob", "envs": "missing-equals"},
    {"name": "missing-launch", "unset": ["LAUNCH"]},
    {"name": "mkdir-failure", "fault": ["makedirs", 1]},
    {"name": "open-failure", "fault": ["open", 1]},
    {"name": "spawn-failure", "fault": ["spawn", 1]},
    {"name": "thread-failure", "fault": ["thread_start", 1]},
    {"name": "stop-graceful", "action": "stop"},
    {"name": "stop-kill", "action": "stop", "wait_timeout": True},
    {"name": "stop-dead", "action": "stop", "poll": [0]},
    {"name": "stop-busy-port", "action": "stop", "port": [0]},
    {"name": "stop-port-retry", "action": "stop", "port": [0, 0, 1]},
    {"name": "stop-signal-error", "action": "stop", "fault": ["signal", 1]},
    {"name": "stop-log-close-error", "action": "stop", "fault": ["close", 1]},
    {"name": "stop-copy-error", "action": "stop", "fault": ["copyfileobj", 1]},
    {"name": "stop-remove-error", "action": "stop", "fault": ["remove", 1]},
    {"name": "stop-port-error", "action": "stop", "fault": ["connect_ex", 1]},
    {"name": "stop-fuser-error", "action": "stop", "port": [0], "fault": ["run", 1]},
    {"name": "watchdog-low", "action": "watchdog", "meminfo": "MemAvailable: 1024 kB\n"},
    {"name": "watchdog-equal", "action": "watchdog", "meminfo": "MemAvailable: 262144 kB\n"},
    {"name": "watchdog-invalid-limit", "action": "watchdog", "env": {"WATCHDOG_MIN_AVAIL_MB": "x"}},
    {"name": "watchdog-missing-available", "action": "watchdog", "meminfo": "MemFree: 1 kB\n"},
    {"name": "watchdog-stopping", "action": "watchdog", "stopping": True},
    {
        "name": "watchdog-write-error",
        "action": "watchdog",
        "meminfo": "MemAvailable: 0 kB\n",
        "fault": ["write", 1],
    },
    {"name": "post-unicode", "action": "post"},
    {"name": "post-timeout", "action": "post", "timeout": 3.5},
    {"name": "post-http-error", "action": "post", "reply": "OSError"},
    {"name": "post-bad-json", "action": "post", "reply": "not json"},
]


def observe(case):
    effects, engine, result, error = Effects(case), None, None, None
    original = api(effects)
    try:
        action = case.get("action", "start")
        if action == "post":
            result = original.post(
                8080,
                "/completion",
                {"prompt": "你", "n_predict": 4},
                **({"timeout": case["timeout"]} if "timeout" in case else {}),
            )
        else:
            engine = original.Engine(
                "/case/inst", 8080, 16, "2:2", case.get("envs", "K=7"), ["--ctx-size", "64"]
            )
            if action == "stop":
                engine.stop()
            elif action == "watchdog":
                engine.stopping = case.get("stopping", False)
                engine._watchdog()
            result = {
                key: getattr(engine, key)
                for key in ("dir", "port", "tier", "cpus", "envs", "ok", "start_s", "stopping")
            }
    except BaseException as exc:
        error = [type(exc).__name__, str(exc)]
    return copy.deepcopy(
        scalar(
            {
                "name": case["name"],
                "events": effects.events,
                "result": result,
                "error": error,
                "files": effects.files,
                "streams": [[s.fixture_path, s.closed] for s in effects.streams],
                "process_returncode": effects.process.returncode,
            }
        )
    )


def generate():
    return {"source": "controller_f.py:Engine/post/TRACE_ENV", "cases": [observe(c) for c in CASES]}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
