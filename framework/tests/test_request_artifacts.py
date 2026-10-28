"""Reference-first whole request/artifact commands; all effects are supplied in memory."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import fnmatch
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_log_records import REFERENCE as REFERENCE_LOG
from test_loop_model import Effects, model_files, reference_api, snapshot, trace
from test_offline_planner import REFERENCE_PREFILL, exact, runtime_key

FIXTURES = Path(__file__).parent / "fixtures"
ORACLES = FIXTURES / "oracle"
GOLDENS = FIXTURES / "request-artifacts-goldens.json"


def frame(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def chat_frames(tokens=20, prompt=32, reply="synthetic"):
    return [
        frame(
            {
                "timings": {"prompt_n": prompt, "cache_n": 0, "prompt_ms": 25.0, "predicted_n": n},
                "choices": [{"delta": {"content": reply if n == 1 else ""}}],
            }
        )
        for n in range(1, tokens + 1)
    ] + ["data: [DONE]\n\n"]


class RequestEffects(Effects):
    def __init__(self, case):
        super().__init__(case)
        self.data.setdefault("/case/script.json", json.dumps([{"text": "One."}, {"text": "Two?"}]))
        self.data.setdefault("/case/tokens.json", "[32, 17]")
        self.ns, self.histories = {}, []
        self.http_index, self.ns_value = 0, 1_000_000_000
        self.responses = copy.deepcopy(case.get("responses", []))
        self.globs = copy.deepcopy(case.get("globs", {}))
        self.disk_reads = 0
        self.env = {"PORT": "8080", **self.env}

    def open(self, path, mode="r", *, errors=None, encoding=None):
        self.event("open_encoding", path, encoding)
        if path == "/proc/diskstats":
            seq = self.case.get("diskstats", ["8 0 toy 0 0 100 0\n"])
            self.data[path] = seq[min(self.disk_reads, len(seq) - 1)]
            self.disk_reads += 1
        return super().open(path, mode, errors=errors)

    def stat(self, path):
        if path in self.case.get("devices", {}):
            self.event("stat", path)
            return SimpleNamespace(st_dev=self.case["devices"][path])
        return super().stat(path)

    def major(self, value):
        self.event("major", value)
        return value // 256

    def minor(self, value):
        self.event("minor", value)
        return value % 256

    def glob(self, pattern):
        self.event("glob", pattern)
        if pattern in self.globs:
            return self.globs[pattern][:]
        candidates = dict.fromkeys(p.rsplit("/", 1)[0] for p in self.data if "/" in p)
        return [
            p
            for p in candidates
            if len(p.split("/")) == len(pattern.split("/")) and fnmatch.fnmatchcase(p, pattern)
        ]

    def time_ns(self):
        self.event("time_ns")
        self.ns_value += 20_000_000
        return self.ns_value

    def getitem_env(self, key):
        self.event("environ_getitem", key)
        return self.env[key]

    def callback(self, value):
        self.event("callback", value)
        if self.case.get("callback_mutate"):
            self.histories[-1].append({"role": "user", "content": "callback mutation"})

    def turn(self, function, *args):
        history = args[1]
        alias = next((i for i, h in enumerate(self.histories) if h is history), None)
        if alias is None:
            alias = len(self.histories)
            self.histories.append(history)
        self.event("turn_call", alias, history, args[0], args[2], len(args) > 3)
        return function(*args)

    def urlopen(self, request, timeout):
        body = json.loads(request.data)
        self.event(
            "urlopen", request.full_url, request.data.decode(), request.header_items(), timeout
        )
        index = self.http_index
        self.http_index += 1
        path = request.full_url.split(":8080", 1)[-1]
        configured = self.responses.pop(0) if self.responses else None
        if configured is None:
            if path == "/v1/chat/completions":
                prompt = self.case.get("prompt_n", 32)
                configured = {"lines": chat_frames(body["max_tokens"], prompt)}
            elif path == "/apply-template":
                configured = {"json": {"prompt": "|".join(m["content"] for m in body["messages"])}}
            elif path == "/tokenize":
                configured = {"json": {"tokens": list(range(len(body["content"])))}}
            else:
                configured = {
                    "json": {
                        "timings": {"prompt_n": self.case.get("warm_n", 128), "prompt_ms": 500.0}
                    }
                }
        h = self

        class Response:
            def __enter__(self):
                h.event("http_enter", index)
                return self

            def __exit__(self, kind, value, tb):
                h.event("http_exit", index, kind.__name__ if kind else None)
                return False

            def read(self):
                h.event("http_read", index)
                if "raw" in configured:
                    return configured["raw"].encode()
                return json.dumps(configured.get("json"), ensure_ascii=False).encode()

            def __iter__(self):
                for n, line in enumerate(configured.get("lines", [])):
                    h.event("http_next", index, n)
                    if n == 0 and h.engines:
                        e = h.engines[-1]
                        h.data[e.dir + "/server.log"] += trace(
                            h.case.get("prompt_n", 32),
                            h.case.get("fixture_tokens", body.get("max_tokens", 20)),
                            epoch=index + 1,
                        )
                    yield line.encode()
                h.event("http_end", index)

        return Response()


def execute_reference(name, h, api, argv=()):
    """Execute the complete frozen script, replacing imports only with supplied ports."""
    source = ORACLES / (name + "_reference.py")
    fake_sys = SimpleNamespace(argv=[name + ".py", *argv], path=[], exit=sys.exit)

    class Environ:
        get = staticmethod(h.getenv)

        def __getitem__(self, key):
            return h.getitem_env(key)

    path = SimpleNamespace(
        dirname=os.path.dirname, abspath=os.path.abspath, basename=os.path.basename, exists=h.exists
    )
    fake_os = SimpleNamespace(
        path=path, stat=h.stat, major=h.major, minor=h.minor, environ=Environ()
    )

    class Parser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            return super().parse_args(fake_sys.argv[1:] if args is None else args, namespace)

    client = h.ns.get("client")
    modules = {
        "argparse": SimpleNamespace(ArgumentParser=Parser),
        "sys": fake_sys,
        "os": fake_os,
        "time": SimpleNamespace(time=h.clock, time_ns=h.time_ns),
        "glob": SimpleNamespace(glob=h.glob),
        "urllib.request": SimpleNamespace(
            request=SimpleNamespace(Request=urllib.request.Request, urlopen=h.urlopen)
        ),
        "prefill": REFERENCE_PREFILL,
        "logparse": REFERENCE_LOG,
        "machine": SimpleNamespace(Machine=api.make_machine),
        "controller": SimpleNamespace(Controller=api.Controller),
    }
    if client:
        modules["client"] = SimpleNamespace(turn=lambda *a: h.turn(client["turn"], *a))

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        return (
            modules[name]
            if name in modules
            else builtins.__import__(name, globals, locals, fromlist, level)
        )

    ns = {
        "__file__": str(source),
        "__name__": "frozen_request_artifact",
        "__builtins__": {**vars(builtins), "__import__": importing},
        "open": h.open,
    }
    h.ns[name] = ns
    exec(compile(source.read_text(), str(source), "exec"), ns)
    return ns


def argv_for(kind, case):
    if kind == "report":
        return ["/case/run"]
    if kind == "conversation":
        return [
            "--boot",
            "/case/run",
            "--script",
            "/case/script.json",
            "--tokens",
            "/case/tokens.json",
            "--decode",
            "20",
            "--window",
            "1-20",
            "--mode",
            case.get("mode", "static"),
            "--start",
            "1,all",
            "--out",
            "/case/run/conv-static",
            *case.get("argv", []),
        ]
    return [
        "--port",
        "8080",
        "--script",
        "/case/script.json",
        "--max-tokens",
        "20",
        "--out",
        "/case/client.jsonl",
        *case.get("argv", []),
    ]


@contextlib.contextmanager
def original_api(h):
    with reference_api(h) as api:
        execute_reference("client", h, api)
        yield SimpleNamespace(
            model=api,
            turn=lambda *a: h.turn(h.ns["client"]["turn"], *a),
            script=lambda argv: script_reference(h, api, argv),
            bootstrap=lambda argv: execute_reference("boot_client", h, api, argv),
            conversation=lambda argv: execute_reference("converse", h, api, argv),
            report=lambda argv: execute_reference("compare_online", h, api, argv),
        )


def script_reference(h, api, argv):
    ns = execute_reference("client", h, api, argv)
    fn = ns["turn"]
    ns["turn"] = lambda *a: h.turn(fn, *a)
    return ns["main"]()


def pipeline(h, api, case):
    """Actual reference producer output feeds the already frozen fit→plan→settings call graph."""
    from test_recorded_calibration import observe_reference as fit
    from test_recorded_calibration import synthetic_files

    raw = synthetic_files({})
    raw["client.jsonl"] = h.data["/case/client.jsonl"]
    device = model_files()["probe/device.json"]
    fit_case = {
        "runs": {},
        "files": {"/case/a/" + k: v for k, v in raw.items()},
        "chain": True,
        "device": device,
    }
    result = fit(fit_case)
    if result["outcome"] != ["return"] or result["chain"]["outcome"] != ["return"]:
        return result
    writes = result["writes"]
    h.data["/case/run/insitu.json"] = writes["/case/insitu.json"]
    h.data["/case/run/model.json"] = writes["/case/model.json"]
    h.data["/case/run/plan.json"] = result["chain"]["writes"]["plan.json"]
    api.conversation(argv_for("conversation", {**case, "argv": case.get("conversation_argv", [])}))
    api.report(["/case/run"])
    return result


def observe(case, api_factory=original_api):
    h = RequestEffects(case)
    output = SimpleNamespace(write=h.output, flush=lambda: h.event("stdout_flush"))
    result, chain, snap = None, None, None
    with api_factory(h) as api, contextlib.redirect_stdout(output):
        try:
            kind = case["kind"]
            if kind == "turn":
                args = [
                    8080,
                    copy.deepcopy(case.get("history", [{"role": "user", "content": "你好"}])),
                    case.get("max_tokens", 20),
                ]
                if case.get("callback", True):
                    args.append(h.callback)
                result = api.turn(*args)
            elif kind == "script":
                result = api.script(argv_for(kind, case))
            elif kind == "bootstrap":
                api.bootstrap(argv_for(kind, case))
                if case.get("pipeline"):
                    chain = pipeline(h, api, case)
            elif kind == "conversation":
                api.conversation(argv_for(kind, case))
                if case.get("report"):
                    api.report(["/case/run"])
            elif kind == "report":
                api.report(argv_for(kind, case))
            outcome = ["return", exact(result)]
        except (Exception, SystemExit) as error:
            outcome = [type(error).__name__, str(error)]
        snap = snapshot(None, api.model, h)
        states = {
            key: exact(value)
            for key, value in h.ns.get("boot_client", {}).items()
            if key in ("text", "msgs", "devices", "drive", "counts", "hist", "total")
        }
    return copy.deepcopy(
        {
            "outcome": outcome,
            "events": h.events,
            "stdout": h.stdout.getvalue(),
            "files": {p: v for p, v in h.data.items() if p not in h.initial or v != h.initial[p]},
            "histories": exact(h.histories),
            "snapshot": snap,
            "state": states,
            "chain": chain,
        }
    )


def data():
    result = json.loads(GOLDENS.read_text()) if GOLDENS.exists() else {"cases": []}
    boundary = FIXTURES / "request-artifacts-return-boundary.json"
    if boundary.exists():
        corrections = json.loads(boundary.read_text())["events"]
        for case in result["cases"]:
            for runtime, events in corrections.get(case["name"], {}).items():
                case["expected"][runtime]["events"] = events
    return result


@pytest.mark.parametrize("case", data()["cases"], ids=lambda c: c["name"])
def test_reference_request_artifact(case):
    assert observe(case) == case["expected"][runtime_key()]


def test_request_source_bytes():
    manifest = json.loads((ORACLES / "request-artifacts-manifest.json").read_text())
    for spec in manifest["sources"].values():
        assert hashlib.sha256((ORACLES / spec["oracle"]).read_bytes()).hexdigest() == spec["sha256"]


def test_returned_snapshot_does_not_alias_later_stream_cleanup(monkeypatch):
    retained = []
    original = RequestEffects

    class Retained(original):
        def __init__(self, case):
            super().__init__(case)
            retained.append(self)

    monkeypatch.setattr(sys.modules[__name__], "RequestEffects", Retained)
    result = observe({"kind": "conversation"})
    before = copy.deepcopy(result)
    h = retained[0]
    assert not h.controllers[-1].events.closed
    h.controllers[-1].events.close()
    assert h.events[-1] == ["close", {"tuple": ["/case/run/conv-static/events.jsonl"]}]
    assert result == before
