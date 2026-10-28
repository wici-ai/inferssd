"""F-first complete observation/session/controller transcripts, with no native effects."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_log_records import F as F_LOG
from test_offline_planner import F_MACHINE, F_PLACEMENT, F_PREFILL, digest, exact, load, state

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = FIXTURES / "loop-model-goldens.json"


def model_files():
    return copy.deepcopy(
        json.loads((FIXTURES / "offline-planner-goldens.json").read_text())["cases"][0]["files"]
    )


def trace(prompt=32, tokens=20, start=None, epoch=1, partial=False, ragged=False):
    start = prompt if start is None else start
    lines = [
        f"[kimi_cpu_prefill_graph] event=activate requested_tokens={prompt} configured_max=128\n",
        '[kimi_cpu_sink_profile] {"phase":"prefill","event":"layer","compute_wall_us":12000}\n',
        f"[moe_prefill_copy] bytes=1048576 read_us=500 passes=1 tokens={prompt}\n",
    ]
    for i in range(tokens):
        for layer in range(1 if ragged and i == 1 else 2):
            base = dict(phase="decode", event="layer", epoch=epoch, token=start + i, layer=layer)
            lines.append(
                "[kimi_cpu_sink_trace] "
                + json.dumps(
                    dict(
                        base,
                        selected_experts=[i % 7, i % 7 + 1],
                        vram_hit=[True, False],
                        cache_hit=[False, True],
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            lines.append(
                "[kimi_cpu_sink_profile] "
                + json.dumps(
                    dict(
                        base,
                        total_us=1500 + i,
                        wait_io_us=500,
                        compute_wall_us=500,
                        other_wall_us=500 + i,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
    text = "".join(lines)
    return text.rstrip("\n") if partial else text


def reply(prompt=32, tokens=20, cache=0):
    return {
        "timings": {"prompt_n": prompt, "cache_n": cache, "prompt_ms": 25.0, "predicted_n": tokens},
        "tok_s": [i * 0.01 + i * i * 0.000001 for i in range(tokens)],
        "wall_s": 1.5,
        "reply": "synthetic",
    }


class Effects:
    def __init__(self, case):
        self.case = case
        self.data = {"/case/run/" + p: json.dumps(v) for p, v in model_files().items()}
        self.data.update(copy.deepcopy(case.get("files", {})))
        self.initial = copy.deepcopy(self.data)
        self.mtimes = {}
        self.events, self.engines, self.streams = [], [], []
        self.clock_value, self.fault = 100.0, case.get("fault")
        self.counts = {}
        self.stdout = io.StringIO()
        self.machines = []
        self.controllers = []
        self.env = dict(case.get("env", {}))

    def event(self, op, *args):
        self.events.append([op, exact(args)])
        key = (op, str(args[0]) if args else "")
        self.counts[key] = self.counts.get(key, 0) + 1
        f = self.fault
        if f and f[0] == op and f[1] in (key[1], "*") and (len(f) < 3 or f[2] == self.counts[key]):
            kind = RuntimeError if len(f) > 3 and f[3] == "RuntimeError" else OSError
            raise kind(f"configured {op} failure: {key[1]}")

    def open(self, path, mode="r", *, errors=None):
        self.event("open", path, mode, errors)
        if mode not in ("w", "a") and path not in self.data:
            raise FileNotFoundError(f"missing synthetic file: {path}")
        binary = "b" in mode
        raw = self.data.get(path, "")
        if mode == "w":
            raw = ""
        if binary and isinstance(raw, str):
            raw = raw.encode()
        if not binary and isinstance(raw, bytes):
            raw = raw.decode(errors=errors or "strict")
        owner = self
        base = io.BytesIO if binary else io.StringIO

        class Stream(base):
            def read(self, *a):
                owner.event("read", path, a)
                return super().read(*a)

            def seek(self, *a):
                owner.event("seek", path, a)
                return super().seek(*a)

            def __next__(self):
                owner.event("next", path)
                return super().__next__()

            def write(self, text):
                try:
                    owner.event("write", path, text)
                except OSError:
                    super().write(text[:1])
                    owner.data[path] = self.getvalue()
                    raise
                n = super().write(text)
                owner.data[path] = self.getvalue()
                return n

            def flush(self):
                owner.event("flush", path)
                return super().flush()

            def close(self):
                if not self.closed:
                    super().close()
                    owner.event("close", path)

            def __del__(self):
                try:
                    self.close()
                except Exception as error:
                    owner.events.append(["ignored_close", type(error).__name__, str(error)])

        stream = Stream(raw)
        if mode == "a":
            stream.seek(0, 2)
        if mode in ("w", "a"):
            self.data[path] = stream.getvalue()
        self.streams.append((path, weakref.ref(stream)))
        return stream

    def gzip_open(self, path, mode="r", *, errors=None):
        self.event("gzip_open", path, mode, errors)
        return self.open(path, mode, errors=errors)

    def exists(self, path):
        self.event("exists", path)
        return path in self.data

    def stat(self, path):
        self.event("stat", path)
        if path not in self.data:
            raise FileNotFoundError(f"missing synthetic file: {path}")
        raw = self.data[path]
        raw = raw.encode() if isinstance(raw, str) else raw
        return SimpleNamespace(st_size=len(raw), st_mtime_ns=self.mtimes.get(path, 100))

    def makedirs(self, path, exist_ok=False):
        self.event("makedirs", path, exist_ok)

    def getenv(self, key, default=None):
        self.event("getenv", key, default)
        return self.env.get(key, default)

    def clock(self):
        self.event("clock")
        self.clock_value += 0.25
        return self.clock_value

    def output(self, text):
        self.event("stdout", text)
        return self.stdout.write(text)

    def post(self, *args):
        self.event("post", *args)
        return {"ignored": "response"}

    def engine(self, d, port, tier, cpus, envs, args):
        self.event("engine", d, port, tier, cpus, envs, args)
        h = self
        index = len(self.engines)

        class Engine:
            def __init__(self):
                self.dir, self.port, self.tier, self.cpus, self.envs = d, port, tier, cpus, envs
                self.ok = index not in h.case.get("bad_start", [])
                self.start_s = h.case.get("start_s", 2.0)
                self.is_alive = True

            def alive(self):
                h.event("alive", self.dir)
                return self.is_alive

            def stop(self):
                h.event("stop", self.dir)
                self.is_alive = False
                if h.case.get("mutate_stop"):
                    self.tier, self.cpus = 1, "split"

        e = Engine()
        self.engines.append(e)
        self.data.setdefault(d + "/server.log", "")
        return e


class Pricing:
    """Only controlled external model responses for exact threshold/ordering cases."""

    def __init__(self, h):
        self.h = h
        self.slope = 0.0
        self.P = {}
        self.PF = {}
        self.routing = [[[[0]]]]
        self.configs = h.case.get("configs", [[1, "stay"], [2, "other"]])

    def predict(self, *args):
        self.h.event("predict", *args)
        _tier, p, rem = args[1], args[2], args[4]
        cost = self.h.case.get("costs", {}).get(p, 100.0)
        return [{"prefill_s": cost, "decode_tps": 1.0, "hit_v": 0.5} for _ in rem]


@contextlib.contextmanager
def f_api(h):
    saved = sys.path[:]
    try:
        with patch.dict(sys.modules, prefill=F_PREFILL, logparse=F_LOG):
            online = load("online")
        with patch.dict(sys.modules, online=online, machine=F_MACHINE):
            controller = load("controller")
    finally:
        sys.path[:] = saved

    def machine(run):
        h.event("machine", run)
        m = F_MACHINE.Machine(run)
        h.machines.append(m)
        return m

    def make_controller(*args, **kwargs):
        instance = controller.Controller.__new__(controller.Controller)
        h.controllers.append(instance)
        controller.Controller.__init__(instance, *args, **kwargs)
        return instance

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(F_MACHINE, "open", h.open, create=True))
        stack.enter_context(
            patch.object(F_MACHINE, "os", SimpleNamespace(path=SimpleNamespace(exists=h.exists)))
        )
        stack.enter_context(
            patch.object(
                F_PLACEMENT,
                "default_threads",
                lambda dev: int(dev["default_threads"]) if dev.get("default_threads") else 8,
            )
        )
        stack.enter_context(patch.object(F_LOG, "open", h.open, create=True))
        stack.enter_context(patch.object(F_LOG, "gzip", SimpleNamespace(open=h.gzip_open)))
        stack.enter_context(patch.object(online, "open", h.open, create=True))
        stack.enter_context(
            patch.object(
                online, "os", SimpleNamespace(path=SimpleNamespace(exists=h.exists), stat=h.stat)
            )
        )
        stack.enter_context(patch.object(controller, "Machine", machine))
        stack.enter_context(patch.object(controller, "Engine", h.engine))
        stack.enter_context(patch.object(controller, "post", h.post))
        stack.enter_context(patch.object(controller, "open", h.open, create=True))
        stack.enter_context(
            patch.object(
                controller,
                "os",
                SimpleNamespace(environ=SimpleNamespace(get=h.getenv), makedirs=h.makedirs),
            )
        )
        stack.enter_context(patch.object(controller, "time", SimpleNamespace(time=h.clock)))
        yield SimpleNamespace(
            online=online,
            make_machine=machine,
            Controller=make_controller,
            cache_read=online._read_arm_cached,
            arms=online._ARMS,
            live=online._LIVE,
            traced={
                v.__code__: "machine." + k
                for k, v in vars(F_MACHINE.Machine).items()
                if callable(v)
            },
        )


def cache_state(api):
    return {
        "arms": [[exact(k), digest(v)] for k, v in api.arms.items()],
        "live": [[p, offset, digest(vars(parser))] for p, (parser, offset) in api.live.items()],
    }


def snapshot(ctl, api, h):
    if ctl is None and h.controllers:
        ctl = h.controllers[-1]
    if ctl is None:
        return {"cache": cache_state(api)}
    fields = {
        k: getattr(ctl, k)
        for k in (
            "out",
            "port",
            "args",
            "loop",
            "w0",
            "w1",
            "decode",
            "horizon",
            "keeps_history",
            "insts",
            "asked",
            "n_turns",
            "switch_s",
            "seen",
        )
        if hasattr(ctl, k)
    }
    e = getattr(ctl, "engine", None)
    fields["engine"] = dict(vars(e)) if e else None
    fields["model_digest"] = digest(state(ctl.m)) if hasattr(ctl, "m") else None
    fields["events_closed"] = ctl.events.closed if hasattr(ctl, "events") else None
    return {"controller": exact(fields), "cache": cache_state(api)}


def observe(case, api_factory=f_api):
    h = Effects(case)
    steps = []
    ctl = None
    m = None
    previous = []
    aliases = []
    machine_calls = []
    stdout = SimpleNamespace(write=h.output, flush=lambda: h.event("stdout_flush"))
    prior = sys.getprofile()
    with api_factory(h) as api, patch.object(sys, "stdout", stdout):

        def profile(frame, event, arg):
            if event == "call" and frame.f_code in api.traced:
                machine_calls.append(api.traced[frame.f_code])

        sys.setprofile(profile)
        try:
            for action in case["actions"]:
                kind, *args = copy.deepcopy(action)
                try:
                    if kind == "create":
                        kw = {"window": (1, 20), "decode": 20, **(args[0] if args else {})}
                        ctl = api.Controller(
                            "/case/run",
                            "/case/conv",
                            18080,
                            case.get("start", [4, "all"]),
                            ["--seed", "1"],
                            **kw,
                        )
                        value = None
                    elif kind == "machine":
                        m = api.make_machine("/case/run")
                        value = None
                    elif kind == "pricing":
                        m = Pricing(h)
                        value = None
                    elif kind == "put":
                        h.data[args[0]] = args[1]
                        value = None
                    elif kind == "append":
                        h.data[args[0]] = h.data.get(args[0], "") + args[1]
                        value = None
                    elif kind == "mtime":
                        h.mtimes[args[0]] = args[1]
                        value = None
                    elif kind == "delete":
                        del h.data[args[0]]
                        value = None
                    elif kind == "fault":
                        h.fault = args[0]
                        h.counts = {}
                        value = None
                    elif kind == "cache":
                        value = api.cache_read(*args)
                        aliases.append([kind, [value is p for p in previous]])
                        previous.append(value)
                    elif kind == "served":
                        value = api.online.served(m, *args)
                    elif kind in ("fit_slope", "feedback", "decide"):
                        value = getattr(api.online, kind)(m, *args)
                    elif kind == "expect":
                        value = api.online.expect(*args)
                    elif kind == "deliver":
                        options = args[0] if args else {}
                        log = trace(**options)
                        h.data[ctl.engine.dir + "/server.log"] += log
                        rec = reply(
                            options.get("prompt", 32),
                            options.get("tokens", 20),
                            options.get("start", options.get("prompt", 32))
                            - options.get("prompt", 32),
                        )
                        rec.update(case.get("record_override", {}))
                        ctl.after(options.get("prompt", 32), rec)
                        value = None
                    elif kind == "kill":
                        ctl.engine.is_alive = False
                        value = None
                    elif kind == "before":
                        value = ctl.before(*args)
                    elif kind == "prepare":
                        value = ctl.prepare()
                        aliases.append(
                            [
                                kind,
                                value[0] is ctl.seen[1],
                                value[1] is ctl.seen[2],
                                [value[0] is p for p in previous],
                            ]
                        )
                        previous.append(value[0])
                    elif kind == "note":
                        value = ctl.note(args[0])
                        aliases.append(["note_argument", exact(args[0])])
                    elif kind == "close":
                        value = ctl.close()
                    elif kind == "state":
                        value = ctl.context
                    else:
                        raise AssertionError(kind)
                    steps.append([kind, "return", exact(value), snapshot(ctl, api, h)])
                except (Exception, SystemExit) as error:
                    steps.append([kind, type(error).__name__, str(error), snapshot(ctl, api, h)])
        finally:
            sys.setprofile(prior)
        # Observe while Controller and streams remain alive. Implicit fixture destruction
        # is deliberately outside the transcript; close()'s event-stream state is explicit.
        result = {
            "steps": steps,
            "events": h.events,
            "stdout": h.stdout.getvalue(),
            "aliases": aliases,
            "files": {p: v for p, v in h.data.items() if h.initial.get(p) != v},
            "machine_calls": machine_calls,
            "model_state": state(m)
            if m is not None and not isinstance(m, Pricing)
            else (exact({"slope": m.slope}) if m else None),
            "final": snapshot(ctl, api, h),
        }
        return copy.deepcopy(result)


def data():
    return json.loads(GOLDENS.read_text())


def runtime():
    v = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if v not in ("3.9", "3.12"):
        pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")
    return v


@pytest.mark.parametrize(
    "case", data()["cases"] if GOLDENS.exists() else [], ids=lambda c: c["name"]
)
def test_f_loop(case):
    assert observe(case) == case["expected"][runtime()]


def test_f_loop_sources():
    for row in json.loads((FIXTURES / "oracle/loop-manifest.json").read_text())["files"]:
        assert (
            hashlib.sha256((FIXTURES / "oracle" / row["file"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
