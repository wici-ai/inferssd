from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ORACLES = FIXTURES / "oracle"


def load(name):
    spec = importlib.util.spec_from_file_location(f"planner_{name}_F", ORACLES / f"{name}_f.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved_path = sys.path[:]
try:
    F_SIM, F_PREFILL, F_PLACEMENT = load("sim"), load("prefill"), load("placement")
    with patch.dict(sys.modules, sim=F_SIM, prefill=F_PREFILL, placement=F_PLACEMENT):
        F_SETTINGS = load("settings")
        with patch.dict(sys.modules, settings=F_SETTINGS):
            F_MACHINE = load("machine")
finally:
    sys.path[:] = saved_path


def exact(value):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {"items": [[exact(k), exact(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {type(value).__name__: [exact(v) for v in value]}
    return value


def digest(value):
    return hashlib.sha256(json.dumps(exact(value), ensure_ascii=False).encode()).hexdigest()


class Watched(dict):
    def __init__(self, data, events, name):
        self.events, self.name = events, name
        super().__init__((k, watched(v, events, f"{name}.{k}")) for k, v in data.items())

    def __getitem__(self, key):
        self.events.append(["get", self.name, key])
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.events.append(["optional", self.name, key])
        return super().get(key, default)


def watched(data, events, name):
    if isinstance(data, dict):
        return Watched(data, events, name)
    if isinstance(data, list):
        return [watched(v, events, f"{name}[{i}]") for i, v in enumerate(data)]
    return data


def state(machine):
    result = {}
    for key, value in vars(machine).items():
        if key.startswith("_artifact") or key == "_host_threads":
            continue
        if key == "_replays":
            value = {k: {**vars(r), "tier": vars(r.tier)} for k, r in value.items()}
        result[key] = value
    return exact(result)


class Transcript:
    def __init__(self, case):
        self.case = case
        self.files = copy.deepcopy(case["files"])
        self.events, self.reads, self.loaded = [], [], []
        self.writes = {}
        self.clocks = iter(case.get("clock", [100.0, 101.25]))

    def relative(self, path):
        return str(path).removeprefix("/case/run/")

    def read_json(self, path):
        name = self.relative(path)
        self.events.append(["read_json", name])
        if name not in self.files:
            raise FileNotFoundError(f"missing artifact: {name}")
        raw = self.files[name]
        parsed = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
        value = watched(parsed, self.reads, name)
        self.loaded.append(value)
        return value

    def exists(self, path):
        name = self.relative(path)
        self.events.append(["exists", name])
        return name in self.files

    def open(self, path, mode="r"):
        if mode == "r":
            return SimpleNamespace(name=str(path))
        name = self.relative(path)
        self.events.append(["open_write", name])
        if self.case.get("fail_open"):
            raise OSError("configured output open failure")
        transcript = self

        class Writer(io.StringIO):
            def write(self, text):
                if transcript.case.get("fail_write"):
                    raise OSError("configured output write failure")
                transcript.writes[name] = transcript.writes.get(name, "") + text
                transcript.files[name] = transcript.writes[name]
                return len(text)

        return Writer()

    def write_json(self, path, value):
        json.dump(value, self.open(path, "w"), indent=1)

    def replace(self, source, destination):
        self.events.append(["replace", self.relative(source), self.relative(destination)])
        if self.case.get("fail_replace"):
            raise OSError("configured replace failure")
        self.writes[self.relative(destination)] = self.writes.pop(self.relative(source))

    def clock(self):
        self.events.append(["clock"])
        return next(self.clocks)

    def host(self):
        self.events.append(["host"])
        if self.case.get("host_error"):
            raise RuntimeError("configured host failure")
        return self.case.get("host_threads", 8)


def actions(machine, case):
    out = []
    for item in case.get("actions", []):
        method, kwargs = item["method"], copy.deepcopy(item.get("kwargs", {}))
        if item.get("routing"):
            kwargs["segs"] = machine.routing[: item.get("segments", len(machine.routing))]
        if item.get("pending_iterator"):
            kwargs["prompts"] = iter(kwargs["prompts"])
        try:
            value = getattr(machine, method)(**kwargs)
            out.append(
                [method, exact(value) if method != "replay" else "replay", digest(state(machine))]
            )
        except Exception as error:
            out.append([method, type(error).__name__, str(error), digest(state(machine))])
    return out


def observe_f(case):
    transcript, instances = Transcript(case), []
    output = io.StringIO()
    calls = []

    class Captured(F_MACHINE.Machine):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

    names = {v.__code__: k for k, v in vars(F_MACHINE.Machine).items() if callable(v)}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in names:
            calls.append(names[frame.f_code])

    fake_json = SimpleNamespace(
        load=lambda stream: transcript.read_json(stream.name), dump=json.dump, dumps=json.dumps
    )
    machine_api = SimpleNamespace(
        Machine=Captured, layer_faults=F_MACHINE.layer_faults, row_faults=F_MACHINE.row_faults
    )
    original_host = F_PLACEMENT.default_threads

    def host(dev):
        if dev.get("default_threads"):
            return int(dev["default_threads"])
        return transcript.host()

    previous_profile, previous_path = sys.getprofile(), sys.path[:]
    outcome, action_results = None, []
    argv = ["plan.py", "/case/run", "-o", "/case/run/plan.json"] + case.get("argv", [])
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(patch.dict(sys.modules, machine=machine_api))
            stack.enter_context(patch.object(F_MACHINE, "open", transcript.open, create=True))
            stack.enter_context(patch.object(F_MACHINE, "json", fake_json))
            stack.enter_context(
                patch.object(
                    F_MACHINE, "os", SimpleNamespace(path=SimpleNamespace(exists=transcript.exists))
                )
            )
            F_PLACEMENT.default_threads = host
            sys.setprofile(profile)
            if case.get("mode", "plan") == "machine":
                pending = case.get("pending")
                if case.get("pending_iterator"):
                    pending = iter(pending)
                m = Captured("/case/run", pending=pending, workload=case.get("workload"))
                action_results = actions(m, case)
                outcome = ["return"]
            else:
                with (
                    patch.object(sys, "argv", argv),
                    patch.dict(
                        sys.modules, json=fake_json, time=SimpleNamespace(time=transcript.clock)
                    ),
                ):
                    exec(
                        compile((ORACLES / "plan_f.py").read_text(), "frozen-plan.py", "exec"),
                        {"__file__": "/oracle/plan_f.py", "open": transcript.open},
                    )
                outcome = ["return"]
                if case.get("reload", True):
                    with (
                        patch.object(F_SETTINGS, "open", transcript.open, create=True),
                        patch.object(F_SETTINGS, "json", fake_json),
                        patch.object(
                            F_SETTINGS,
                            "os",
                            SimpleNamespace(
                                path=SimpleNamespace(abspath=lambda p: p),
                                replace=transcript.replace,
                            ),
                        ),
                        patch.dict(sys.modules, logparse=SimpleNamespace(ledger=lambda _: None)),
                        patch.object(
                            sys, "argv", ["settings.py", "/case/run", "-o", "/case/settings.env"]
                        ),
                    ):
                        F_SETTINGS.main()
    except (Exception, SystemExit) as error:
        outcome = [type(error).__name__, str(error)]
    finally:
        F_PLACEMENT.default_threads = original_host
        sys.path[:] = previous_path
        sys.setprofile(previous_profile)
    return {
        "outcome": outcome,
        "events": transcript.events,
        "stdout": output.getvalue(),
        "writes": transcript.writes,
        "states": [digest(state(m)) for m in instances],
        "aliases": [
            [getattr(m, "P", None) is getattr(m, "ins", {}).get("placements")] for m in instances
        ],
        "actions": action_results,
        "calls": calls,
        "mapping_reads_sha256": digest(transcript.reads),
        "mapping_read_count": len(transcript.reads),
        "loaded_after": [digest(x) for x in transcript.loaded],
    }


DATA = json.loads((FIXTURES / "offline-planner-goldens.json").read_text())


def runtime_key():
    key = f"{sys.version_info.major}.{sys.version_info.minor}"
    if key not in ("3.9", "3.12"):
        pytest.skip("F byte observations currently recorded for Python 3.9 and 3.12 only")
    return key


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["name"])
def test_f_frozen(case):
    assert observe_f(case) == case["expected"][runtime_key()]


def test_plan_oracle_identity():
    manifest = json.loads((ORACLES / "planner-manifest.json").read_text())
    assert hashlib.sha256((ORACLES / "plan_f.py").read_bytes()).hexdigest() == manifest["sha256"]


def observe_candidate(case):
    import argparse

    from ssdash.model import machine as candidate
    from ssdash.plan.planner import run_plan
    from ssdash.plan.planner_types import PlanArguments

    transcript, instances = Transcript(case), []
    output, calls = io.StringIO(), []

    class Captured(candidate.Machine):
        def __init__(self, artifacts, *args, **kwargs):
            instances.append(self)
            if isinstance(artifacts, str):
                artifacts = transcript
            kwargs["host_threads"] = transcript.host
            super().__init__(artifacts, *args, **kwargs)

    names = {v.__code__: k for k, v in vars(candidate.Machine).items() if callable(v)}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in names:
            calls.append(names[frame.f_code])

    original_profile, original_path = sys.getprofile(), sys.path[:]
    outcome, action_results = None, []
    try:
        with contextlib.redirect_stdout(output):
            sys.setprofile(profile)
            if case.get("mode", "plan") == "machine":
                pending = case.get("pending")
                if case.get("pending_iterator"):
                    pending = iter(pending)
                m = Captured(transcript, pending=pending, workload=case.get("workload"))
                action_results = actions(m, case)
                outcome = ["return"]
            else:
                parser = argparse.ArgumentParser()
                parser.add_argument("--prompts", required=True)
                parser.add_argument("--decode", type=int, default=512)
                parser.add_argument("--window", default="129-512")
                parser.add_argument("--known", action="store_true")
                parser.add_argument("--pending", default="")
                a = parser.parse_args(case.get("argv", []))
                run_plan(
                    PlanArguments(
                        a.prompts, a.decode, a.window, a.known, a.pending, "/case/run/plan.json"
                    ),
                    transcript,
                    host_threads=transcript.host,
                    clock=transcript.clock,
                    emit=print,
                    write_json=transcript.write_json,
                    factory=Captured,
                )
                outcome = ["return"]
                if case.get("reload", True):
                    from ssdash.plan.settings_command import SettingsArguments, run_settings

                    def write_settings(path, text):
                        with transcript.open(f"{path}.tmp", "w") as stream:
                            stream.write(text)
                        transcript.replace(f"{path}.tmp", path)

                    run_settings(
                        SettingsArguments("/case/run", out="/case/settings.env"),
                        SimpleNamespace(read_json=transcript.read_json, exists=transcript.exists),
                        host_threads=transcript.host,
                        ledger=lambda _: None,
                        absolute_path=lambda p: p,
                        write_text=write_settings,
                        stdout=sys.stdout.write,
                        factory=Captured,
                    )
    except (Exception, SystemExit) as error:
        outcome = [type(error).__name__, str(error)]
    finally:
        sys.path[:] = original_path
        sys.setprofile(original_profile)
    return {
        "outcome": outcome,
        "events": transcript.events,
        "stdout": output.getvalue(),
        "writes": transcript.writes,
        "states": [digest(state(m)) for m in instances],
        "aliases": [
            [getattr(m, "P", None) is getattr(m, "ins", {}).get("placements")] for m in instances
        ],
        "actions": action_results,
        "calls": calls,
        "mapping_reads_sha256": digest(transcript.reads),
        "mapping_read_count": len(transcript.reads),
        "loaded_after": [digest(x) for x in transcript.loaded],
    }


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["name"])
def test_new_complete_callgraph(case):
    assert observe_candidate(case) == case["expected"][runtime_key()]
