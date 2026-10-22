from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE_PATH = FIXTURES / "oracle/prefill_f.py"
SPEC = importlib.util.spec_from_file_location("prefill_completion_f_oracle", ORACLE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def exact(value):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {"items": [[exact(k), exact(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [exact(v) for v in value]}
    if isinstance(value, list):
        return {"list": [exact(v) for v in value]}
    return value


class ObservedDict(dict):
    def __init__(self, path, events):
        super().__init__()
        self.path, self.events = path, events

    def __getitem__(self, key):
        self.events.append(["getitem", self.path, key])
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.events.append(["get", self.path, key, exact(default)])
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.events.append(["setitem", self.path, key, exact(value)])
        return super().__setitem__(key, value)


def watched(value, path, events, memo):
    if isinstance(value, (dict, list)) and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = ObservedDict(path, events)
        memo[id(value)] = result
        for k, v in value.items():
            dict.__setitem__(result, k, watched(v, f"{path}.{k}", events, memo))
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(watched(v, f"{path}[{i}]", events, memo) for i, v in enumerate(value))
        return result
    return value


def object_paths(value, path, output):
    if isinstance(value, (dict, list)):
        output.setdefault(id(value), []).append(path)
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in entries:
            object_paths(item, f"{path}.{key}", output)


FIELDS = {
    "union_points": ("N", "L", "out"),
    "rest_fit": ("sp", "sn", "sr", "spp", "snn", "spn", "spr", "snr", "det", "a", "d"),
    "build": (
        "E",
        "L",
        "k",
        "r_d",
        "link",
        "rates",
        "b_e",
        "role",
        "dec",
        "prompt",
        "scale",
        "union",
    ),
    "graph_line": ("role", "rs", "n", "dev", "host"),
    "graph_pass_max": ("g", "room", "per_token"),
    "drive_factor": ("lo", "hi", "f"),
    "feedback": ("out", "pc", "rd", "fs", "kw", "f"),
    "check": ("rows", "p", "s", "first", "own", "cross"),
}
DEPENDENCIES = (
    "coverage",
    "pass_seconds",
    "seconds",
    "cut",
    "lane_layer",
    "coupled",
    "break_even",
    "threshold",
    "served",
    "offloaded",
    "cold_share",
)


def observe(module, case):
    args = copy.deepcopy(case["arguments"])
    if case.get("shared_memory"):
        args["ins"]["graph"]["memory"] = args["ins"]["placements"]["all"]["memory"]
    if case.get("shared_model_memory"):
        args["m"]["graph_memory"] = args["m"]["memory"]["all"]
    reads, memo = [], {}
    args = watched(args, "args", reads, memo)
    before = exact(args)
    input_objects = {}
    object_paths(args, "args", input_objects)
    callbacks = []
    how_count = 0

    def how(placement):
        nonlocal how_count
        how_count += 1
        callbacks.append(["how", placement])
        spec = case.get("how", {})
        if how_count == spec.get("raise_on"):
            raise RuntimeError("configured how failure")
        if how_count == spec.get("mutate_on"):
            args["m"]["cov_warm"] = spec["cov_warm"]
        value = spec.get("placements", {}).get(placement, [None, None])
        callbacks.append(["how_return", exact(tuple(value))])
        return tuple(value)

    def time_of(factor):
        spec = case["time_of"]
        callbacks.append(["time_of", exact(factor)])
        if sum(row[0] == "time_of" for row in callbacks) == spec.get("raise_on"):
            raise RuntimeError("configured time_of failure")
        value = spec["base"] + spec["slope"] * factor
        if "floor" in spec:
            value = max(spec["floor"], value)
        callbacks.append(["time_return", exact(value)])
        return value

    names = [*FIELDS, *DEPENDENCIES]
    functions = {getattr(module, name).__code__: name for name in names}
    fn = getattr(module, case["function"])
    calls, state = [], {}

    def capture(frame, event, arg):
        if event == "call" and frame.f_code in functions:
            calls.append(functions[frame.f_code])
        if event == "return" and frame.f_code is fn.__code__:
            state.update(
                {key: frame.f_locals[key] for key in FIELDS[fn.__name__] if key in frame.f_locals}
            )

    kwargs = dict(args)
    if case.get("point_iterator"):
        kwargs["points"] = iter(kwargs["points"])
    if case["function"] == "drive_factor":
        kwargs["time_of"] = time_of
    if case["function"] == "feedback":
        kwargs["how"] = how
    previous = sys.getprofile()
    sys.setprofile(capture)
    result = None
    try:
        try:
            result = fn(**kwargs)
            outcome = {"return": result}
            if case.get("compose"):
                outcome["composed"] = {
                    "seconds": module.seconds(result, "all", 65, result["tier_at_warm"], False),
                    "threshold": module.threshold(result, "all", [65, 40], reply=16),
                    "graph_line": module.graph_line(result, "all"),
                    "graph_pass_max": module.graph_pass_max(result, "all", 512),
                    "check": module.check(result),
                }
        except (
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            IndexError,
            RuntimeError,
        ) as error:
            outcome = {"error": type(error).__name__, "message": str(error)}
    finally:
        sys.setprofile(previous)
    output_objects = {}
    object_paths(result, "result", output_objects)
    aliases = [
        [paths, input_objects[identity]]
        for identity, paths in output_objects.items()
        if identity in input_objects
    ]
    return {
        "outcome": exact(outcome),
        "state": exact(state),
        "aliases": aliases,
        "input_before_sha256": hashlib.sha256(json.dumps(before).encode()).hexdigest(),
        "input_after_sha256": hashlib.sha256(json.dumps(exact(args)).encode()).hexdigest(),
        "callback_count": len(callbacks),
        "callbacks_sha256": hashlib.sha256(json.dumps(callbacks).encode()).hexdigest(),
        "callbacks_first": callbacks[:6],
        "callbacks_last": callbacks[-6:],
        "mapping_read_count": len(reads),
        "mapping_reads_sha256": hashlib.sha256(json.dumps(reads).encode()).hexdigest(),
        "call_counts": dict(sorted(Counter(calls).items())),
        "call_order_sha256": hashlib.sha256(json.dumps(calls).encode()).hexdigest(),
    }


GOLDENS = json.loads((FIXTURES / "prefill-completion-goldens.json").read_text())


def test_frozen_oracle_identity():
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == GOLDENS["oracle_sha256"]
    assert GOLDENS["frozen_F"] == "8b12192920b339a4ef31c9006d1ebe20304ce45a"


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_prefill_completion(case):
    key = f"{sys.version_info.major}.{sys.version_info.minor}"
    if key not in GOLDENS["runtimes"]:
        pytest.skip("No frozen transcript recorded for this Python minor version")
    assert observe(ORACLE, case) == case["expected"][key]
