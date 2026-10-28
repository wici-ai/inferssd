from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from inferssd.model import prefill_model

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE_PATH = FIXTURES / "oracle/prefill_f.py"
SPEC = importlib.util.spec_from_file_location("prefill_model_f_oracle", ORACLE_PATH)
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
    if isinstance(value, set):
        return {"set": sorted((exact(v) for v in value), key=lambda x: json.dumps(x))}
    return value


class ObservedDict(dict):
    def __init__(self, data, path, events):
        super().__init__(data)
        self.path, self.events = path, events

    def __getitem__(self, key):
        self.events.append(["getitem", self.path, key])
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.events.append(["get", self.path, key, exact(default)])
        return super().get(key, default)


def watched(value, path, events):
    if isinstance(value, dict):
        return ObservedDict(
            {k: watched(v, f"{path}.{k}", events) for k, v in value.items()}, path, events
        )
    if isinstance(value, list):
        return [watched(v, f"{path}[{i}]", events) for i, v in enumerate(value)]
    return value


FIELDS = {
    "coverage": ("cap", "E", "k", "i", "u"),
    "pass_seconds": ("b", "cov", "read", "U", "routes", "us", "cpu_us", "cpu"),
    "seconds": ("tot", "cpu", "rb", "q", "graph"),
    "break_even": ("states", "lo", "hi", "slope", "R", "a"),
    "threshold": ("ps", "t", "cand", "forms", "time", "best"),
}


def observe(module, case):
    """Observe the original call graph, mapping access order and failure point."""
    args = copy.deepcopy(case["arguments"])
    reads = []
    if "m" in args:
        args["m"] = watched(args["m"], "m", reads)
    if case.get("iterator"):
        args["turns"] = iter(args["turns"])
    fn = getattr(module, case["function"])
    names = [*FIELDS, "cut", "served", "offloaded", "cold_share", "lane_layer"]
    functions = {getattr(module, name).__code__: name for name in names}
    functions[module.lane_layer.__globals__["coupled"].__code__] = "coupled"
    calls, state = [], {}

    def capture(frame, event, arg):
        if event == "call" and frame.f_code in functions:
            calls.append(functions[frame.f_code])
        if event == "return" and frame.f_code is fn.__code__:
            state.update(
                {
                    key: frame.f_locals[key]
                    for key in FIELDS[case["function"]]
                    if key in frame.f_locals
                }
            )

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        try:
            outcome = {"return": fn(**args)}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            outcome = {"error": type(error).__name__, "message": str(error)}
    finally:
        sys.setprofile(previous)
    return {
        "outcome": exact(outcome),
        "state": exact(state),
        "mapping_read_count": len(reads),
        "mapping_reads_sha256": hashlib.sha256(json.dumps(reads).encode()).hexdigest(),
        "call_counts": dict(sorted(Counter(calls).items())),
        "call_order_sha256": hashlib.sha256(json.dumps(calls).encode()).hexdigest(),
    }


GOLDENS = json.loads((FIXTURES / "prefill-model-goldens.json").read_text())
OLDER_GOLDENS = json.loads((FIXTURES / "prefill-model-python39.json").read_text())


def frozen_expected(case):
    # Select only measured runtimes; future arithmetic is not inferred from a range.
    if sys.version_info[:2] == (3, 9):
        return OLDER_GOLDENS["overrides"].get(case["name"], case["expected"])
    if sys.version_info[:2] == (3, 12):
        return case["expected"]
    pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")


def test_frozen_oracle_identity():
    manifest = json.loads((FIXTURES / "oracle/prefill-manifest.json").read_text())
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["revision"] == "8b12192920b339a4ef31c9006d1ebe20304ce45a"
    assert OLDER_GOLDENS["oracle_sha256"] == GOLDENS["oracle_sha256"]
    assert (
        OLDER_GOLDENS["base_goldens_sha256"]
        == hashlib.sha256((FIXTURES / "prefill-model-goldens.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_prefill_model(case):
    assert observe(ORACLE, case) == frozen_expected(case)


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_typed_prefill_model(case):
    assert observe(prefill_model, case) == observe(ORACLE, case)
