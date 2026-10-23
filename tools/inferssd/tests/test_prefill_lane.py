from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from inferssd.model import prefill_lane

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE_PATH = FIXTURES / "oracle/prefill_f.py"
SPEC = importlib.util.spec_from_file_location("prefill_lane_f_oracle", ORACLE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def exact(value):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(item) for item in value]
    return value


def observe(module, case):
    """Capture F's own return state, without calculating its timing or policy."""
    fn = getattr(module, case["function"])
    fields = (
        ("high", "low", "n", "r")
        if case["function"] == "coupled"
        else (
            "N",
            "R",
            "c",
            "g_dev",
            "g_host",
            "measured_host",
            "copy_us",
            "chosen",
            "priced",
            "d",
            "rows",
            "drive_us_per_mb",
            "drive",
            "fill",
            "overhead",
            "times",
            "cpus",
            "copied",
            "gpu",
            "lowest",
            "m",
        )
    )
    state = {}

    def capture(frame, event, arg):
        if event == "return" and frame.f_code is fn.__code__:
            state.update({key: frame.f_locals[key] for key in fields if key in frame.f_locals})

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        result = fn(**case["arguments"])
    finally:
        sys.setprofile(previous)
    return {"return": exact(result), "state": exact(state)}


GOLDENS = json.loads((FIXTURES / "prefill-lane-goldens.json").read_text())


def test_frozen_oracle_identity():
    manifest = json.loads((FIXTURES / "oracle/prefill-manifest.json").read_text())
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["revision"] == "8b12192920b339a4ef31c9006d1ebe20304ce45a"


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_prefill_lane(case):
    assert observe(ORACLE, case) == case["expected"]


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_typed_prefill_lane(case):
    assert observe(prefill_lane, case) == case["expected"]
