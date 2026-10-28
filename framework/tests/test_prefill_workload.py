from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from inferssd.model import prefill_workload

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE_PATH = FIXTURES / "oracle/prefill_reference.py"
SPEC = importlib.util.spec_from_file_location("prefill_workload_reference", ORACLE_PATH)
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
    """Call the reference functions directly; keep generator consumption visible."""
    args = dict(case["arguments"])
    if case.get("iterator"):
        key = "ps" if case["function"] == "cold_share" else "turns"
        args[key] = iter(args[key])
    if case["function"] == "workload":
        ps = module.offloaded(**args)
        result = {
            "served": module.served(args["turns"], args.get("reply", 0)),
            "offloaded": ps,
            "cold_share": module.cold_share(ps),
        }
    else:
        result = getattr(module, case["function"])(**args)
    return exact(result)


GOLDENS = json.loads((FIXTURES / "prefill-workload-goldens.json").read_text())


def test_frozen_oracle_identity():
    manifest = json.loads((FIXTURES / "oracle/prefill-manifest.json").read_text())
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["revision"] == "8b12192920b339a4ef31c9006d1ebe20304ce45a"


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_prefill_workload(case):
    assert observe(ORACLE, case) == case["expected"]


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_typed_prefill_workload(case):
    assert observe(prefill_workload, case) == case["expected"]
