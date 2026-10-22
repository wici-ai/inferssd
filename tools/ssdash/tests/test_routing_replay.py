from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from ssdash.model import routing_replay
from ssdash.model.decode_cost import LayerResponse, price_decode

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = json.loads((FIXTURES / "routing-replay-goldens.json").read_text())
ORACLE_PATH = FIXTURES / "oracle/sim_f.py"
SPEC = importlib.util.spec_from_file_location("routing_replay_f_oracle", ORACLE_PATH)
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


def tier_state(tier):
    return {
        "cap": tier.cap,
        "minimum": tier.min,
        "k": tier.k,
        "ls": list(tier.ls.items()),
        "counts": [(key, tier.count(key)) for key in tier.ls],
        "residents": sorted(tier.res),
        "heap": tier.heap,
        "step": tier.step,
        "admissions": tier.admissions,
    }


def state(obj, kind):
    if kind == "tier":
        return tier_state(obj)
    if kind == "ram":
        return {
            "slots": obj.slots,
            "ls": list(obj.ls.items()),
            "residents": list(obj.res.items()),
            "clock": obj.clock,
        }
    return {
        "tier": tier_state(obj.tier),
        "skip": obj.skip,
        "rows": obj.rows,
        "segment_rows": obj.seg_rows,
        "done": obj.done,
        "uploads": obj.uploads,
        "v": obj.v,
        "n_exp": obj.n_exp,
        "behind": None if obj.behind is None else list(obj.behind.items()),
        "steps": obj.steps,
        "behind_cache": list(obj._behind_hits.items()),
        "stats": obj.stats(),
    }


def transcript(module, case):
    """Observe public results and mutable state; never implement cache decisions."""
    kind = case["kind"]
    if kind == "simulate":
        return exact(module.simulate(case["routing"], **case["parameters"]))
    cls = {"tier": module.Tier, "ram": module.RamCache, "replay": module.Replay}[kind]
    try:
        obj = cls(**case["parameters"])
    except (ZeroDivisionError, ValueError) as error:
        return {"constructor_error": type(error).__name__}
    rows = [{"initial": exact(state(obj, kind))}]
    for operation in case["operations"]:
        try:
            if kind == "tier":
                obj.step = operation["step"]
                result = obj.access(operation["layer"], operation["experts"])
            elif kind == "ram":
                result = obj.access(operation["step"], operation["experts"])
            elif operation["action"] == "advance":
                result = "self" if obj.advance(operation["routing"]) is obj else "wrong object"
            else:
                result = obj.behind_hits(operation["slots"])
            row = {"result": exact(result)}
        except (ValueError, AttributeError) as error:
            row = {"error": type(error).__name__, "message": str(error)}
        row["state"] = exact(state(obj, kind))
        rows.append(row)
    return rows


def test_replay_oracle_identity():
    manifest = json.loads((FIXTURES / "oracle/sim-manifest.json").read_text())
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert GOLDENS["oracle_sha256"] == manifest["sha256"]
    assert GOLDENS["source_revision"] == "8b12192920b339a4ef31c9006d1ebe20304ce45a"


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_frozen_f_state_transcript(case):
    assert transcript(ORACLE, case) == case["legacy"]


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda case: case["name"])
def test_typed_replay_state_exact(case):
    assert transcript(routing_replay, case) == case["legacy"]


def test_incremental_rows_match_simulate_and_cost_leaf():
    case = next(
        case for case in GOLDENS["cases"] if case["name"] == "simulate_same_segment_routing"
    )
    routing, parameters = case["routing"], case["parameters"]
    replay = routing_replay.Replay(**parameters, behind=True)
    for count in range(1, len(routing) + 1):
        replay.advance(routing[:count])
        replay.behind_hits(1)
    assert exact((replay.rows, replay.stats())) == case["legacy"]
    costs = json.loads((FIXTURES / "decode-cost-goldens.json").read_text())["price_cases"][0][
        "costs"
    ]
    assert exact(
        price_decode(replay.rows, LayerResponse.from_legacy(costs), 1.0).to_mapping()
    ) == exact(ORACLE.price(replay.rows, costs, 1.0))


def test_state_goldens_include_semantic_boundaries():
    by_name = {case["name"]: case for case in GOLDENS["cases"]}
    zero = by_name["tier_zero_capacity_still_counts_duplicates"]["legacy"]
    assert zero[-1]["state"]["admissions"] == 0 and zero[-1]["state"]["ls"]
    threshold = by_name["tier_minimum_threshold_and_decay"]["legacy"]
    assert threshold[2]["state"]["admissions"] == 0
    assert threshold[3]["state"]["admissions"] == 2
    duplicates = by_name["ram_locked_hits_and_duplicate_clock"]["legacy"]
    assert duplicates[2]["result"] == 2  # Duplicate resident IDs count twice.
    assert duplicates[2]["state"]["clock"] == 5
    replay = by_name["replay_length_only_prefix_and_rejected_shrink"]["legacy"]
    assert replay[3]["state"] == replay[2]["state"]  # Changed contents, same lengths.
    assert replay[4]["error"] == replay[5]["error"] == "ValueError"
    assert replay[6]["state"]["behind_cache"] == []


def test_routing_has_no_host_dependency_or_input_mutation(monkeypatch):
    case = next(case for case in GOLDENS["cases"] if case["kind"] == "simulate")
    original = json.dumps(case)

    def denied(*args, **kwargs):
        raise AssertionError("routing replay consulted a host dependency")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr("builtins.open", denied)
    monkeypatch.setattr("os.getenv", denied)
    monkeypatch.setattr("os.cpu_count", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    assert transcript(routing_replay, case) == case["legacy"]
    assert json.dumps(case) == original
