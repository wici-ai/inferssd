from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inferssd.model.decode_cost import LayerResponse, fit_response, price_decode

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = json.loads((FIXTURES / "decode-cost-goldens.json").read_text())
ORACLE_PATH = FIXTURES / "oracle/sim_f.py"
SPEC = importlib.util.spec_from_file_location("decode_cost_f_oracle", ORACLE_PATH)
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


def check_result(call, expected):
    if "error" in expected:
        errors = {
            "IndexError": IndexError,
            "TypeError": TypeError,
            "ZeroDivisionError": ZeroDivisionError,
        }
        with pytest.raises(errors[expected["error"]]):
            call()
    else:
        assert exact(call()) == expected["exact"]


def test_oracle_identity():
    manifest = json.loads((FIXTURES / "oracle/sim-manifest.json").read_text())
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["sha256"] == GOLDENS["oracle_sha256"]


@pytest.mark.parametrize("case", GOLDENS["price_cases"], ids=lambda case: case["name"])
def test_frozen_f_price(case):
    check_result(
        lambda: ORACLE.price(case["rows"], case["costs"], case["cpu_scale"]), case["legacy"]
    )


@pytest.mark.parametrize("case", GOLDENS["price_cases"], ids=lambda case: case["name"])
def test_typed_price_exact(case):
    costs = LayerResponse.from_legacy(case["costs"])
    rows = tuple(tuple(tuple(pair) for pair in row) for row in case["rows"])
    check_result(lambda: price_decode(rows, costs, case["cpu_scale"]).to_mapping(), case["legacy"])


@pytest.mark.parametrize("case", GOLDENS["fit_cases"], ids=lambda case: case["name"])
def test_frozen_f_fit(case):
    check_result(lambda: ORACLE.lstsq(case["x"], case["y"]), case["legacy"])


@pytest.mark.parametrize("case", GOLDENS["fit_cases"], ids=lambda case: case["name"])
def test_typed_fit_exact(case):
    check_result(lambda: fit_response(case["x"], case["y"]), case["legacy"])


def test_response_deep_freeze_and_optional_states():
    by_name = {
        case["name"]: LayerResponse.from_legacy(case["costs"]) for case in GOLDENS["price_cases"]
    }
    absent = by_name["missing_ready_counts"]
    assert absent.ready_given_nx is None
    assert by_name["empty_ready_counts"].ready_given_nx == ()
    assert not absent.ready_share_all_present
    assert by_name["capped_unused_null_share"].ready_share_all_present
    assert by_name["capped_unused_null_share"].ready_share_all is None
    assert by_name["uncapped_zero_share"].ready_share_all == 0.0
    with pytest.raises(FrozenInstanceError):
        absent.pf_k = 5
    with pytest.raises(TypeError):
        absent.layer_table[0][1][0] = 2
    with pytest.raises(TypeError):
        by_name["twenty_samples_empirical"].ready_given_nx[0][1][0] = 2


def test_pure_costs_do_not_consult_host_or_mutate_inputs(monkeypatch):
    case = GOLDENS["price_cases"][0]
    original = json.dumps(case)
    response = LayerResponse.from_legacy(case["costs"])

    def denied(*args, **kwargs):
        raise AssertionError("cost model consulted a host dependency")

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.setattr("builtins.open", denied)
    monkeypatch.setattr("os.getenv", denied)
    monkeypatch.setattr("os.cpu_count", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    result = price_decode(case["rows"], response, case["cpu_scale"])
    assert exact(result.to_mapping()) == case["legacy"]["exact"]
    assert fit_response([[1, 0], [1, 1], [1, 2]], [1, 3, 5]) == (1.0, 2.0)
    assert json.dumps(case) == original


def test_oracle_distinguishes_empirical_threshold_and_column_order():
    prices = {case["name"]: case["legacy"] for case in GOLDENS["price_cases"]}
    fits = {case["name"]: case["legacy"] for case in GOLDENS["fit_cases"]}
    assert (
        prices["nineteen_samples_fallback"]["exact"] != prices["twenty_samples_empirical"]["exact"]
    )
    assert prices["fifth_sample_measured"]["exact"] != prices["missing_ready_counts"]["exact"]
    assert fits["dependent_columns"]["exact"] != fits["dependent_reordered"]["exact"]
    assert fits["near_dependent_below"]["exact"] != fits["near_dependent_above"]["exact"]
    assert all(math.isfinite(value) for value in fits["near_dependent_above"]["value"])
