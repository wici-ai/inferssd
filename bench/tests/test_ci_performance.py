"""Observed performance policy; synthetic records are never hardware evidence."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import compare
from ci.contract import load, policy_read
import test_device_ci

policy = test_device_ci.policy
synthetic = test_device_ci.synthetic


@pytest.fixture
def pair():
    result = {
        "input_tokens": 100, "prompt_ids": [7] * 100, "prompt_sha256": "a" * 64,
        "output_tokens": 256, "tokens": [11] * 256,
        "timings": {"prompt_n": 100, "prompt_ms": 1000},
        "metrics": {"decode_first128_tps": 10, "decode_second128_tps": 10,
                    "decode_tps": 10, "ttft_s": 1, "server_prefill_tps": 100},
    }
    receipt = {
        "model_identity": {"shards": [{"sha256": "b" * 64}]},
        "host": {"hardware": "synthetic", "os": "synthetic", "memory_budget_bytes": 1024,
                 "os_page_cache_treatment": "uncontrolled",
                 "effective_settings": {"native_launch_policy": {"typed_values": {"cpu.threads": 2}}}},
        "build": {"loaded_libraries": [{"path": "libsynthetic.so"}]},
        "server": {"pid": 1, "start_identity": "base"},
    }
    head_receipt = copy.deepcopy(receipt)
    head_receipt["server"] = {"pid": 2, "start_identity": "candidate"}
    return result, copy.deepcopy(result), receipt, head_receipt


@pytest.mark.parametrize("metric", compare.METRICS)
def test_even_small_slowdown_is_inconclusive_not_permitted_regression(pair, policy, metric):
    a, b, ar, br = pair
    b["metrics"][metric] *= 1.001 if metric == "ttft_s" else .999
    if metric == "server_prefill_tps":
        b["timings"]["prompt_ms"] = b["input_tokens"] * 1000 / b["metrics"][metric]
        b["metrics"][metric] = b["input_tokens"] * 1000 / b["timings"]["prompt_ms"]
    result = compare.compare_pair(a, b, ar, br, policy)
    assert result["status"] == "inconclusive" and not result["passed"]
    assert result["observed_slowdowns"] == [metric]
    assert result["diagnostic_required"] and result["evidence_valid"]


def test_equal_samples_pass_only_observed_gate(pair, policy):
    result = compare.compare_pair(*pair, policy)
    assert result["passed"] and result["status"] == "passed"
    assert not result["diagnostic_required"]
    assert "not statistical proof" in result["interpretation"]


def test_faster_different_output_is_inconclusive_and_retains_metrics(pair, policy):
    a, b, ar, br = pair
    b["tokens"][64] = 12
    for metric in compare.METRICS:
        b["metrics"][metric] *= .5 if metric == "ttft_s" else 2
    b["timings"]["prompt_ms"] /= 2
    result = compare.compare_pair(a, b, ar, br, policy)
    assert not result["passed"] and result["status"] == "inconclusive"
    assert result["diagnostic_required"] and result["evidence_valid"]
    assert result["observed_slowdowns"] == []
    assert all(value["passed"] for value in result["metrics"].values())
    assert result["metrics"]["decode_tps"]["candidate"] == 20
    assert result["output_workload"] == {
        "same_tokens": False, "first_difference_one_based": 65,
        "base_token": 11, "candidate_token": 12,
    }


def test_same_tokens_with_different_probability_floats_can_compare(pair, policy):
    pair[0]["probabilities"] = [{"id": 11, "logprob": -.1}]
    pair[1]["probabilities"] = [{"id": 11, "logprob": -.10001}]
    result = compare.compare_pair(*pair, policy)
    assert result["passed"] and result["output_workload"]["same_tokens"]


@pytest.mark.parametrize("tokens", [None, [], [11] * 255, [True] * 256, [-1] * 256])
def test_missing_or_invalid_output_tokens_are_invalid_evidence(pair, policy, tokens):
    if tokens is None:
        pair[1].pop("tokens")
    else:
        pair[1]["tokens"] = tokens
    with pytest.raises(ValueError, match="output token evidence"):
        compare.compare_pair(*pair, policy)


@pytest.mark.parametrize("value", [None, 0, -1, True, float("nan"), float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_native_prefill_cannot_disappear_from_gate(pair, policy, value, side):
    pair[side]["metrics"]["server_prefill_tps"] = value
    with pytest.raises(ValueError, match="metric"):
        compare.compare_pair(*pair, policy)


def test_prefill_cannot_be_replaced_with_client_ttft_rate(pair, policy):
    pair[1]["metrics"]["server_prefill_tps"] = 101
    with pytest.raises(ValueError, match="native prompt"):
        compare.compare_pair(*pair, policy)


def test_metric_ratio_overflow_is_invalid_not_a_speedup(pair, policy):
    pair[0]["metrics"]["decode_tps"] = 1e-320
    with pytest.raises(ValueError, match="ratio"):
        compare.compare_pair(*pair, policy)


@pytest.mark.parametrize("key", ["decode_regression_fraction", "ttft_regression_fraction", "prefill_regression_fraction"])
def test_policy_cannot_reintroduce_a_positive_allowed_regression(policy, key):
    policy["thresholds"][key] = .01
    with pytest.raises(ValueError, match="no permitted"):
        policy_read(policy)


@pytest.mark.parametrize("tamper", ["run-id", "ledger", "tier-order", "native-time", "interrupted"])
def test_current_complete_attempt_required(synthetic, tamper):
    roots, plan, policy = synthetic
    folder = roots[0] / "muse/candidate"
    if tamper == "ledger":
        (folder / "attempts.jsonl").write_text("")
    elif tamper == "native-time":
        path = folder / "512/result.json"
        result = load(path)
        result["timings"]["prompt_ms"] *= 2
        path.write_text(json.dumps(result))
    else:
        path = folder / "summary.json"
        summary = load(path)
        if tamper == "run-id":
            summary["run_id"] = "another-attempt"
        elif tamper == "interrupted":
            summary["interrupted_signals"] = [15]
        else:
            summary["order"] = ["512", "512"]
        path.write_text(json.dumps(summary))
    report = compare.aggregate(roots, plan, policy)
    assert report["status"] == "failed" and not report["passed"]
    row = next(row for row in report["devices"][0]["rows"] if row["model"] == "muse")
    assert row["status"] == "failed" and not row["evidence_valid"]


def test_observed_native_prefill_slowdown_blocks_complete_three_device_report(synthetic):
    roots, plan, policy = synthetic
    folder = roots[0] / "muse/candidate"
    arm = folder / "512"
    records = [json.loads(line) for line in (arm / "result.events.jsonl").read_text().splitlines()]
    records[-1]["event"]["timings"]["prompt_ms"] *= 1.001
    (arm / "result.events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (arm / "result.events.sse").write_text("".join("data: " + json.dumps(row["event"]) + "\n\n" for row in records))
    result = load(arm / "result.json")
    rebuilt = compare.run_final.stream_chat.summarize(
        records, result["start_ns"], result["end_ns"], result["prompt_ids"], 256,
    )
    compare.run_final.final_chat.validate_timings(rebuilt)
    result.update(rebuilt)
    (arm / "result.json").write_text(json.dumps(result))
    summary, manifest = load(folder / "summary.json"), load(folder / "manifest.json")
    summary["rows"][0]["result"] = result
    test_device_ci.measurement_manifest(folder, manifest["config"], summary["rows"])
    report = compare.aggregate(roots, plan, policy)
    assert report["status"] == "inconclusive" and not report["passed"]
    row = next(row for row in report["devices"][0]["rows"] if row["model"] == "muse")
    assert row["evidence_valid"] and row["observed_slowdowns"] == ["server_prefill_tps"]
