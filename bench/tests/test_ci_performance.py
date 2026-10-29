"""Observed performance policy; synthetic records are never hardware evidence."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import compare
from ci.contract import load, noise_bands, policy_read
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
                 "effective_settings": {"native_launch_policy": {"typed_values": {"prefetch.depth": 2}}}},
        "build": {"loaded_libraries": [{"path": "libsynthetic.so"}]},
        "server": {"pid": 1, "start_identity": "base"},
    }
    head_receipt = copy.deepcopy(receipt)
    head_receipt["server"] = {"pid": 2, "start_identity": "candidate"}
    return result, copy.deepcopy(result), receipt, head_receipt


@pytest.mark.parametrize("metric", compare.METRICS)
def test_single_pair_slowdown_is_marked_for_review_not_a_regression(pair, policy, metric):
    a, b, ar, br = pair
    b["metrics"][metric] *= 1.001 if metric == "ttft_s" else .999
    if metric == "server_prefill_tps":
        b["timings"]["prompt_ms"] = b["input_tokens"] * 1000 / b["metrics"][metric]
        b["metrics"][metric] = b["input_tokens"] * 1000 / b["timings"]["prompt_ms"]
    result = compare.compare_pair(a, b, ar, br, policy)
    assert result["status"] == "needs-review" and not result["passed"]
    assert result["observed_slowdowns"] == [metric]
    assert result["diagnostic_required"] and result["evidence_valid"]


def test_equal_samples_pass_only_observed_gate(pair, policy):
    result = compare.compare_pair(*pair, policy)
    assert result["passed"] and result["status"] == "passed"
    assert not result["diagnostic_required"]
    assert "decided from all of its pairs together" in result["interpretation"]


def test_pair_reports_signed_changes_and_exact_prompt_conditions(pair, policy):
    a, b, ar, br = pair
    b["metrics"]["decode_first128_tps"] = 12
    b["metrics"]["decode_second128_tps"] = 11
    b["metrics"]["ttft_s"] = .9
    result = compare.compare_pair(a, b, ar, br, policy)
    assert result["metrics"]["decode_first128_tps"]["change_percent"] == pytest.approx(20)
    assert result["metrics"]["decode_second128_tps"]["change_percent"] == pytest.approx(10)
    assert result["metrics"]["ttft_s"]["change_percent"] == pytest.approx(-10)
    assert result["metrics"]["ttft_s"]["direction"] == "lower-is-better"
    assert result["measurement_identity"]["prompt_sha256"] == a["prompt_sha256"]
    assert result["measurement_identity"]["output_tokens"] == 256
    assert len(result["measurement_identity"]["conditions_sha256"]) == 64


def test_faster_different_output_needs_review_and_retains_metrics(pair, policy):
    a, b, ar, br = pair
    b["tokens"][64] = 12
    for metric in compare.METRICS:
        b["metrics"][metric] *= .5 if metric == "ttft_s" else 2
    b["timings"]["prompt_ms"] /= 2
    result = compare.compare_pair(a, b, ar, br, policy)
    assert not result["passed"] and result["status"] == "needs-review"
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


def band(**changes):
    value = dict(sigma_log={metric: 0.01 for metric in compare.METRICS}, pairs=8,
                 statistic="rms-paired-log-ratio", conditions_sha256="c" * 64, engine_revision="a" * 40,
                 calibration_jobs=["f" * 32], date="2026-10-29")
    value.update(changes)
    return value


def test_shipped_policy_holds_gate_constants_but_no_bands(policy):
    thresholds = policy["thresholds"]
    assert thresholds["mode"] == "calibrated-noise-band" and "bands" not in thresholds
    assert thresholds["pairs"] == 2 and thresholds["order"] == "ABBA" and "k" not in thresholds
    assert thresholds["floor_fraction"] == {metric: 0.01 for metric in compare.METRICS}


@pytest.mark.parametrize("damage", ["old-mode", "extra-key", "pairs", "bool-pairs", "order", "fixed-k",
                                    "floor-negative", "floor-one", "floor-missing", "bands-in-policy"])
def test_policy_thresholds_are_explicit_and_finite(policy, damage):
    thresholds = policy["thresholds"]
    if damage == "old-mode": thresholds["mode"] = "observed-no-slowdown"
    elif damage == "extra-key": thresholds["decode_regression_fraction"] = 0
    elif damage == "pairs": thresholds["pairs"] = 1
    elif damage == "bool-pairs": thresholds["pairs"] = True
    elif damage == "order": thresholds["order"] = "ABAB"
    elif damage == "fixed-k": thresholds["k"] = 3  # The multiplier is the fixed t quantile, not policy.
    elif damage == "floor-negative": thresholds["floor_fraction"]["ttft_s"] = -0.01
    elif damage == "floor-one": thresholds["floor_fraction"]["decode_tps"] = 1
    elif damage == "floor-missing": del thresholds["floor_fraction"]["server_prefill_tps"]
    else: thresholds["bands"] = {}  # Bands are operator data sealed per job, never policy.
    with pytest.raises(ValueError):
        policy_read(policy)


@pytest.mark.parametrize("damage", [None, "sigma-missing", "sigma-negative", "sigma-inf", "few-pairs", "statistic",
                                    "conditions", "revision", "no-jobs", "bad-job", "duplicate-job", "date",
                                    "impossible-date", "extra"])
def test_noise_band_entries_are_complete_and_well_formed(damage):
    value = band()
    if damage == "sigma-missing": del value["sigma_log"]["ttft_s"]
    elif damage == "sigma-negative": value["sigma_log"]["ttft_s"] = -1
    elif damage == "sigma-inf": value["sigma_log"]["ttft_s"] = float("inf")
    elif damage == "few-pairs": value["pairs"] = 7  # Two default calibration runs give 8.
    elif damage == "statistic": value["statistic"] = "median"
    elif damage == "conditions": value["conditions_sha256"] = "C" * 64
    elif damage == "revision": value["engine_revision"] = "a" * 39
    elif damage == "no-jobs": value["calibration_jobs"] = []
    elif damage == "bad-job": value["calibration_jobs"] = ["f" * 31]
    elif damage == "duplicate-job": value["calibration_jobs"] = ["f" * 32] * 2
    elif damage == "date": value["date"] = "29/10/2026"
    elif damage == "impossible-date": value["date"] = "2026-02-30"
    elif damage == "extra": value["k"] = 3
    bands = {"cloud-5090": {"muse": {"512": value}}}
    if damage is None:
        noise_bands(bands)
    else:
        with pytest.raises(ValueError):
            noise_bands(bands)


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
    assert report["status"] == "needs-review" and not report["passed"]
    row = next(row for row in report["devices"][0]["rows"] if row["model"] == "muse")
    assert row["evidence_valid"] and row["observed_slowdowns"] == ["server_prefill_tps"]


def test_legacy_single_pair_paths_describe_what_they_actually_decide(synthetic):
    from ci.private_controller import check_body
    roots, plan, policy = synthetic
    report = compare.aggregate(roots, plan, policy)
    assert report["interpretation"] == compare.LEGACY_INTERPRETATION
    assert all(device["interpretation"] == compare.LEGACY_INTERPRETATION for device in report["devices"])
    controller = dict(repository="wici-ai/inferssd-device-ci", run_id=1, run_attempt=1)
    summary = check_body(dict(controller=controller, head="b" * 40), report, 88)["output"]["summary"]
    for text in (compare.LEGACY_INTERPRETATION, summary):
        assert "one base/candidate pair" in text.lower() and "noise" not in text and "band" not in text
