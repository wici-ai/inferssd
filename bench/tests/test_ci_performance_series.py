# SPDX-License-Identifier: Apache-2.0
"""PR runs against baseline history, baseline jobs and the history tool; synthetic data only."""

import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import compare, history, impact
from ci.contract import load

POLICY = load(Path(__file__).resolve().parents[1] / "ci/policy.json")
BASE, CANDIDATE = "a" * 40, "c" * 40
CELL = dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32, base=BASE, candidate=CANDIDATE)
GATE_ORDER = ["candidate", "base", "candidate", "candidate"]
BASELINE_ORDER = ["candidate"] * 4
TOKENS = [11] * 256
SPREAD = (0.99, 1.01, 0.995, 1.005)  # run-to-run spread of a baseline job, as factors

RECEIPT = {
    "model_identity": {"shards": [{"sha256": "b" * 64}]},
    "host": {"hardware": "synthetic", "os": "synthetic", "memory_budget_bytes": 1024,
             "os_page_cache_treatment": "uncontrolled",
             "effective_settings": {"native_launch_policy": {"typed_values": {"prefetch.depth": 2},
                                                             "template_sha256": "c" * 64},
                                    "native_resource_policy": {"cpus_allowed_list": "0-3", "memlock_bytes": 1024,
                                                               "cgroup_limits": {"memory_max": "1024"}}}},
    "build": {"loaded_libraries": [{"path": "libsynthetic.so"}]},
}
CONDITIONS = compare.conditions_digest(compare.history_conditions(RECEIPT))
# A characteristics file as a baseline job's profile stage printed it.
MACHINE = json.dumps(dict(schema="synthetic", sections=dict(cpu=dict(status="ok"), gpus=dict(status="failed"))))
MACHINE_SHA = compare.hashlib.sha256(MACHINE.encode()).hexdigest()
BASE_METRICS = dict(decode_first128_tps=10.0, decode_second128_tps=10.0, decode_tps=10.0, ttft_s=1.0,
                    server_prefill_tps=100.0)
BUILD = dict(freeze_manifest_sha256="1" * 64, binary_sha256="2" * 64, cache_key="3" * 64,
             build_config_sha256="4" * 64, toolchain={"CMAKE_CXX_COMPILER": "/usr/bin/g++-12"})


def record(commit=BASE, spread=SPREAD, scale=1.0, conditions=CONDITIONS, device="cloud-5090", model="muse",
           tier="512", measured_at="2026-10-28T10:00:00Z", job="e" * 32, tokens=TOKENS, base=BASE_METRICS,
           characteristics=MACHINE_SHA):
    """One verified baseline record: four runs of `commit`, each metric scaled by a spread factor."""
    runs = []
    for factor in spread:
        metrics = {metric: value * factor * scale for metric, value in base.items()}
        runs.append(dict(metrics=metrics, logs={metric: compare.ln(value) for metric, value in metrics.items()},
                         tokens_sha256=compare.tokens_digest(list(tokens))))
    identity = dict(prompt_sha256="a" * 64, prompt_ids_sha256="d" * 64, input_tokens=100, output_tokens=256,
                    conditions_sha256=conditions)
    return dict(schema_version=1, purpose="baseline-history", device=device, model=model, tier=tier, commit=commit,
                conditions_sha256=conditions, characteristics_sha256=characteristics, runs=runs,
                measurement_identity=identity,
                build=dict(BUILD), job_id=job, run_id=7, run_attempt=1, measured_at=measured_at)


def sealed(cells, commit=BASE, base=BASE_METRICS):
    """Sealed history with one base record per {(device, model, tier): conditions_sha256}."""
    return [record(commit, conditions=conditions, device=device, model=model, tier=tier, base=base)
            for (device, model, tier), conditions in sorted(cells.items())]


def result(values=None, tokens=None, start=0):
    metrics = dict(BASE_METRICS)
    metrics.update(values or {})
    prompt_ms = 100000 / metrics["server_prefill_tps"]
    metrics["server_prefill_tps"] = 100 * 1000 / prompt_ms
    return {"input_tokens": 100, "prompt_ids": [7] * 100, "prompt_sha256": "a" * 64, "output_tokens": 256,
            "tokens": list(tokens or TOKENS), "timings": {"prompt_n": 100, "prompt_ms": prompt_ms},
            "metrics": metrics, "start_ns": start, "end_ns": start + 10}


def protocol_for(purpose="regression-gate"):
    return dict(tier="512", prompt_sha256="0" * 64, output_tokens=256, purpose=purpose,
                order=list(BASELINE_ORDER if purpose == "baseline" else GATE_ORDER))


def series(changes=None, purpose="regression-gate", token_change=None):
    """All timed runs of one cell; `changes` maps (run, side) to metric overrides."""
    results, receipts = {}, {}
    for position, key in enumerate(compare.schedule(protocol_for(purpose)["order"])):
        tokens = list(TOKENS)
        if token_change is not None and key == token_change:
            tokens[100] = 12
        results[key] = result((changes or {}).get(key), tokens, start=position * 100)
        receipts[key] = dict(copy.deepcopy(RECEIPT), server=dict(pid=position + 1, start_identity=str(key)))
    return results, receipts


def write_arm(folder, tier, config, offset):
    """Real final-protocol artifacts for one timed arm, on a shared job clock."""
    import final_chat
    import stream_chat
    from test_final import write_artifacts
    write_artifacts(folder, tier, config)
    events = folder / "result.events.jsonl"
    records = [json.loads(line) for line in events.read_text().splitlines()]
    for row in records:
        row["monotonic_ns"] += offset
    events.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = json.loads((folder / "result.json").read_text())
    rebuilt = stream_chat.summarize(records, result["start_ns"] + offset, result["end_ns"] + offset,
                                    result["prompt_ids"], 256)
    final_chat.validate_timings(rebuilt)
    result.update(json.loads(json.dumps(rebuilt)))
    (folder / "result.json").write_text(json.dumps(result))


def shift_arm(folder, delta):
    """Move a retained arm on the job clock, keeping its raw stream and summary consistent."""
    import final_chat
    import stream_chat
    events = folder / "result.events.jsonl"
    records = [json.loads(line) for line in events.read_text().splitlines()]
    for row in records:
        row["monotonic_ns"] += delta
    events.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = json.loads((folder / "result.json").read_text())
    rebuilt = stream_chat.summarize(records, result["start_ns"] + delta, result["end_ns"] + delta,
                                    result["prompt_ids"], 256)
    final_chat.validate_timings(rebuilt)
    result.update(json.loads(json.dumps(rebuilt)))
    (folder / "result.json").write_text(json.dumps(result))
    arm = json.loads((folder / "benchmark.json").read_text())
    arm["result"] = result
    (folder / "benchmark.json").write_text(json.dumps(arm))


def decide(changes=None, history_records=None, purpose="regression-gate", cell=CELL, token_change=None,
           environment=None, policy=None):
    results, receipts = series(changes, purpose, token_change)
    return compare.decide_cell(results, receipts, policy or copy.deepcopy(POLICY), cell=cell,
                               protocol=protocol_for(purpose),
                               history=[record()] if history_records is None else history_records,
                               environment=environment)


def scaled(fraction, metric="decode_tps"):
    """Metric override that makes a run `fraction` slower (negative: faster)."""
    value = BASE_METRICS[metric]
    return {metric: value * (1 + fraction) if metric == "ttft_s" else value / (1 + fraction)}


def candidates(override):
    return {(k, "candidate"): override for k in (1, 2, 3)}


def sigma(spread=SPREAD):
    logs = [math.log(factor) for factor in spread]
    centre = sum(logs) / len(logs)
    return math.sqrt(sum((value - centre) ** 2 for value in logs) / (len(logs) - 1))


def test_run_order_is_candidate_base_candidate_candidate_and_baseline_four_runs():
    assert compare.schedule(GATE_ORDER) == [(1, "candidate"), (1, "base"), (2, "candidate"), (3, "candidate")]
    assert compare.schedule(BASELINE_ORDER) == [(k, "candidate") for k in (1, 2, 3, 4)]
    for bad in ([], ["base", "other"], "ABBA", None):
        with pytest.raises(ValueError):
            compare.schedule(bad)


def test_outcome_orders_failed_regressed_review_baseline_passed():
    order = ["failed", "regressed", "needs-review", "baseline", "passed"]
    for index, status in enumerate(order):
        assert compare.outcome([dict(status=value) for value in order[index:]]) == status
    assert compare.outcome([]) == "passed"
    with pytest.raises(ValueError):
        compare.outcome([dict(status="inconclusive")])


@pytest.mark.parametrize("df,t", [(3, 9.219), (4, 6.621), (5, 5.508), (6, 4.905), (12, 3.765), (16, 3.545),
                                  (17, 3.545), (45, 3.199), (64, 3.122), (70, 3.122), (128, 3.060), (129, 3.0),
                                  (1000, 3.0)])
def test_t_lookup_uses_the_nearest_lower_df_down_to_three(df, t):
    assert compare.t_quantile(df) == t
    for low in (2, 0, True, None, 3.0):
        with pytest.raises(ValueError):
            compare.t_quantile(low)


def test_pooled_variance_is_within_commit_and_df_is_runs_minus_one_per_record():
    first, second = record(), record("b" * 40, spread=(0.98, 1.02, 1.0, 1.0), scale=1.3)
    value, df = compare.pooled([first, second], "decode_tps")
    logs = [[run["logs"]["decode_tps"] for run in item["runs"]] for item in (first, second)]
    squares = [(x - sum(row) / 4) ** 2 for row in logs for x in row]
    assert df == 6 and value == pytest.approx(math.sqrt(sum(squares) / 6), rel=1e-12)
    # A commit's level (scale) does not enter: only the spread within each commit.
    assert compare.pooled([record(scale=2.0)], "decode_tps")[0] == pytest.approx(sigma(), rel=1e-12)
    with pytest.raises(ValueError):
        compare.pooled([], "decode_tps")


def test_identical_runs_pass_with_every_value_shown():
    cell = decide()
    assert cell["status"] == "passed" and cell["passed"] and cell["regressions"] == cell["improvements"] == []
    assert cell["review_reasons"] == [] and cell["order"] == ["r1/candidate", "r1/base", "r2/candidate", "r3/candidate"]
    entry = cell["metrics"]["decode_tps"]
    assert entry["history"] == [10.0 * factor for factor in SPREAD] and entry["base_run"] == 10.0
    assert entry["candidate"] == [10.0] * 3 and entry["df"] == 3 and entry["k"] == 9.219
    assert entry["sigma_log"] == pytest.approx(sigma(), rel=1e-12)
    expected = max(math.log(1.01), 9.219 * sigma() * math.sqrt(1 / 3 + 1 / 5))
    assert entry["threshold"] == pytest.approx(expected, rel=1e-12) and entry["result"] == "within-noise"
    assert entry["anchor_limit"] == pytest.approx(9.219 * sigma() * math.sqrt(1.25), rel=1e-12)
    assert entry["base_run_matches_history"] and not entry["beyond_threshold"] and not entry["improved_beyond_threshold"]
    assert cell["history_pool"] == [dict(commit=BASE, job_id="e" * 32, measured_at="2026-10-28T10:00:00Z")]
    assert cell["decision_rule"]["gated_metrics"] == ["decode_tps", "ttft_s", "server_prefill_tps"]
    assert cell["decision_rule"]["baseline_history_sha256"] == impact.fingerprint([record()])
    assert json.loads(json.dumps(cell)) == cell


@pytest.mark.parametrize("metric", compare.METRICS)
def test_consistent_slowdown_beyond_noise_regresses_only_deciding_metrics(metric):
    cell = decide(candidates(scaled(0.10, metric)))
    entry = cell["metrics"][metric]
    assert entry["difference"] > entry["threshold"] and entry["beyond_threshold"] and entry["all_candidates_worse"]
    expected = 10.0 if metric == "ttft_s" else -100 * 0.10 / 1.10
    # Base estimate is the history mean and the in-run base; SPREAD is centred near 1.
    assert entry["change_percent"] == pytest.approx(expected, abs=0.05)
    if metric in compare.GATED_METRICS:
        assert cell["status"] == "regressed" and cell["regressions"] == [metric] and entry["result"] == "regressed"
    else:
        # A decode half-window is a sub-window of whole decode: shown, never deciding.
        assert cell["status"] == "passed" and cell["regressions"] == [] and cell["review_reasons"] == []
        assert entry["result"] == "informational" and entry["gated"] is False


@pytest.mark.parametrize("metric", compare.GATED_METRICS)
def test_faster_beyond_noise_is_labelled_improved_and_still_passes(metric):
    cell = decide(candidates(scaled(-0.10, metric)))
    entry = cell["metrics"][metric]
    assert entry["difference"] < -entry["threshold"] and entry["improved_beyond_threshold"]
    assert entry["result"] == "improved" and cell["improvements"] == [metric]
    assert cell["status"] == "passed" and cell["regressions"] == [] and cell["review_reasons"] == []
    # Within the threshold either way it is within noise, never improved.
    small = decide(candidates(scaled(-0.02, metric)))
    assert small["metrics"][metric]["result"] == "within-noise" and small["improvements"] == []
    # A half-window that is faster is information only.
    half = decide(candidates(scaled(-0.10, "decode_first128_tps")))
    assert half["metrics"]["decode_first128_tps"]["result"] == "informational" and half["improvements"] == []


def test_disagreeing_candidates_beyond_noise_need_review_not_regression():
    cell = decide({(1, "candidate"): scaled(0.40), (2, "candidate"): scaled(0.40), (3, "candidate"): scaled(-0.01)})
    entry = cell["metrics"]["decode_tps"]
    assert entry["beyond_threshold"] and not entry["all_candidates_worse"]
    assert cell["status"] == "needs-review" and cell["review_reasons"] == ["candidates-disagree:decode_tps"]
    assert entry["result"] == "candidates-disagree" and cell["regressions"] == []


@pytest.mark.parametrize("changes,status,reasons", [
    (candidates(scaled(0.005)), "passed", []),                     # consistent, below the threshold
    (candidates({"ttft_s": 1.1}), "regressed", []),                # higher TTFT is worse
    (candidates({"ttft_s": 0.9}), "passed", []),                   # lower TTFT is better
    ({(1, "candidate"): scaled(0.10, "decode_second128_tps")}, "passed", []),  # information only
])
def test_threshold_direction_and_deciding_metrics(changes, status, reasons):
    cell = decide(changes)
    assert cell["status"] == status and cell["review_reasons"] == reasons


@pytest.mark.parametrize("pool,fraction,status", [
    (1, 0.06, "passed"), (1, 0.08, "regressed"),        # df 3: t 9.219, threshold about 6.1%
    (4, 0.02, "passed"), (4, 0.03, "regressed"),        # df 12: t 3.765, threshold about 2.5%
])
def test_threshold_tightens_as_baseline_records_accumulate(pool, fraction, status):
    records = [record()] + [record(commit, measured_at="2026-10-2" + str(index) + "T10:00:00Z")
                            for index, commit in enumerate(("b" * 40, "d" * 40, "e" * 40)[:pool - 1])]
    cell = decide(candidates(scaled(fraction)), records)
    entry = cell["metrics"]["decode_tps"]
    df = 3 * pool
    assert entry["df"] == df and entry["k"] == compare.t_quantile(df)
    expected = max(math.log(1.01), compare.t_quantile(df) * sigma() * math.sqrt(1 / 3 + 1 / 5))
    assert entry["threshold"] == pytest.approx(expected, rel=1e-9)
    assert cell["status"] == status


def test_floor_bounds_the_threshold_when_history_has_no_spread():
    cell = decide(candidates(scaled(0.005)), [record(spread=(1, 1, 1, 1))])
    assert cell["metrics"]["decode_tps"]["sigma_log"] == 0.0
    assert cell["metrics"]["decode_tps"]["threshold"] == pytest.approx(math.log(1.01)) and cell["status"] == "passed"
    assert decide(candidates(scaled(0.02)), [record(spread=(1, 1, 1, 1))])["status"] == "regressed"


def test_pool_is_every_record_under_the_same_measured_conditions():
    others = [record("%040x" % index, measured_at="2026-10-%02dT10:00:00Z" % index) for index in range(1, 21)]
    cell = decide(None, [record(), *others, record("b" * 40, conditions="0" * 64)])
    assert len(cell["history_pool"]) == 21 and BASE in {item["commit"] for item in cell["history_pool"]}
    assert cell["metrics"]["decode_tps"]["df"] == 63 and cell["metrics"]["decode_tps"]["k"] == 3.140
    # Pool order is by measurement time, so the values do not depend on how records were sealed.
    shuffled = decide(None, [*reversed(others), record()])
    ordered = decide(None, [record(), *others])
    assert shuffled["metrics"] == ordered["metrics"] and shuffled["history_pool"] == ordered["history_pool"]


def test_base_run_far_from_its_history_needs_review_and_blocks_a_regression():
    changes = dict(candidates(scaled(0.30)))
    changes[1, "base"] = scaled(0.20)
    cell = decide(changes)
    entry = cell["metrics"]["decode_tps"]
    assert not entry["base_run_matches_history"] and abs(entry["base_run_log"] - entry["history_mean_log"]) > entry["anchor_limit"]
    assert cell["status"] == "needs-review" and cell["review_reasons"][0] == "base-run-differs-from-baseline-history:decode_tps"
    assert entry["result"] == "regressed"  # the metric comparison is still shown
    # Within the anchor limit the base run only moves the base estimate.
    near = decide({(1, "base"): scaled(0.05)})
    assert near["metrics"]["decode_tps"]["base_run_matches_history"] and near["status"] == "passed"


@pytest.mark.parametrize("drift,anchored", [(-0.013, True), (0.013, True), (-0.03, False)])
def test_anchor_allows_at_least_two_percent_drift_from_a_steady_history(drift, anchored):
    # One baseline job's runs agree to 0.1%; the in-run base of a later job drifts by drift.
    steady = [record(spread=(0.999, 1.001, 0.9995, 1.0005))]
    cell = decide({(1, "base"): scaled(drift, "ttft_s")}, steady)
    entry = cell["metrics"]["ttft_s"]
    assert entry["anchor_limit"] == pytest.approx(math.log(1.02), rel=1e-12)
    assert entry["base_run_matches_history"] is anchored
    assert (cell["review_reasons"] == []) is anchored
    if not anchored:
        assert cell["review_reasons"] == ["base-run-differs-from-baseline-history:ttft_s"]


@pytest.mark.parametrize("state", ["missing", "other-device", "other-model", "other-tier", "other-commit", "conditions"])
def test_missing_or_mismatched_history_needs_review_even_for_a_large_slowdown(state):
    records = {"missing": [], "other-device": [record(device="dgx-spark")], "other-model": [record(model="deepseek")],
               "other-tier": [record(tier="4096")], "other-commit": [record("b" * 40)],
               "conditions": [record(conditions="0" * 64)]}[state]
    cell = decide(candidates(scaled(0.5)), records)
    assert cell["status"] == "needs-review" and cell["regressions"] == []
    reason = "baseline-conditions-differ" if state == "conditions" else "no-baseline-history"
    assert cell["review_reasons"] == [reason] and cell["history_pool"] == []
    assert all(entry["result"] == ("no-baseline-history" if entry["gated"] else "informational")
               and entry["threshold"] is None for entry in cell["metrics"].values())


@pytest.mark.parametrize("run", [(1, "candidate"), (1, "base"), (3, "candidate")])
def test_any_different_token_sequence_is_recorded_without_review(run):
    # Generated tokens are not yet required to match (compare.REQUIRE_SAME_TOKENS).
    cell = decide(token_change=run)
    assert cell["status"] == "passed" and cell["review_reasons"] == []
    assert cell["output_workload"]["same_tokens"] is False and cell["output_workload"]["same_tokens_required"] is False
    # The first run is the reference; a changed first run differs from every other run.
    assert cell["output_workload"]["reference_run"] == "r1/candidate"
    # The history's tokens match the unchanged runs, so they differ from a changed reference too.
    differing = (["r1/base", "r2/candidate", "r3/candidate", "baseline-history"] if run == (1, "candidate")
                 else [compare.arm_name(*run)])
    assert [item["run"] for item in cell["output_workload"]["differences"]] == differing
    assert all(item.get("first_difference_one_based", 101) == 101 for item in cell["output_workload"]["differences"])


def test_history_token_sequence_differences_are_recorded_without_review():
    other = list(TOKENS)
    other[5] = 3
    cell = decide(None, [record(tokens=other)])
    assert cell["status"] == "passed" and cell["review_reasons"] == []
    assert cell["output_workload"]["differences"][0]["run"] == "baseline-history"


def test_regression_keeps_other_review_reasons_visible():
    cell = decide(candidates(scaled(0.10)), environment=dict(status="needs-review", reasons=["r2/candidate:swapout-activity"]))
    assert cell["status"] == "regressed" and cell["review_reasons"] == ["host:r2/candidate:swapout-activity"]


@pytest.mark.parametrize("slowdown", [False, True])
def test_flagged_host_check_needs_review_and_never_hides_a_regression(slowdown):
    environment = dict(status="needs-review", reasons=["r2/candidate:swap-activity"])
    cell = decide(candidates(scaled(0.10)) if slowdown else None, environment=environment)
    assert cell["status"] == ("regressed" if slowdown else "needs-review")
    assert cell["review_reasons"] == ["host:r2/candidate:swap-activity"] and cell["environment"] == environment


@pytest.mark.parametrize("damage", ["missing-run", "extra-run", "order", "reused-server", "conditions", "tokens-short",
                                    "protocol-purpose", "protocol-order", "cell", "host-status", "zero-metric",
                                    "history-logs", "duplicate-history", "history-runs", "baseline-two-commits"])
def test_inconsistent_or_incomplete_cell_is_invalid_evidence(damage):
    results, receipts = series()
    protocol, cell, records = protocol_for(), dict(CELL), [record()]
    environment = None
    if damage == "missing-run": del results[2, "candidate"]
    elif damage == "extra-run": results[4, "candidate"] = results[3, "candidate"]
    elif damage == "order": results[1, "base"]["start_ns"], results[2, "candidate"]["start_ns"] = 200, 100
    elif damage == "reused-server": receipts[1, "base"]["server"] = receipts[1, "candidate"]["server"]
    elif damage == "conditions": receipts[3, "candidate"]["host"]["memory_budget_bytes"] = 2048
    elif damage == "tokens-short": results[1, "base"]["tokens"] = TOKENS[:128]
    elif damage == "protocol-purpose": protocol["purpose"] = "best-of"
    elif damage == "protocol-order": protocol["order"] = ["base", "candidate"]
    elif damage == "cell": del cell["job_id"]
    elif damage == "host-status": environment = dict(status="inconclusive", reasons=[])
    elif damage == "zero-metric": results[1, "candidate"]["metrics"]["decode_tps"] = 0.0
    elif damage == "history-logs": records[0]["runs"][0]["logs"]["ttft_s"] = 0.5
    elif damage == "duplicate-history": records.append(copy.deepcopy(records[0]))
    elif damage == "history-runs": records[0]["runs"] = records[0]["runs"][:3]
    else: protocol = protocol_for("baseline")
    with pytest.raises((ValueError, KeyError)):
        compare.decide_cell(results, receipts, copy.deepcopy(POLICY), cell=cell, protocol=protocol,
                            history=records, environment=environment)


def test_mean_is_correctly_rounded_on_every_python():
    # Python 3.12 sum() compensates and 3.9 does not; the replay must not depend on either.
    import functools
    import operator
    values = [1e16, 1.0, -1e16, 1.0]
    naive = functools.reduce(operator.add, values, 0.0) / len(values)
    assert compare.mean(values) == math.fsum(values) / len(values) == 0.5
    assert naive == 0.25
    entry = decide({(k, "candidate"): {"decode_tps": value} for k, value in zip((1, 2, 3), (9.7, 9.7, 9.9))})
    logs = entry["metrics"]["decode_tps"]["candidate_logs"]
    assert entry["metrics"]["decode_tps"]["difference"] == math.fsum(logs) / 3 - entry["metrics"]["decode_tps"]["base_estimate"]


def test_derived_values_do_not_depend_on_the_platform_math_library():
    # Decimal logarithms are correctly rounded in software, so a device run and
    # the hosted replay compute identical values.
    for ratio in (0.9, 1.0000001, 1.05, 2.5, 1e-3):
        value = compare.ln(ratio)
        assert value == compare.ln(ratio) and value == pytest.approx(math.log(ratio), rel=1e-15, abs=1e-300)
    assert compare.exp(0.5) == pytest.approx(math.exp(0.5), rel=1e-15)


BASELINE_CELL = dict(CELL, candidate=BASE)


def baseline_cell(changes=None, history_records=(), **kwargs):
    if changes is None:
        changes = {(k, "candidate"): {metric: value * factor for metric, value in BASE_METRICS.items()}
                   for k, factor in zip((1, 2, 3, 4), SPREAD)}
    return decide(changes, list(history_records), purpose="baseline", cell=BASELINE_CELL, **kwargs)


def test_baseline_job_reports_four_values_and_sigma_after_adding_its_record():
    alone = baseline_cell()
    entry = alone["metrics"]["decode_tps"]
    assert alone["status"] == "baseline" and not alone["passed"] and alone["regressions"] == []
    assert entry["values"] == pytest.approx([10.0 * factor for factor in SPREAD]) and entry["df"] == 3
    assert entry["sigma_log"] == pytest.approx(sigma(), rel=1e-9)
    assert alone["order"] == ["r1/candidate", "r2/candidate", "r3/candidate", "r4/candidate"]
    assert alone["interpretation"] == compare.BASELINE_INTERPRETATION
    prior = [record("b" * 40, spread=(0.97, 1.03, 1.0, 1.0)), record("d" * 40, conditions="0" * 64)]
    joined = baseline_cell(history_records=prior)["metrics"]["decode_tps"]
    # Only the same-conditions record from another commit joins: df 3 + 3.
    assert joined["df"] == 6 and joined["sigma_log"] > entry["sigma_log"]


def test_baseline_job_with_a_host_flag_needs_review_and_different_tokens_do_not():
    assert baseline_cell(token_change=(3, "candidate"))["status"] == "baseline"
    flagged = baseline_cell(environment=dict(status="needs-review", reasons=["r2/candidate:swap-activity"]))
    assert flagged["status"] == "needs-review" and flagged["review_reasons"] == ["host:r2/candidate:swap-activity"]


def test_baseline_plan_selects_every_cell_for_one_commit_and_replays_exactly():
    value = impact.plan([], POLICY, base=BASE, head=BASE, candidate=BASE, baseline=True)
    assert value["purpose"] == "baseline-history" and value["changed_paths"] == []
    assert len(impact.selected(value)) == 6
    assert all(cell["performance"] == dict(required=True, tiers=["512"], output_tokens=256) for cell in value["cells"])
    assert impact.validate(copy.deepcopy(value), POLICY) == value
    assert not impact.selected(impact.plan([], POLICY, base=BASE, head=BASE, candidate=BASE))
    for change in ("purpose", "path", "revision"):
        bad = copy.deepcopy(value)
        if change == "purpose": bad["purpose"] = "impact-selection"
        elif change == "path": bad["changed_paths"] = ["engine/src/llama.cpp"]
        else: bad["candidate"] = CANDIDATE
        with pytest.raises(ValueError):
            impact.validate(bad, POLICY)
    with pytest.raises(ValueError):
        impact.plan(["engine/src/llama.cpp"], POLICY, base=BASE, head=BASE, candidate=BASE, baseline=True)
    with pytest.raises(ValueError):
        impact.plan([], POLICY, base=BASE, head="b" * 40, candidate="b" * 40, baseline=True)


def job_config(protocol, mode="functional-numerical-performance", plan=None, history_records=None):
    from ci import selection_contract as contract
    from test_ci_selection import template
    value = plan or impact.plan([], POLICY, base=BASE, head=BASE, candidate=BASE, baseline=True)
    config = template(value)
    config.update(validation_mode=mode, execution_devices=["cloud-5090"], performance_protocol=protocol,
                  baseline_history=history_records or [])
    for side in config["sides"].values():
        side["build"]["toolchain"] = {"CMAKE_CXX_COMPILER": "/usr/bin/g++-12"}
        for data in side["models"].values():
            if mode == contract.FUNCTIONAL_MODE:
                data["benchmarks"] = {}
            for spec in data["benchmarks"].values():
                spec["preflight_argv"] = []
    # Real device templates build one reviewed target list whatever the plan selects.
    targets = sorted({target for target, *_ in impact.CASES.values()})
    for side in config["sides"].values():
        side["build"].update(targets=["llama-server", *targets], image_relatives=["bin/" + name for name in targets])
    sides = contract.timed_sides(protocol) if mode != contract.FUNCTIONAL_MODE else ("candidate",)
    manifest = dict(job_id="f" * 32, source_revisions={side: value[side] for side in sides})
    return contract, config, manifest, value


def test_baseline_job_binds_with_its_protocol_and_stages_four_candidate_runs():
    from ci import selection_contract
    contract, config, manifest, value = job_config(selection_contract.performance_protocol("baseline"))
    config, manifest = contract.bind(config, manifest, value, POLICY)
    assert {name for name in manifest["stages"] if "-benchmark-" in name} == {
        "candidate-" + model + "-r" + str(k) + "-benchmark-512" for model in ("deepseek", "muse") for k in range(1, 5)}
    assert not any(name.startswith("base-") for name in manifest["stages"])
    assert contract.timed_protocol(config)["order"] == BASELINE_ORDER


def test_pr_job_stages_three_candidate_runs_and_one_base_run():
    from ci import selection_contract as contract
    value = impact.plan(["engine/src/ssd/model-adapter.cpp"], POLICY, base=BASE, head="b" * 40, candidate=CANDIDATE)
    contract, config, manifest, value = job_config(contract.performance_protocol(), plan=value)
    config, manifest = contract.bind(config, manifest, value, POLICY)
    timed = {name for name in manifest["stages"] if "-benchmark-" in name}
    models = {name.split("-")[1] for name in timed}
    assert timed == {side + "-" + model + "-r" + str(k) + "-benchmark-512" for model in models
                     for side, runs in (("candidate", (1, 2, 3)), ("base", (1,))) for k in runs}


@pytest.mark.parametrize("damage", ["gate-protocol", "functional", "baseline-for-change", "order", "history-missing"])
def test_baseline_and_gate_protocols_cannot_be_swapped(damage):
    from ci import selection_contract as contract
    protocol = contract.performance_protocol("baseline")
    mode, plan = "functional-numerical-performance", None
    if damage == "gate-protocol": protocol = contract.performance_protocol()
    elif damage == "functional": mode = contract.FUNCTIONAL_MODE
    elif damage == "baseline-for-change":
        plan = impact.plan(["engine/src/llama.cpp"], POLICY, base=BASE, head="b" * 40, candidate=CANDIDATE)
    elif damage == "order": protocol = dict(protocol, order=["candidate"] * 5)
    contract, config, manifest, value = job_config(protocol, mode, plan)
    if damage == "history-missing": del config["baseline_history"]
    with pytest.raises(ValueError):
        contract.bind(config, manifest, value, POLICY)


def test_build_cache_key_is_the_same_for_a_pr_base_and_the_baseline_build_of_that_commit():
    from ci import selection_contract as contract
    pr = impact.plan(["engine/src/ssd/model-adapter.cpp"], POLICY, base=BASE, head="b" * 40, candidate=CANDIDATE)
    _, gate, _, _ = job_config(contract.performance_protocol(), plan=pr)
    _, baseline, _, _ = job_config(contract.performance_protocol("baseline"))
    for config in (gate, baseline):
        for side, spec in config["sides"].items():
            spec["build"]["environment"] = {"LD_LIBRARY_PATH": "/build/" + side + "/bin:/opt/toolchain/lib64"}
    assert contract.build_cache_key(gate, "base") == contract.build_cache_key(baseline, "candidate")
    assert contract.build_cache_key(gate, "candidate") != contract.build_cache_key(gate, "base")
    gate["sides"]["base"]["build"]["cmake_flags"] = ["-DGGML_CUDA=OFF"]
    assert contract.build_cache_key(gate, "base") != contract.build_cache_key(baseline, "candidate")


def test_build_cache_offer_and_store_bind_into_the_manifest_and_are_rechecked():
    from ci import selection_contract as contract
    _, config, manifest, value = job_config(contract.performance_protocol("baseline"))
    config["cache_build_as"] = contract.build_cache_key(config, "candidate")
    bound_config, bound_manifest = contract.bind(copy.deepcopy(config), copy.deepcopy(manifest), value, POLICY)
    assert bound_manifest["cache_as"] == {"candidate": config["cache_build_as"]}
    bad = dict(config, cache_build_as="0" * 64)
    with pytest.raises(ValueError):
        contract.bind(bad, copy.deepcopy(manifest), value, POLICY)
    pr = impact.plan(["engine/src/ssd/model-adapter.cpp"], POLICY, base=BASE, head="b" * 40, candidate=CANDIDATE)
    _, config, manifest, value = job_config(contract.performance_protocol(), plan=pr)
    config["base_build"] = dict(key=contract.build_cache_key(config, "base"), freeze_manifest_sha256="1" * 64,
                                source_job="e" * 32)
    bound_config, bound_manifest = contract.bind(copy.deepcopy(config), copy.deepcopy(manifest), value, POLICY)
    assert bound_manifest["prebuilt_offer"] == {"base": dict(key=config["base_build"]["key"], manifest_sha256="1" * 64)}
    tampered = copy.deepcopy(bound_manifest)
    tampered["prebuilt_offer"]["base"]["manifest_sha256"] = "2" * 64
    with pytest.raises(ValueError):
        contract.bound(bound_config, tampered, "f" * 32)
    for damage in ("key", "candidate-side"):
        bad = copy.deepcopy(config)
        if damage == "key": bad["base_build"]["key"] = "0" * 64
        else: bad["cache_build_as"] = contract.build_cache_key(bad, "candidate")  # only a baseline job stores
        with pytest.raises(ValueError):
            contract.bind(bad, copy.deepcopy(manifest), value, POLICY)


def report_for(cells, commit=BASE, job="f" * 32):
    return dict(purpose="operator-selected-report-only", base=commit, head=commit, candidate=commit,
                cells=cells, errors=[])


def baseline_row(model="muse", job="f" * 32, machine=MACHINE, **kwargs):
    """One verified row of a baseline job's hosted report, with the characteristics file its job profiled."""
    cell = baseline_cell(**kwargs)
    cell["cell"] = dict(cell["cell"], model=model, job_id=job)
    cell["build"] = dict(BUILD)
    cell["derived_launch"] = dict(sides=dict(candidate=dict(characteristics_sha256=MACHINE_SHA)), changes=[])
    profile = dict(sha256=compare.hashlib.sha256(machine.encode()).hexdigest(), returncode=1, side="candidate",
                   sections=dict(cpu="ok", gpus="failed"), characteristics=machine)
    return dict(device="cloud-5090", model=model, job_id=job, status=cell["status"],
                performance=[dict(cell, tier="512")], machine_profile=profile)


def store(root, records):
    """A baseline-history directory holding these records and the characteristics file they name."""
    history.write_characteristics(root, "cloud-5090", MACHINE)
    for item in records:
        history.write(root, item)
    return root


def test_history_tool_ingests_verified_baseline_cells_and_skips_others(tmp_path):
    flagged = baseline_row("deepseek", environment=dict(status="needs-review", reasons=["r2/candidate:swapout-activity"]))
    report = report_for([baseline_row(), flagged, dict(device="dgx-spark", model="muse", status="deferred-not-in-this-run",
                                                         performance="deferred-not-in-this-run")])
    records, skipped, machine = history.records_from_report(
        report, device="cloud-5090", job_id="f" * 32, commit=BASE, run_id=12, run_attempt=1,
        measured_at="2026-10-29T08:00:00Z")
    assert [item["model"] for item in records] == ["muse"]
    assert skipped == [dict(cell="cloud-5090/deepseek/512", status="needs-review",
                            reasons=["host:r2/candidate:swapout-activity"])]
    value = records[0]
    assert [run["metrics"]["decode_tps"] for run in value["runs"]] == pytest.approx([10.0 * f for f in SPREAD])
    assert value["build"] == BUILD and value["conditions_sha256"] == CONDITIONS and value["run_id"] == 12
    # The record names the characteristics file the job profiled; that file is stored first, by its hash.
    assert machine == MACHINE and value["characteristics_sha256"] == MACHINE_SHA
    root = tmp_path / "baseline-history"
    with pytest.raises(ValueError, match="characteristics .* are missing"):
        history.write(root, value)
    assert history.write_characteristics(root, "cloud-5090", machine) == "stored"
    assert history.write_characteristics(root, "cloud-5090", machine) == "present"
    saved = history.characteristics_path(root, "cloud-5090", MACHINE_SHA)
    assert saved == root / "cloud-5090/characteristics" / (MACHINE_SHA + ".json")
    assert saved.read_text() == MACHINE and saved.stat().st_mode & 0o777 == 0o400
    assert history.write(root, value) == "stored" and history.write(root, value) == "present"
    assert history.load_all(root) == [value]
    path = history.record_path(root, value)
    assert path == root / "cloud-5090/muse/512" / (BASE + "." + CONDITIONS + ".json")
    assert path.stat().st_mode & 0o777 == 0o400
    # One baseline job per commit: a second, different record of the same key is refused.
    with pytest.raises(ValueError, match="different record already exists"):
        history.write(root, dict(value, job_id="1" * 32))
    assert history.load_all(root) == [value]


@pytest.mark.parametrize("damage", ["other-commit", "gate-cell", "other-job", "not-a-report", "no-characteristics",
                                    "characteristics-hash", "other-characteristics", "two-profiles"])
def test_history_tool_refuses_a_report_that_is_not_this_baseline_job(damage):
    row = baseline_row()
    report = report_for([row])
    if damage == "other-commit": report["candidate"] = CANDIDATE
    elif damage == "gate-cell": row["performance"][0]["protocol"] = protocol_for()
    elif damage == "other-job": row["job_id"] = "1" * 32
    elif damage == "no-characteristics": del row["machine_profile"]["characteristics"]
    elif damage == "characteristics-hash": row["machine_profile"]["characteristics"] += " "
    elif damage == "other-characteristics":
        row["performance"][0]["derived_launch"]["sides"]["candidate"]["characteristics_sha256"] = "0" * 64
    elif damage == "two-profiles":
        other = baseline_row("deepseek", machine=MACHINE + " ")
        other["performance"][0]["derived_launch"]["sides"]["candidate"]["characteristics_sha256"] = \
            other["machine_profile"]["sha256"]
        report["cells"].append(other)
    else: report = dict(cells="not a list")
    with pytest.raises(ValueError):
        history.records_from_report(report, device="cloud-5090", job_id="f" * 32, commit=BASE,
                                    run_id=12, run_attempt=1, measured_at="2026-10-29T08:00:00Z")


def test_history_files_are_verified_on_load_and_pruned_newest_first(tmp_path):
    root = store(tmp_path / "baseline-history", [])
    for day in range(1, 6):
        history.write(root, record("%040x" % day, measured_at="2026-10-%02dT10:00:00Z" % day))
    removed = history.prune(root, 2, dry_run=True)
    assert len(removed) == 3 and len(history.load_all(root)) == 5
    history.prune(root, 2)
    assert [item["commit"] for item in history.load_all(root)] == ["%040x" % 4, "%040x" % 5]
    assert history.characteristics_path(root, "cloud-5090", MACHINE_SHA).exists()
    wrong = root / "cloud-5090/muse/512" / ("f" * 40 + "." + CONDITIONS + ".json")
    wrong.write_text(json.dumps(record("%040x" % 5)))
    with pytest.raises(ValueError, match="file name differs"):
        history.load_all(root)
    wrong.unlink()
    tampered = history.record_path(root, record("%040x" % 5))
    value = json.loads(tampered.read_text())
    value["runs"][0]["metrics"]["decode_tps"] = 11.0
    tampered.chmod(0o600)
    tampered.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="logs differ"):
        history.load_all(root)


def test_a_pr_job_takes_the_characteristics_of_the_most_recent_baseline_job(tmp_path):
    root = store(tmp_path / "baseline-history", [record("1" * 40, measured_at="2026-10-01T10:00:00Z", job="1" * 32)])
    newer = MACHINE.replace("synthetic", "reprofiled")
    sha = compare.hashlib.sha256(newer.encode()).hexdigest()
    history.write_characteristics(root, "cloud-5090", newer)
    history.write(root, record("2" * 40, model="deepseek", measured_at="2026-10-02T10:00:00Z", job="2" * 32,
                               characteristics=sha))
    records = history.load_all(root)
    assert history.latest_characteristics(root, records, "cloud-5090") == dict(
        sha256=sha, job_id="2" * 32, commit="2" * 40, text=newer)
    with pytest.raises(ValueError, match="dgx-spark has no baseline machine characteristics yet; a baseline job "
                                         "must run on it first"):
        history.latest_characteristics(root, records, "dgx-spark")
    # Pruning the older job's record removes the characteristics file only it named.
    history.write(root, record("3" * 40, measured_at="2026-10-03T10:00:00Z", job="3" * 32, characteristics=sha))
    assert history.prune(root, 1) == ["cloud-5090/muse/512/" + "1" * 40 + "." + CONDITIONS + ".json",
                                      "cloud-5090/characteristics/" + MACHINE_SHA + ".json"]
    assert history.latest_characteristics(root, history.load_all(root), "cloud-5090")["commit"] == "3" * 40


@pytest.mark.parametrize("damage", ["tampered", "missing", "misnamed"])
def test_a_tampered_or_missing_characteristics_file_is_refused(tmp_path, damage):
    root = store(tmp_path / "baseline-history", [record()])
    path = history.characteristics_path(root, "cloud-5090", MACHINE_SHA)
    path.chmod(0o600)
    if damage == "tampered":
        path.write_text(MACHINE.replace("ok", "failed"))
    elif damage == "missing":
        path.unlink()
    else:
        path.rename(path.with_name("0" * 64 + ".json"))
    with pytest.raises(ValueError, match="differ from their hash" if damage != "missing" else "are missing"):
        history.load_all(root)
    assert history.main(["--root", str(root), "verify"]) == 1


def test_history_cli_lists_verifies_prunes_and_ingests(tmp_path, capsys):
    root = tmp_path / "baseline-history"
    report = tmp_path / "aggregate.json"
    report.write_text(json.dumps(report_for([baseline_row()])))
    arguments = ["--root", str(root), "ingest", "--report", str(report), "--device", "cloud-5090", "--job-id", "f" * 32,
                 "--commit", BASE, "--measured-at", "2026-10-29T08:00:00Z", "--run-id", "12"]
    assert history.main(arguments) == 0
    out = capsys.readouterr().out
    assert "stored characteristics " + MACHINE_SHA[:12] in out and "stored cloud-5090/muse/512/" in out
    assert history.main(["--root", str(root), "verify"]) == 0
    assert "verified 1 baseline records" in capsys.readouterr().out
    assert history.main(["--root", str(root), "list"]) == 0
    assert BASE[:12] in capsys.readouterr().out
    assert history.main(["--root", str(root), "prune", "--keep", "1"]) == 0
    assert history.main(arguments[:-2] + ["--run-id", "12", "--commit", CANDIDATE]) == 1


@pytest.mark.parametrize("damage", [None, "envelope", "child-report", "config-edit", "characteristics-envelope",
                                    "characteristics-unsealed"])
def test_aggregate_replays_with_exactly_the_history_named_by_the_envelope(tmp_path, monkeypatch, damage):
    """Synthetic retained child: only the seal checks before the evidence replay are exercised."""
    from ci import selected_aggregate
    from ci import selection_contract
    from ci.contract import digest, save
    records = [record()]
    contract, config, manifest, value = job_config(selection_contract.performance_protocol("baseline"),
                                                   history_records=records)
    if damage == "characteristics-envelope":
        # Sealed like the history: the envelope must name the hash of the file in the config.
        config["machine_characteristics"] = dict(sha256=MACHINE_SHA, text=MACHINE, job_id="e" * 32, commit=BASE,
                                                 host_path="/opt/inferssd-ci-integrations/machine.json",
                                                 runtime_path="/models/machine.json")
    config, manifest = contract.bind(config, manifest, value, POLICY)
    root = tmp_path / "cloud-5090" / ("f" * 32)
    root.mkdir(parents=True)
    save(root / "root-manifest.json", manifest)
    config["manifest_sha256"] = digest(root / "root-manifest.json")
    if damage == "config-edit":
        config["baseline_history"] = []
    save(root / "selected-config.json", config)
    job = dict(device="cloud-5090", job_id="f" * 32, source_revision=value["candidate"],
               config_sha256=digest(root / "selected-config.json"), selection_plan_sha256=impact.fingerprint(value),
               control_revision="d" * 40, control_manifest_sha256="e" * 64)
    scope = contract.execution_scope(config)
    report = dict(status="baseline", cleanup_confirmed=True, purpose="selected-integration",
                  selection_plan_sha256=impact.fingerprint(value), device="cloud-5090", job_id="f" * 32,
                  control_revision="d" * 40, control_manifest_sha256="e" * 64, selection_parent="1-1",
                  authorization_path="live-private-parent", performance_protocol=config["performance_protocol"],
                  baseline_history_sha256="0" * 64 if damage == "child-report" else impact.fingerprint(records),
                  base_build=None, **scope, **{key: value[key] for key in ("base", "head", "candidate")})
    save(root / "integration.json", report)
    save(root / "job-finished.json", dict(cleanup_confirmed=True))
    save(tmp_path / "cloud-5090" / "dispatch.json", dict(selection_parent="1-1", revision=value["candidate"],
         **{key: job[key] for key in ("device", "job_id", "config_sha256", "selection_plan_sha256",
                                      "control_revision", "control_manifest_sha256")}))
    monkeypatch.setattr(selected_aggregate, "sealed", lambda *a: None)
    expected = "0" * 64 if damage == "envelope" else impact.fingerprint(records)
    machine = {"characteristics-envelope": "0" * 64, "characteristics-unsealed": MACHINE_SHA}.get(damage)
    with pytest.raises((ValueError, OSError)) as caught:
        selected_aggregate.device_report(root, value, POLICY, job, dict(conclusion="success", run_attempt=1),
                                         scope, expected, machine)
    # Editing the history after sealing already breaks the root-manifest config hash.
    envelope_error = "sealed machine characteristics differ from the operator envelope"
    expected_error = {None: None, "config-edit": "selected source binding differs",
                      "characteristics-envelope": envelope_error, "characteristics-unsealed": envelope_error}.get(
        damage, "sealed baseline history differs")
    if expected_error is None:
        assert "performance-preflight.json" in str(caught.value) or "admission" in str(caught.value)
    else:
        assert expected_error in str(caught.value)
