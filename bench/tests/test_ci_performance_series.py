# SPDX-License-Identifier: Apache-2.0
"""Interleaved-pair decisions, A/A calibration and noise bands; synthetic data only."""

import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import compare, impact, noise_band
from ci.contract import load, noise_bands

POLICY = load(Path(__file__).resolve().parents[1] / "ci/policy.json")
CELL = dict(device="cloud-5090", model="muse", tier="512", job_id="f" * 32, base="a" * 40, candidate="c" * 40)


def band(conditions, sigma=0.01, **changes):
    value = dict(sigma_log={metric: sigma for metric in compare.METRICS}, pairs=8,
                 statistic="rms-paired-log-ratio", conditions_sha256=conditions, engine_revision="a" * 40,
                 calibration_jobs=["e" * 32], date="2026-10-29")
    value.update(changes)
    return value


def banded(cells, sigma=0.01, pairs=8):
    """Sealed noise bands with one band per {(device, model, tier): conditions_sha256}."""
    value = {}
    for (device, model, tier), conditions in cells.items():
        value.setdefault(device, {}).setdefault(model, {})[tier] = band(conditions, sigma, pairs=pairs)
    return noise_bands(value)


RECEIPT = {
    "model_identity": {"shards": [{"sha256": "b" * 64}]},
    "host": {"hardware": "synthetic", "os": "synthetic", "memory_budget_bytes": 1024,
             "os_page_cache_treatment": "uncontrolled",
             "effective_settings": {"native_launch_policy": {"typed_values": {"prefetch.depth": 2}}}},
    "build": {"loaded_libraries": [{"path": "libsynthetic.so"}]},
}
CONDITIONS = compare.conditions_digest(compare.comparable(RECEIPT))


def result(values=None, tokens=None, start=0):
    metrics = dict(decode_first128_tps=10.0, decode_second128_tps=10.0, decode_tps=10.0, ttft_s=1.0,
                   server_prefill_tps=100.0)
    metrics.update(values or {})
    prompt_ms = 100000 / metrics["server_prefill_tps"]
    metrics["server_prefill_tps"] = 100 * 1000 / prompt_ms
    return {"input_tokens": 100, "prompt_ids": [7] * 100, "prompt_sha256": "a" * 64, "output_tokens": 256,
            "tokens": list(tokens or [11] * 256), "timings": {"prompt_n": 100, "prompt_ms": prompt_ms},
            "metrics": metrics, "start_ns": start, "end_ns": start + 10}


def series(changes=None, bands=None, pairs=2, purpose="regression-gate", cell=CELL, token_change=None):
    """Build a complete cell; `changes` maps (pair, side) to metric overrides."""
    results, receipts = {}, {}
    for position, key in enumerate(compare.schedule(pairs)):
        tokens = [11] * 256
        if token_change is not None and key == token_change:
            tokens[100] = 12
        results[key] = result((changes or {}).get(key), tokens, start=position * 100)
        receipts[key] = dict(copy.deepcopy(RECEIPT), server=dict(pid=position + 1, start_identity=str(key)))
    bands = bands if bands is not None else banded({("cloud-5090", "muse", "512"): CONDITIONS})
    policy = copy.deepcopy(POLICY)
    compared = [compare.compare_pair(results[k, "base"], results[k, "candidate"],
                                     receipts[k, "base"], receipts[k, "candidate"], policy)
                for k in range(1, pairs + 1)]
    protocol = dict(tier="512", prompt_sha256="0" * 64, output_tokens=256, samples_per_side=pairs,
                    order="ABBA", purpose=purpose)
    return results, receipts, compared, policy, protocol, bands


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


def decide(*args, **kwargs):
    results, receipts, compared, policy, protocol, bands = series(*args, **kwargs)
    return compare.compare_series(results, receipts, compared, policy, cell=kwargs.get("cell", CELL),
                                  protocol=protocol, bands=bands)


def slower(fraction, metric="decode_tps"):
    """Metric override that makes the candidate `fraction` slower."""
    base = result()["metrics"][metric]
    return {metric: base * (1 + fraction) if metric == "ttft_s" else base / (1 + fraction)}


def both(override):
    return {(1, "candidate"): override, (2, "candidate"): override}


def test_schedule_is_abba_and_pair_k_starts_with_base_when_odd():
    assert compare.schedule(2) == [(1, "base"), (1, "candidate"), (2, "candidate"), (2, "base")]
    assert [side[0] for _, side in compare.schedule(4)] == list("bccbbccb")
    with pytest.raises(ValueError):
        compare.schedule(True)


def test_outcome_orders_failed_regressed_review_calibrated_passed():
    order = ["failed", "regressed", "needs-review", "calibrated", "passed"]
    for index, status in enumerate(order):
        assert compare.outcome([dict(status=value) for value in order[index:]]) == status
    assert compare.outcome([]) == "passed"
    with pytest.raises(ValueError):
        compare.outcome([dict(status="inconclusive")])


def test_identical_arms_pass_with_explicit_per_pair_values():
    cell = decide()
    assert cell["status"] == "passed" and cell["passed"] and cell["regressions"] == [] and cell["review_reasons"] == []
    assert cell["order"] == ["r1/base", "r1/candidate", "r2/candidate", "r2/base"]
    metric = cell["metrics"]["decode_tps"]
    assert metric["base"] == metric["candidate"] == [10.0, 10.0] and metric["d"] == [0.0, 0.0]
    assert metric["threshold"] == pytest.approx(4.277 * 0.01 / math.sqrt(2)) and metric["result"] == "within-band"
    assert cell["noise_band"]["status"] == "matched" and cell["noise_band"]["pairs"] == 8
    assert cell["decision_rule"]["k"] == 4.277 and len(cell["pairs"]) == 2
    assert cell["decision_rule"]["gated_metrics"] == ["decode_tps", "ttft_s", "server_prefill_tps"]
    assert cell["decision_rule"]["noise_bands_sha256"] == impact.fingerprint(
        banded({("cloud-5090", "muse", "512"): CONDITIONS}))
    assert json.loads(json.dumps(cell)) == cell


@pytest.mark.parametrize("metric", compare.METRICS)
def test_consistent_slowdown_beyond_the_band_regresses_only_deciding_metrics(metric):
    cell = decide(both(slower(0.05, metric)))
    entry = cell["metrics"][metric]
    assert entry["d"] == [pytest.approx(math.log(1.05))] * 2 and entry["all_pairs_worse"]
    assert entry["mean_d"] > entry["threshold"] and entry["beyond_band"]
    expected = 5.0 if metric == "ttft_s" else -100 * 0.05 / 1.05
    assert entry["change_percent"] == pytest.approx(expected)
    if metric in compare.GATED_METRICS:
        assert cell["status"] == "regressed" and cell["regressions"] == [metric] and entry["result"] == "regressed"
    else:
        # A decode half-window is a sub-window of whole decode: shown, never deciding.
        assert cell["status"] == "passed" and cell["regressions"] == [] and cell["review_reasons"] == []
        assert entry["result"] == "informational" and entry["gated"] is False


@pytest.mark.parametrize("pairs,k", [(8, 4.277), (9, 4.095), (16, 3.545), (17, 3.545), (31, 3.292), (64, 3.122),
                                     (65, 3.0), (400, 3.0)])
def test_band_multiplier_is_the_fixed_t_quantile_for_the_pooled_pairs(pairs, k):
    assert compare.band_multiplier(pairs) == k
    cell = decide(bands=banded({("cloud-5090", "muse", "512"): CONDITIONS}, pairs=pairs))
    assert cell["decision_rule"]["k"] == k
    assert cell["metrics"]["ttft_s"]["threshold"] == pytest.approx(max(math.log(1.01), k * 0.01 / math.sqrt(2)))
    for low in (7, True, None):
        with pytest.raises(ValueError):
            compare.band_multiplier(low)


def test_disagreeing_pairs_beyond_the_band_need_review_not_regression():
    cell = decide({(1, "candidate"): {"decode_tps": 8.0}, (2, "candidate"): {"decode_tps": 10.5}})
    entry = cell["metrics"]["decode_tps"]
    assert entry["d"][0] > 0 > entry["d"][1] and entry["mean_d"] > entry["threshold"]
    assert cell["status"] == "needs-review" and cell["review_reasons"] == ["pairs-disagree:decode_tps"]
    assert entry["result"] == "pairs-disagree" and cell["regressions"] == []


@pytest.mark.parametrize("changes,status,reasons", [
    (both(slower(0.005)), "passed", []),                       # consistent, below the threshold
    (both({"decode_tps": 11.0}), "passed", []),                # faster
    (both({"ttft_s": 0.9}), "passed", []),                     # lower TTFT is better
    (both({"ttft_s": 1.1}), "regressed", []),                  # higher TTFT is worse
    # Pair 2 is equal (d = 0), so not every pair is slower.
    ({(1, "candidate"): slower(0.10)}, "needs-review", ["pairs-disagree:decode_tps"]),
    # An informational half-window never needs review on its own.
    ({(1, "candidate"): slower(0.10, "decode_first128_tps")}, "passed", []),
])
def test_band_threshold_direction_and_strictly_worse_pairs(changes, status, reasons):
    cell = decide(changes)
    assert cell["status"] == status and cell["review_reasons"] == reasons


@pytest.mark.parametrize("sigma,fraction,status", [
    (0.0, 0.005, "passed"), (0.0, 0.02, "regressed"),          # floor ln(1.01) dominates
    (0.05, 0.15, "passed"), (0.05, 0.18, "regressed"),          # k*sigma/sqrt(2) = 0.151 dominates
])
def test_threshold_is_max_of_floor_and_scaled_sigma(sigma, fraction, status):
    bands = banded({("cloud-5090", "muse", "512"): CONDITIONS}, sigma=sigma)
    cell = decide(both(slower(fraction)), bands=bands)
    expected = max(math.log(1.01), 4.277 * sigma / math.sqrt(2))
    assert cell["metrics"]["decode_tps"]["threshold"] == pytest.approx(expected)
    assert cell["status"] == status


@pytest.mark.parametrize("band_state", ["missing", "other-device", "other-tier", "conditions"])
def test_missing_or_mismatched_band_needs_review_even_for_a_large_slowdown(band_state):
    cells = {"missing": {}, "other-device": {("dgx-spark", "muse", "512"): CONDITIONS},
             "other-tier": {("cloud-5090", "muse", "4096"): CONDITIONS},
             "conditions": {("cloud-5090", "muse", "512"): "0" * 64}}[band_state]
    cell = decide(both(slower(0.5)), bands=banded(cells))
    assert cell["status"] == "needs-review" and cell["regressions"] == []
    reason = "noise-band-conditions-differ" if band_state == "conditions" else "no-noise-band"
    assert cell["review_reasons"] == [reason]
    assert cell["noise_band"]["status"] == ("conditions-differ" if band_state == "conditions" else "missing")
    assert all(entry["result"] == ("no-matching-band" if entry["gated"] else "informational")
               and entry["threshold"] is None for entry in cell["metrics"].values())


@pytest.mark.parametrize("arm", [(1, "candidate"), (2, "base"), (2, "candidate")])
def test_any_different_token_sequence_needs_review_between_or_within_sides(arm):
    cell = decide(token_change=arm)
    assert cell["status"] == "needs-review" and cell["review_reasons"] == ["generated-tokens-differ"]
    assert cell["output_workload"]["differences"] == [dict(arm=compare.arm_name(*arm), first_difference_one_based=101,
                                                           reference_token=11, token=12)]


def test_regression_keeps_other_review_reasons_visible():
    cell = decide(both(slower(0.05)), token_change=(2, "base"))
    assert cell["status"] == "regressed" and cell["review_reasons"] == ["generated-tokens-differ"]


@pytest.mark.parametrize("slowdown", [False, True])
def test_flagged_host_pair_needs_review_and_never_hides_a_regression(slowdown):
    results, receipts, compared, policy, protocol, bands = series(both(slower(0.05)) if slowdown else None)
    environment = dict(status="needs-review", reasons=["candidate:swap-activity"])
    compared[1] = dict(compared[1], environment=environment)
    compared[0] = dict(compared[0], environment=dict(status="passed", reasons=[]))
    cell = compare.compare_series(results, receipts, compared, policy, cell=CELL, protocol=protocol, bands=bands)
    assert cell["status"] == ("regressed" if slowdown else "needs-review")
    assert cell["review_reasons"] == ["host:r2:candidate:swap-activity"]


def test_arms_out_of_abba_order_fail_on_the_order_check_itself():
    results, receipts, compared, policy, protocol, bands = series()
    # Swap only when the two pair-2 arms ran; identities and values stay valid.
    first, second = results[2, "candidate"], results[2, "base"]
    first["start_ns"], second["start_ns"] = second["start_ns"], first["start_ns"]
    first["end_ns"], second["end_ns"] = second["end_ns"], first["end_ns"]
    with pytest.raises(ValueError, match="interleaved order"):
        compare.compare_series(results, receipts, compared, policy, cell=CELL, protocol=protocol, bands=bands)


@pytest.mark.parametrize("damage", ["missing-arm", "extra-pair", "order", "reused-server", "pair-values", "pair-tokens",
                                    "conditions", "policy-pairs", "protocol-order", "protocol-purpose", "cell",
                                    "host-status", "zero-metric", "invalid-bands"])
def test_inconsistent_or_incomplete_series_is_invalid_evidence(damage):
    results, receipts, compared, policy, protocol, bands = series()
    cell = dict(CELL)
    if damage == "missing-arm": del results[2, "base"]
    elif damage == "extra-pair": compared.append(compared[0])
    elif damage == "order": results[2, "candidate"]["start_ns"], results[2, "base"]["start_ns"] = 300, 200
    elif damage == "reused-server": receipts[2, "base"]["server"] = receipts[1, "base"]["server"]
    elif damage == "pair-values": compared[1]["metrics"]["decode_tps"]["candidate"] = 9.0
    elif damage == "pair-tokens": compared[0]["output_workload"]["same_tokens"] = False
    elif damage == "conditions": compared[1]["measurement_identity"] = dict(compared[1]["measurement_identity"], conditions_sha256="0" * 64)
    elif damage == "policy-pairs": policy["thresholds"]["pairs"] = 3
    elif damage == "protocol-order": protocol["order"] = "ABAB"
    elif damage == "protocol-purpose": protocol["purpose"] = "best-of"
    elif damage == "cell": del cell["job_id"]
    elif damage == "host-status": compared[0]["environment"] = dict(status="inconclusive", reasons=[])
    elif damage == "invalid-bands": bands["cloud-5090"]["muse"]["512"]["pairs"] = 2
    else: results[1, "candidate"]["metrics"]["decode_tps"] = 0.0
    with pytest.raises(ValueError):
        compare.compare_series(results, receipts, compared, policy, cell=cell, protocol=protocol, bands=bands)


def test_derived_values_do_not_depend_on_the_platform_math_library():
    # Decimal logarithms are correctly rounded in software, so a device run and
    # the hosted replay compute identical values.
    for ratio in (0.9, 1.0000001, 1.05, 2.5, 1e-3):
        value = compare.ln(ratio)
        assert value == compare.ln(ratio) and value == pytest.approx(math.log(ratio), rel=1e-15, abs=1e-300)
    assert compare.exp(0.5) == pytest.approx(math.exp(0.5), rel=1e-15)


def calibration_cell(job="f" * 32, pairs=4, scale=0.0, device="cloud-5090", changes=None, **kwargs):
    if changes is None:
        changes = {(k, "candidate"): {"decode_tps": 10.0 * (1 + scale * (k % 2 - 0.5))} for k in range(1, pairs + 1)}
    return decide(changes, pairs=pairs, purpose="noise-calibration",
                  cell=dict(CELL, device=device, candidate=CELL["base"], job_id=job), **kwargs)


def test_mean_and_rms_are_correctly_rounded_on_every_python():
    # Python 3.12 sum() compensates and 3.9 does not; the replay must not depend on either.
    import functools
    import operator
    cell = calibration_cell(changes={(k, "candidate"): {"decode_tps": value}
                                     for k, value in enumerate((9.7, 9.7, 9.7, 9.9), 1)})
    entry = cell["metrics"]["decode_tps"]
    naive = functools.reduce(operator.add, entry["d"], 0.0) / 4
    assert entry["mean_d"] == math.fsum(entry["d"]) / 4 == 0.025356989576906793
    assert naive == 0.025356989576906797 != entry["mean_d"]
    assert entry["rms_d"] == math.sqrt(math.fsum(d * d for d in entry["d"]) / 4)


def test_calibration_reports_per_pair_noise_without_a_decision():
    cell = calibration_cell(scale=0.02)
    assert cell["status"] == "calibrated" and not cell["passed"] and cell["regressions"] == []
    assert "noise_band" not in cell and "decision_rule" not in cell
    entry = cell["metrics"]["decode_tps"]
    assert len(entry["d"]) == 4 and entry["rms_d"] == pytest.approx(math.sqrt(sum(d * d for d in entry["d"]) / 4))
    assert "threshold" not in entry and cell["interpretation"] == compare.CALIBRATION_INTERPRETATION
    assert cell["order"] == ["r1/base", "r1/candidate", "r2/candidate", "r2/base",
                             "r3/base", "r3/candidate", "r4/candidate", "r4/base"]


def test_calibration_with_different_tokens_needs_review_and_two_revisions_are_invalid():
    assert calibration_cell(token_change=(3, "base"))["status"] == "needs-review"
    with pytest.raises(ValueError):
        decide(pairs=4, purpose="noise-calibration")


def test_calibration_selects_every_cell_for_one_revision_and_replays_exactly():
    value = impact.plan([], POLICY, base="a" * 40, head="a" * 40, candidate="a" * 40, calibration=True)
    assert value["purpose"] == "noise-calibration" and value["changed_paths"] == []
    assert len(impact.selected(value)) == 6
    assert all(cell["performance"] == dict(required=True, tiers=["512"], output_tokens=256) for cell in value["cells"])
    assert all(cell["numerical"] == impact.numerical_cases(cell["device"], [cell["model"]]) for cell in value["cells"])
    assert impact.validate(copy.deepcopy(value), POLICY) == value
    ordinary = impact.plan([], POLICY, base="a" * 40, head="a" * 40, candidate="a" * 40)
    assert not impact.selected(ordinary)
    for change in ("purpose", "path", "revision"):
        bad = copy.deepcopy(value)
        if change == "purpose": bad["purpose"] = "impact-selection"
        elif change == "path": bad["changed_paths"] = ["engine/src/llama.cpp"]
        else: bad["candidate"] = "c" * 40
        with pytest.raises(ValueError):
            impact.validate(bad, POLICY)
    with pytest.raises(ValueError):
        impact.plan(["engine/src/llama.cpp"], POLICY, base="a" * 40, head="a" * 40, candidate="a" * 40, calibration=True)
    with pytest.raises(ValueError):
        impact.plan([], POLICY, base="a" * 40, head="b" * 40, candidate="b" * 40, calibration=True)


def calibration_config(protocol, mode="functional-numerical-performance", plan=None):
    from ci import selection_contract as contract
    from test_ci_selection import template
    value = plan or impact.plan([], POLICY, base="a" * 40, head="a" * 40, candidate="a" * 40, calibration=True)
    config = template(value)
    config.update(validation_mode=mode, execution_devices=["cloud-5090"], performance_protocol=protocol, noise_bands={})
    for side in config["sides"].values():
        for data in side["models"].values():
            if mode == contract.FUNCTIONAL_MODE:
                data["benchmarks"] = {}
            for spec in data["benchmarks"].values():
                spec["preflight_argv"] = []
    manifest = dict(job_id="f" * 32, source_revisions={side: value[side] for side in ("base", "candidate")})
    return contract, config, manifest, value


def test_calibration_binds_only_with_its_protocol_and_stages_every_arm():
    from ci import selection_contract
    contract, config, manifest, value = calibration_config(selection_contract.performance_protocol(4))
    config, manifest = contract.bind(config, manifest, value, POLICY)
    assert {name for name in manifest["stages"] if "-benchmark-" in name} == {
        side + "-" + model + "-r" + str(k) + "-benchmark-512"
        for side in ("base", "candidate") for model in ("deepseek", "muse") for k in range(1, 5)}
    assert contract.timed_protocol(config)["samples_per_side"] == 4


@pytest.mark.parametrize("damage", ["gate-protocol", "functional", "calibration-for-change", "pairs-out-of-range"])
def test_calibration_and_gate_protocols_cannot_be_swapped(damage):
    from ci import selection_contract as contract
    protocol = contract.performance_protocol(4)
    mode, plan = "functional-numerical-performance", None
    if damage == "gate-protocol": protocol = contract.performance_protocol()
    elif damage == "functional": mode = contract.FUNCTIONAL_MODE
    elif damage == "calibration-for-change":
        plan = impact.plan(["engine/src/llama.cpp"], POLICY, base="a" * 40, head="b" * 40, candidate="c" * 40)
    else: protocol = dict(protocol, samples_per_side=5)
    contract, config, manifest, value = calibration_config(protocol, mode, plan)
    with pytest.raises(ValueError):
        contract.bind(config, manifest, value, POLICY)


def test_noise_band_tool_pools_calibration_pairs_into_a_valid_policy_band():
    first, second = calibration_cell("1" * 32, scale=0.02), calibration_cell("2" * 32, scale=0.04)
    report = dict(purpose="operator-selected-report-only", cells=[dict(
        device="cloud-5090", model="muse", status="calibrated", performance=[dict(second, tier="512", evidence={})])])
    bands, skipped = noise_band.bands([first, report], date="2026-10-29")
    value = bands["cloud-5090"]["muse"]["512"]
    pooled = first["metrics"]["decode_tps"]["d"] + second["metrics"]["decode_tps"]["d"]
    assert value["pairs"] == 8 and skipped == []
    assert value["sigma_log"]["decode_tps"] == math.sqrt(math.fsum(d * d for d in pooled) / 8)
    assert value["sigma_log"]["ttft_s"] == 0.0
    assert value["calibration_jobs"] == ["1" * 32, "2" * 32] and value["engine_revision"] == CELL["base"]
    assert value["conditions_sha256"] == CONDITIONS and value["date"] == "2026-10-29"
    # The printed object is exactly a valid operator bands file, usable by the gate.
    assert noise_bands(bands) == bands
    assert decide(bands=bands)["noise_band"]["status"] == "matched"


@pytest.mark.parametrize("damage", ["gate-result", "duplicate", "revisions", "conditions", "tampered-d",
                                    "unknown-input"])
def test_noise_band_tool_refuses_inputs_that_cannot_be_pooled(damage):
    first, second = calibration_cell("1" * 32), calibration_cell("2" * 32)
    if damage == "gate-result": second = decide()
    elif damage == "duplicate": second = copy.deepcopy(first)
    elif damage == "revisions": second["cell"] = dict(second["cell"], base="b" * 40, candidate="b" * 40)
    elif damage == "conditions": second["measurement_identity"] = dict(second["measurement_identity"], conditions_sha256="0" * 64)
    elif damage == "tampered-d": second["metrics"]["ttft_s"]["d"][0] = 0.5
    elif damage == "unknown-input": second = dict(cells="not a list")
    with pytest.raises((ValueError, KeyError, TypeError)):
        noise_band.bands([first, second], date="2026-10-29")


def test_short_failed_and_flagged_cells_are_skipped_with_a_note_not_dropped_or_fatal():
    good = [calibration_cell("1" * 32), calibration_cell("2" * 32)]
    short = calibration_cell("3" * 32, device="dgx-spark")
    # A host check flag during calibration needs review, and its d values are not pooled.
    results, receipts, compared, policy, protocol, bands = series(
        pairs=4, purpose="noise-calibration", cell=dict(CELL, candidate=CELL["base"], job_id="4" * 32))
    compared[2] = dict(compared[2], environment=dict(status="needs-review", reasons=["base:swap-activity"]))
    flagged = compare.compare_series(results, receipts, compared, policy, cell=dict(CELL, candidate=CELL["base"],
                                     job_id="4" * 32), protocol=protocol, bands=bands)
    assert flagged["status"] == "needs-review" and flagged["review_reasons"] == ["host:r3:base:swap-activity"]
    report = dict(purpose="operator-selected-report-only", errors=[dict(device="local-mbp", error="cleanup unconfirmed")],
                  cells=[dict(device="cloud-5090", model="muse", status="needs-review", performance=[flagged]),
                         dict(device="local-mbp", model="muse", status="failed", performance="not-verified",
                              error="cleanup unconfirmed"),
                         dict(device="dgx-spark", model="muse", status="deferred-not-in-this-run",
                              performance="deferred-not-in-this-run"),
                         dict(device="dgx-spark", model="tiny", status="not-required", reason="not selected")])
    result, skipped = noise_band.bands([*good, short, report], date="2026-10-29")
    assert result["cloud-5090"]["muse"]["512"]["pairs"] == 8 and "dgx-spark" not in result
    assert {(item["cell"], item["status"]) for item in skipped} == {
        ("cloud-5090/muse/512/" + "4" * 32, "needs-review"), ("local-mbp", "error"),
        ("local-mbp/muse", "failed"), ("dgx-spark/muse", "deferred-not-in-this-run"),
        ("dgx-spark/muse/512", "too-few-pairs")}
    assert any("4 pooled pairs; a band needs at least 8" in item["reasons"][0] for item in skipped)


def test_noise_band_cli_skips_flagged_cells_and_prints_the_bands_fragment(tmp_path, capsys):
    paths = []
    cells = (calibration_cell("1" * 32), calibration_cell("2" * 32), calibration_cell("5" * 32, token_change=(1, "candidate")))
    for index, cell in enumerate(cells):
        path = tmp_path / ("cell-" + str(index) + ".json")
        path.write_text(json.dumps(cell))
        paths.append(str(path))
    assert noise_band.main([*paths, "--date", "2026-10-30"]) == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["cloud-5090"]["muse"]["512"]["calibration_jobs"] == ["1" * 32, "2" * 32]
    assert "skipped cloud-5090/muse/512/" + "5" * 32 + " (needs-review: generated-tokens-differ)" in captured.err
    assert noise_band.main([paths[0]]) == 1  # 4 pairs alone are too few for a band
    assert "a band needs at least 8" in capsys.readouterr().err


def test_noise_band_cli_merges_an_existing_bands_file_without_losing_other_cells(tmp_path, capsys):
    existing = banded({("dgx-spark", "deepseek", "512"): "d" * 64, ("cloud-5090", "muse", "512"): "e" * 64})
    (tmp_path / "noise-bands.json").write_text(json.dumps(existing))
    cell, other = tmp_path / "cell.json", tmp_path / "other.json"
    cell.write_text(json.dumps(calibration_cell("1" * 32)))
    other.write_text(json.dumps(calibration_cell("2" * 32)))
    assert noise_band.main([str(cell), str(other), "--merge", str(tmp_path / "noise-bands.json"), "--date", "2026-10-30"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dgx-spark"] == existing["dgx-spark"]
    assert printed["cloud-5090"]["muse"]["512"]["conditions_sha256"] == CONDITIONS
    (tmp_path / "noise-bands.json").write_text(json.dumps({"other-device": {}}))
    assert noise_band.main([str(cell), str(other), "--merge", str(tmp_path / "noise-bands.json")]) == 1


@pytest.mark.parametrize("damage", ["unknown-device", "unknown-model", "unknown-tier", "not-object", "bad-band"])
def test_sealed_bands_are_keyed_strictly_by_cell(damage):
    value = {"unknown-device": {"other": {}}, "unknown-model": {"cloud-5090": {"tiny": {}}},
             "unknown-tier": {"cloud-5090": {"muse": {"2048": band(CONDITIONS)}}}, "not-object": [],
             "bad-band": {"cloud-5090": {"muse": {"512": band(CONDITIONS, pairs=3)}}}}[damage]
    with pytest.raises(ValueError):
        noise_bands(value)


@pytest.mark.parametrize("damage", [None, "envelope", "child-report", "config-edit"])
def test_aggregate_replays_with_exactly_the_bands_named_by_the_envelope(tmp_path, monkeypatch, damage):
    """Synthetic retained child: only the seal checks before the evidence replay are exercised."""
    from ci import selected_aggregate
    from ci.contract import digest, save
    bands = banded({("cloud-5090", "muse", "512"): CONDITIONS})
    contract, config, manifest, value = calibration_config(__import__("ci.selection_contract").selection_contract
                                                           .performance_protocol(4))
    config["noise_bands"] = bands
    config, manifest = contract.bind(config, manifest, value, POLICY)
    root = tmp_path / "cloud-5090" / ("f" * 32)
    root.mkdir(parents=True)
    save(root / "root-manifest.json", manifest)
    config["manifest_sha256"] = digest(root / "root-manifest.json")
    if damage == "config-edit":
        config["noise_bands"] = {}
    save(root / "selected-config.json", config)
    job = dict(device="cloud-5090", job_id="f" * 32, source_revision=value["candidate"],
               config_sha256=digest(root / "selected-config.json"), selection_plan_sha256=impact.fingerprint(value),
               control_revision="d" * 40, control_manifest_sha256="e" * 64)
    scope = contract.execution_scope(config)
    report = dict(status="calibrated", cleanup_confirmed=True, purpose="selected-integration",
                  selection_plan_sha256=impact.fingerprint(value), device="cloud-5090", job_id="f" * 32,
                  control_revision="d" * 40, control_manifest_sha256="e" * 64, selection_parent="1-1",
                  authorization_path="live-private-parent", performance_protocol=config["performance_protocol"],
                  noise_bands_sha256="0" * 64 if damage == "child-report" else impact.fingerprint(bands),
                  **scope, **{key: value[key] for key in ("base", "head", "candidate")})
    save(root / "integration.json", report)
    save(root / "job-finished.json", dict(cleanup_confirmed=True))
    save(tmp_path / "cloud-5090" / "dispatch.json", dict(selection_parent="1-1", revision=value["candidate"],
         **{key: job[key] for key in ("device", "job_id", "config_sha256", "selection_plan_sha256",
                                      "control_revision", "control_manifest_sha256")}))
    monkeypatch.setattr(selected_aggregate, "sealed", lambda *a: None)
    expected = "0" * 64 if damage == "envelope" else impact.fingerprint(bands)
    with pytest.raises((ValueError, OSError)) as caught:
        selected_aggregate.device_report(root, value, POLICY, job, dict(conclusion="success", run_attempt=1),
                                         scope, expected)
    # Editing bands after sealing already breaks the root-manifest config hash.
    expected_error = {None: None, "config-edit": "selected source binding differs"}.get(damage, "sealed noise bands differ")
    if expected_error is None:
        assert "performance-preflight.json" in str(caught.value)  # Seal accepted; fails at the evidence replay.
    else:
        assert expected_error in str(caught.value)
