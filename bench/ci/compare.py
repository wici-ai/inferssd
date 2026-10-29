"""Compare interleaved base/candidate pairs against calibrated noise bands.

Invalid evidence fails. A cell regresses only when its mean paired log ratio is
beyond the measured noise threshold and every pair is slower.
"""

import argparse
from decimal import Context, Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import compare_outputs
import run_final
from ci.contract import DEVICES, MODELS, PERFORMANCE_METRICS, digest, load, noise_bands, policy_read, runner_config, save

_spec = importlib.util.spec_from_file_location(
    "inferssd_comparison_test_gate", BENCH / "device/gate.py"
)
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)

METRICS = PERFORMANCE_METRICS
SIDES = ("base", "candidate")
ORDER = "ABBA"
REGRESSION_GATE = "regression-gate"
CALIBRATION = "noise-calibration"
# Most severe first. "calibrated" only appears in A/A calibration runs.
SEVERITY = ("failed", "regressed", "needs-review", "calibrated", "passed")
SUCCESS = ("passed", "calibrated")
# Only these metrics can make a cell regressed or out of band. The two decode
# half-windows are sub-windows of whole decode and are shown for information.
GATED_METRICS = ("decode_tps", "ttft_s", "server_prefill_tps")
# One-sided Student-t quantiles whose upper tail equals the normal 3-sigma tail
# (P = 0.99865), keyed by the band's pooled pairs (its degrees of freedom).
# Values are rounded up, a degree count between entries uses the nearest lower
# entry, and above 64 the normal value 3.0 applies. A fixed table keeps the
# multiplier identical on every device and in the hosted replay.
BAND_QUANTILE = "one-sided Student-t, P = 0.99865 (normal 3 sigma), df = pooled band pairs"
T_QUANTILES = ((8, 4.277), (9, 4.095), (10, 3.957), (11, 3.850), (12, 3.765), (13, 3.695),
               (14, 3.636), (15, 3.587), (16, 3.545), (18, 3.476), (20, 3.423), (24, 3.345),
               (28, 3.292), (32, 3.253), (40, 3.199), (48, 3.165), (56, 3.140), (64, 3.122))
NORMAL_QUANTILE = 3.0
LEGACY_INTERPRETATION = (
    "One base/candidate pair per cell, each arm a fresh server with one completion. Any "
    "slower metric or different generated tokens needs review; a single pair is an "
    "observation, not a statistical test. No automatic retries or selection of a best "
    "attempt. OS cache treatment is recorded, not assumed cold."
)
PAIR_INTERPRETATION = (
    "One base/candidate pair, each arm a fresh server with one completion. "
    "needs-review here marks any slower metric or different generated tokens in this "
    "pair alone; a cell is decided from all of its pairs together. OS cache treatment "
    "is recorded, not assumed cold."
)
INTERPRETATION = (
    "Each cell runs interleaved base/candidate pairs in ABBA order (base, candidate, "
    "candidate, base), each arm a fresh server with one completion. Every arm is kept; "
    "nothing is retried or selected. For each metric, d is the per-pair log ratio, signed "
    "so that a positive value means the candidate is slower. The cell is regressed when, "
    "for whole decode, TTFT or native prefill, the mean d is above the calibrated noise "
    "threshold and every pair is slower; the decode half-windows are shown for information. "
    "It needs review when such a threshold is exceeded but the pairs disagree, generated "
    "tokens differ, no noise band matches the measured conditions, or the host check "
    "flagged an arm. "
    "Otherwise it passes: no slowdown beyond measured noise, which is not proof that none "
    "exists. OS cache treatment is recorded, not assumed cold."
)
CALIBRATION_INTERPRETATION = (
    "Calibration run: base and candidate are the same revision, so d measures run-to-run "
    "noise. No pass or regression decision is made; the per-pair values feed the noise "
    "bands used by later comparisons."
)


def outcome(rows):
    statuses = {row["status"] for row in rows}
    unknown = statuses - set(SEVERITY)
    if unknown:
        raise ValueError("unknown cell status: " + ", ".join(sorted(map(str, unknown))))
    return next((status for status in SEVERITY if status in statuses), "passed")


def schedule(pairs):
    """Executed arm order: pair k runs base first when k is odd, candidate first otherwise."""
    if type(pairs) is not int or pairs < 1:
        raise ValueError("positive pair count required")
    return [(k, side) for k in range(1, pairs + 1) for side in (SIDES if k % 2 else SIDES[::-1])]


def arm_name(k, side):
    return "r" + str(k) + "/" + side


def band_multiplier(pairs):
    """Threshold multiplier for a band calibrated from `pairs` pooled A/A pairs."""
    if type(pairs) is not int or pairs < T_QUANTILES[0][0]:
        raise ValueError("noise band has too few pairs for a decision")
    if pairs > T_QUANTILES[-1][0]:
        return NORMAL_QUANTILE
    return [value for df, value in T_QUANTILES if df <= pairs][-1]  # nearest lower entry


def mean(values):
    """Correctly rounded everywhere: plain sum() of floats changed in Python 3.12."""
    return math.fsum(values) / len(values)


def rms(values):
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


# Device runs and the hosted replay use different C math libraries. Decimal's
# ln/exp are correctly rounded in software, so every derived value is
# bit-identical wherever the evidence is recomputed.
EXACT = Context(prec=40)


def ln(value):
    return float(EXACT.ln(Decimal(value)))


def exp(value):
    return float(EXACT.exp(Decimal(value)))


def log_ratio(metric, base, candidate):
    """Signed so that a positive value means the candidate is slower."""
    ratio = candidate / base
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("non-finite or underflowed metric ratio " + metric)
    value = ln(ratio)
    # Adding 0.0 turns a negated zero into plain 0.0 for readable reports.
    return (value if metric == "ttft_s" else -value) + 0.0


def validate_metrics(result):
    for metric in METRICS:
        number = result["metrics"].get(metric)
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError("missing or anomalous metric " + metric)
    native = result["timings"]
    prompt_ms = native.get("prompt_ms")
    if (type(prompt_ms) not in (int, float) or not math.isfinite(prompt_ms)
            or prompt_ms <= 0 or native.get("prompt_n") != result["input_tokens"]
            or result["metrics"]["server_prefill_tps"] != result["input_tokens"] * 1000 / prompt_ms):
        raise ValueError("prefill must come from matching native prompt tokens/time")


def comparable(receipt, comparison_binding=None, side=None):
    if comparison_binding is not None:
        from ci.reference import comparable as reference_comparable
        return reference_comparable(receipt, comparison_binding, side)
    host = receipt["host"]
    settings = dict(host["effective_settings"])
    settings.pop(
        "native_profile_sha256", None
    )  # actual branch binary/revision must differ
    if (
        "native_launch_policy" not in settings
        or "typed_values" not in settings["native_launch_policy"]
    ):
        raise ValueError(
            "comparison requires real typed configuration and exact launch policy"
        )
    return {
        "model": receipt["model_identity"],
        "host": {
            k: host[k]
            for k in (
                "hardware",
                "os",
                "memory_budget_bytes",
                "os_page_cache_treatment",
            )
        },
        "settings": settings,
        "image_names": sorted(
            Path(x["path"]).name for x in receipt["build"]["loaded_libraries"]
        ),
    }


def conditions_digest(conditions):
    """SHA-256 of canonical JSON, as for the plan and sealed noise bands.

    Applied to `comparable()` it is the measured-conditions identity that a
    noise band is calibrated under.
    """
    return hashlib.sha256(json.dumps(conditions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cell(root, model, side, tier, revision, device, *, expected_order=None):
    folder = root / model / side
    summary = load(folder / "summary.json")
    manifest = load(folder / "manifest.json")
    if (
        summary.get("purpose") != "ci"
        or manifest.get("purpose") != "ci"
        or summary.get("complete") is not True
        or summary.get("samples_per_tier") != 1
        or manifest.get("samples_per_tier") != 1
        or not isinstance(manifest.get("run_id"), str)
        or not manifest["run_id"]
        or summary.get("run_id") != manifest["run_id"]
        or summary.get("interrupted_signals") != []
        or summary.get("order") != manifest.get("order")
        or manifest.get("order") not in (["512"], ["512", "4096"])
        or (expected_order is not None and manifest["order"] != list(expected_order))
        or len(summary["rows"]) != len(summary["order"])
        or [row.get("tier") for row in summary["rows"]] != summary["order"]
    ):
        raise ValueError("incomplete, duplicate or non-CI attempt")
    attempts = [json.loads(line) for line in (folder / "attempts.jsonl").read_text().splitlines()]
    if attempts != summary["rows"]:
        raise ValueError("summary does not retain the complete attempt ledger")
    rows = [row for row in summary["rows"] if row.get("tier") == tier]
    if len(rows) != 1 or rows[0].get("status") != "passed":
        raise ValueError("missing or failed model/tier attempt")
    config = manifest["config"]
    if (
        config["revision"] != revision
        or config["device"] != device
        or config["model"] != model
    ):
        raise ValueError("measurement source/device/model identity mismatch")
    result = run_final.read_result(
        folder / tier / "result.json", tier, config["request_timeout_s"]
    )
    receipt = run_final.validate_receipt(folder / tier / "wrapper.json", config, tier)
    if rows[0].get("result") != result or rows[0].get(
        "wrapper_receipt_sha256"
    ) != digest(folder / tier / "wrapper.json"):
        raise ValueError("summary differs from verified raw measurement/receipt")
    if not rows[0].get("owned_group_cleanup_confirmed") or not rows[0].get(
        "wrapper_cleanup_confirmed"
    ):
        raise ValueError("cleanup was not confirmed")
    validate_metrics(result)
    return result, receipt


def compare_pair(base, head, base_receipt, head_receipt, policy, *, comparison_binding=None):
    policy_read(policy)
    validate_metrics(base)
    validate_metrics(head)
    conditions = comparable(base_receipt, comparison_binding, "base")
    if conditions != comparable(head_receipt, comparison_binding, "candidate"):
        raise ValueError(
            "resource, placement, model, template, page-cache treatment or image set changed"
        )
    if (
        base["prompt_ids"] != head["prompt_ids"]
        or base["prompt_sha256"] != head["prompt_sha256"]
    ):
        raise ValueError("authoritative template IDs or fixed prompt changed")
    if [base_receipt["server"]["pid"], base_receipt["server"]["start_identity"]] == [
        head_receipt["server"]["pid"],
        head_receipt["server"]["start_identity"],
    ]:
        raise ValueError("baseline and candidate reused a server process")
    for result in (base, head):
        tokens = result.get("tokens")
        if (type(result.get("output_tokens")) is not int or result["output_tokens"] != 256
                or not isinstance(tokens, list) or len(tokens) != result["output_tokens"]
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("performance comparison requires complete output token evidence")
    difference = next((i for i, (before, after) in enumerate(zip(base["tokens"], head["tokens"]))
                       if before != after), None)
    output_workload = {
        "same_tokens": difference is None,
        "first_difference_one_based": None if difference is None else difference + 1,
        "base_token": None if difference is None else base["tokens"][difference],
        "candidate_token": None if difference is None else head["tokens"][difference],
    }
    metrics, failures = {}, []
    for key in METRICS:
        before, after = base["metrics"][key], head["metrics"][key]
        ratio = after / before
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("non-finite or underflowed metric ratio " + key)
        passed = after <= before if key == "ttft_s" else after >= before
        metrics[key] = {
            "base": before,
            "candidate": after,
            "ratio": ratio,
            "change_percent": (ratio - 1) * 100,
            "direction": "lower-is-better" if key == "ttft_s" else "higher-is-better",
            "limit_ratio": 1.0,
            "passed": passed,
        }
        if not passed:
            failures.append(key)
    return {
        "passed": not failures and output_workload["same_tokens"],
        "status": "needs-review" if failures or difference is not None else "passed",
        "evidence_valid": True,
        "metrics": metrics,
        # Retain the old field for consumers; these are observations, not a
        # claim that a single pair established a reproducible regression.
        "regressions": failures,
        "observed_slowdowns": failures,
        "diagnostic_required": bool(failures) or difference is not None,
        "output_workload": output_workload,
        "interpretation": PAIR_INTERPRETATION,
        "actual_input_tokens": head["input_tokens"],
        "measurement_identity": {
            "prompt_sha256": base["prompt_sha256"],
            "prompt_ids_sha256": hashlib.sha256(json.dumps(base["prompt_ids"], separators=(",", ":")).encode()).hexdigest(),
            "input_tokens": base["input_tokens"],
            "output_tokens": 256,
            "conditions_sha256": conditions_digest(conditions),
        },
        "prefill_tps": {
            "base": base["metrics"]["server_prefill_tps"],
            "candidate": head["metrics"]["server_prefill_tps"],
        },
    }


def first_difference(reference, tokens):
    return next((i for i, (before, after) in enumerate(zip(reference, tokens)) if before != after), None)


def compare_series(results, receipts, pairs, policy, *, cell, protocol, bands):
    """Decide one cell from every planned arm; nothing is dropped, retried or selected.

    `results` and `receipts` map (pair, side) to each arm's verified measurement and
    comparison receipt. `pairs` holds the per-pair `compare_pair` results in pair
    order (with the Mac host check applied). `cell` names the device, model, tier,
    job and both source revisions. `bands` are the noise bands sealed into the job;
    a cell without one cannot pass or regress. Any inconsistency is invalid evidence.
    """
    policy_read(policy)
    noise_bands(bands)
    thresholds = policy["thresholds"]
    count, purpose = protocol.get("samples_per_side"), protocol.get("purpose")
    if (protocol.get("order") != ORDER or purpose not in (REGRESSION_GATE, CALIBRATION)
            or type(count) is not int or count < 2):
        raise ValueError("timed cells require the interleaved paired protocol")
    if set(cell) != {"device", "model", "tier", "job_id", "base", "candidate"}:
        raise ValueError("series cell identity is incomplete")
    if purpose == CALIBRATION and cell["base"] != cell["candidate"]:
        raise ValueError("noise calibration compares one revision with itself")
    if purpose == REGRESSION_GATE and (thresholds["pairs"] != count or thresholds["order"] != ORDER):
        raise ValueError("decision policy and executed protocol differ")
    order = schedule(count)
    if (set(results) != set(order) or set(receipts) != set(order)
            or not isinstance(pairs, list) or len(pairs) != count):
        raise ValueError("every planned arm and pair is required; none may be dropped or added")
    previous = None
    for key in order:
        start, end = results[key].get("start_ns"), results[key].get("end_ns")
        if (type(start) is not int or type(end) is not int or start > end
                or (previous is not None and start < previous)):
            raise ValueError("timed arms did not run in the declared interleaved order")
        previous = end
    servers = [json.dumps(receipts[key]["server"], sort_keys=True) for key in order]
    if len(set(servers)) != len(servers):
        raise ValueError("timed arms reused a server process")
    identity = pairs[0].get("measurement_identity")
    reasons, host = [], []
    for k, pair in enumerate(pairs, 1):
        if pair.get("evidence_valid") is not True or pair.get("measurement_identity") != identity:
            raise ValueError("pair evidence or measurement conditions differ within the cell")
        for metric in METRICS:
            if any(pair["metrics"][metric][side] != results[k, side]["metrics"][metric] for side in SIDES):
                raise ValueError("pair evidence differs from its retained arms")
        if pair["output_workload"]["same_tokens"] != (results[k, "base"]["tokens"] == results[k, "candidate"]["tokens"]):
            raise ValueError("pair token evidence differs from its retained arms")
        environment = pair.get("environment")
        if environment is not None:
            if environment.get("status") not in ("passed", "needs-review"):
                raise ValueError("unrecognized host check status")
            host.extend("r" + str(k) + ":" + reason for reason in environment["reasons"])
    reference = results[1, "base"]["tokens"]
    differences = []
    for k, side in order:
        index = first_difference(reference, results[k, side]["tokens"])
        if index is not None:
            differences.append(dict(arm=arm_name(k, side), first_difference_one_based=index + 1,
                                    reference_token=reference[index], token=results[k, side]["tokens"][index]))
    band = bands.get(cell["device"], {}).get(cell["model"], {}).get(cell["tier"])
    if purpose == CALIBRATION:
        noise = None
    elif band is None:
        noise = dict(status="missing")
        reasons.append("no-noise-band")
    else:
        noise = dict({key: value for key, value in band.items() if key != "sigma_log"},
                     status="matched" if band["conditions_sha256"] == identity["conditions_sha256"] else "conditions-differ")
        if noise["status"] != "matched":
            reasons.append("noise-band-conditions-differ")
    multiplier = band_multiplier(band["pairs"]) if noise is not None and noise["status"] == "matched" else None
    metrics, regressions = {}, []
    for metric in METRICS:
        base = [results[k, "base"]["metrics"][metric] for k in range(1, count + 1)]
        candidate = [results[k, "candidate"]["metrics"][metric] for k in range(1, count + 1)]
        d = [log_ratio(metric, before, after) for before, after in zip(base, candidate)]
        average = mean(d)
        entry = dict(direction="lower-is-better" if metric == "ttft_s" else "higher-is-better",
                     gated=metric in GATED_METRICS, base=base, candidate=candidate, d=d, mean_d=average,
                     change_percent=(exp(average if metric == "ttft_s" else -average) - 1) * 100,
                     all_pairs_worse=all(value > 0 for value in d))
        if purpose == CALIBRATION:
            entry["rms_d"] = rms(d)
        elif noise["status"] == "matched":
            sigma = band["sigma_log"][metric]
            floor = float(EXACT.ln(Decimal(1) + Decimal(thresholds["floor_fraction"][metric])))
            threshold = max(floor, multiplier * sigma / math.sqrt(count))
            beyond = average > threshold
            result = ("regressed" if entry["all_pairs_worse"] else "pairs-disagree") if beyond else "within-band"
            entry.update(sigma_log=sigma, threshold=threshold, beyond_band=beyond,
                         result=result if entry["gated"] else "informational")
            if entry["result"] == "regressed":
                regressions.append(metric)
            elif entry["result"] == "pairs-disagree":
                reasons.append("pairs-disagree:" + metric)
        else:
            entry.update(sigma_log=None, threshold=None, beyond_band=None,
                         result="no-matching-band" if entry["gated"] else "informational")
        metrics[metric] = entry
    if differences:
        reasons.append("generated-tokens-differ")
    reasons.extend("host:" + reason for reason in host)
    if purpose == CALIBRATION:
        status = "needs-review" if reasons else "calibrated"
    else:
        status = "regressed" if regressions else "needs-review" if reasons else "passed"
    result = dict(status=status, passed=status == "passed", evidence_valid=True, cell=dict(cell),
                  protocol=dict(protocol), order=[arm_name(k, side) for k, side in order],
                  statistic="mean-paired-log-ratio", metrics=metrics, regressions=regressions,
                  review_reasons=reasons,
                  output_workload=dict(same_tokens=not differences, reference_arm=arm_name(1, "base"),
                                       differences=differences),
                  measurement_identity=identity, pairs=list(pairs),
                  interpretation=CALIBRATION_INTERPRETATION if purpose == CALIBRATION else INTERPRETATION)
    if purpose == REGRESSION_GATE:
        result.update(noise_band=noise, decision_rule=dict(
            k=multiplier, k_quantile=BAND_QUANTILE, floor_fraction=thresholds["floor_fraction"], pairs=count,
            gated_metrics=list(GATED_METRICS),
            threshold="max(ln(1 + floor_fraction), k * sigma_log / sqrt(pairs))",
            noise_bands_sha256=conditions_digest(bands)))
    return result


def device_report(root, plan, policy):
    root = Path(root)
    manifest = load(root / "ci-manifest.json")
    device = manifest["device"]
    if plan.get("tiers") not in (["512"], ["512", "4096"]):
        raise ValueError("invalid or duplicate planned prompt tiers")
    if (
        device not in DEVICES
        or manifest.get("purpose") != "ci"
        or manifest.get("plan") != plan
        or manifest.get("policy_sha256") != policy["_sha256"]
    ):
        raise ValueError("CI artifact belongs to another source/policy/device plan")
    if digest(root / "runner-config.json") != plan["device_config_sha256"][device]:
        raise ValueError(
            "artifact device config is not the provisioned reviewed config"
        )
    config = runner_config(load(root / "runner-config.json"), device)
    bindings = config.get("comparison_bindings")
    if bindings is not None:
        from ci.reference import validate_binding
        if not isinstance(bindings, dict) or set(bindings) != {*MODELS, "tiny"}:
            raise ValueError("reference bindings must cover tiny and both performance models")
        for binding in bindings.values():
            validate_binding(binding)
            if any(binding["sides"][side]["revision"] != plan[side] for side in ("base", "candidate")):
                raise ValueError("reference binding differs from the current exact base/merge candidate")
    if "controller" in plan:
        from ci.isolated_device import REQUIRED_GATES
        isolated = load(root / "isolation-receipt.json")
        description = isolated.get("controller_isolation", {})
        if (isolated.get("plan") != plan or isolated.get("device") != device
                or isolated.get("status") != "passed" or isolated.get("cleanup_confirmed") is not True
                or description.get("proof_sha256") != config.get("isolation", {}).get("proof_sha256")
                or description.get("candidate_access") != "sandboxed-stages"
                or any(description.get("gates", {}).get(g) is not True for g in REQUIRED_GATES)):
            raise ValueError("missing or inconsistent protected broker execution/isolation receipt")
    for side in ("base", "candidate"):
        build = load(root / ("build-" + side + ".json"))
        if build.get("status") != "passed" or build.get("revision") != plan[side]:
            raise ValueError("missing exact-source build identity")
        for stage in ("build", "validation"):
            ownership = load(root / ("stage-" + stage + "-" + side) / "ownership.json")
            if ownership.get("status") != "passed" or any(
                ownership.get("server", {}).get(k) is not True
                for k in ("owned_processes_gone", "resource_scope_gone")
            ):
                raise ValueError("native build/test stage resource cleanup unconfirmed")
        validation = load(root / ("validation-" + side + ".json"))
        rebuilt = GATE.compare(
            GATE.read_results(root / ("validation-" + side + ".xml")),
            config["validation"]["baseline"],
        )
        if any(validation.get(k) != v for k, v in rebuilt.items()):
            raise ValueError(
                "validation summary differs from complete raw test results"
            )
        if (
            validation.get("passed") is not True
            or validation.get("counts", {}).get("skipped") != 0
        ):
            raise ValueError(
                "small-model numerical/lifecycle validation did not pass completely"
            )
        if (
            validation["revision"] != plan[side]
            or validation.get("cleanup_confirmed") is not True
        ):
            raise ValueError("validation source or cleanup mismatch")
    if (
        load(root / "build-base.json")["toolchain"]
        != load(root / "build-candidate.json")["toolchain"]
    ):
        raise ValueError("actual toolchain changed between base/candidate")
    a, ar = cell(root, "tiny", "base", "512", plan["base"], device, expected_order=["512"])
    b, br = cell(root, "tiny", "candidate", "512", plan["candidate"], device, expected_order=["512"])
    numerical = compare_outputs.compare(a, b, max_input_tokens=len(a["prompt_ids"]))
    tiny_binding = bindings["tiny"] if bindings is not None else None
    if (
        comparable(ar, tiny_binding, "base") != comparable(br, tiny_binding, "candidate")
        or not numerical["exact"]
        or load(root / "tiny-numerical.json") != numerical
    ):
        raise ValueError(
            "tiny numerical gate did not reproduce exact tokens/probabilities"
        )
    rows = []
    for model in MODELS:
        for tier in plan["tiers"]:
            row = {
                "device": device,
                "model": model,
                "tier": tier,
                "samples_per_side": 1,
            }
            try:
                base, base_receipt = cell(
                    root, model, "base", tier, plan["base"], device, expected_order=plan["tiers"]
                )
                head, head_receipt = cell(
                    root, model, "candidate", tier, plan["candidate"], device, expected_order=plan["tiers"]
                )
                binding = bindings[model] if bindings is not None else None
                row.update(compare_pair(base, head, base_receipt, head_receipt, policy, comparison_binding=binding))
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(passed=False, status="failed", evidence_valid=False,
                           error=str(error), anomalous=True)
            rows.append(row)
    return {
        "schema_version": 1,
        "purpose": "ci",
        "device": device,
        "plan": plan,
        "passed": all(x["passed"] for x in rows),
        "status": outcome(rows),
        "rows": rows,
        "thresholds": policy["thresholds"],
        "interpretation": LEGACY_INTERPRETATION,
    }


def aggregate(roots, plan, policy):
    reports = [device_report(root, plan, policy) for root in roots]
    if sorted(x["device"] for x in reports) != sorted(DEVICES):
        raise ValueError(
            "merge gate requires exactly one report from every real device"
        )
    return {
        "schema_version": 1,
        "purpose": "ci",
        "passed": all(x["passed"] for x in reports),
        "status": outcome(reports),
        "plan": plan,
        "devices": reports,
        "thresholds": policy["thresholds"],
        "formal_samples": 0,
        "interpretation": LEGACY_INTERPRETATION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--device-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        policy = policy_read(load(args.policy))
        policy["_sha256"] = digest(args.policy)
        report = aggregate(args.device_result, load(args.plan), policy)
        save(args.output, report)
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        save(
            args.output,
            {
                "purpose": "ci",
                "passed": False,
                "status": "failed",
                "error": str(error),
                "formal_samples": 0,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
