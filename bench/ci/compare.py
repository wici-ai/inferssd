"""Decide performance cells from repeated runs against device baseline history.

Invalid evidence fails. A cell regresses only when the candidate mean of a
deciding metric is beyond the noise threshold measured on the same device and
every candidate run is slower than the base estimate.
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
from ci.contract import DEVICES, MODELS, PERFORMANCE_METRICS, digest, history_record, load, policy_read, runner_config, save

_spec = importlib.util.spec_from_file_location(
    "inferssd_comparison_test_gate", BENCH / "device/gate.py"
)
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)

METRICS = PERFORMANCE_METRICS
SIDES = ("base", "candidate")
REGRESSION_GATE = "regression-gate"
BASELINE = "baseline"
# Most severe first. "baseline" only appears in baseline jobs of a main commit.
SEVERITY = ("failed", "regressed", "needs-review", "baseline", "passed")
SUCCESS = ("passed", "baseline")
# Only these metrics can make a cell regressed or needs-review on their values.
# The two decode half-windows are sub-windows of whole decode, shown for information.
GATED_METRICS = ("decode_tps", "ttft_s", "server_prefill_tps")
# One-sided Student-t quantiles whose upper tail equals the normal 3-sigma tail
# (P = 0.99865), keyed by degrees of freedom of the pooled baseline variance.
# Values are rounded up, a count between entries uses the nearest lower entry,
# and above 128 the normal value 3.0 applies. A fixed table keeps the
# multiplier identical on every device and in the hosted replay.
# The anchor check allows the in-run base at least this drift from its history:
# the runs of one baseline job agree more closely than two jobs hours apart.
ANCHOR_FLOOR_FRACTION = "0.02"
T_QUANTILE = "one-sided Student-t, P = 0.99865 (normal 3 sigma), df = pooled baseline degrees of freedom"
T_QUANTILES = ((3, 9.219), (4, 6.621), (5, 5.508), (6, 4.905), (7, 4.530), (8, 4.277), (9, 4.095),
               (10, 3.957), (11, 3.850), (12, 3.765), (13, 3.695), (14, 3.636), (15, 3.587),
               (16, 3.545), (18, 3.476), (20, 3.423), (24, 3.345), (28, 3.292), (32, 3.253),
               (40, 3.199), (48, 3.165), (56, 3.140), (64, 3.122), (72, 3.108), (80, 3.097),
               (96, 3.081), (128, 3.060))
NORMAL_QUANTILE = 3.0
LEGACY_INTERPRETATION = (
    "One base/candidate pair per cell, each arm a fresh server with one completion. Any "
    "slower metric or different generated tokens needs review; a single pair is an "
    "observation, not a statistical test. No automatic retries or selection of a best "
    "attempt. OS cache treatment is recorded, not assumed cold."
)
PAIR_INTERPRETATION = (
    "One base/candidate pair, each arm a fresh server with one completion. Any slower "
    "metric or different generated tokens needs review; a single pair is an observation, "
    "not a statistical test. OS cache treatment is recorded, not assumed cold."
)
INTERPRETATION = (
    "Each cell runs the candidate three times and the base once, in the order candidate, "
    "base, candidate, candidate, every run a fresh server with one completion. All runs are "
    "kept; nothing is retried or selected. The base is also measured four times by the "
    "baseline job of the base commit on the same device. Values are compared as natural "
    "logs, signed so that a positive difference means the candidate is slower. The cell is "
    "regressed when, for whole decode, TTFT or native prefill, the candidate mean is above "
    "the base estimate by more than the noise threshold and every candidate run is slower; "
    "the decode half-windows are shown for information. It needs review when the in-run "
    "base differs from its baseline history, the threshold is exceeded but the candidate "
    "runs disagree, generated tokens differ, no baseline history matches the measured "
    "conditions, or the host check flagged a run. Otherwise it passes: no slowdown beyond "
    "measured noise, which is not proof that none exists. A deciding metric whose candidate "
    "mean is below the base estimate by more than the same threshold is labelled improved "
    "(faster beyond noise); within the threshold either way it is within noise. The label "
    "does not change the status. Where a side's launch was derived on the device, that side's "
    "own launch strategy chose its settings and thread pools from the machine characteristics "
    "of the device's most recent baseline job; what the PR's launch sets differently is listed "
    "with the cell, and it is not a reason to refuse the comparison. OS cache treatment is "
    "recorded, not assumed cold."
)
BASELINE_INTERPRETATION = (
    "Baseline job: one build of a main commit measured four times, every run a fresh "
    "server with one completion. No pass or regression decision is made; the verified "
    "values join the device's baseline history used by later comparisons."
)


def outcome(rows):
    statuses = {row["status"] for row in rows}
    unknown = statuses - set(SEVERITY)
    if unknown:
        raise ValueError("unknown cell status: " + ", ".join(sorted(map(str, unknown))))
    return next((status for status in SEVERITY if status in statuses), "passed")


def schedule(order):
    """Executed (run index per side, side) sequence for a protocol's side order."""
    if not isinstance(order, list) or not order or any(side not in SIDES for side in order):
        raise ValueError("explicit timed run order required")
    seen = {side: 0 for side in SIDES}
    result = []
    for side in order:
        seen[side] += 1
        result.append((seen[side], side))
    return result


def arm_name(k, side):
    return "r" + str(k) + "/" + side


def t_quantile(df):
    """Threshold multiplier for a pooled variance with `df` degrees of freedom."""
    if type(df) is not int or df < T_QUANTILES[0][0]:
        raise ValueError("baseline history has too few degrees of freedom for a decision")
    if df > T_QUANTILES[-1][0]:
        return NORMAL_QUANTILE
    return [value for key, value in T_QUANTILES if key <= df][-1]  # nearest lower entry


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
    """Conditions two runs must share to be compared.

    A run whose launch was derived on the device carries `derived_launch`: its
    side's own strategy chose the settings, thread pools and server arguments
    from the job's one characteristics file. Those choices are results to report,
    so the typed values and argv leave the comparison; the launch request, the
    template, the machine characteristics and everything else must still match.
    """
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
    conditions = {
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
    derived = receipt.get("derived_launch")
    if derived is not None:
        policy = settings["native_launch_policy"]
        if {key: entry["value"] for key, entry in derived["settings"].items()} != policy["typed_values"]:
            raise ValueError("derived launch settings differ from the typed values the server ran")
        settings["native_launch_policy"] = {key: value for key, value in policy.items() if key not in ("args", "typed_values")}
        conditions["machine_characteristics_sha256"] = derived["characteristics_sha256"]
    return conditions


def machine_policy(resources):
    """The job's machine policy as the device template's `host.resources` states it: memory budget and CPU set.

    On Linux the budget is the cgroup memory limit and the memory lock limit,
    and the CPU set is the allowed CPU list. The Mac executor has a sampled
    memory budget and pins no CPUs. What a platform does not state is None.
    """
    return {"memory_budget_bytes": resources.get("memory_budget_bytes"),
            "memory_max": resources.get("cgroup_limits", {}).get("memory_max"),
            "memlock_bytes": resources.get("memlock_bytes"), "cpus": resources.get("cpus_allowed_list")}


def history_conditions(receipt):
    """What baseline history is keyed on: the machine, its OS, model files, chat template and machine policy.

    Settings are not part of it. Each commit derives its own from the machine
    characteristics, so the history of main stays usable for a PR whose
    strategy derives different settings. The memory budget and CPU set they are
    derived within are part of it: a change to either starts a new history.
    """
    host = receipt["host"]
    return {"model": receipt["model_identity"], "hardware": host["hardware"], "os": host["os"],
            "template_sha256": host["effective_settings"]["native_launch_policy"]["template_sha256"],
            "machine_policy": machine_policy(host["effective_settings"]["native_resource_policy"])}


def conditions_digest(conditions):
    """SHA-256 of canonical JSON, as for the plan and the sealed baseline history.

    Applied to `history_conditions()` it is the identity a baseline record was
    measured under.
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
        "same_tokens_required": REQUIRE_SAME_TOKENS,
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
        "passed": not failures and (output_workload["same_tokens"] or not REQUIRE_SAME_TOKENS),
        "status": "needs-review" if failures or (difference is not None and REQUIRE_SAME_TOKENS) else "passed",
        "evidence_valid": True,
        "metrics": metrics,
        # Retain the old field for consumers; these are observations, not a
        # claim that a single pair established a reproducible regression.
        "regressions": failures,
        "observed_slowdowns": failures,
        "diagnostic_required": bool(failures) or (difference is not None and REQUIRE_SAME_TOKENS),
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


# Generated tokens are recorded for every run but, for now, not required to match.
# Where the CUDA GPU tier computes experts, the experts it holds when each is needed
# depend on timing, so runs of one build can generate different tokens. A
# difference is reported in output_workload and does not make a cell need review.
REQUIRE_SAME_TOKENS = False


def first_difference(reference, tokens):
    return next((i for i, (before, after) in enumerate(zip(reference, tokens)) if before != after), None)


def signed_log(metric, value):
    """Natural log of a measured value, signed so that a larger result is worse."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid measured value for " + metric)
    log = ln(value)
    return (log if metric == "ttft_s" else -log) + 0.0


def tokens_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def history_checked(records):
    """Validate sealed baseline records; their logs are recomputed, never trusted."""
    if not isinstance(records, list):
        raise ValueError("sealed baseline history must be a list of records")
    seen = set()
    for record in records:
        history_record(record)
        key = tuple(record[name] for name in ("device", "model", "tier", "commit", "conditions_sha256"))
        if key in seen:
            raise ValueError("sealed baseline history repeats a record")
        seen.add(key)
        for run in record["runs"]:
            if any(run["logs"][metric] != ln(run["metrics"][metric]) for metric in METRICS):
                raise ValueError("baseline record logs differ from its measured values")
    return records


def pooled(records, metric):
    """Pooled within-commit standard deviation of the log values, and its degrees of freedom.

    Each record holds the four runs of one commit's baseline job.
    """
    squares, df = [], 0
    for record in records:
        logs = [run["logs"][metric] for run in record["runs"]]
        centre = mean(logs)
        squares.extend((value - centre) * (value - centre) for value in logs)
        df += len(logs) - 1
    if df < 1:
        raise ValueError("pooled baseline variance needs at least two runs of one commit")
    return math.sqrt(math.fsum(squares) / df), df


def arms_checked(results, receipts, order):
    """Every planned run is present, ran in order on its own server, under equal conditions."""
    keys = schedule(order)
    if set(results) != set(keys) or set(receipts) != set(keys):
        raise ValueError("every planned timed run is required; none may be dropped or added")
    previous = None
    for key in keys:
        start, end = results[key].get("start_ns"), results[key].get("end_ns")
        if (type(start) is not int or type(end) is not int or start > end
                or (previous is not None and start < previous)):
            raise ValueError("timed runs did not run in the declared order")
        previous = end
    servers = [json.dumps(receipts[key]["server"], sort_keys=True) for key in keys]
    if len(set(servers)) != len(servers):
        raise ValueError("timed runs reused a server process")
    first, conditions = results[keys[0]], None
    for key in keys:
        result = results[key]
        validate_metrics(result)
        tokens = result.get("tokens")
        if (type(result.get("output_tokens")) is not int or result["output_tokens"] != 256
                or not isinstance(tokens, list) or len(tokens) != 256
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("performance comparison requires complete output token evidence")
        if result["prompt_ids"] != first["prompt_ids"] or result["prompt_sha256"] != first["prompt_sha256"]:
            raise ValueError("authoritative template IDs or fixed prompt changed between runs")
        current = comparable(receipts[key])
        if conditions is not None and current != conditions:
            raise ValueError("resource, placement, model, template, page-cache treatment or image set changed between runs")
        conditions = current
    return {"prompt_sha256": first["prompt_sha256"],
            "prompt_ids_sha256": hashlib.sha256(json.dumps(first["prompt_ids"], separators=(",", ":")).encode()).hexdigest(),
            "input_tokens": first["input_tokens"], "output_tokens": 256,
            "conditions_sha256": conditions_digest(history_conditions(receipts[keys[0]]))}


def launches_checked(receipts, keys):
    """Each side's derived launch, the same for all of that side's runs; None for fixed launches."""
    launches = {}
    for k, side in keys:
        value = receipts[k, side].get("derived_launch")
        if launches.setdefault(side, value) != value:
            raise ValueError("runs of one side used different derived launches")
    if all(value is None for value in launches.values()):
        return None
    from ci.staged_launch import changes
    changed = changes(launches["base"], launches["candidate"]) if len(launches) == 2 else []
    return dict(sides=launches, changes=changed)


def decide_cell(results, receipts, policy, *, cell, protocol, history, environment=None):
    """Decide one cell from all of its timed runs and the sealed baseline history.

    `results` and `receipts` map (run index, side) to each run's verified measurement
    and comparison receipt. `history` is the list of baseline records sealed into the
    job. `environment` is the Mac host check over all runs, or None. Nothing is dropped,
    retried or selected; any inconsistency is invalid evidence.
    """
    policy_read(policy)
    thresholds = policy["thresholds"]
    history_checked(history)
    purpose = protocol.get("purpose")
    if purpose not in (REGRESSION_GATE, BASELINE):
        raise ValueError("timed cells require the gate or baseline protocol")
    if set(cell) != {"device", "model", "tier", "job_id", "base", "candidate"}:
        raise ValueError("cell identity is incomplete")
    if purpose == BASELINE and cell["base"] != cell["candidate"]:
        raise ValueError("a baseline job measures one main commit")
    order = protocol["order"]
    keys = schedule(order)
    identity = arms_checked(results, receipts, order)
    launches = launches_checked(receipts, keys)
    reference = results[keys[0]]["tokens"]
    differences = []
    for k, side in keys:
        index = first_difference(reference, results[k, side]["tokens"])
        if index is not None:
            differences.append(dict(run=arm_name(k, side), first_difference_one_based=index + 1,
                                    reference_token=reference[index], token=results[k, side]["tokens"][index]))
    host = []
    if environment is not None:
        if environment.get("status") not in ("passed", "needs-review") or not isinstance(environment.get("reasons"), list):
            raise ValueError("unrecognized host check result")
        host = ["host:" + reason for reason in environment["reasons"]]
    same_cell = [record for record in history if (record["device"], record["model"], record["tier"])
                 == (cell["device"], cell["model"], cell["tier"])]
    # The noise pool: every record of this cell measured under the same conditions.
    matching = sorted((record for record in same_cell if record["conditions_sha256"] == identity["conditions_sha256"]),
                      key=lambda record: (record["measured_at"], record["commit"]))
    floor = {metric: float(EXACT.ln(Decimal(1) + Decimal(thresholds["floor_fraction"][metric]))) for metric in METRICS}
    anchor_floor = float(EXACT.ln(Decimal(1) + Decimal(ANCHOR_FLOOR_FRACTION)))
    metrics, regressions, improvements, blocking, reasons = {}, [], [], [], []
    if purpose == BASELINE:
        runs = [results[k, side] for k, side in keys]
        prior = [record for record in matching if record["commit"] != cell["candidate"]]
        current = dict(runs=[dict(logs={metric: ln(run["metrics"][metric]) for metric in METRICS}) for run in runs])
        for metric in METRICS:
            logs = [run["logs"][metric] for run in current["runs"]]
            sigma, df = pooled([*prior, current], metric)
            metrics[metric] = dict(direction="lower-is-better" if metric == "ttft_s" else "higher-is-better",
                                   gated=metric in GATED_METRICS, values=[run["metrics"][metric] for run in runs],
                                   logs=logs, mean_log=mean(logs), sigma_log=sigma, df=df)
        pool = [dict(commit=record["commit"], job_id=record["job_id"], measured_at=record["measured_at"]) for record in prior]
    else:
        base_record = next((record for record in matching if record["commit"] == cell["base"]), None)
        if base_record is None and matching and not any(record["commit"] == cell["base"] for record in same_cell):
            # A main commit that changed no performance-relevant file has no baseline
            # job of its own: its PRs compare against the cell's most recent record.
            base_record = matching[-1]
        pool = []
        if base_record is None:
            blocking.append("baseline-conditions-differ" if any(record["commit"] == cell["base"] for record in same_cell)
                            else "no-baseline-history")
        else:
            pool = [dict(commit=record["commit"], job_id=record["job_id"], measured_at=record["measured_at"])
                    for record in matching]
            if any(run["tokens_sha256"] != tokens_digest(reference) for run in base_record["runs"]):
                differences.append(dict(run="baseline-history", tokens_sha256=[run["tokens_sha256"] for run in base_record["runs"]]))
        candidates = [k for k, side in keys if side == "candidate"]
        for metric in METRICS:
            entry = dict(direction="lower-is-better" if metric == "ttft_s" else "higher-is-better",
                         gated=metric in GATED_METRICS, base_run=results[1, "base"]["metrics"][metric],
                         candidate=[results[k, "candidate"]["metrics"][metric] for k in candidates],
                         history=None if base_record is None else [run["metrics"][metric] for run in base_record["runs"]])
            if base_record is not None:
                history_logs = [signed_log(metric, value) for value in entry["history"]]
                anchor = signed_log(metric, entry["base_run"])
                candidate_logs = [signed_log(metric, value) for value in entry["candidate"]]
                sigma, df = pooled(matching, metric)
                k = t_quantile(df)
                base_estimate = mean([*history_logs, anchor])
                difference = mean(candidate_logs) - base_estimate
                anchor_limit = max(anchor_floor, k * sigma * math.sqrt(1 + 1 / len(history_logs)))
                standard_error = sigma * math.sqrt(1 / len(candidate_logs) + 1 / (len(history_logs) + 1))
                threshold = max(floor[metric], k * standard_error)
                anchored = abs(anchor - mean(history_logs)) <= anchor_limit
                beyond = difference > threshold
                worse = all(value > base_estimate for value in candidate_logs)
                # Faster beyond the same threshold is reported, never a status of its own.
                improved = difference < -threshold
                result = (("regressed" if worse else "candidates-disagree") if beyond
                          else "improved" if improved else "within-noise")
                entry.update(history_mean_log=mean(history_logs), base_run_log=anchor, candidate_logs=candidate_logs,
                             base_estimate=base_estimate, difference=difference,
                             change_percent=(exp(difference if metric == "ttft_s" else -difference) - 1) * 100,
                             sigma_log=sigma, df=df, k=k, anchor_limit=anchor_limit, base_run_matches_history=anchored,
                             threshold=threshold, beyond_threshold=beyond, all_candidates_worse=worse,
                             improved_beyond_threshold=improved,
                             result=result if entry["gated"] else "informational")
                if entry["gated"] and not anchored:
                    blocking.append("base-run-differs-from-baseline-history:" + metric)
                if entry["result"] == "regressed":
                    regressions.append(metric)
                elif entry["result"] == "improved":
                    improvements.append(metric)
                elif entry["result"] == "candidates-disagree":
                    reasons.append("candidates-disagree:" + metric)
            else:
                entry.update(threshold=None, result="no-baseline-history" if entry["gated"] else "informational")
            metrics[metric] = entry
    if differences and REQUIRE_SAME_TOKENS:
        reasons.append("generated-tokens-differ")
    reasons.extend(host)
    if purpose == BASELINE:
        status = "needs-review" if reasons else "baseline"
    else:
        # A device whose base no longer matches its history cannot establish a regression.
        status = "needs-review" if blocking else "regressed" if regressions else "needs-review" if reasons else "passed"
    result = dict(status=status, passed=status == "passed", evidence_valid=True, cell=dict(cell),
                  protocol=dict(protocol), order=[arm_name(k, side) for k, side in keys],
                  statistic="pooled-within-commit-log-variance", metrics=metrics, regressions=regressions,
                  improvements=improvements,
                  review_reasons=[*blocking, *reasons],
                  output_workload=dict(same_tokens=not differences, same_tokens_required=REQUIRE_SAME_TOKENS,
                                       reference_run=arm_name(*keys[0]),
                                       tokens_sha256=tokens_digest(reference), differences=differences),
                  measurement_identity=identity, history_pool=pool,
                  **({} if purpose == BASELINE else dict(
                      baseline_commit=None if base_record is None else base_record["commit"])),
                  decision_rule=dict(t_quantile=T_QUANTILE, floor_fraction=thresholds["floor_fraction"],
                                     gated_metrics=list(GATED_METRICS),
                                     baseline_history_sha256=conditions_digest(history),
                                     threshold="max(ln(1 + floor_fraction), t_df * sigma * sqrt(1/candidate runs + 1/(history runs + 1)))",
                                     anchor="|base run - history mean| <= max(ln(1 + " + ANCHOR_FLOOR_FRACTION
                                            + "), t_df * sigma * sqrt(1 + 1/history runs))"),
                  interpretation=BASELINE_INTERPRETATION if purpose == BASELINE else INTERPRETATION)
    if launches is not None:
        result["derived_launch"] = launches
    if environment is not None:
        result["environment"] = environment
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
