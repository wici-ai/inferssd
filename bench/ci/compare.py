"""Block invalid evidence, changed output or slowdown; n=1 is not a variance test."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import compare_outputs
import run_final
from ci.contract import DEVICES, MODELS, digest, load, policy_read, runner_config, save

_spec = importlib.util.spec_from_file_location(
    "inferssd_comparison_test_gate", BENCH / "device/gate.py"
)
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)

METRICS = (
    "decode_first128_tps", "decode_second128_tps", "decode_tps", "ttft_s",
    "server_prefill_tps",
)
INTERPRETATION = (
    "One attempt per side/cell. Passed means no observed slowdown in these samples, "
    "not statistical proof of no regression. Any slowdown is inconclusive and blocks "
    "acceptance pending a bounded independent diagnosis; no automatic retries or "
    "selection of a best attempt. OS cache treatment is recorded, not assumed cold."
    " Different generated token sequences are inconclusive even when every metric improves."
)


def outcome(rows):
    statuses = {row["status"] for row in rows}
    if "failed" in statuses:
        return "failed"
    return "inconclusive" if "inconclusive" in statuses else "passed"


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
    if (comparable(base_receipt, comparison_binding, "base")
            != comparable(head_receipt, comparison_binding, "candidate")):
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
            "limit_ratio": 1.0,
            "passed": passed,
        }
        if not passed:
            failures.append(key)
    return {
        "passed": not failures and output_workload["same_tokens"],
        "status": "inconclusive" if failures or difference is not None else "passed",
        "evidence_valid": True,
        "metrics": metrics,
        # Retain the old field for consumers; these are observations, not a
        # claim that a single pair established a reproducible regression.
        "regressions": failures,
        "observed_slowdowns": failures,
        "diagnostic_required": bool(failures) or difference is not None,
        "output_workload": output_workload,
        "interpretation": INTERPRETATION,
        "actual_input_tokens": head["input_tokens"],
        "prefill_tps": {
            "base": base["metrics"]["server_prefill_tps"],
            "candidate": head["metrics"]["server_prefill_tps"],
        },
    }


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
        "interpretation": INTERPRETATION,
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
        "interpretation": INTERPRETATION,
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
