"""Rebuild selected acceptance from raw artifacts, never from a green job alone."""

from pathlib import Path, PurePosixPath

import run_final
from ci.compare import comparable, compare_pair
from ci.contract import digest, load
from ci.impact import CASES, MARKERS, fingerprint, validate
from ci.selection_contract import bound
from ci.staged_numerical import validate_csv
from ci.selected_preflight import retained


def sealed(root, report):
    manifest = root / "evidence-manifest.json"
    if digest(manifest) != report["evidence_manifest_sha256"]:
        raise ValueError("retained evidence manifest hash differs")
    files = load(manifest)["files"]
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in (".", "..") for part in path.parts):
            raise ValueError("evidence path escapes its artifact")
        target = root / relative
        if target.is_symlink() or digest(target) != expected:
            raise ValueError("selected raw evidence missing or changed")


def device_report(root, plan, policy, job, child):
    root = Path(root)
    report = load(root / "integration.json")
    sealed(root, report)
    dispatch = load(root.parent / "dispatch.json")
    if (not dispatch.get("selection_parent")
            or report.get("selection_parent") != dispatch["selection_parent"]
            or report.get("authorization_path") != "live-private-parent"):
        raise ValueError("manual operator execution cannot satisfy a selected parent")
    config = load(root / "selected-config.json")
    manifest = load(root / "root-manifest.json")
    needs = bound(config, manifest, job["job_id"])
    if (config["selection_plan"] != plan or digest(root / "selected-config.json") != job["config_sha256"]
            or digest(root / "root-manifest.json") != config["manifest_sha256"]):
        raise ValueError("selected artifact differs from root-staged configuration/plan")
    if (report.get("status") != "passed" or report.get("cleanup_confirmed") is not True
            or load(root / "job-finished.json").get("cleanup_confirmed") is not True
            or report.get("purpose") != "selected-integration"
            or report.get("selection_plan_sha256") != fingerprint(plan)
            or report.get("device") != job["device"] or report.get("job_id") != job["job_id"]
            or any(report.get(key) != plan[key] for key in ("base", "head", "candidate"))
            or report.get("control_revision") != job["control_revision"]
            or report.get("control_manifest_sha256") != job["control_manifest_sha256"]
            or child.get("conclusion") != "success" or child.get("run_attempt") != 1):
        raise ValueError("selected result/source/controller/cleanup identity is not a clean single attempt")
    if digest(root / "performance-preflight.json") != report.get("performance_preflight_sha256"):
        raise ValueError("selected preflight gate missing or changed")
    preflight = retained(root, config, needs)
    gate = load(root / "numerical/selected-numerical.json")
    if set(gate) != set(needs["numerical"]):
        raise ValueError("selected numerical cases missing or unexpected")
    for case, value in gate.items():
        target, operation, params = CASES[case]
        folder = root / "numerical" / case
        stage = load(folder / "stage.json")
        if (value.get("status") != "passed" or stage.get("returncode") != 0
                or stage.get("cleanup_confirmed") is not True
                or Path(value["binary"]["runtime_path"]).name != target):
            raise ValueError("selected real numerical stage failed or was not clean")
        raw = (folder / "stdout.log").read_bytes()
        if operation:
            count = validate_csv(raw, config["numerical"]["backend"], operation, params)
        else:
            marker = MARKERS[case]
            count = sum(line.startswith(marker) for line in raw.splitlines())
            if count != 1:
                raise ValueError("selected GPU oracle did not execute")
        if value["tests_executed"] != count:
            raise ValueError("numerical summary differs from actual executed rows")
    rows = []
    for model in needs["models"]:
        smoke = load(root / ("native-" + model) / "smoke.json")
        if (smoke.get("status") != "passed" or smoke.get("cleanup_confirmed") is not True
                or smoke.get("completion_requests") != 1):
            raise ValueError("selected functional model smoke missing or failed")
        response = load(root / ("native-" + model) / "response.json")
        if response.get("tokens_predicted") != 8 or response.get("stop_type") != "limit" or response.get("truncated") is not False:
            raise ValueError("selected model did not complete its actual short request")
        performance = []
        for tier in needs["performance"].get(model, {}).get("tiers", []):
            data, receipts = {}, {}
            for side in ("base", "candidate"):
                folder = root / model / tier / side
                arm = load(folder / "benchmark.json")
                if (arm.get("status") != "passed" or arm.get("cleanup_confirmed") is not True
                        or arm.get("completion_requests") != 1 or arm.get("revision") != plan[side]
                        or arm.get("tier") != tier or arm.get("model") != model):
                    raise ValueError("selected benchmark arm is missing, failed, repeated or unclean")
                spec = config["sides"][side]["models"][model]["benchmarks"][tier]
                data[side] = run_final.read_result(folder / "result.json", tier, spec["request_timeout_s"])
                receipts[side] = load(folder / "comparison-receipt.json")
                if comparable(receipts[side]) != comparable(preflight[model, tier, side]):
                    raise ValueError("timed runtime differs from retained short preflight")
                if arm["result"] != data[side]:
                    raise ValueError("measurement summary differs from complete raw SSE evidence")
            compared = compare_pair(data["base"], data["candidate"], receipts["base"], receipts["candidate"], policy)
            if load(root / model / tier / "comparison.json") != compared or compared["passed"] is not True:
                raise ValueError("selected performance failed or is inconclusive")
            performance.append(dict(tier=tier, **compared))
        rows.append(dict(device=job["device"], model=model, status="passed", functional="passed",
                         numerical="passed", performance=performance or "not-required"))
    return rows


def aggregate(plan, policy, artifacts, jobs, children):
    validate(plan, policy)
    rows, errors = [], []
    for cell in plan["cells"]:
        if not cell["required"]:
            rows.append(dict(device=cell["device"], model=cell["model"], status="not-required", reason=cell["reason"]))
    for device in dict.fromkeys(cell["device"] for cell in plan["cells"] if cell["required"]):
        try:
            rows += device_report(Path(artifacts) / device / jobs[device]["job_id"], plan, policy, jobs[device], children[device])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(device=device, error=str(error)))
    return dict(schema_version=1, purpose="selected-ci", passed=not errors,
                selection_plan_sha256=fingerprint(plan), cells=rows, errors=errors, formal_samples=0)
