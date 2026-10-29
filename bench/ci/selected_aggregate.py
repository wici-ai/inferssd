"""Rebuild selected acceptance from raw artifacts, never from a green job alone."""

from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

import run_final
from ci.compare import BASELINE, SEVERITY, SUCCESS, arm_name, comparable, decide_cell, outcome, schedule
from ci.contract import digest, load
from ci.impact import CASES, MARKERS, fingerprint, validate
from ci.selection_contract import (arm_prefix, bound, execution_scope, FUNCTIONAL_MODE, PERFORMANCE_MODE,
                                   baseline_build_identity, sealed_history, timed_protocol)
from ci.staged_numerical import validate_csv
from ci.staged_benchmark import verify_stopped, verify_runtime
from ci.staged_checkpoint import mapped_files, native_settings
from ci.selected_preflight import arm_launch, launch as preflight_launch, retained
from ci.selected_receipt import (retained_typed_settings, mac_job_root, mac_runtime_images, mac_stopped,
                                 MAC_RESOURCE_ENFORCEMENT, normalize)
from ci import model_admission, mac_host_gate, staged_launch


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


def functional_retained(root, config, model, cell):
    """Recheck retained root probes; do not pretend to observe the device here."""
    folder = root / ("native-" + model)
    if digest(folder / "runtime-receipt.json") != cell.get("runtime_receipt_sha256"):
        raise ValueError("functional runtime receipt missing or changed")
    return runtime_retained(root, config, "candidate", model, folder,
                            config["sides"]["candidate"]["models"][model]["smoke"],
                            folder / "runtime-receipt.json")


def runtime_retained(root, config, side, model, folder, launch, receipt_path):
    """Bind each retained timed/preflight receipt to its actual root observations."""
    verify_stopped(load(folder / "stopped.json"))
    runtime, started = load(receipt_path), load(folder / "started.json")
    build = load(root / ("build-" + side) / "build-receipt.json")
    data = config["sides"][side]["models"][model]
    resources = config["host"]["resources"]
    effective = runtime["host"]["effective_settings"]
    identity = data["model_identity"]
    if model_admission.enabled(config):
        identity = dict(identity, readonly_admission=model_admission.binding(config))
    revision = config["sides"][side].get("revision", config["revision"])
    if (runtime["model_identity"] != identity or runtime["build"]["revision"] != revision
            or runtime["server"] != dict(pid=started["worker_pid"], start_identity=started["birth_identity"])
            or runtime["host"]["hardware"] != config["host"]["hardware"]
            or runtime["host"]["os"] != config["host"]["os"]
            or effective["native_resource_policy"] != resources
            or effective["native_launch_policy"]["template_sha256"] != data["template"]["sha256"]
            or not retained_typed_settings(native_settings(folder / "stderr.log")["values"],
                effective["native_launch_policy"]["typed_values"], config["sides"][side], model, build,
                root / ("build-" + side) / "freeze-manifest.json", config.get("manifest_path"))):
        raise ValueError("retained typed inputs/host/receipt identity differs")
    resolved = data.get("derived")
    if resolved is not None and (
            effective["native_launch_policy"]["typed_values"] != normalize(resolved["effective"]["values"],
                                                                         config["sides"][side]["paths"]["frozen"])
            or runtime.get("derived_launch") != staged_launch.summary(config, side, model, resolved)):
        raise ValueError("retained receipt differs from the side's resolved launch")
    if config.get("paths", {}).get("platform") == "darwin":
        mac_host_gate.retained_arm(folder, mac_host_gate.policy(config))
        manifest = load(root / "root-manifest.json")
        probes = [load(folder / name) for name in ("runtime-before.json", "runtime-after.json")]
        images = mac_runtime_images(config, manifest, side, model, launch, started, probes)[0]
        mac_stopped(config, manifest, load(folder / "stopped.json"))
        expected = [{"path": image["runtime_path"], "sha256": image["sha256"]}
                    for image in [build["binary"], *build["images"]] if image["runtime_path"] in images]
        if (build["binary"]["runtime_path"] not in images
                or sorted(runtime["build"]["loaded_libraries"], key=lambda x: x["path"]) != sorted(expected, key=lambda x: x["path"])
                or effective["ambient_environment"] != launch["environment"]
                or effective.get("native_resource_enforcement") != MAC_RESOURCE_ENFORCEMENT):
            raise ValueError("retained Mac loaded images or declared environment differs")
        known = {image["path"] for image in expected}
        systems = effective.get("system_images", [])
        if {image["path"] for image in systems} != set(images) - known or any(not image["path"].startswith(
                ("/usr/", "/lib/", "/lib64/", "/System/Library/", "/Library/Apple/", "/opt/inferssd-ci-toolchains/"))
                for image in systems):
            raise ValueError("retained Mac system-image inventory differs")
        return runtime
    previous = None
    for name in ("runtime-before.json", "runtime-after.json"):
        probe = load(folder / name)
        verify_runtime(SimpleNamespace(**started), probe)
        if (probe["cmdline"] != launch["argv"] or probe["executable"] != build["binary"]["runtime_path"]
                or probe["executable_sha256"] != build["binary"]["sha256"]
                or probe["cpus_allowed_list"] != resources["cpus_allowed_list"]
                or probe["cgroup_limits"] != resources["cgroup_limits"]):
            raise ValueError("retained functional executable/argv/resources differ")
        lock = re.search(r"^Max locked memory\s+(\S+)\s+(\S+)\s+bytes\s*$", probe["proc_limits"], re.M)
        environment = dict(probe["environment"])
        if (not lock or lock.groups() != (str(resources["memlock_bytes"]),) * 2
                or environment.pop("PWD", None) != launch["cwd"]):
            raise ValueError("retained functional memlock/cwd differs")
        for key, value in launch["environment"].items():
            if environment.pop(key, None) != value:
                raise ValueError("retained functional explicit environment differs")
        maps = mapped_files(probe)
        if environment != effective["ambient_environment"] or previous is not None and maps != previous:
            raise ValueError("retained functional ambient environment/image identities changed")
        if any(item["path"] not in maps for item in runtime["build"]["loaded_libraries"]):
            raise ValueError("retained functional loaded images are absent from root maps")
        previous = maps
    return runtime


def performance_retained(root, config, model, tier, plan, policy, preflight, job_id):
    """Reconstruct every timed run of one cell, then its decision with the sealed history."""
    protocol = timed_protocol(config)
    data, receipts, evidence, host_states = {}, {}, {}, {}
    host_policy = mac_host_gate.policy(config) if config.get("paths", {}).get("platform") == "darwin" else None
    job_root = mac_job_root(config["manifest_path"]) if host_policy is not None else None
    for index, side in schedule(protocol["order"]):
        folder = root / model / tier / ("r" + str(index)) / side
        arm = load(folder / "benchmark.json")
        if (arm.get("status") != "passed" or arm.get("cleanup_confirmed") is not True
                or arm.get("completion_requests") != 1 or arm.get("revision") != plan[side]
                or arm.get("tier") != tier or arm.get("model") != model or arm.get("run") != index):
            raise ValueError("selected timed run is missing, failed, repeated or unclean")
        if load(folder / "started.json").get("stage_id") != arm_prefix(side, model, index) + "-benchmark-" + tier:
            raise ValueError("timed run evidence belongs to another stage")
        spec = config["sides"][side]["models"][model]["benchmarks"][tier]
        launch = arm_launch(spec, side, index, required=config.get("validation_mode") == PERFORMANCE_MODE,
                            job_root=job_root)
        data[index, side] = run_final.read_result(folder / "result.json", tier, spec["request_timeout_s"])
        if config.get("performance_protocol") is not None:
            if data[index, side]["prompt_sha256"] != config["performance_protocol"]["prompt_sha256"]:
                raise ValueError("timed prompt differs from the sealed performance protocol")
        receipts[index, side] = runtime_retained(root, config, side, model, folder, launch,
                                                 folder / "comparison-receipt.json")
        if host_policy is not None:
            host_states[index, side] = mac_host_gate.retained_arm(folder, host_policy)
        if comparable(receipts[index, side]) != comparable(preflight[model, tier, side]):
            raise ValueError("timed runtime differs from retained short preflight")
        if arm["result"] != data[index, side]:
            raise ValueError("measurement summary differs from complete raw SSE evidence")
        names = ["result.json", "comparison-receipt.json", "benchmark.json", "started.json", "stopped.json",
                 "runtime-before.json", "runtime-after.json", "result.events.jsonl", "result.events.sse", "stderr.log"]
        if host_policy is not None:
            names += ["host-quiet.json", "host-state-before-request.json", "host-state-after-request.json",
                      "host-state-after-cleanup.json"]
        evidence["r" + str(index) + "/" + side] = {
            name: dict(path=str((folder / name).relative_to(root)), sha256=digest(folder / name)) for name in names}
    environment = (mac_host_gate.cell({arm_name(*key): value for key, value in host_states.items()})
                   if host_policy is not None else None)
    decision = decide_cell(data, receipts, policy, protocol=protocol, history=sealed_history(config),
                           environment=environment,
                           cell=dict(device=config["device"], model=model, tier=tier, job_id=job_id,
                                     base=plan["base"], candidate=plan["candidate"]))
    if protocol["purpose"] == BASELINE:
        decision["build"] = baseline_build_identity(config, load(root / "build-candidate" / "build-receipt.json"))
    if load(root / model / tier / "comparison.json") != decision:
        raise ValueError("selected cell decision differs from verified raw runs")
    return dict(tier=tier, **decision, sources={side: plan[side] for side in ("base", "candidate")},
                evidence=evidence)


def device_report(root, plan, policy, job, child, scope=None, history_sha256=None):
    root = Path(root)
    report = load(root / "integration.json")
    sealed(root, report)
    dispatch = load(root.parent / "dispatch.json")
    if (not dispatch.get("selection_parent")
            or report.get("selection_parent") != dispatch["selection_parent"]
            or report.get("authorization_path") != "live-private-parent"):
        raise ValueError("manual operator execution cannot satisfy a selected parent")
    if (any(dispatch.get(key) != job[key] for key in ("device", "job_id", "config_sha256",
            "selection_plan_sha256", "control_revision", "control_manifest_sha256"))
            or dispatch.get("revision") != job["source_revision"]):
        raise ValueError("retained child dispatch differs from its root-staged job/source")
    config = load(root / "selected-config.json")
    manifest = load(root / "root-manifest.json")
    if execution_scope(config) != scope:
        raise ValueError("retained execution scope differs from the parent acceptance mode")
    needs = bound(config, manifest, job["job_id"])
    if (config["selection_plan"] != plan or digest(root / "selected-config.json") != job["config_sha256"]
            or digest(root / "root-manifest.json") != config["manifest_sha256"]):
        raise ValueError("selected artifact differs from root-staged configuration/plan")
    if (report.get("status") not in SEVERITY[1:] or report.get("cleanup_confirmed") is not True
            or load(root / "job-finished.json").get("cleanup_confirmed") is not True
            or report.get("purpose") != "selected-integration"
            or report.get("selection_plan_sha256") != fingerprint(plan)
            or report.get("device") != job["device"] or report.get("job_id") != job["job_id"]
            or any(report.get(key) != plan[key] for key in ("base", "head", "candidate"))
            or report.get("control_revision") != job["control_revision"]
            or report.get("control_manifest_sha256") != job["control_manifest_sha256"]
            or child.get("conclusion") != ("success" if report.get("status") in SUCCESS else "failure")
            or child.get("run_attempt") != 1):
        raise ValueError("selected result/source/controller/cleanup identity is not a clean single attempt")
    if scope is not None and any(report.get(key) != value for key, value in scope.items()):
        raise ValueError("executed report scope differs from its root-staged inputs")
    if scope is not None and scope["validation_mode"] == PERFORMANCE_MODE:
        # bound() above already admitted the sealed config protocol for this plan.
        if report.get("performance_protocol") != timed_protocol(config):
            raise ValueError("required interleaved 512/256 performance protocol differs")
    # The operator envelope names the history it sealed into every device config;
    # the child decided with it and this replay recomputes with the same records.
    history = fingerprint(sealed_history(config))
    if (scope is not None and history != history_sha256) or report.get("baseline_history_sha256") != history:
        raise ValueError("sealed baseline history differs from the operator envelope or the executed job")
    base_build = load(root / "build-base" / "build-receipt.json") if (root / "build-base").exists() else None
    offer = config.get("base_build")
    expected_source = None
    if base_build is not None:
        adopted = base_build.get("adopted")
        if (adopted is not None and (offer is None or adopted != dict(key=offer["key"],
                                                                       manifest_sha256=offer["freeze_manifest_sha256"])
                                     or base_build["freeze"]["manifest_sha256"] != offer["freeze_manifest_sha256"])):
            raise ValueError("adopted base build differs from the sealed cache offer")
        expected_source = "cache" if adopted is not None else "built"
    if (report.get("base_build") or {}).get("source") != expected_source:
        raise ValueError("reported base build source differs from the retained build receipt")
    model_admission.retained(root, config, manifest, report)
    # Derived launches are rebuilt from the retained profile and resolution output
    # before any server evidence is checked against them.
    machine = staged_launch.retained(root, config, needs) if staged_launch.derived(config) else None
    if report.get("machine_profile") != machine:
        raise ValueError("reported machine profile differs from its retained stage output")
    if digest(root / "performance-preflight.json") != report.get("performance_preflight_sha256"):
        raise ValueError("selected preflight gate missing or changed")
    preflight = retained(root, config, needs)
    for cell in load(root / "performance-preflight.json")["cells"]:
        model, tier = cell["model"], cell["tier"]
        for side, data in cell["sides"].items():
            folder = root / data["raw"]
            model_data = config["sides"][side]["models"][model]
            launch = model_data["smoke"] if data["reused_functional"] else preflight_launch(
                model_data["benchmarks"][tier], side, required=config.get("validation_mode") == PERFORMANCE_MODE,
                job_root=config["sides"][side]["paths"]["job_root"] if config["paths"]["platform"] == "darwin" else None)
            runtime_retained(root, config, side, model, folder, launch,
                             root / "preflight" / model / tier / side / "comparison-receipt.json")
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
        if scope is not None:
            rows_for_model = [cell for cell in report["cells"] if cell.get("model") == model]
            if len(rows_for_model) != 1:
                raise ValueError("functional model receipt is missing or repeated")
            functional_retained(root, config, model, rows_for_model[0])
        performance = []
        for tier in needs["performance"].get(model, {}).get("tiers", []):
            performance.append(performance_retained(root, config, model, tier, plan, policy, preflight, job["job_id"]))
        required = next(cell["performance"] for cell in plan["cells"]
                        if cell["device"] == job["device"] and cell["model"] == model)
        deferred = scope is not None and scope["validation_mode"] == FUNCTIONAL_MODE and required["required"]
        measured = [item["tier"] for item in performance]
        deferred_tiers = [tier for tier in required["tiers"] if tier not in measured] if scope is not None else []
        if performance:
            claimed = next(cell for cell in report["cells"] if cell["model"] == model)["performance"]
            if claimed != [{key: value for key, value in item.items() if key not in ("sources", "evidence")}
                           for item in performance]:
                raise ValueError("integration performance cells differ from verified series")
        rows.append(dict(device=job["device"], model=model, job_id=job["job_id"],
                         status=outcome(performance) if performance else "passed", functional="passed",
                         numerical="passed", performance=performance or (
                             "deferred-not-in-this-run" if deferred else "not-required"),
                         performance_deferred_tiers=deferred_tiers))
        if machine is not None:
            rows[-1]["machine_profile"] = machine
    if outcome(rows) != report["status"]:
        raise ValueError("integration outcome differs from verified required evidence")
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
            rows += failed_cells(plan, device, str(error))
    status = "failed" if errors else outcome([row for row in rows if row["status"] != "not-required"])
    return dict(schema_version=1, purpose="selected-ci", passed=status == "passed", status=status,
                selection_plan_sha256=fingerprint(plan), cells=rows, errors=errors, formal_samples=0)


def failed_cells(plan, device, error):
    return [dict(device=device, model=cell["model"], status="failed", functional="not-verified",
                 numerical="not-verified", performance="not-verified", error=error)
            for cell in plan["cells"] if cell["required"] and cell["device"] == device]


def operator_aggregate(plan, policy, artifacts, jobs, children, scope, history_sha256):
    """Report this authorized run without claiming the entire impact is covered."""
    validate(plan, policy)
    if execution_scope(scope) != scope:
        raise ValueError("explicit operator execution scope required")
    devices = set(scope["execution_devices"])
    if set(jobs) != devices or set(children) != devices:
        raise ValueError("operator results must exactly cover the declared devices")
    rows, errors = [], []
    for cell in plan["cells"]:
        if not cell["required"]:
            rows.append(dict(device=cell["device"], model=cell["model"], status="not-required", reason=cell["reason"]))
        elif cell["device"] not in devices:
            status = "deferred-offline" if cell["device"] == "local-mbp" else "deferred-not-in-this-run"
            rows.append(dict(device=cell["device"], model=cell["model"], status=status,
                             functional=status, numerical=status,
                             performance="deferred-not-in-this-run" if cell["performance"]["required"] else "not-required"))
    for device in scope["execution_devices"]:
        try:
            actual = device_report(Path(artifacts) / device / jobs[device]["job_id"], plan, policy,
                                   jobs[device], children[device], scope, history_sha256)
            expected = {cell["model"] for cell in plan["cells"] if cell["required"] and cell["device"] == device}
            if len(actual) != len(expected) or {row["model"] for row in actual} != expected:
                raise ValueError("selected device/model cells missing, repeated or unexpected")
            rows += actual
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(device=device, error=str(error)))
            rows += failed_cells(plan, device, str(error))
    executed = [row for row in rows if row["device"] in devices and row["status"] != "not-required"]
    status = "failed" if errors else outcome(executed)
    complete = not errors and all(cell["status"] in (*SUCCESS, "not-required")
                                  and cell.get("performance") != "deferred-not-in-this-run"
                                  and not cell.get("performance_deferred_tiers") for cell in rows)
    linux = [row for row in executed if row["device"] != "local-mbp"]
    mac = [row for row in executed if row["device"] == "local-mbp"]
    return dict(schema_version=1, purpose="operator-selected-report-only", passed=status == "passed",
                status=status, execution_scoped_status=status,
                linux_scoped_status=outcome(linux) if linux else "not-in-this-run",
                mac_scoped_status=outcome(mac) if mac else "not-in-this-run",
                full_impact_coverage_complete=complete, pr_publication="not-enabled",
                source_identity="sealed-source-not-current-pr", baseline_history_sha256=history_sha256, **scope,
                selection_plan_sha256=fingerprint(plan), cells=rows, errors=errors, formal_samples=0)
