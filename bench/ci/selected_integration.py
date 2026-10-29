"""Execute exactly the selected cells through one already isolated broker lease."""

import os
from pathlib import Path
import time

import run_final
from ci.compare import BASELINE, SUCCESS, arm_name, comparable, decide_cell, outcome, schedule
from ci.contract import digest, load, save
from ci.private_controller import API, source_manifest
from ci.isolated_device import manifest_read_path
from ci.selection_lifetime import OBSERVATION_BUDGET_S, ParentWatch
from ci.selected_receipt import mac_job_root, receipt
from ci import selected_preflight, model_admission, mac_host_gate, staged_launch
from ci.impact import fingerprint
from ci.selection_contract import (PERFORMANCE_MODE, arm_prefix, baseline_build_identity, execution_scope,
                                   job_sides, protected_binding, runtime_requirements, sealed_history,
                                   timed_protocol)
from ci.stage_session import StageSession
from ci.staged_benchmark import benchmark
from ci.staged_build import run_build
from ci.staged_numerical import selected_numerical
from ci.staged_observations import observe
from ci.staged_smoke import collect_log, smoke


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, purpose="selected-integration", status="failed",
                  cleanup_confirmed=False, formal_samples=0, cells=[])
    session, watch = None, None
    admitted = False
    try:
        config = protected_binding(args.config, args.config_sha256, args.selection_plan_sha256, args.job_id)
        control = Path(__file__).resolve().parents[2]
        if (source_manifest(control) != config["control_revision"]
                or digest(control / "control-source.json") != config["control_manifest_sha256"]):
            raise ValueError("root-installed controller identity differs from admitted deployment")
        plan, device = config["selection_plan"], config["device"]
        report.update(selection_plan_sha256=args.selection_plan_sha256, device=device,
                      base=plan["base"], head=plan["head"], candidate=plan["candidate"],
                      config_sha256=args.config_sha256, control_revision=config["control_revision"],
                      control_manifest_sha256=config["control_manifest_sha256"], job_id=args.job_id)
        (output / "selected-config.json").write_bytes(args.config.read_bytes())
        manifest_path = manifest_read_path(config)
        manifest = load(manifest_path)
        (output / "root-manifest.json").write_bytes(manifest_path.read_bytes())
        needs = runtime_requirements(config)
        history = sealed_history(config)
        report["baseline_history_sha256"] = fingerprint(history)
        host_policy = mac_host_gate.policy(config) if config["paths"]["platform"] == "darwin" else None
        host_options = {"host_policy": host_policy} if host_policy is not None else {}
        scope = execution_scope(config)
        if scope is not None:
            report.update(scope)
            if "performance_protocol" in config:
                report["performance_protocol"] = config["performance_protocol"]
        # A derived launch is complete only once its side resolved it on the device.
        derived = staged_launch.derived(config)
        pairs = selected_preflight.declarations(config, manifest, needs)
        admitted = model_admission.enabled(config)
        model_admission.prepare(config, manifest, output)
        parent = getattr(args, "selection_parent", "")
        report.update(selection_parent=parent,
                      authorization_path="live-private-parent" if parent else "operator-root-staged")
        if parent:
            repository = config["policy"]["controller_repository"]
            sha = os.environ["GITHUB_WORKFLOW_SHA"]
            if (os.environ.get("GITHUB_REPOSITORY") != repository
                    or os.environ.get("GITHUB_REF") != "refs/heads/main"
                    or os.environ.get("GITHUB_SHA") != sha
                    or os.environ.get("GITHUB_WORKFLOW_REF") != repository + "/.github/workflows/integration-smoke.yml@refs/heads/main"):
                raise ValueError("selected child must use the protected private main workflow")
            watch = ParentWatch(API(repository, os.environ["GITHUB_TOKEN"], timeout=OBSERVATION_BUDGET_S), parent, sha)
            watch.start()
        session = StageSession(args.socket, args.job_id, controller_uid=os.getuid())
        if watch is not None:
            watch.attach(session)
        opened = session.open()
        save(output / "job-opened.json", opened)
        if opened["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("active broker job differs from root-staged selection")
        model_admission.probe(session, config, manifest, "before", output)
        builds = {}
        sides = job_sides(config, needs)
        offer = config.get("base_build")
        adopted = opened.get("prebuilt", {}).get("base") if offer is not None else None
        if offer is None and opened.get("prebuilt"):
            raise ValueError("broker adopted a cached build that was not offered")
        if adopted is not None and (adopted.get("key") != offer["key"]
                                    or adopted.get("manifest_sha256") != offer["freeze_manifest_sha256"]):
            raise ValueError("broker adopted a cached build other than the offered one")
        if "base" in sides:
            report["base_build"] = dict(source="cache" if adopted is not None else "built", offered=offer,
                                        refused=opened.get("prebuilt_refused", {}).get("base"))
        # The candidate always builds; the base reuses its verified cached build when offered.
        machine = None
        for side in sides:
            spec = config["sides"][side]
            if side == "base" and adopted is not None:
                builds[side] = run_build(session, side, spec["build"], spec["paths"], output / ("build-" + side),
                                         spec["revision"], adopted=adopted)
            else:
                alias = spec.get("alias")
                if alias:
                    target = output / (side + "-alias")
                    target.mkdir()
                    stage = session.start(side + "-alias", "build", alias["argv"], alias.get("environment", {}), alias["cwd"], alias["timeout_s"])
                    completed = session.wait(stage, deadline=time.monotonic() + alias["timeout_s"])
                    save(target / "stage.json", completed)
                    collect_log(session, stage, target)
                    if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
                        raise RuntimeError("sandboxed alias preparation failed")
                builds[side] = run_build(session, side, spec["build"], spec["paths"], output / ("build-" + side), spec["revision"])
            if derived:
                # A baseline job profiles the machine once with the commit it
                # measures. A PR job takes the sealed file of its device's most
                # recent baseline job instead. Each side then resolves its own
                # launches from that one characteristics file.
                if machine is None:
                    machine = (staged_launch.adopt(config) if staged_launch.sealed(config) is not None else
                               staged_launch.profile(session, config, side, needs["models"], output / "machine"))
                    report["machine_profile"] = machine
                for model in staged_launch.launch_models(config, needs, side):
                    resolved = staged_launch.resolve(session, config, side, model, machine,
                                                     output / "launch" / (side + "-" + model))
                    staged_launch.fill(config, side, model, resolved, staged_launch.tiers(needs, model))
        if derived:
            pairs = selected_preflight.declarations(config, manifest, needs)
        if len(builds) == 2 and builds["base"]["toolchain"] != builds["candidate"]["toolchain"]:
            raise ValueError("actual compiler paths differ between compared arms")
        numeric = selected_numerical(session, config["numerical"], builds["candidate"], needs["numerical"], output / "numerical")
        functional, cells = {}, {}
        for model in needs["models"]:
            data = config["sides"]["candidate"]["models"][model]
            native = dict(data["smoke"], revision=plan["candidate"])
            if native["argv"][0] != builds["candidate"]["binary"]["runtime_path"]:
                raise ValueError("functional executable differs from sealed candidate")
            result = smoke(session, native, output / ("native-" + model), stage_id="candidate-smoke-" + model, **host_options)
            if result["status"] != "passed":
                raise RuntimeError("selected functional model request failed")
            functional[model] = dict(launch=native, output=output / ("native-" + model), result=result)
            observation = native.get("observations")
            if observation:
                argv = native["argv"]
                if argv.count("--ssd-telemetry") != 1 or argv[argv.index("--ssd-telemetry") + 1] != observation["path"]:
                    raise ValueError("observations do not bind actual native telemetry")
                observe(session, observation, model, output / ("observations-" + model),
                        platform=config["paths"]["platform"])
            cells[model] = dict(model=model, functional="passed", numerical={case: numeric[case] for case in needs["numerical"]}, performance=[])
            if scope is not None:
                folder = output / ("native-" + model)
                receipt(config, manifest, "candidate", model, native, builds["candidate"], folder,
                        destination=folder / "runtime-receipt.json")
                cells[model]["runtime_receipt_sha256"] = digest(folder / "runtime-receipt.json")
        preflight = selected_preflight.run(session, config, manifest, pairs, builds, functional, output)
        report["performance_preflight_sha256"] = digest(output / "performance-preflight.json")
        protocol = timed_protocol(config)
        job_root = mac_job_root(config["manifest_path"]) if config["paths"]["platform"] == "darwin" else None
        for model in needs["models"]:
            cell = cells[model]
            report["cells"].append(cell)
            for tier in needs["performance"].get(model, {}).get("tiers", []):
                # Every planned run executes once in the fixed order and is kept.
                # A slower or different result is a retained outcome, never a
                # reason to retry a run or drop it from the cell.
                measurements, receipts, host_states = {}, {}, {}
                for index, side in schedule(protocol["order"]):
                    spec = config["sides"][side]["models"][model]["benchmarks"][tier]
                    launch = dict(selected_preflight.arm_launch(
                        spec, side, index, required=config.get("validation_mode") == PERFORMANCE_MODE,
                        job_root=job_root), model=model, revision=plan[side], tier=tier,
                        stage_prefix=arm_prefix(side, model, index), run=index)
                    if launch["argv"][0] != builds[side]["binary"]["runtime_path"]:
                        raise ValueError("performance executable differs from sealed side")
                    folder = output / model / tier / ("r" + str(index)) / side
                    measured = benchmark(session, side, launch, folder, **host_options)
                    if measured["status"] != "passed" or measured["cleanup_confirmed"] is not True:
                        raise RuntimeError("selected timed run failed; no retry")
                    measurements[index, side] = run_final.read_result(folder / "result.json", tier, spec["request_timeout_s"])
                    receipts[index, side] = receipt(config, manifest, side, model, launch, builds[side], folder)
                    if host_policy is not None:
                        host_states[index, side] = mac_host_gate.retained_arm(folder, host_policy)
                    if comparable(receipts[index, side]) != comparable(preflight[model, tier, side]):
                        raise ValueError("timed runtime differs from its successful short preflight")
                environment = (mac_host_gate.cell({arm_name(*key): value for key, value in host_states.items()})
                               if host_policy is not None else None)
                decision = decide_cell(measurements, receipts, config["policy"], protocol=protocol, history=history,
                                       environment=environment,
                                       cell=dict(device=device, model=model, tier=tier, job_id=args.job_id,
                                                 base=plan["base"], candidate=plan["candidate"]))
                if protocol["purpose"] == BASELINE:
                    decision["build"] = baseline_build_identity(config, builds["candidate"])
                save(output / model / tier / "comparison.json", decision)
                cell["performance"].append(dict(tier=tier, **decision))
        model_admission.probe(session, config, manifest, "after", output)
        if watch is not None:
            watch.check()
        report["status"] = outcome([item for cell in report["cells"] for item in cell["performance"]])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if session is None or session.closed:
                raise RuntimeError("positive broker cleanup unavailable")
            finished = session.finish()
            save(output / "job-finished.json", finished)
            if finished.get("cleanup_confirmed") is not True:
                raise RuntimeError("broker cleanup not positively confirmed")
            report["cleanup_confirmed"] = True
        except Exception as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        if admitted and report["cleanup_confirmed"]:
            try:
                report["model_admission_sha256"] = model_admission.finalize(output, config, manifest)
            except Exception as error:
                report.update(status="failed", model_admission_error=f"{type(error).__name__}: {error}")
        if session is not None:
            session.close()
        if watch is not None:
            try:
                watch.stop()
            except Exception as error:
                report.update(status="failed", parent_error=repr(error))
        files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
        save(output / "evidence-manifest.json", dict(schema_version=1, files=files))
        report["evidence_manifest_sha256"] = digest(output / "evidence-manifest.json")
        save(output / "integration.json", report)
    return 0 if report["status"] in SUCCESS else 1
