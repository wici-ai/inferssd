"""Execute exactly the selected cells through one already isolated broker lease."""

import os
from pathlib import Path
import time

import run_final
from ci.compare import compare_pair
from ci.contract import digest, load, save
from ci.impact import device_requirements
from ci.private_controller import source_manifest
from ci.selected_receipt import receipt
from ci.selection_contract import protected_binding
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
    session = None
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
        manifest = load(config["manifest_path"])
        (output / "root-manifest.json").write_bytes(Path(config["manifest_path"]).read_bytes())
        needs = device_requirements(plan, device)
        session = StageSession(args.socket, args.job_id, controller_uid=os.getuid())
        opened = session.open()
        save(output / "job-opened.json", opened)
        if opened["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("active broker job differs from root-staged selection")
        builds = {}
        sides = ("base", "candidate") if needs["performance"] else ("candidate",)
        for side in sides:
            spec = config["sides"][side]
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
        if len(builds) == 2 and builds["base"]["toolchain"] != builds["candidate"]["toolchain"]:
            raise ValueError("actual compiler paths differ between compared arms")
        numeric = selected_numerical(session, config["numerical"], builds["candidate"], needs["numerical"], output / "numerical")
        for model in needs["models"]:
            data = config["sides"]["candidate"]["models"][model]
            native = dict(data["smoke"], revision=plan["candidate"])
            if native["argv"][0] != builds["candidate"]["binary"]["runtime_path"]:
                raise ValueError("functional executable differs from sealed candidate")
            result = smoke(session, native, output / ("native-" + model), stage_id="candidate-smoke-" + model)
            if result["status"] != "passed":
                raise RuntimeError("selected functional model request failed")
            observation = native.get("observations")
            if observation:
                argv = native["argv"]
                if argv.count("--ssd-telemetry") != 1 or argv[argv.index("--ssd-telemetry") + 1] != observation["path"]:
                    raise ValueError("observations do not bind actual native telemetry")
                observe(session, observation, model, output / ("observations-" + model))
            cell = dict(model=model, functional="passed", numerical={case: numeric[case] for case in needs["numerical"]}, performance=[])
            for tier in needs["performance"].get(model, {}).get("tiers", []):
                measurements, receipts = {}, {}
                for side in ("base", "candidate"):
                    spec = config["sides"][side]["models"][model]["benchmarks"][tier]
                    launch = dict(spec, model=model, revision=plan[side], tier=tier, stage_prefix=side + "-" + model)
                    if launch["argv"][0] != builds[side]["binary"]["runtime_path"]:
                        raise ValueError("performance executable differs from sealed side")
                    folder = output / model / tier / side
                    measured = benchmark(session, side, launch, folder)
                    if measured["status"] != "passed" or measured["cleanup_confirmed"] is not True:
                        raise RuntimeError("selected single benchmark arm failed; no retry")
                    measurements[side] = run_final.read_result(folder / "result.json", tier, spec["request_timeout_s"])
                    receipts[side] = receipt(config, manifest, side, model, launch, builds[side], folder)
                compared = compare_pair(measurements["base"], measurements["candidate"], receipts["base"], receipts["candidate"], config["policy"])
                save(output / model / tier / "comparison.json", compared)
                cell["performance"].append(dict(tier=tier, **compared))
                if compared["passed"] is not True:
                    raise RuntimeError("observed slowdown is inconclusive and blocks acceptance")
            report["cells"].append(cell)
        report["status"] = "passed"
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
        if session is not None:
            session.close()
        files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
        save(output / "evidence-manifest.json", dict(schema_version=1, files=files))
        report["evidence_manifest_sha256"] = digest(output / "evidence-manifest.json")
        save(output / "integration.json", report)
    return 0 if report["status"] == "passed" else 1
