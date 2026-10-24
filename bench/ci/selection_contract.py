"""One selection identity from hosted admission through the root broker job.

The manifest contains the config payload hash (excluding manifest_sha256), and
the config contains the final manifest file hash. This avoids a circular hash.
No runner can create either protected file or change the selected requirements.
"""

import copy
import re
from pathlib import PurePosixPath

from ci.contract import DEVICES, digest, load, policy_read
from ci.impact import CASES, device_requirements, fingerprint, validate


JOB = re.compile(r"[0-9a-f]{32}")


def payload_hash(config):
    return fingerprint({k: v for k, v in config.items() if k != "manifest_sha256"})


def requirements(config):
    policy = policy_read(config["policy"])
    plan = validate(config["selection_plan"], policy)
    if config["selection_plan_sha256"] != fingerprint(plan):
        raise ValueError("selected plan hash differs")
    device = config["device"]
    if device not in DEVICES:
        raise ValueError("unknown physical device")
    needs = device_requirements(plan, device)
    if not needs["models"]:
        raise ValueError("unselected device must not be dispatched")
    if (config.get("schema_version") != 1 or config.get("purpose") != "selected-integration"
            or config["revision"] != plan["candidate"]):
        raise ValueError("selected configuration/source identity differs")
    platform = "darwin" if device == "local-mbp" else "linux"
    if config["numerical"]["backend"] != ("MTL0" if platform == "darwin" else "CUDA0"):
        raise ValueError("selected numerical backend differs from physical device")
    for side in ("base", "candidate"):
        spec = config["sides"][side]
        if (spec["revision"] != plan[side] or spec["paths"]["platform"] != platform
                or set(spec["models"]) != set(needs["models"])):
            raise ValueError("staged side does not cover the exact selected source/models")
        for model, data in spec["models"].items():
            tiers = needs["performance"].get(model, {}).get("tiers", [])
            if set(data.get("benchmarks", {})) != set(tiers):
                raise ValueError("performance arms must exactly match the selected tiers")
            if side == "candidate" and data["smoke"]["model"] != model:
                raise ValueError("functional request model differs from selected cell")
        required_targets = {CASES[case][0] for case in needs["numerical"]}
        if side == "candidate" and not required_targets <= set(spec["build"]["targets"]):
            raise ValueError("build omits selected numerical targets")
        if side == "candidate" and not required_targets <= {PurePosixPath(p).name for p in spec["build"]["image_relatives"]}:
            raise ValueError("sealed build identity omits selected numerical images")
    if config["paths"] != config["sides"]["candidate"]["paths"]:
        raise ValueError("workflow platform guard must use the exact candidate paths")
    from ci.model_admission import enabled
    enabled(config)
    return needs


def bind(config, manifest, plan, policy):
    """Operator-only data preparation; writes nothing and performs no deployment."""
    config, manifest = copy.deepcopy(config), copy.deepcopy(manifest)
    validate(plan, policy_read(policy))
    config.update(schema_version=1, purpose="selected-integration", policy=policy,
                  selection_plan=plan, selection_plan_sha256=fingerprint(plan),
                  revision=plan["candidate"],
                  paths=config["sides"]["candidate"]["paths"])
    config.pop("manifest_sha256", None)
    needs = requirements(config)
    sides = ("base", "candidate") if needs["performance"] else ("candidate",)
    if manifest.get("source_revisions") != {side: plan[side] for side in sides}:
        raise ValueError("operator manifest must stage the exact required source revisions")
    if config["paths"]["platform"] == "linux":
        manifest["stages"] = linux_stages(config, needs, sides)
    else:
        manifest["selection_resource_policy"] = config["host"]["resources"]
    manifest["selection"] = dict(plan_sha256=fingerprint(plan),
                                 config_payload_sha256=payload_hash(config),
                                 device=config["device"],
                                 source_revisions={k: plan[k] for k in ("base", "head", "candidate")})
    return config, manifest


def manifest_sources(config, sides):
    return {"sources": {side: config["sides"][side]["paths"]["source"] for side in sides}}


def linux_stages(config, needs, sides):
    """Exact root allowlist for the fixed trusted calls, with no arbitrary argv."""
    stages = {}

    def add(name, kind, side, executable, cwd, gpu):
        stages[name] = dict(kind=kind, side=side, executables=[executable], cwd=cwd, gpu=gpu)

    for side in sides:
        spec = config["sides"][side]
        paths, build = spec["paths"], spec["build"]
        for name in ("configure", "build"):
            add(side + "-" + name, "build", side, build["cmake"], paths["cwd"], False)
        binary = str(PurePosixPath(paths["frozen"]) / build["binary_relative"])
        add(side + "-version", "test", side, binary, paths["frozen"], True)
        if spec.get("alias"):
            alias = spec["alias"]
            add(side + "-alias", "build", side, alias["argv"][0], alias["cwd"], False)
        for model in needs["models"]:
            data = spec["models"][model]
            if side == "candidate":
                add("candidate-smoke-" + model, "server", side, binary, data["smoke"]["cwd"], True)
                if data["smoke"].get("observations"):
                    add("candidate-observations-" + model, "test", side, "/usr/bin/cat", data["smoke"]["observations"]["cwd"], False)
            for tier, arm in data.get("benchmarks", {}).items():
                add(side + "-" + model + "-preflight-" + tier, "server", side, binary, arm["cwd"], True)
                add(side + "-" + model + "-benchmark-" + tier, "server", side, binary, arm["cwd"], True)
    root = config["sides"]["candidate"]["paths"]["frozen"]
    for case in needs["numerical"]:
        add("candidate-numerical-" + case, "test", "candidate", root + "/bin/" + CASES[case][0], config["numerical"]["cwd"], True)
    from ci.model_admission import enabled, probe_spec
    if enabled(config):
        for when in ("before", "after"):
            probe = probe_spec(config, manifest_sources(config, sides), when)
            add(probe["stage_id"], probe["kind"], "candidate", probe["argv"][0], probe["cwd"], False)
    return stages


def bound(config, manifest, job_id):
    needs = requirements(config)
    plan = config["selection_plan"]
    expected = dict(plan_sha256=fingerprint(plan), config_payload_sha256=payload_hash(config),
                    device=config["device"],
                    source_revisions={k: plan[k] for k in ("base", "head", "candidate")})
    if (not JOB.fullmatch(job_id) or manifest.get("job_id") != job_id
            or manifest.get("selection") != expected):
        raise ValueError("root broker manifest/config/selected source binding differs")
    return needs


def protected_binding(path, expected_config, expected_plan, job_id):
    from ci.isolated_device import protected
    config_path = protected(path)
    if digest(config_path) != expected_config:
        raise ValueError("root-staged configuration hash differs")
    config = load(config_path)
    if config["selection_plan_sha256"] != expected_plan:
        raise ValueError("workflow expected plan is missing or differs")
    manifest_path = protected(config["manifest_path"])
    if digest(manifest_path) != config["manifest_sha256"]:
        raise ValueError("root broker manifest hash differs")
    bound(config, load(manifest_path), job_id)
    return config
