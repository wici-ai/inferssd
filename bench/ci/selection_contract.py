"""One selection identity from hosted admission through the root broker job.

The manifest contains the config payload hash (excluding manifest_sha256), and
the config contains the final manifest file hash. This avoids a circular hash.
No runner can create either protected file or change the selected requirements.
"""

import copy
import re
from pathlib import PurePosixPath

import final_chat

from ci.contract import DEVICES, digest, load, policy_read
from ci.impact import BASELINE, CASES, device_requirements, fingerprint, validate


JOB = re.compile(r"[0-9a-f]{32}")
OPERATOR_DEVICES = DEVICES
FUNCTIONAL_MODE = "functional-numerical"
PERFORMANCE_MODE = "functional-numerical-performance"
GATE = "regression-gate"
BASELINE_JOB = "baseline"
# Timed runs per model, in execution order. A PR runs its candidate three times
# and its base once, not first. A baseline job builds one main commit (as the
# candidate side) and measures it four times.
RUN_ORDERS = {GATE: ["candidate", "base", "candidate", "candidate"], BASELINE_JOB: ["candidate"] * 4}


def performance_protocol(purpose=GATE):
    """The fixed operator performance protocol; it never adds formal samples."""
    if purpose not in RUN_ORDERS:
        raise ValueError("operator performance purpose must be the gate or a baseline job")
    return dict(tier="512", prompt_sha256=final_chat.PROMPTS["512"][1], output_tokens=256,
                purpose=purpose, order=list(RUN_ORDERS[purpose]))


def protocol_checked(protocol, *, baseline):
    """Accept only the fixed gate protocol, or the baseline protocol for a baseline plan."""
    if not isinstance(protocol, dict) or type(protocol.get("output_tokens")) is not int:
        raise ValueError("operator performance requires the fixed 512/256 run protocol")
    if protocol != performance_protocol(BASELINE_JOB if baseline else GATE):
        raise ValueError("operator performance requires the fixed 512/256 run protocol")
    return protocol


def timed_protocol(config):
    """The sealed protocol of an operator run; other configurations use the gate."""
    protocol = config.get("performance_protocol")
    if protocol is None:
        return performance_protocol()
    return protocol_checked(protocol, baseline=config.get("selection_plan", {}).get("purpose") == BASELINE)


def timed_sides(protocol):
    """Sides that are built and timed: a baseline job builds only the candidate side."""
    return tuple(side for side in ("base", "candidate") if side in protocol["order"])


def run_count(protocol, side):
    return protocol["order"].count(side)


def job_sides(config, needs):
    """Sides staged and built: both for a PR with timed cells, the candidate otherwise.

    A baseline job builds its one main commit once, as the candidate side.
    """
    if not needs["performance"]:
        return ("candidate",)
    return timed_sides(timed_protocol(config))


def build_config_sha256(spec):
    """Side-independent build settings that determine the frozen output.

    A side's own source, scratch and frozen paths (for example the
    `LD_LIBRARY_PATH` naming `/build/base/bin`) are written as placeholders, so
    the base side of a PR and the candidate side of the baseline job that built
    the same commit have the same identity. Jobs, timeout and log bound do not
    change the output.
    """
    paths = spec["paths"]
    names = sorted({paths[name]: name for name in ("frozen", "source", "build", "cwd")}.items(),
                   key=lambda item: -len(item[0]))
    pattern = re.compile("|".join(re.escape(path) + r"(?![^/:\s=\"])" for path, _ in names))
    placeholders = dict(names)

    def neutral(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: "{" + placeholders[match.group(0)] + "}", value)
        if isinstance(value, list):
            return [neutral(item) for item in value]
        if isinstance(value, dict):
            return {key: neutral(item) for key, item in value.items()}
        return value

    build = {key: neutral(value) for key, value in spec["build"].items()
             if key not in ("jobs", "timeout_s", "output_max_bytes")}
    return fingerprint(dict(platform=paths["platform"], build=build))


def build_cache_key(config, side):
    """Device build-cache key: exact commit, build configuration and reviewed toolchain."""
    spec = config["sides"][side]
    return fingerprint(dict(schema_version=1, revision=spec["revision"], build_config_sha256=build_config_sha256(spec),
                            toolchain=spec["build"]["toolchain"]))


def baseline_build_identity(config, build):
    """Build identity a baseline record keeps: what a later PR can adopt as its base."""
    spec = config["sides"]["candidate"]
    return dict(freeze_manifest_sha256=build["freeze"]["manifest_sha256"], binary_sha256=build["binary"]["sha256"],
                cache_key=build_cache_key(config, "candidate"), build_config_sha256=build_config_sha256(spec),
                toolchain=build["toolchain"])


def sealed_history(config):
    """Baseline records sealed into this job by the operator.

    Operator performance configs always carry the field (possibly empty), so a
    dropped field cannot be confused with a device that has no history yet.
    """
    from ci.compare import history_checked
    return history_checked(config.get("baseline_history", []))


def arm_prefix(side, model, run):
    """Stage prefix of one timed run; the broker stage is `<prefix>-benchmark-<tier>`."""
    if side not in ("base", "candidate") or type(run) is not int or run < 1:
        raise ValueError("explicit timed run side and index required")
    return side + "-" + model + "-r" + str(run)


def execution_scope(value):
    """An explicit run scope never edits the canonical impact requirements."""
    if not any(key in value for key in ("validation_mode", "execution_devices")):
        return None
    devices = value.get("execution_devices")
    if (value.get("validation_mode") not in (FUNCTIONAL_MODE, PERFORMANCE_MODE) or not isinstance(devices, list)
            or not devices or any(not isinstance(device, str) or device not in OPERATOR_DEVICES for device in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("operator scope requires a nonempty unique physical device subset")
    return dict(validation_mode=value["validation_mode"], execution_devices=devices)


def runtime_requirements(config):
    needs = device_requirements(config["selection_plan"], config["device"])
    scope = execution_scope(config)
    if scope is not None:
        if config["device"] not in scope["execution_devices"]:
            raise ValueError("device is outside the sealed operator execution scope")
        if scope["validation_mode"] == FUNCTIONAL_MODE:
            needs = dict(needs, performance={})
        else:
            # Keep the canonical plan unchanged, including any required 4K
            # coverage. This explicitly bounded operator run executes 512 only.
            needs = dict(needs, performance={model: dict(required=value["required"],
                tiers=["512"], output_tokens=256) for model, value in needs["performance"].items()})
    return needs


def selection_identity(config):
    plan = config["selection_plan"]
    identity = dict(plan_sha256=fingerprint(plan), config_payload_sha256=payload_hash(config),
                    device=config["device"],
                    source_revisions={k: plan[k] for k in ("base", "head", "candidate")})
    scope = execution_scope(config)
    if scope is not None:
        identity.update(scope)
    return identity


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
    needs = runtime_requirements(config)
    scope = execution_scope(config)
    baseline = plan["purpose"] == BASELINE
    if baseline and (scope is None or scope["validation_mode"] != PERFORMANCE_MODE):
        raise ValueError("a baseline job requires the explicit operator performance scope")
    if scope is not None and scope["validation_mode"] == PERFORMANCE_MODE:
        protocol_checked(config.get("performance_protocol"), baseline=baseline)
        if "baseline_history" not in config:
            raise ValueError("operator performance requires explicitly sealed baseline history")
    sealed_history(config)
    if "base_build" in config:
        offer = config["base_build"]
        if (baseline or config["paths"]["platform"] != "linux" or not isinstance(offer, dict)
                or set(offer) != {"key", "freeze_manifest_sha256", "source_job"}
                or offer["key"] != build_cache_key(config, "base")
                or not isinstance(offer["freeze_manifest_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", offer["freeze_manifest_sha256"]) is None
                or not isinstance(offer["source_job"], str) or JOB.fullmatch(offer["source_job"]) is None):
            raise ValueError("a cached base build is offered only to a Linux PR job, for its exact base build")
    if "cache_build_as" in config and (not baseline or config["paths"]["platform"] != "linux"
                                       or config["cache_build_as"] != build_cache_key(config, "candidate")):
        raise ValueError("only a Linux baseline job stores its build, under its exact cache key")
    if not needs["models"]:
        raise ValueError("unselected device must not be dispatched")
    if (config.get("schema_version") != 1 or config.get("purpose") != "selected-integration"
            or config["revision"] != plan["candidate"]):
        raise ValueError("selected configuration/source identity differs")
    platform = "darwin" if device == "local-mbp" else "linux"
    if config["numerical"]["backend"] != ("MTL0" if platform == "darwin" else "CUDA0"):
        raise ValueError("selected numerical backend differs from physical device")
    if platform == "darwin":
        match = re.fullmatch(r"/Library/InferSSD-CI/private/job-manifests/([0-9a-f]{32})\.json", config["manifest_path"])
        resources = config["host"]["resources"]
        if (match is None or set(resources) != {"max_processes", "memory_budget_bytes", "host_reserve_bytes"}
                or any(type(v) is not int or v <= 0 for v in resources.values())):
            raise ValueError("Mac requires its fixed root manifest and explicit sampled resource policy")
        mac_root = "/Library/InferSSD-CI/jobs/" + match[1]
    for side in ("base", "candidate"):
        spec = config["sides"][side]
        if (spec["revision"] != plan[side] or spec["paths"]["platform"] != platform
                or set(spec["models"]) != set(needs["models"])):
            raise ValueError("staged side does not cover the exact selected source/models")
        if platform == "darwin" and (spec["paths"].get("job_root") != mac_root
                or spec["paths"]["build"] != mac_root + "/work/" + side
                or spec["paths"]["frozen"] != mac_root + "/frozen/" + side
                or spec["paths"]["source"] != "/Library/InferSSD-CI/inputs/operator-" + match[1] + "/" + side):
            raise ValueError("Mac source/build/freeze paths must bind this exact job side")
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
    sides = job_sides(config, needs)
    if manifest.get("source_revisions") != {side: plan[side] for side in sides}:
        raise ValueError("operator manifest must stage the exact required source revisions")
    if config["paths"]["platform"] == "linux":
        manifest["stages"] = linux_stages(config, needs, sides)
        # The broker re-hashes a cached tree before adopting it, and only a
        # baseline job of a main commit stores its frozen build for later jobs.
        if "base_build" in config:
            manifest["prebuilt_offer"] = {"base": dict(key=config["base_build"]["key"],
                                                       manifest_sha256=config["base_build"]["freeze_manifest_sha256"])}
        if "cache_build_as" in config:
            manifest["cache_as"] = {"candidate": config["cache_build_as"]}
    else:
        manifest["selection_resource_policy"] = config["host"]["resources"]
        lock = manifest.get("selection_lock_identity")
        if not isinstance(lock, list) or len(lock) != 2 or any(type(v) is not int or v < 0 for v in lock):
            raise ValueError("Mac manifest must bind the original root device lease")
    manifest["selection"] = selection_identity(config)
    return config, manifest


def manifest_sources(config, sides):
    return {"sources": {side: config["sides"][side]["paths"]["source"] for side in sides}}


def linux_stages(config, needs, sides):
    """Exact root allowlist for the fixed trusted calls, with no arbitrary argv."""
    stages = {}

    def add(name, kind, side, executable, cwd, gpu):
        stages[name] = dict(kind=kind, side=side, executables=[executable], cwd=cwd, gpu=gpu)

    protocol = timed_protocol(config)
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
                for run in range(1, run_count(protocol, side) + 1):
                    add(arm_prefix(side, model, run) + "-benchmark-" + tier, "server", side, binary, arm["cwd"], True)
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
    expected = selection_identity(config)
    if (not JOB.fullmatch(job_id) or manifest.get("job_id") != job_id
            or manifest.get("selection") != expected):
        raise ValueError("root broker manifest/config/selected source binding differs")
    offer = ({"base": dict(key=config["base_build"]["key"], manifest_sha256=config["base_build"]["freeze_manifest_sha256"])}
             if "base_build" in config else None)
    cache_as = {"candidate": config["cache_build_as"]} if "cache_build_as" in config else None
    if manifest.get("prebuilt_offer") != offer or manifest.get("cache_as") != cache_as:
        raise ValueError("root broker build-cache instructions differ from the sealed configuration")
    if config["paths"]["platform"] == "darwin":
        lock = manifest.get("selection_lock_identity")
        if (manifest.get("selection_resource_policy") != config["host"]["resources"]
                or not isinstance(lock, list) or len(lock) != 2
                or any(type(v) is not int or v < 0 for v in lock)):
            raise ValueError("Mac root resource/lease binding differs")
    return needs


def protected_binding(path, expected_config, expected_plan, job_id):
    from ci.isolated_device import protected, protected_manifest
    config_path = protected(path)
    if digest(config_path) != expected_config:
        raise ValueError("root-staged configuration hash differs")
    config = load(config_path)
    if config["selection_plan_sha256"] != expected_plan:
        raise ValueError("workflow expected plan is missing or differs")
    manifest_path = protected_manifest(config)
    bound(config, load(manifest_path), job_id)
    return config
