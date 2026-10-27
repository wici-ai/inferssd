#!/usr/bin/env python3
"""Prepare operator-authorized device CI data; keep gh credentials on this host.

Preparation exports immutable Git objects and explicit reviewed device inputs.
It does not install, connect to a device, acquire a lease or execute source code.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import HASH, digest, load, policy_read, revision, save
from ci.impact import fingerprint, plan, device_requirements
from ci.selection_contract import JOB, bind, bound, runtime_requirements, performance_protocol


LINUX_DEVICES = ("cloud-5090", "dgx-spark")
DEVICES = (*LINUX_DEVICES, "local-mbp")
MODE = "functional-numerical-performance"
LEGACY_MODE = "functional-numerical"
PURPOSE = "operator-selected-report-only"
MAX_INPUT = 60 * 1024
WORKFLOW = ".github/workflows/device-performance.yml"
MAC_PYTHON = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"


def device_layout(device, job_id):
    if device == "local-mbp":
        base = "/Library/InferSSD-CI"
        return dict(platform="darwin", source=base + "/inputs/operator-" + job_id,
                    manifest=base + "/private/job-manifests/" + job_id + ".json",
                    config=base + "/integrations/" + job_id + ".json",
                    stager=base + "/controller/current/bench/ci/operator_mac_stage.py")
    if device not in LINUX_DEVICES:
        raise ValueError("unknown operator device")
    return dict(platform="linux", source="/var/lib/inferssd-ci-inputs/operator-" + job_id,
                manifest="/var/lib/inferssd-ci-broker/jobs/" + job_id + "/manifest.json",
                config="/opt/inferssd-ci-integrations/" + job_id + ".json",
                stager="/opt/inferssd-ci-control/current/bench/ci/operator_stage.py")


def machine_identity(value):
    if (not isinstance(value, dict) or set(value) != {"hardware_uuid_sha256", "hardware_model", "cpu_brand"}
            or HASH.fullmatch(value.get("hardware_uuid_sha256", "")) is None
            or any(not isinstance(value[k], str) or not value[k] or "\0" in value[k]
                   for k in ("hardware_model", "cpu_brand"))):
        raise ValueError("explicit reviewed Mac physical machine identity required")
    return value


def parsed(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key " + key)
            value[key] = item
        return value

    def nonfinite(_):
        raise ValueError("nonfinite JSON constant")

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            nonfinite(text)
        return value

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=number)


class Gh:
    """Call the existing gh login without reading or exporting its token."""

    def __init__(self, executable="gh"):
        self.executable = executable

    def call(self, path, data=None):
        argv = [self.executable, "api", "--hostname", "github.com", path]
        raw = None
        if data is not None:
            argv += ["--method", "POST", "--input", "-", "-H", "X-GitHub-Api-Version: 2026-03-10"]
            raw = json.dumps(data, allow_nan=False).encode()
        result = subprocess.run(argv, input=raw, capture_output=True, timeout=45)
        if result.returncode:
            # gh diagnostics may mention the active auth configuration. Do not
            # retain/forward authentication output into the device package.
            raise RuntimeError("existing gh login/API request failed: " + path)
        return parsed(result.stdout) if result.stdout.strip() else None


def git(source, *args):
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError("Git object operation failed: " + args[0])
    return result.stdout


def access(api, policy):
    user = api.call("user")
    actor, actor_id = user.get("login"), user.get("id")
    if not isinstance(actor, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", actor):
        raise ValueError("existing gh account identity missing")
    if type(actor_id) is not int or actor_id <= 0:
        raise ValueError("existing gh account numeric identity missing")
    permissions = {}
    for repository in (policy["repository"], policy["controller_repository"]):
        value = api.call("repos/" + repository + "/collaborators/" + actor + "/permission")
        permission = value.get("permission")
        if permission not in ("admin", "maintain", "write"):
            raise ValueError("operator needs write permission on " + repository)
        permissions[repository] = permission
    metadata = api.call("repos/" + policy["controller_repository"])
    if metadata.get("private") is not True or metadata.get("default_branch") != "main":
        raise ValueError("controller must be the private main repository")
    controller = revision(api.call("repos/" + policy["controller_repository"] + "/branches/main")["commit"]["sha"])
    return dict(login=actor, id=actor_id, permissions=permissions), controller


def source_identity(source, identities, api, repository):
    """Compare complete local Git objects with the source API, not a PR label."""
    objects = {}
    for name in ("base", "head", "candidate"):
        sha = revision(identities[name])
        if git(source, "rev-parse", sha + "^{commit}").decode().strip() != sha:
            raise ValueError("source identity is not an exact commit")
        tree = git(source, "rev-parse", sha + "^{tree}").decode().strip()
        parents = git(source, "show", "-s", "--format=%P", sha).decode().split()
        remote = api.call("repos/" + repository + "/git/commits/" + sha)
        if (remote.get("sha") != sha or remote.get("tree", {}).get("sha") != tree
                or [item.get("sha") for item in remote.get("parents", [])] != parents):
            raise ValueError("local source object differs from authenticated source repository")
        objects[name] = dict(revision=sha, tree=tree, parents=parents)
    base, head, candidate = (identities[k] for k in ("base", "head", "candidate"))
    if candidate == head:
        git(source, "merge-base", "--is-ancestor", base, head)
    elif objects["candidate"]["parents"] != [base, head]:
        raise ValueError("candidate must be head or the exact base/head merge")
    paths, diffs = [], {}
    for name, target in (("head", head), ("candidate", candidate)):
        changed = git(source, "diff", "--name-only", "-z", "--no-renames", "--no-ext-diff", base, target)
        paths.extend(item.decode("utf-8") for item in changed.split(b"\0") if item)
        diffs[name] = git(source, "diff", "--binary", "--full-index", "--no-renames", "--no-ext-diff", "--no-textconv", base, target)
    # A merge commit can contain resolution changes absent from its head. Cover
    # both complete diffs rather than silently omit candidate-only code.
    full_diff = b"BASE -> HEAD\n" + diffs["head"] + b"\nBASE -> CANDIDATE\n" + diffs["candidate"]
    return dict(commits=objects, diff_sha256=hashlib.sha256(full_diff).hexdigest(),
                component_diff_sha256={key: hashlib.sha256(value).hexdigest() for key, value in diffs.items()}), sorted(set(paths)), full_diff


def relative(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute()
            or any(x in ("", ".", "..") for x in value.split("/"))):
        raise ValueError("canonical package-relative input name required")
    return value


def input_file(root, name):
    path = root / relative(name)
    if path.is_symlink() or not path.is_file():
        raise ValueError("explicit regular reviewed input file required")
    # Input documents cannot escape through a symlinked ancestor either.
    if path.resolve() != root.resolve().joinpath(*PurePosixPath(name).parts):
        raise ValueError("reviewed input path has a symlink ancestor")
    return path


def substitute(value, job_id):
    if isinstance(value, str):
        return value.replace("{{job_id}}", job_id)
    if isinstance(value, list):
        return [substitute(x, job_id) for x in value]
    if isinstance(value, dict):
        return {key: substitute(item, job_id) for key, item in value.items()}
    return value


def replace_strings(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(x, replacements) for x in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def export_source(source, candidate, output, side="candidate"):
    """One batch Git blob reader; no checkout, hooks, export-ignore or source execution."""
    rows = []
    for item in git(source, "ls-tree", "-rz", candidate).split(b"\0"):
        if not item:
            continue
        descriptor, name = item.split(b"\t", 1)
        mode, kind, blob = descriptor.decode("ascii").split()
        name = relative(name.decode("utf-8"))
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError("source export requires only regular tracked Git files")
        rows.append(dict(path=name, mode=mode, blob=blob))
    if side not in ("base", "candidate"):
        raise ValueError("fixed source side required")
    archive_name = side + "-source.tar.gz"
    archive_path = output / archive_name
    process = subprocess.Popen(["git", "-C", str(source), "cat-file", "--batch"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        with archive_path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    process.stdin.write((row["blob"] + "\n").encode())
                    process.stdin.flush()
                    blob, kind, size = process.stdout.readline().decode("ascii").split()
                    size = int(size)
                    if blob != row["blob"] or kind != "blob" or not 0 <= size <= 128 << 20:
                        raise ValueError("Git source blob response differs")
                    row["bytes"] = size
                    item = tarfile.TarInfo(row["path"])
                    item.size, item.mode = size, int(row["mode"], 8) & 0o777
                    archive.addfile(item, process.stdout)
                    if process.stdout.read(1) != b"\n":
                        raise ValueError("Git batch blob framing differs")
        process.stdin.close()
        if process.wait(timeout=15):
            raise ValueError("Git batch source export failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    value = dict(revision=candidate, tree=git(source, "rev-parse", candidate + "^{tree}").decode().strip(),
                 archive=archive_name, archive_sha256=digest(archive_path), files=rows)
    save(output / ("source-map.json" if side == "candidate" else "base-source-map.json"), value)
    return value


def normalize(template, manifest, declared, job_id, identities, mode=MODE):
    if mode not in (MODE, LEGACY_MODE):
        raise ValueError("explicit supported operator validation mode required")
    old_job = manifest["job_id"]
    template, manifest = (replace_strings(value, [(old_job, job_id), ("{{job_id}}", job_id),
        ("/JOB/", "/" + job_id + "/")]) for value in (template, manifest))
    sides = ("base", "candidate") if mode == MODE else ("candidate",)
    layout = device_layout(template["device"], job_id)
    mac = layout["platform"] == "darwin"
    if mac and mode != MODE:
        raise ValueError("Mac operator staging requires the explicit performance mode")
    root = layout["source"]
    source_roots = {side: root + "/" + side if mode == MODE else root for side in sides}
    manifest["sources"] = source_roots
    manifest["source_revisions"] = {side: identities[side] for side in sides}
    manifest["builds"] = {side: side for side in sides}
    mounts = {model: manifest["models"][model] for model in template["sides"]["candidate"]["models"]}
    cache_treatment = (template["host"].get("os_page_cache_treatment") if mode == MODE else
        "not-forced/uncontrolled; functional/numerical only; no cache-drop or cold-cache assertion")
    if not isinstance(cache_treatment, str) or not cache_treatment.strip():
        raise ValueError("explicit shared OS page cache treatment required")
    template["host"]["os_page_cache_treatment"] = cache_treatment
    for side, previous in list(template["sides"].items()):
        spec = replace_strings(previous, [(previous["revision"], identities[side])])
        if mac and side in sides:
            spec = replace_strings(spec, [(spec["paths"]["source"], source_roots[side])])
        template["sides"][side] = spec
        for model, data in spec["models"].items():
            if side in sides:
                prefix = spec["paths"]["source"].rstrip("/") + "/" if mac else "/source/" + side + "/"
                path = data["template"]["runtime_path"]
                if not path.startswith(prefix):
                    raise ValueError("template must belong to its exact source side")
                template_path = source_roots[side] + "/" + relative(path[len(prefix):])
                data["template"]["host_path"] = template_path
                settings = data["settings"]
                old_template, old_settings = path, settings["runtime_path"]
                settings["host_path"] = (root + "/settings-" + model + "-" + side + ".ssd" if mac else
                    "/opt/inferssd-ci-integrations/" + job_id + "-" + model + "-" + side + "-settings.ssd")
                settings["runtime_path"] = settings["host_path"] if mac else "/models/settings-" + model + "-" + side + ".ssd"
                if mac:
                    data["template"]["runtime_path"] = template_path
                    for launch in [data["smoke"], *data.get("benchmarks", {}).values()]:
                        for key in ("argv", "preflight_argv"):
                            if key in launch:
                                launch[key] = [template_path if x == old_template else settings["runtime_path"]
                                               if x == old_settings else x for x in launch[key]]
                mounts["settings-" + model + "-" + side + ".ssd"] = settings["host_path"]
            if "model_identity" not in data:
                data["model_identity"] = copy.deepcopy(template["model_identity"][model])
            for row in data["model_identity"]["shards"]:
                if "stat" not in row:
                    row["stat"] = {key: row[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")}
                if "sha256" not in row:
                    row["sha256"] = row["historical_sha256"]
            data["smoke"]["os_page_cache_treatment"] = cache_treatment
            if mode == LEGACY_MODE and data.get("benchmarks"):
                raise ValueError("operator functional/numerical input must not contain timed arms")
        if mac and side in sides:
            spec["paths"]["source"] = source_roots[side]
    manifest["models"] = mounts
    template["manifest_path"] = layout["manifest"]
    if mac:
        manifest["input_roots"] = [root]
        manifest["model_roots"] = sorted({mounts[model] for model in template["sides"]["candidate"]["models"]})
        manifest["selection_lock_identity"] = declared["lock_identity"]
    if mode == MODE:
        template["performance_protocol"] = performance_protocol()
    for key in ("control_revision", "control_manifest_sha256"):
        template[key] = declared[key]
    revision(template["control_revision"])
    if HASH.fullmatch(template["control_manifest_sha256"]) is None:
        raise ValueError("explicit installed control manifest hash required")
    return template, manifest


def staged_settings_name(model, side, mode):
    return "settings-" + model + ("-" + side if mode == MODE else "") + ".ssd"


def paired_settings(base, candidate, *, device=None, job_id=None):
    # Only the side's source/build namespace is an identity difference. Every
    # other byte (including typed resource values) must remain identical.
    if device == "local-mbp":
        if not isinstance(job_id, str) or JOB.fullmatch(job_id) is None:
            raise ValueError("exact Mac job namespace required")
        root = ("/Library/InferSSD-CI/jobs/" + job_id + "/frozen/").encode()
        return base.replace(root + b"base/", root + b"candidate/") == candidate
    return base.replace(b"/build/base/", b"/build/candidate/").replace(b"/source/base/", b"/source/candidate/") == candidate


def root_policy(config, manifest, declared):
    value = dict(schema_version=1, purpose="operator-stage-policy", device=config["device"],
        broker_config_sha256=config["broker_config_sha256"], lock_identity=declared["lock_identity"],
        control_revision=config["control_revision"], control_manifest_sha256=config["control_manifest_sha256"],
        resources_sha256=fingerprint(config["host"]["resources"]), min_free_bytes=declared["min_free_bytes"],
        models={model: dict(mount=manifest["models"][model], identity_sha256=fingerprint(data["model_identity"]),
                           settings_sha256=data["settings"]["sha256"])
                for model, data in config["sides"]["candidate"]["models"].items()},
        model_admission={key: config[key] for key in ("model_admission_path", "model_admission_sha256",
            "model_mount_proof_path", "model_mount_proof_sha256", "prior_model_job_id") if key in config})
    if config["device"] == "local-mbp":
        value.update(broker_device=declared["broker_device"], machine_identity=machine_identity(declared["machine_identity"]))
        for model in value["models"].values():
            model["settings_template_sha256"] = model.pop("settings_sha256")
    else:
        value["gpu_uuid"] = declared["gpu_uuid"]
    if config.get("validation_mode") == MODE:
        value.update(performance_protocol=performance_protocol(), side_settings_sha256={side: {
            model: data["settings"]["sha256"] for model, data in config["sides"][side]["models"].items()}
            for side in ("base", "candidate")})
        if config["device"] == "local-mbp":
            value["side_settings_template_sha256"] = value.pop("side_settings_sha256")
    return value


def prepare(args, api=None):
    mode = getattr(args, "validation_mode", MODE)
    if mode not in (MODE, LEGACY_MODE):
        raise ValueError("explicit supported operator validation mode required")
    if mode == MODE:
        import final_chat
        final_chat.prompt_path("512")  # Trusted recorder input, never PR source.
    policy = policy_read(load(args.policy))
    specification = load(args.inputs)
    execution = list(args.execution_devices)
    if (not execution or len(set(execution)) != len(execution) or not set(execution) <= set(DEVICES)
            or specification.get("schema_version") != 1
            or not set(execution) <= set(specification.get("devices", {}))
            or not set(specification.get("devices", {})) <= set(DEVICES)):
        raise ValueError("explicit reviewed inputs for each selected device required")
    actor, workflow_sha = access(api or Gh(args.gh), policy)
    api = api or Gh(args.gh)
    if workflow_sha != revision(specification["controller_workflow_sha"]):
        raise ValueError("private controller main changed from reviewed input")
    identities = {key: revision(getattr(args, key)) for key in ("base", "head", "candidate")}
    source, paths, full_diff = source_identity(args.source, identities, api, policy["repository"])
    impact = plan(paths, policy, **identities)
    execution = [device for device in execution if device_requirements(impact, device)["models"]]
    if not execution:
        raise ValueError("impact requires no enabled device; do not stage or dispatch")
    root = args.inputs.resolve().parent
    devices, jobs, inputs, pending_assets = {}, {}, {}, {}
    for device in execution:
        declared = specification["devices"][device]
        job_id = declared.get("job_id", uuid.uuid4().hex)
        if not isinstance(job_id, str) or JOB.fullmatch(job_id) is None:
            raise ValueError("fresh 32-hex job ID required")
        template_path, manifest_path = (input_file(root, declared[key]) for key in ("template", "manifest"))
        template, manifest = normalize(*(load(path) for path in (template_path, manifest_path)), declared, job_id, identities, mode)
        if template.get("device") != device or manifest.get("job_id") != job_id:
            raise ValueError("device/job template differs from selected slot")
        template.update(execution_devices=execution, validation_mode=mode, selection_plan=impact)
        admission_policy = root_policy(template, manifest, declared)
        if device == "local-mbp":
            from ci.operator_mac_stage import settings_payload
            for side, spec in template["sides"].items():
                for model, data in spec["models"].items():
                    name = staged_settings_name(model, side, mode)
                    raw = input_file(root, declared["rootstage_files"][name]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != data["settings"]["sha256"]:
                        raise ValueError("Mac settings template differs from its reviewed hash")
                    raw = settings_payload(raw, job_id, side)
                    pending_assets[device, name] = raw
                    data["settings"]["sha256"] = hashlib.sha256(raw).hexdigest()
        if mode == MODE:
            for side, spec in template["sides"].items():
                for data in spec["models"].values():
                    prefix = template["sides"][side]["paths"]["source"].rstrip("/") + "/" if device == "local-mbp" else "/source/" + side + "/"
                    relative_template = relative(data["template"]["runtime_path"][len(prefix):])
                    data["template"]["sha256"] = hashlib.sha256(git(args.source, "show", identities[side] + ":" + relative_template)).hexdigest()
        needs = runtime_requirements(template)
        for spec in template["sides"].values():
            spec["models"] = {name: data for name, data in spec["models"].items() if name in needs["models"]}
            for model, data in spec["models"].items():
                tiers = needs["performance"].get(model, {}).get("tiers", [])
                if not set(tiers) <= set(data.get("benchmarks", {})):
                    raise ValueError("required performance launch is missing from reviewed profile")
                data["benchmarks"] = {tier: data["benchmarks"][tier] for tier in tiers}
        manifest["models"] = {name: path for name, path in manifest["models"].items() if name in needs["models"] or
            any(name == "settings-" + model + "-" + side + ".ssd" for model in needs["models"] for side in ("base", "candidate"))}
        active_sides = ("base", "candidate") if needs["performance"] else ("candidate",)
        for key in ("sources", "source_revisions", "builds"):
            manifest[key] = {side: manifest[key][side] for side in active_sides}
        manifest["models"] = {name: path for name, path in manifest["models"].items() if name in needs["models"] or
            any(name == "settings-" + model + "-" + side + ".ssd" for model in needs["models"] for side in active_sides)}
        if device == "local-mbp":
            manifest["model_roots"] = sorted({manifest["models"][model] for model in needs["models"]})
        config, manifest = bind(template, manifest, impact, policy)
        if mode == MODE:
            from ci.selected_preflight import declarations
            declarations(config, manifest, needs)
        if config.get("validation_mode") != mode:
            raise ValueError("trusted selection contract does not support the explicit execution mode")
        # Functional/numerical mode must never silently retain timed arms.
        if mode == LEGACY_MODE and (manifest.get("source_revisions") != {"candidate": identities["candidate"]}
                or any(data.get("benchmarks") for spec in config["sides"].values() for data in spec["models"].values())
                or any("benchmark" in name or "preflight" in name for name in manifest.get("stages", {}))):
            raise ValueError("functional/numerical preparation cannot authorize a performance stage")
        lock = declared["lock_identity"]
        if not isinstance(lock, list) or len(lock) != 2 or any(type(x) is not int or x < 0 for x in lock):
            raise ValueError("actual original device/inode lease identity required")
        if device == "local-mbp":
            machine_identity(declared.get("machine_identity"))
            if not isinstance(declared.get("broker_device"), str) or not declared["broker_device"]:
                raise ValueError("explicit installed Mac broker device required")
        elif not isinstance(declared["gpu_uuid"], str) or not declared["gpu_uuid"].startswith("GPU-"):
            raise ValueError("explicit actual GPU UUID required")
        devices[device] = dict(config=config, manifest=manifest, policy=admission_policy)
        assets = {}
        for name, local in declared.get("rootstage_files", {}).items():
            path = input_file(root, local)
            if path.suffix == ".gguf" or path.stat().st_size > 128 << 20:
                raise ValueError("root staging accepts small source/settings metadata, never model payload")
            assets[relative(name)] = dict(local_path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path))
            if (device, name) in pending_assets:
                raw = pending_assets[device, name]
                assets[name] = dict(local_path=str((args.output / device / name).resolve()),
                                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        sides = ("base", "candidate") if needs["performance"] else ("candidate",)
        expected_assets = {staged_settings_name(model, side, mode) for side in sides for model in needs["models"]}
        if not expected_assets <= set(assets):
            raise ValueError("explicit settings payloads for exactly the selected sides/models required")
        assets = {name: row for name, row in assets.items() if name in expected_assets}
        for side in sides:
            for model, data in config["sides"][side]["models"].items():
                name = staged_settings_name(model, side, mode)
                if assets[name]["sha256"] != data["settings"]["sha256"]:
                    raise ValueError("settings payload differs from declared side hash")
        if needs["performance"]:
            for model in needs["models"]:
                base_data, candidate_data = (config["sides"][side]["models"][model] for side in ("base", "candidate"))
                def payload(side):
                    name = staged_settings_name(model, side, mode)
                    return pending_assets[device, name] if device == "local-mbp" else Path(assets[name]["local_path"]).read_bytes()
                if (base_data["model_identity"] != candidate_data["model_identity"] or not paired_settings(
                        payload("base"), payload("candidate"), device=device, job_id=job_id)):
                    raise ValueError("base/candidate model or effective settings differ")
        minimum = declared["min_free_bytes"]
        if type(minimum) is not int or minimum <= 0:
            raise ValueError("explicit positive staging free-space reserve required")
        if not HASH.fullmatch(config.get("broker_config_sha256", "")):
            raise ValueError("explicit existing broker configuration hash required")
        transport_profile(declared["ssh"], device=device)
        inputs[device] = dict(job_id=job_id, reviewed_template_sha256=digest(template_path),
                              reviewed_manifest_sha256=digest(manifest_path), files=assets,
                              ssh=declared["ssh"], min_free_bytes=minimum)
        if device == "local-mbp":
            inputs[device].update(machine_identity=declared["machine_identity"], lock_identity=lock)
    if len({row["manifest"]["job_id"] for row in devices.values()}) != len(execution):
        raise ValueError("each device needs a distinct fresh job")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "policy.json", policy)
    save(args.output / "source-identity.json", source)
    save(args.output / "changed-paths.json", paths)
    (args.output / "source.diff").write_bytes(full_diff)
    save(args.output / "selection-plan.json", impact)
    required_sides = {side for row in devices.values() for side in row["manifest"]["sources"]}
    source_maps = {side: export_source(args.source, identities[side], args.output, side) for side in ("base", "candidate") if side in required_sides}
    source_map = source_maps["candidate"]
    for device, row in devices.items():
        folder = args.output / device
        folder.mkdir()
        for name in inputs[device]["files"]:
            if (device, name) in pending_assets:
                (folder / name).write_bytes(pending_assets[device, name])
        save(folder / "manifest.json", row["manifest"])
        row["config"]["manifest_sha256"] = digest(folder / "manifest.json")
        bound(row["config"], row["manifest"], row["manifest"]["job_id"])
        save(folder / "config.json", row["config"])
        save(folder / "policy.review.json", row["policy"])
        jobs[device] = dict(device=device, job_id=row["manifest"]["job_id"],
            source_revision=identities["candidate"], config_sha256=digest(folder / "config.json"),
            selection_plan_sha256=fingerprint(impact), control_revision=row["config"]["control_revision"],
            control_manifest_sha256=row["config"]["control_manifest_sha256"])
        inputs[device].update(config_path=str((folder / "config.json").resolve()),
            config_sha256=jobs[device]["config_sha256"], manifest_path=str((folder / "manifest.json").resolve()),
            manifest_sha256=row["config"]["manifest_sha256"])
    envelope = dict(schema_version=1, purpose=PURPOSE, repository=policy["repository"],
        **identities, prepared_by=actor, controller_repository=policy["controller_repository"],
        controller_workflow_sha=workflow_sha, selection=impact, selection_plan_sha256=fingerprint(impact),
        execution_devices=execution, validation_mode=mode, jobs=jobs, policy_sha256=fingerprint(policy))
    if len(json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode()) > MAX_INPUT:
        raise ValueError("complete prepared envelope exceeds workflow input bound; never truncate impact")
    save(args.output / "envelope.json", envelope)
    for device, row in devices.items():
        helper_sha = digest(Path(__file__).with_name("operator_mac_stage.py" if device == "local-mbp" else "operator_stage.py"))
        folder = args.output / device
        files = dict(inputs[device]["files"])
        for name, path in (("config.json", folder / "config.json"), ("manifest.json", folder / "manifest.json"),
                           ("source-map.json", args.output / "source-map.json"),
                           ("candidate-source.tar.gz", args.output / "candidate-source.tar.gz"),
                           ("envelope.json", args.output / "envelope.json")):
            files[name] = dict(local_path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path))
        if "base" in row["manifest"]["sources"]:
            for name in ("base-source-map.json", "base-source.tar.gz"):
                path = args.output / name
                files[name] = dict(local_path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path))
        request = dict(schema_version=1, job_id=inputs[device]["job_id"], device=device,
             candidate=identities["candidate"], source_tree=source_map["tree"],
             broker_config_sha256=row["config"]["broker_config_sha256"],
             min_free_bytes=inputs[device]["min_free_bytes"], operator_stage_sha256=helper_sha,
             policy_sha256=digest(folder / "policy.review.json"),
             files={name: {key: value[key] for key in ("bytes", "sha256")} for name, value in files.items()})
        if mode == MODE:
            request["sources"] = {side: {key: source_maps[side][key] for key in ("revision", "tree")} for side in row["manifest"]["sources"]}
        save(folder / "root-input.json", request)
        inputs[device].update(files=files, root_input_path=str((folder / "root-input.json").resolve()),
                             root_input_sha256=digest(folder / "root-input.json"), policy_sha256=digest(folder / "policy.review.json"))
    save(args.output / "rootstage-inputs.json", dict(schema_version=1, purpose="operator-rootstage-inputs",
         envelope_sha256=digest(args.output / "envelope.json"), source_identity_sha256=digest(args.output / "source-identity.json"),
         selection_plan_sha256=fingerprint(impact), devices=inputs))
    files = {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    save(args.output / "prepared-files.json", dict(schema_version=1, files=files))
    return envelope


def transport_profile(value, *, device="cloud-5090"):
    argv = value.get("argv")
    if (not isinstance(argv, list) or not argv or Path(argv[0]).name != "ssh" or "-T" not in argv
            or any(not isinstance(x, str) or not x or "\0" in x for x in argv)
            or any(x in argv for x in ("-t", "-tt", "-A"))):
        raise ValueError("reviewed SSH profile must disable PTY and agent forwarding")
    if "-a" not in argv:
        raise ValueError("SSH profile must explicitly disable agent forwarding")
    expected = [MAC_PYTHON if device == "local-mbp" else "/usr/bin/python3", "-I",
                *(["-B"] if device == "local-mbp" else []), device_layout(device, "0" * 32)["stager"]]
    root_argv = value.get("root_argv")
    if device == "local-mbp" and (value.get("sudo_password_required") is not False
            or value.get("sudo_mode", "root") != "root"):
        raise ValueError("Mac background stage requires the fixed forced SSH entry without sudo wrapping")
    if value.get("sudo_password_required") is True:
        expected = ["/usr/bin/sudo", "-S", "-p", "", *expected]
    elif value.get("sudo_password_required") is not False:
        raise ValueError("explicit root/sudo authentication mode required")
    elif value.get("sudo_mode") == "nopasswd":
        expected = ["/usr/bin/sudo", "-n", *expected]
    elif value.get("sudo_mode", "root") != "root":
        raise ValueError("unknown fixed sudo mode")
    if root_argv != expected:
        raise ValueError("fixed reviewed root stage executable required")
    return argv, root_argv


def verified(prepared):
    expected = load(prepared / "prepared-files.json")["files"]
    for name, sha in expected.items():
        path = input_file(prepared, name)
        if digest(path) != sha:
            raise ValueError("prepared source/config/package changed")
    return load(prepared / "envelope.json"), load(prepared / "rootstage-inputs.json")


def claim(path, value):
    """Durable once-only witness before an irreversible external action."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def send_stage(device, data, *, password=None):
    argv, root_argv = transport_profile(data["ssh"], device=device)
    if data["ssh"]["sudo_password_required"] and (not isinstance(password, str) or "\n" in password or "\r" in password):
        raise ValueError("one in-memory sudo password required")
    request = Path(data["root_input_path"])
    if digest(request) != data["root_input_sha256"]:
        raise ValueError("reviewed root stage input changed")
    members = {"root-input.json": dict(local_path=str(request), bytes=request.stat().st_size,
                                      sha256=data["root_input_sha256"]), **data["files"]}
    # Verify every local member before connecting, not after sending a prefix.
    for row in members.values():
        path = Path(row["local_path"])
        if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("root-stage input member changed")
    command = shlex.join([*root_argv, "--expected-sha256", data["root_input_sha256"]])
    # Authentication stays local to ssh; gh/API credentials and loader overrides
    # are neither in argv/package nor inherited for accidental SendEnv rules.
    environment = {key: value for key, value in os.environ.items()
                   if key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "SSH_AUTH_SOCK")}
    process = subprocess.Popen([*argv, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=environment)
    try:
        if data["ssh"]["sudo_password_required"]:
            process.stdin.write((password + "\n").encode())
            process.stdin.flush()
        with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, row in members.items():
                item = tarfile.TarInfo(name)
                item.size, item.mode = row["bytes"], 0o400
                with Path(row["local_path"]).open("rb") as source:
                    archive.addfile(item, source)
        process.stdin.close()
        process.stdin = None
        raw, _ = process.communicate(timeout=300)
        if process.returncode or len(raw) > 1 << 20:
            raise RuntimeError("root staging failed; retain fresh job and stop without retry")
        receipt = parsed(raw)
        if receipt.get("device") != device:
            raise ValueError("root stage returned another physical device")
        return receipt
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def stage(args, sender=send_stage, *, password=None):
    envelope, inputs = verified(args.prepared)
    for device in envelope["execution_devices"]:
        data, job = inputs["devices"][device], envelope["jobs"][device]
        # O_EXCL witnesses prevent an ambiguous SSH result being retried by run.
        claim(args.prepared / device / "staging-attempt.json", dict(device=device, job_id=job["job_id"],
             root_input_sha256=data["root_input_sha256"]))
        receipt = sender(device, data, password=password)
        validate_root_receipt(envelope, inputs, device, receipt)
        save(args.prepared / device / "root-stage-receipt.json", receipt)
    return envelope


def validate_root_receipt(envelope, inputs, device, receipt):
    job, data = envelope["jobs"][device], inputs["devices"][device]
    required = dict(status="passed", root_uid=0, device=device, job_id=job["job_id"],
        candidate=envelope["candidate"], root_input_sha256=data["root_input_sha256"],
        envelope_sha256=inputs["envelope_sha256"], config_sha256=job["config_sha256"],
        manifest_sha256=data["manifest_sha256"], control_revision=job["control_revision"],
        control_manifest_sha256=job["control_manifest_sha256"], broker_config_unchanged=True,
        existing_coordination_unmodified=True)
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("actual root staging receipt/source/config/control identity differs")
    if envelope.get("validation_mode") == MODE:
        request = load(Path(data["root_input_path"]))
        observed = receipt.get("sources", {})
        if set(observed) != set(request["sources"]) or any(
                observed[side].get("root_owned_readonly") is not True or any(observed[side].get(key) != value
                    for key, value in expected.items()) for side, expected in request["sources"].items()):
            raise ValueError("root staging did not seal the exact required source sides")
    if receipt.get("source", {}).get("root_owned_readonly") is not True:
        raise ValueError("root staging did not seal its source")
    if receipt.get("policy_sha256") != data["policy_sha256"]:
        raise ValueError("root stage did not use the reviewed local admission policy")
    if device == "local-mbp":
        recovered = receipt.get("broker_recovery", {})
        request = load(Path(data["root_input_path"]))
        if (receipt.get("machine_identity") != data["machine_identity"]
                or receipt.get("original_lock") != data["lock_identity"]
                or receipt.get("resource_enforcement") != "sampled-UMA-guard-not-hard-cgroup"
                or recovered.get("status") != "ready"
                or recovered.get("config_sha256") != request["broker_config_sha256"]):
            raise ValueError("actual Mac physical/resource/recovery receipt differs")


def dispatch(args, api=None):
    if (args.prepared / "dispatch-attempt.json").exists():
        raise FileExistsError("prepared job already has a dispatch attempt; retain it without retry")
    envelope, inputs = verified(args.prepared)
    policy = policy_read(load(args.prepared / "policy.json"))
    api = api or Gh(args.gh)
    actor, workflow_sha = access(api, policy)
    if (any(actor[k] != envelope["prepared_by"][k] for k in ("login", "id"))
            or workflow_sha != envelope["controller_workflow_sha"]):
        raise ValueError("operator account or controller main changed; prepare a new job")
    for device in envelope["execution_devices"]:
        validate_root_receipt(envelope, inputs, device, load(args.prepared / device / "root-stage-receipt.json"))
    workflow = api.call("repos/" + policy["controller_repository"] + "/actions/workflows/device-performance.yml")
    if workflow.get("path") != WORKFLOW or workflow.get("state") != "active":
        raise ValueError("fixed report-only controller workflow is not active")
    body = dict(ref="main", inputs=dict(prepared_envelope=json.dumps(envelope, separators=(",", ":"), allow_nan=False)),
                return_run_details=True)
    claim(args.prepared / "dispatch-attempt.json", dict(workflow_sha=workflow_sha,
         envelope_sha256=digest(args.prepared / "envelope.json"), workflow_id=workflow["id"]))
    result = api.call("repos/" + policy["controller_repository"] + "/actions/workflows/" + str(workflow["id"]) + "/dispatches", body)
    if not isinstance(result, dict) or type(result.get("workflow_run_id")) is not int or result["workflow_run_id"] <= 0:
        raise ValueError("dispatch did not return an exact run ID; no retry or guessed run")
    run = api.call("repos/" + policy["controller_repository"] + "/actions/runs/" + str(result["workflow_run_id"]))
    if (run.get("id") != result["workflow_run_id"] or run.get("workflow_id") != workflow["id"]
            or run.get("path") != WORKFLOW or run.get("event") != "workflow_dispatch"
            or run.get("head_branch") != "main" or run.get("head_sha") != workflow_sha
            or run.get("run_attempt") != 1 or run.get("actor", {}).get("id") != actor["id"]):
        raise ValueError("dispatched controller run identity differs")
    save(args.prepared / "dispatched-run.json", run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("prepare", "stage", "dispatch", "run"):
        command = sub.add_parser(mode)
        if mode in ("prepare", "run"):
            for key in ("source", "inputs", "output"):
                command.add_argument("--" + key, type=Path, required=True)
            for key in ("base", "head", "candidate"):
                command.add_argument("--" + key, required=True)
            command.add_argument("--policy", type=Path, default=Path(__file__).with_name("policy.json"))
            command.add_argument("--validation-mode", choices=(MODE, LEGACY_MODE), default=MODE)
            command.add_argument("--execution-devices", nargs="+", choices=DEVICES, default=list(DEVICES))
        else:
            command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--gh", default="gh", help="Existing authorized gh executable; no token export")
        if mode in ("stage", "run"):
            command.add_argument("--sudo-prompt", action="store_true", help="One hidden in-memory sudo password, never stored or passed as an argument")
    args = parser.parse_args()
    if args.command in ("prepare", "run"):
        prepare(args)
        args.prepared = args.output
    if args.command in ("stage", "run"):
        envelope, inputs = verified(args.prepared)
        needed = any(inputs["devices"][d]["ssh"]["sudo_password_required"] for d in envelope["execution_devices"])
        if needed and not args.sudo_prompt:
            parser.error("Spark root staging requires --sudo-prompt; no password fallback")
        password = None
        if needed:
            import getpass
            password = getpass.getpass("Existing operator sudo password (memory only): ")
        try:
            stage(args, password=password)
        finally:
            password = None
    if args.command in ("dispatch", "run"):
        value = dispatch(args)
        print(json.dumps(dict(run_id=value["id"], html_url=value.get("html_url")), indent=2))


if __name__ == "__main__":
    main()
