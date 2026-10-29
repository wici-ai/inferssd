#!/usr/bin/env python3
"""One Linux Muse 512/256 comparison with explicit broker lease provenance.

Operator configuration binds the root manifest, both source revisions, build and
benchmark specs, model file identities, host/resources and the explicit reference
mapping. Only declared null profile/image/alias hashes are completed from actual
evidence; settings, image roles and source identities are never inferred.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import io
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_final

from ci.compare import compare_pair
from ci.contract import digest, load, policy_read, revision, save
from ci.isolated_device import protected, protected_manifest
from ci.reference import LEGACY_REVISION, identity, validate_binding
from ci.resume_base import recover
from ci.stage_session import StageSession, unique_object
from ci.staged_benchmark import benchmark
from ci.staged_build import run_build, sealed_file
from ci.staged_smoke import collect_log


def checked_hash(declared, actual):
    if declared is not None and declared != actual:
        raise ValueError("declared hash differs from the actual retained artifact")
    return actual


def check_declaration(binding):
    """Check structure before expensive work, without claiming unknown hashes."""
    structural = copy.deepcopy(binding)
    for side in structural["sides"].values():
        if side["profile_sha256"] is None:
            side["profile_sha256"] = "0" * 64
        for image in side["images"]:
            if image["sha256"] is None:
                image["sha256"] = "0" * 64
        side["launch_policy"]["alias_sha256"] = [
            "0" * 64 if item is None else item
            for item in side["launch_policy"]["alias_sha256"]]
    validate_binding(structural)


def file_identity(path):
    path = protected(path)
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("model identity requires a regular protected file")
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def model_mount(manifest, host_path, runtime_path):
    path, runtime = Path(host_path), Path(runtime_path)
    return sum(path.is_relative_to(Path(root)) and
               runtime == Path("/models") / name / path.relative_to(Path(root))
               for name, root in manifest["models"].items()) == 1


def models(config, manifest):
    """No GGUF payload read: exact root-bound files and unchanged stat identities."""
    shards = config["model_identity"]["shards"]
    if len(shards) != 22 or len({x["runtime_path"] for x in shards}) != 22:
        raise ValueError("checkpoint requires the exact 22 Muse shards")
    result = []
    for item in shards:
        path = Path(item["host_path"])
        if not model_mount(manifest, path, item["runtime_path"]) or path.suffix != ".gguf":
            raise ValueError("model path is not the exact root-manifest readonly mount")
        actual = file_identity(path)
        if any(item.get(key) != value for key, value in actual.items()):
            raise ValueError("root-bound model identity changed")
        result.append(dict(item, **actual))
    return dict(config["model_identity"], shards=result,
                verification="root-bound readonly files; exact stat identity before/after; no payload rehash")


def asset(spec, *, contents=False):
    evidence, raw = sealed_file(spec["host_path"], limit=64 << 20, contents=True)
    if evidence["sha256"] != spec["sha256"]:
        raise ValueError("sealed template/settings differs from reviewed bytes")
    return (evidence, raw) if contents else evidence


def alias_asset(build, side, model):
    root = Path(build["freeze"]["host_path"])
    evidence, raw = sealed_file(root / "assets/muse-alias.tsv", limit=64 << 20, contents=True)
    shards = {item["runtime_path"]: item["bytes"] for item in model["shards"]}
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    # Both pinned generators emit five addressing columns followed by five
    # metadata columns. The engine reads the addressing fields from this table.
    if rows.fieldnames != ["tensor", "expert_idx", "nbytes", "source_path", "offset",
                          "kind", "layer", "type", "aligned", "shard_index"]:
        raise ValueError("unexpected direct-GGUF alias format")
    count = 0
    for row in rows:
        size, offset = int(row["nbytes"]), int(row["offset"])
        if (row["source_path"] not in shards or int(row["expert_idx"]) < 0
                or size <= 0 or offset < 0 or offset + size > shards[row["source_path"]]):
            raise ValueError("alias does not address the reviewed model files")
        count += 1
    if not count:
        raise ValueError("empty expert alias")
    return dict(evidence, runtime_path=f"/build/{side}/assets/muse-alias.tsv", entries=count)


def normalized_args(spec):
    args = spec["benchmark"]["argv"]
    result, index = [], 1
    while index < len(args):
        value = args[index]
        if value in ("--chat-template-file", "--ssd-settings"):
            name = "template" if value == "--chat-template-file" else "settings"
            if index + 1 >= len(args) or args[index + 1] != spec[name]["runtime_path"]:
                raise ValueError("server argv does not use the exact reviewed asset")
            if name == "template":
                result.extend([value, "<verified-identical-template>"])
            index += 2
        else:
            result.append(value)
            index += 1
    if args.count("--chat-template-file") != 1 or args.count("--ssd-settings") != int("settings" in spec):
        raise ValueError("explicit unique template and candidate settings arguments required")
    return result


def native_settings(stderr):
    rows = []
    prefix = b"[ssd.settings] "
    for line in Path(stderr).read_bytes().splitlines():
        if line.startswith(prefix):
            rows.append(json.loads(line[len(prefix):].decode("utf-8"), object_pairs_hook=unique_object))
    if len(rows) != 1 or not isinstance(rows[0].get("values"), dict):
        raise ValueError("candidate must emit exactly one native settings snapshot")
    return rows[0]


def mapped_files(probe):
    files = {}
    for line in probe["proc_maps"].splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if path.endswith(" (deleted)"):
            raise ValueError("running executable image was deleted")
        major, minor = (int(value, 16) for value in fields[3].split(":"))
        observed = (major, minor, int(fields[4]))
        if path in files and files[path] != observed:
            raise ValueError("runtime image path maps multiple file identities")
        files[path] = observed
    return files


def image_identity(host_path, observed):
    value = Path(host_path).stat()
    if (os.major(value.st_dev), os.minor(value.st_dev), value.st_ino) != observed:
        raise ValueError("runtime mapping is not the identified host file")


def system_image(path, observed):
    roots = (Path("/usr"), Path("/lib"), Path("/lib64"),
             Path("/opt/inferssd-ci-toolchains/cuda-12.8"))
    original = Path(path)
    target = original.resolve(strict=True)
    if (not any(original.is_relative_to(root) for root in roots)
            or not any(target.is_relative_to(root.resolve()) for root in roots)):
        raise ValueError("unreviewed executable image outside sealed build or system roots")
    target = protected(target)
    image_identity(target, observed)
    before = target.stat()
    if before.st_size > 2 << 30:
        raise ValueError("system image exceeds bounded hash read")
    value = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("system image changed during observation")
    return dict(path=path, sha256=value.hexdigest())


def validate_candidate(session, spec, build, output):
    """The one optional real-CUDA short gate, from the same sealed build."""
    output.mkdir()
    validation = spec["validation"]
    if validation["argv"] != ["/build/candidate/bin/test-ssd-dense-stream"]:
        raise ValueError("checkpoint only admits the one reviewed CUDA short gate")
    executable = sealed_file(Path(build["freeze"]["host_path"]) / "bin/test-ssd-dense-stream")
    stage = session.start("candidate-dense-stream", "test", validation["argv"],
                          validation["environment"], validation["cwd"], validation["timeout_s"])
    status = session.wait(stage, deadline=time.monotonic() + validation["timeout_s"])
    collect_log(session, stage, output)
    save(output / "stage.json", dict(spec=validation, result=status, executable=executable))
    native = status.get("receipt")
    if (status.get("running") is not False or status.get("returncode") != 0
            or status.get("cleanup_confirmed") is not True
            or status.get("state") in ("failed", "quarantined")
            or isinstance(native, dict) and native.get("passed") is not True):
        raise RuntimeError("required CUDA short gate failed")
    marker = validation.get("required_stdout", "SSD_DENSE_STREAM_OK")
    if marker != "SSD_DENSE_STREAM_OK" or marker not in (output / "stdout.log").read_text():
        raise ValueError("CUDA short gate did not report actual successful execution")


def runtime_receipt(config, side, spec, build, output, binding, model, *, evidence=None):
    evidence = output if evidence is None else evidence
    before, after = (load(evidence / (name + ".json")) for name in ("runtime-before", "runtime-after"))
    resources = config["host"]["resources"]
    ambient = None
    mapped = {}
    for probe in (before, after):
        if (probe["cmdline"] != spec["benchmark"]["argv"]
                or probe["executable"] != build["binary"]["runtime_path"]
                or probe["executable_sha256"] != build["binary"]["sha256"]
                or probe["cpus_allowed_list"] != resources["cpus_allowed_list"]
                or probe["cgroup_limits"] != resources["cgroup_limits"]):
            raise ValueError("native executable, argv or resource envelope differs")
        lock = re.search(r"^Max locked memory\s+(\S+)\s+(\S+)\s+bytes\s*$", probe["proc_limits"], re.M)
        if not lock or lock.groups() != (str(resources["memlock_bytes"]),) * 2:
            raise ValueError("native memlock differs from the reviewed resource envelope")
        environment = dict(probe["environment"])
        if environment.pop("PWD", None) != spec["benchmark"]["cwd"]:
            raise ValueError("native working-directory environment differs")
        for key, value in spec["benchmark"]["environment"].items():
            if environment.pop(key, None) != value:
                raise ValueError("native environment differs from exact launch profile")
        if ambient is not None and environment != ambient:
            raise ValueError("ambient environment changed during the request")
        ambient = environment
        for path, identity_tuple in mapped_files(probe).items():
            if path in mapped and mapped[path] != identity_tuple:
                raise ValueError("mapped image changed during the request")
            mapped[path] = identity_tuple
    images = {item["runtime_path"]: item for item in build["images"]}
    binary = build["binary"]
    image_identity(binary["path"], mapped.pop(binary["runtime_path"]))
    reviewed = binding["sides"][side]
    if {item["path"] for item in reviewed["images"]} != set(images):
        raise ValueError("operator role map must cover the exact sealed build image list")
    actual, alias_provenance = [], []
    for item in reviewed["images"]:
        path, image = item["path"], images[item["path"]]
        host = Path(build["freeze"]["host_path"])
        allowed = {str(Path("/build") / side / Path(member).relative_to(host))
                   for member in image.get("aliases", [image["path"]])}
        observed = set(mapped) & allowed
        if len(observed) != 1:
            raise ValueError("expected exactly one actual mapping within the root freezer alias group")
        actual_path = observed.pop()
        inode = mapped.pop(actual_path)
        image_identity(image["path"], inode)
        alias_provenance.append(dict(declared_path=path, actual_path=actual_path,
                                     inode=list(inode), allowed_paths=sorted(allowed),
                                     freeze_manifest_sha256=build["freeze"]["manifest_sha256"]))
        item["sha256"] = checked_hash(item["sha256"], image["sha256"])
        item["path"] = actual_path
        actual.append(dict(path=actual_path, sha256=image["sha256"]))
    save(output / "image-alias-provenance.json", alias_provenance)
    system = [system_image(path, observation) for path, observation in sorted(mapped.items())]
    launch = copy.deepcopy(reviewed["launch_policy"])
    if launch["args"] != spec["benchmark"]["argv"] or launch["environment"] != spec["benchmark"]["environment"]:
        raise ValueError("reviewed binding does not describe the actual launch")
    if side == "candidate":
        native = native_settings(evidence / "stderr.log")
        _, raw = asset(spec["settings"], contents=True)
        registry = importlib.import_module("inferssd.configuration.settings")
        expected = registry.parse(raw.decode("utf-8"), spec["settings"]["runtime_path"], platform="linux").document()["values"]
        if identity(native["values"]) != identity(expected) or identity(expected) != identity(launch["typed_values"]):
            raise ValueError("sealed settings, native snapshot and reviewed semantics differ")
        save(output / "native-settings.json", native)
    profile = dict(schema_version=1, purpose="staged-native-profile", revision=spec["revision"],
                   launch_policy=launch, binary=build["binary"], images=actual,
                   resource_policy=resources, model_identity=model)
    save(output / "launch-profile.json", profile)
    profile_hash = digest(output / "launch-profile.json")
    reviewed["profile_sha256"] = checked_hash(reviewed["profile_sha256"], profile_hash)
    receipt = dict(purpose="integration-checkpoint-native", revision=spec["revision"],
                   model_identity=model, build=dict(revision=spec["revision"], loaded_libraries=actual),
                   server=dict(pid=before["worker_pid"], start_identity=before["birth_identity"]),
                   host=dict(hardware=config["host"]["hardware"], os=config["host"]["os"],
                             memory_budget_bytes=int(resources["cgroup_limits"]["memory_max"]),
                             os_page_cache_treatment=config["host"]["os_page_cache_treatment"],
                             effective_settings=dict(native_profile_sha256=profile_hash,
                                 native_launch_policy=launch, native_resource_policy=resources,
                                 ambient_environment=ambient, system_images=system,
                                 source_mode="direct-gguf", cache_condition="fresh-server-first-completion")))
    save(output / "comparison-receipt.json", receipt)
    return receipt


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, purpose="integration-checkpoint", device="cloud-5090",
                  model="muse", tier="512", status="failed", performance_acceptance=False,
                  full_device_gate=False, arms={}, config_sha256=args.config_sha256,
                  samples_per_side=1, output_tokens=256, cleanup_confirmed=False)
    session = None
    try:
        path = protected(args.config)
        if digest(path) != args.config_sha256:
            raise ValueError("checkpoint configuration hash differs")
        config = load(path)
        save(output / "checkpoint-config.json", config)
        if (config.get("schema_version") != 1 or config.get("purpose") != report["purpose"]
                or config.get("device") != "cloud-5090"):
            raise ValueError("only the reviewed Cloud 5090 integration checkpoint is supported")
        if "reference_profile" in config:
            report["reference_profile"] = copy.deepcopy(config["reference_profile"])
        policy_read(config["policy"])
        manifest_path = protected_manifest(config)
        manifest = load(manifest_path)
        if manifest["job_id"] != args.job_id:
            raise ValueError("root manifest belongs to another job")
        uname = os.uname()
        if config["host"]["os"] != dict(sysname=uname.sysname, release=uname.release, machine=uname.machine):
            raise ValueError("operator OS fingerprint differs from this host")
        if not config["host"]["hardware"]:
            raise ValueError("root-observed hardware fingerprint is required")
        resources = config["host"]["resources"]
        limit = resources["cgroup_limits"]["memory_max"]
        if (not isinstance(limit, str) or not limit.isdigit() or not 128 << 20 <= int(limit) <= 1 << 50
                or resources["cgroup_limits"]["memory_swap_max"] != "0"):
            raise ValueError("checkpoint requires a host memory limit and no swap")
        binding = copy.deepcopy(config["comparison_binding"])
        check_declaration(binding)
        report["measurement_settings"] = dict(
            expert_gpu_budget_bytes=binding["semantic_settings"]["vram.budget"],
            allocator_reserve_mib=binding["semantic_settings"]["vram.reserve_mib"])
        if config["base"]["revision"] != LEGACY_REVISION:
            raise ValueError("checkpoint base is not the frozen accepted revision")
        for side in ("base", "candidate"):
            spec = config[side]
            revision(spec["revision"])
            if (manifest["source_revisions"][side] != spec["revision"]
                    or binding["sides"][side]["revision"] != spec["revision"]
                    or spec["paths"]["platform"] != "linux"
                    or spec["paths"]["source"] != "/source/" + side
                    or spec["paths"]["frozen"] != "/build/" + side
                    or spec["benchmark"]["model"] != "muse"
                    or spec["benchmark"]["os_page_cache_treatment"] != config["host"]["os_page_cache_treatment"]):
                raise ValueError("side source/model/cache binding differs")
            template = spec["template"]
            relative = Path(template["runtime_path"]).relative_to("/source/" + side)
            if Path(template["host_path"]) != Path(manifest["sources"][side]) / relative:
                raise ValueError("template is outside the bound source mount")
            asset(template)
            layout = binding["sides"][side]["build"]
            if (layout["server_target"] not in spec["build"]["targets"]
                    or layout["binary_relative"] != spec["build"]["binary_relative"]
                    or layout["image_relatives"] != spec["build"]["image_relatives"]
                    or layout["template_relatives"]["muse"] != str(relative)):
                raise ValueError("actual build paths differ from the reviewed layout")
        if normalized_args(config["base"]) != normalized_args(config["candidate"]):
            raise ValueError("shared model/server settings differ between arms")
        baseline = Path(manifest["sources"]["base"])
        source_hashes = {}
        for item in binding["legacy_mapping"].values():
            for evidence in item["inputs"]:
                if evidence["kind"] == "source-default":
                    source = evidence["path"]
                    if source not in source_hashes:
                        source_hashes[source] = sealed_file(baseline / source)["sha256"]
                    if evidence["revision"] != LEGACY_REVISION or evidence["sha256"] != source_hashes[source]:
                        raise ValueError("pinned legacy source evidence differs")
        model = models(config, manifest)
        save(output / "model-identity-before.json", model)
        results, receipts = {}, {}
        if "resumed_base" in config:
            result, receipt, recovered, binary_hash = recover(
                config, args.job_id, output / "base", binding, model,
                receipt=runtime_receipt, alias_asset=alias_asset, models=models)
            results["base"], receipts["base"] = result, receipt
            report["recovered_source"] = recovered
            report["lease_scope"] = "base evidence from a completed earlier lease; candidate alone in the new lease"
            report["arms"]["base"] = dict(revision=config["base"]["revision"], status="recovered",
                completion_requests=1, new_completion_requests=0, binary_sha256=binary_hash)
        session = StageSession(args.socket, args.job_id, controller_uid=os.getuid())
        opened = session.open()
        save(output / "job-opened.json", opened)
        if opened.get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("leased broker manifest differs from the reviewed checkpoint")
        report["current_lease"] = dict(job_id=args.job_id, lease=opened["lease"])
        for side in (("candidate",) if "resumed_base" in config else ("base", "candidate")):
            spec, arm = config[side], output / side
            arm.mkdir()
            report["arms"][side] = dict(revision=spec["revision"], status="failed")
            alias = spec["alias"]
            stage = session.start(side + "-alias", "build", alias["argv"], alias.get("environment", {}), alias["cwd"], alias["timeout_s"])
            status = session.wait(stage, deadline=time.monotonic() + alias["timeout_s"])
            alias_output = arm / "alias"
            alias_output.mkdir()
            collect_log(session, stage, alias_output)
            save(alias_output / "stage.json", status)
            if status.get("returncode") != 0 or status.get("cleanup_confirmed") is not True:
                raise RuntimeError("alias stage failed")
            build = run_build(session, side, spec["build"], spec["paths"], arm / "build", spec["revision"])
            if side == "candidate" and "validation" in spec:
                validate_candidate(session, spec, build, arm / "validation")
            native = dict(spec["benchmark"], revision=spec["revision"])
            if native["argv"][0] != build["binary"]["runtime_path"]:
                raise ValueError("benchmark executable differs from sealed build")
            source = alias_asset(build, side, model)
            save(arm / "alias-identity.json", source)
            launch = binding["sides"][side]["launch_policy"]
            if launch["template_sha256"] != spec["template"]["sha256"] or len(launch["alias_sha256"]) != 1:
                raise ValueError("reference template/alias identity differs")
            launch["alias_sha256"][0] = checked_hash(launch["alias_sha256"][0], source["sha256"])
            if side == "base":
                if native["environment"].get("GGML_MOE_EXPERT_GGUF_ALIAS_TSV") != source["runtime_path"]:
                    raise ValueError("legacy source does not use the verified alias")
            else:
                if launch["typed_values"]["expert.source"] != [source["runtime_path"]]:
                    raise ValueError("typed source does not use the verified alias")
                settings = spec["settings"]
                if not model_mount(manifest, settings["host_path"], settings["runtime_path"]):
                    raise ValueError("candidate settings are outside the root readonly input mount")
                asset(settings)
            measured = benchmark(session, side, native, arm / "benchmark")
            report["arms"][side].update(status=measured["status"], completion_requests=measured["completion_requests"],
                                       binary_sha256=build["binary"]["sha256"])
            if measured["status"] != "passed" or measured["completion_requests"] != 1 or not measured["cleanup_confirmed"]:
                raise RuntimeError("benchmark arm failed; retain its raw evidence")
            result = run_final.read_result(arm / "benchmark/result.json", "512", native["request_timeout_s"])
            if result != measured["result"]:
                raise ValueError("benchmark result differs from the verified raw SSE")
            results[side] = result
            receipts[side] = runtime_receipt(config, side, spec, build, arm / "benchmark", binding, model)
            if models(config, manifest) != model:
                raise ValueError("model identity changed during an arm")
        validate_binding(binding)
        save(output / "resolved-comparison-binding.json", binding)
        report["comparison"] = compare_pair(results["base"], results["candidate"], receipts["base"], receipts["candidate"],
                                             config["policy"], comparison_binding=binding)
        report["status"] = report["comparison"]["status"]
        save(output / "model-identity-after.json", models(config, manifest))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "failed"
    finally:
        if session is not None:
            try:
                cleanup = session.finish()
                save(output / "job-finished.json", cleanup)
                report["cleanup_confirmed"] = cleanup.get("cleanup_confirmed") is True
                if not report["cleanup_confirmed"]:
                    report["status"] = "failed"
            except Exception as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            session.close()
        report["performance_acceptance"] = report["status"] == "passed" and report["cleanup_confirmed"]
        save(output / "checkpoint.json", report)
    return 0 if report["performance_acceptance"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
