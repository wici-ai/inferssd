"""Recover one root-sealed base sample after the known controller-stop misclassification.

This is evidence revalidation, not a sample cache: both source/configuration arms
must remain identical, and the original failed checkpoint is never rewritten.
"""

import copy
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import run_final

from ci.contract import digest, load, save
from ci.reference import LEGACY_REVISION, identity, relative
from ci.staged_benchmark import IDENTITY, verify_runtime, verify_stopped
from ci.staged_build import compiler_cache, frozen_aliases, sealed_file


def same_configuration(old, current):
    """Only a new broker job and the identical settings bytes at a new host path."""
    normalized = []
    for original in (old, current):
        value = copy.deepcopy(original)
        for key in ("manifest_path", "manifest_sha256", "resumed_base"):
            value.pop(key, None)
        value["candidate"]["settings"].pop("host_path")
        normalized.append(value)
    if "resumed_base" in old or identity(normalized[0]) != identity(normalized[1]):
        raise ValueError("resumed base requires the original two arms, host, model and profile")


def sealed_evidence(spec):
    if set(spec) != {"artifact_root", "manifest_path", "manifest_sha256"}:
        raise ValueError("resumed base needs the explicit root-sealed artifact and manifest")
    manifest_file, _ = sealed_file(spec["manifest_path"], limit=4 << 20, contents=True)
    if manifest_file["sha256"] != spec["manifest_sha256"]:
        raise ValueError("resumed evidence manifest hash differs")
    manifest = load(spec["manifest_path"])
    if (manifest.get("schema_version") != 1
            or not re.fullmatch(r"[a-f0-9]{32}", manifest.get("job_id", ""))
            or not re.fullmatch(r"[0-9]+", manifest.get("run_id", ""))
            or type(manifest.get("run_attempt")) is not int or manifest["run_attempt"] < 1
            or not isinstance(manifest.get("files"), dict) or not manifest["files"]):
        raise ValueError("resumed evidence needs its original job/run identity")
    root = Path(spec["artifact_root"])
    total = 0
    for name, expected in manifest["files"].items():
        if not relative(name) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("unsafe sealed evidence manifest entry")
        observed = sealed_file(root / name, limit=64 << 20)
        total += observed["bytes"]
        if observed["sha256"] != expected or total > 256 << 20:
            raise ValueError("resumed evidence differs from its bounded sealed manifest")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if not path.is_dir()}
    if actual != set(manifest["files"]):
        raise ValueError("sealed manifest must cover the complete retained artifact")
    return root / manifest["job_id"], manifest


def completed_base(evidence, old, seal):
    checkpoint = load(evidence / "checkpoint.json")
    measured = load(evidence / "base/benchmark/benchmark.json")
    opened, finished = (load(evidence / (name + ".json")) for name in ("job-opened", "job-finished"))
    if (checkpoint.get("status") != "failed" or checkpoint.get("cleanup_confirmed") is not True
            or set(checkpoint.get("arms", {})) != {"base"}
            or checkpoint.get("error") != "RuntimeError: benchmark arm failed; retain its raw evidence"
            or checkpoint.get("config_sha256") != digest(evidence / "checkpoint-config.json")
            or measured.get("status") != "failed" or measured.get("side") != "base"
            or measured.get("revision") != LEGACY_REVISION
            or measured.get("completion_requests") != 1 or measured.get("samples_per_tier") != 1
            or measured.get("output_tokens") != 256 or measured.get("tier") != "512"
            or measured.get("cleanup_confirmed") is not True
            or measured.get("cleanup_error") != "RuntimeError: native benchmark did not stop normally and cleanly"
            or any(key in measured for key in ("error", "log_errors"))
            or finished.get("cleanup_confirmed") is not True or finished.get("errors") != []
            or opened.get("opened") is not True
            or any(row.get("job_id") != seal["job_id"] or row.get("manifest_sha256") != old["manifest_sha256"]
                   for row in (opened, finished)) or opened["lease"] != finished["lease"]):
        raise ValueError("only the single completed base with the known stop-classification failure is recoverable")
    output = evidence / "base/benchmark"
    stopped, started = load(output / "stopped.json"), load(output / "started.json")
    if (verify_stopped(stopped) != "trusted-controller-stop"
            or finished["stages"].get("base-benchmark-512") != stopped["receipt"]
            or stopped["receipt"]["lease"] != opened["lease"]
            or any(stopped.get(key) != started.get(key) for key in (*IDENTITY, "handle", "stage_id"))):
        raise ValueError("original stage/lease cleanup identity differs")
    for name in ("runtime-before", "runtime-after"):
        verify_runtime(SimpleNamespace(**started), load(output / (name + ".json")))
    result = run_final.read_result(output / "result.json", "512", old["base"]["benchmark"]["request_timeout_s"])
    if (result != measured.get("result") or measured.get("result_artifact") != {
            "path": "result.json", "sha256": digest(output / "result.json")}):
        raise ValueError("original base result differs from its one retained raw stream")
    return result, opened, checkpoint


def retained_build(evidence, spec, opened):
    build = load(evidence / "base/build/build-receipt.json")
    frozen = build["freeze"]
    if (build.get("status") != "passed" or build.get("revision") != spec["revision"]
            or build.get("side") != "base" or frozen.get("side") != "base"
            or frozen.get("path") != "/build/base" or frozen.get("lease") != opened["lease"]
            or Path(frozen["host_path"]) != Path("/var/lib/inferssd-ci-broker/jobs") / opened["job_id"] / "frozen/base"
            or spec["revision"][:7] not in build.get("actual_server_version", "")):
        raise ValueError("retained build is not the original sealed base")
    aliases = frozen_aliases(frozen)
    host = Path(frozen["host_path"])
    items = [build["binary"], *build["images"]]
    names = [spec["build"]["binary_relative"], *spec["build"]["image_relatives"]]
    if len(items) != len(names):
        raise ValueError("retained build image set differs")
    for item, name in zip(items, names):
        expected = dict(sealed_file(host / name, aliases=aliases), runtime_path="/build/base/" + name)
        if item != expected:
            raise ValueError("retained frozen binary/library identity changed")
    cache, raw = sealed_file(host / "CMakeCache.txt", limit=8 << 20, contents=True, aliases=aliases)
    if (cache != build["cmake_cache"] or cache["sha256"] != build["cmake_cache_sha256"]
            or compiler_cache(raw, spec["build"]["toolchain"]) != build["toolchain"]):
        raise ValueError("retained build toolchain identity changed")
    return build


def recover(config, new_job_id, output, binding, model, *, receipt, alias_asset, models):
    """Validate sealed old inputs; write only new derived evidence, run no stage."""
    evidence, seal = sealed_evidence(config["resumed_base"])
    if new_job_id == seal["job_id"]:
        raise ValueError("candidate continuation requires a new job identity")
    old = load(evidence / "checkpoint-config.json")
    same_configuration(old, config)
    result, opened, original = completed_base(evidence, old, seal)
    root_manifest, _ = sealed_file(old["manifest_path"], limit=4 << 20, contents=True)
    if root_manifest["sha256"] != old["manifest_sha256"]:
        raise ValueError("original root job manifest changed")
    if models(old, load(old["manifest_path"])) != model or load(evidence / "model-identity-before.json") != model:
        raise ValueError("resumed base model identity differs")
    build = retained_build(evidence, old["base"], opened)
    source = alias_asset(build, "base", model)
    if source != load(evidence / "base/alias-identity.json"):
        raise ValueError("retained expert alias identity changed")
    launch = binding["sides"]["base"]["launch_policy"]
    if launch["alias_sha256"] not in ([None], [source["sha256"]]):
        raise ValueError("resumed base alias differs from the binding")
    launch["alias_sha256"] = [source["sha256"]]
    # Retain byte-identical failed JSON and raw SSE in the new artifact as well.
    shutil.copytree(evidence / "base", output)
    for directory in (output, *(path for path in output.rglob("*") if path.is_dir())):
        directory.chmod(0o700)
    derived = output / "benchmark"
    actual = receipt(config, "base", old["base"], build, derived, binding, model,
                     evidence=evidence / "base/benchmark")
    recovery = dict(**config["resumed_base"], job_id=seal["job_id"], run_id=seal["run_id"],
                    run_attempt=seal["run_attempt"], lease=opened["lease"],
                    original_status=original["status"], original_error=original["error"],
                    original_config_sha256=original["config_sha256"],
                    recovery="offline revalidation of the unique tier-off CI diagnostic base; original failure retained",
                    formal_final_rows=0,
                    evidence_directory=str(evidence), cleanup_confirmed=True)
    save(output / "recovery.json", recovery)
    return result, actual, recovery, build["binary"]["sha256"]
