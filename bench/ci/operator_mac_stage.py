#!/usr/bin/env python3
"""Seal one reviewed Mac job under the existing root executor policy.

Only this fixed root entry is exposed by the operator's forced SSH command.
It receives data, never executes candidate code, and never creates the job's
worker directory. The executor owns that later, once-only transition.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import operator_stage as common

BASE = Path("/Library/InferSSD-CI")
CONTROL = BASE / "controller/current"
BROKER_CONFIG = BASE / "control/config.json"
RECOVER = BASE / "control/recover.py"
POLICY = BASE / "control/operator-policy.json"
IROOT = BASE / "integrations"
SROOT = BASE / "inputs"
MANIFESTS = BASE / "private/job-manifests"
TIMEOUT = 300
MAC_PYTHON = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"
JOB = re.compile(r"[0-9a-f]{32}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def source_span(raw, prefix):
    raw.decode("utf-8")
    matches = list(re.finditer(rb"(?m)^expert\.source[ \t]*=[ \t]*(\[[^\r\n]*\])[ \t]*$", raw))
    if len(matches) != 1:
        common.fail("one explicit expert.source assignment required")
    match = matches[0]
    values = common.parsed(match.group(1))
    if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str)
            or not values[0].startswith(prefix)):
        common.fail("expert.source must contain one exact frozen-side path")
    common.relative(values[0][len(prefix):])
    return match.span(1)


def settings_payload(template, job, side):
    if not isinstance(job, str) or JOB.fullmatch(job) is None or side not in ("base", "candidate"):
        common.fail("exact Mac settings job/side required")
    prefix = ("/Library/InferSSD-CI/jobs/{{job_id}}/frozen/" + side + "/").encode()
    if template.count(b"{{job_id}}") != 1 or template.count(prefix) != 1:
        common.fail("Mac settings require one exact frozen-side job placeholder")
    begin, end = source_span(template, prefix.decode())
    return template[:begin] + template[begin:end].replace(b"{{job_id}}", job.encode()) + template[end:]


def settings_template(payload, job, side):
    prefix = ("/Library/InferSSD-CI/jobs/" + job + "/frozen/" + side + "/").encode()
    if payload.count(prefix) != 1 or b"{{job_id}}" in payload:
        common.fail("actual settings must consume this exact frozen job/side")
    begin, end = source_span(payload, prefix.decode())
    template = payload[:begin] + payload[begin:end].replace(prefix,
        ("/Library/InferSSD-CI/jobs/{{job_id}}/frozen/" + side + "/").encode()) + payload[end:]
    if settings_payload(template, job, side) != payload:
        common.fail("Mac settings namespace roundtrip differs")
    return template


def machine_identity():
    raw = subprocess.check_output(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], timeout=10)
    values = re.findall(rb'"IOPlatformUUID"\s*=\s*"([A-Fa-f0-9-]{36})"', raw)
    if len(values) != 1:
        common.fail("physical platform UUID is missing or ambiguous")
    return dict(hardware_uuid_sha256=hashlib.sha256(values[0].upper()).hexdigest(),
                hardware_model=subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], timeout=10).decode().strip(),
                cpu_brand=subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=10).decode().strip())


def worker_idle(broker):
    if broker.get("worker_uid") != 62002 or broker.get("runner_uid") != 62001:
        common.fail("installed dedicated Mac account identities differ")
    raw = subprocess.check_output(["/bin/ps", "-axo", "uid=,pid="], timeout=10)
    if len(raw) > 4 << 20:
        common.fail("process inventory exceeds bound")
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 2 or not all(value.isdigit() for value in fields):
            common.fail("process inventory is incomplete")
        if int(fields[0]) == broker["worker_uid"]:
            common.fail("dedicated Mac worker is busy; leave it untouched")


def control_identity(policy):
    path = common.protected(CONTROL / "control-source.json")
    if common.sha(path) != policy["control_manifest_sha256"]:
        common.fail("installed Mac control manifest differs")
    manifest = common.parsed(path.read_bytes())
    if manifest.get("revision") != policy["control_revision"]:
        common.fail("installed Mac control source differs")
    for name, expected in manifest["files"].items():
        if common.sha(common.protected(CONTROL / common.relative(name))) != expected:
            common.fail("installed Mac control file differs")


def ensure_broker(broker, expected_sha):
    path = common.protected(RECOVER)
    if common.sha(path) != broker["installed_files"].get("control/recover.py"):
        common.fail("fixed Mac recovery helper differs from installed configuration")
    python = Path(broker["tools"]["python"])
    info = python.lstat()
    if (str(python) != MAC_PYTHON or not stat.S_ISREG(info.st_mode) or info.st_uid != 0
            or info.st_mode & 0o022 or common.sha(python) != broker["tools"]["files"].get(str(python))):
        common.fail("frozen Xcode Python identity differs")
    result = subprocess.run([str(python), "-I", "-B", str(path), "--ensure"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    if result.returncode or len(result.stdout) > common.RESPONSE_LIMIT:
        detail = result.stderr[:2048].decode("utf-8", "replace")
        common.fail("fixed Mac broker recovery refused: " + detail)
    ready = common.parsed(result.stdout)
    if (ready.get("status") != "ready" or ready.get("protocol") != "mac-executor-v1"
            or ready.get("config_sha256") != expected_sha
            or ready.get("lock_dev_inode") != broker["lock_dev_inode"]
            or ready.get("worker_uid") != broker["worker_uid"] or ready.get("runner_uid") != broker["runner_uid"]
            or ready.get("installed_files") != broker["installed_files"]
            or ready.get("broker_birth", {}).get("uid") != 0):
        common.fail("authenticated Mac broker identity differs from root policy")
    return ready


def models_readonly(config, policy):
    seen = set()
    for model, data in config["sides"]["candidate"]["models"].items():
        root = common.protected(policy["models"][model]["mount"], directory=True)
        if BASE / "models" not in root.parents:
            common.fail("Mac model is outside the original controlled model root")
        if not data["model_identity"].get("shards"):
            common.fail("complete Mac model shard identity required")
        for row in data["model_identity"]["shards"]:
            path = common.protected(row["host_path"])
            if root not in path.parents or path in seen:
                common.fail("duplicate or escaped Mac model shard")
            seen.add(path)
            actual = path.lstat()
            expected = row.get("stat", row)
            if (actual.st_mode & 0o222 or actual.st_nlink != 1
                    or [actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns]
                    != [expected[k] for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]):
                common.fail("root readonly Mac model stat identity differs")
    if not seen:
        common.fail("no actual admitted Mac model shard")
    return len(seen)


def policy_check(policy, request, config, manifest, broker):
    job = request["job_id"]
    sides = common.source_sides(config)
    source = SROOT / ("operator-" + job)
    resources = {key: broker["executor"][key] for key in ("memory_budget_bytes", "host_reserve_bytes", "max_processes")}
    if (policy.get("schema_version") != 1 or policy.get("purpose") != "operator-stage-policy"
            or policy.get("device") != config.get("device") or policy["device"] != "local-mbp"
            or policy["broker_device"] != broker["device"]
            or request["broker_config_sha256"] != policy["broker_config_sha256"]
            or broker["lock_dev_inode"] != policy["lock_identity"]
            or config["control_revision"] != policy["control_revision"]
            or config["control_manifest_sha256"] != policy["control_manifest_sha256"]
            or config["host"]["resources"] != resources
            or common.fingerprint(resources) != policy["resources_sha256"]
            or request["min_free_bytes"] != policy["min_free_bytes"]
            or config.get("validation_mode") != common.PERFORMANCE_MODE
            or config.get("performance_protocol") != common.PERFORMANCE_PROTOCOL
            or policy.get("performance_protocol") != common.PERFORMANCE_PROTOCOL
            or type(config["performance_protocol"].get("output_tokens")) is not int
            or type(config["performance_protocol"].get("samples_per_side")) is not int):
        common.fail("Mac request differs from fixed root device/resources/protocol policy")
    if (manifest.get("selection_resource_policy") != resources
            or manifest.get("selection_lock_identity") != broker["lock_dev_inode"]
            or set(policy.get("side_settings_template_sha256", {})) != {"base", "candidate"}):
        common.fail("Mac manifest resources/lease or settings template policy differs")
    models = config["sides"]["candidate"]["models"]
    if not models or not set(models) <= set(policy["models"]):
        common.fail("Mac models are outside local root admission")
    mounts = {}
    for side in sides:
        if set(config["sides"][side]["models"]) != set(models):
            common.fail("Mac side model sets differ")
        for model, data in config["sides"][side]["models"].items():
            admitted = policy["models"][model]
            settings = str(source / ("settings-" + model + "-" + side + ".ssd"))
            if (common.fingerprint(data["model_identity"]) != admitted["identity_sha256"]
                    or data["settings"]["host_path"] != settings or data["settings"]["runtime_path"] != settings):
                common.fail("Mac model/settings path differs from root admission")
            mounts[model] = admitted["mount"]
            mounts["settings-" + model + "-" + side + ".ssd"] = settings
    if (manifest["models"] != mounts or manifest["sources"] != {side: str(source / side) for side in sides}
            or manifest["builds"] != {side: side for side in sides}
            or manifest.get("input_roots") != [str(source)]
            or manifest.get("model_roots") != sorted({policy["models"][model]["mount"] for model in models})):
        common.fail("Mac manifest host paths differ from root admission")
    if policy.get("model_admission") or any(key in config for key in ("model_admission_path", "model_mount_proof_path", "prior_model_job_id")):
        common.fail("Mac root-owned model contract does not admit the Linux readonly-view seam")


def receive(expected_sha256, incoming=None):
    if os.geteuid() != 0 or sys.platform != "darwin":
        raise PermissionError("root Darwin operator staging required")
    if not isinstance(expected_sha256, str) or HASH.fullmatch(expected_sha256) is None:
        common.fail("exact reviewed root input SHA required")
    deadline = time.monotonic() + TIMEOUT
    with tempfile.TemporaryFile() as spool:
        total = 0
        for block in common.input_chunks(incoming or sys.stdin.buffer, deadline):
            total += len(block)
            if total > common.WIRE_LIMIT:
                common.fail("Mac stage wire exceeds bound")
            spool.write(block)
        spool.seek(0)
        return receive_packet(expected_sha256, spool, deadline)


def receive_packet(expected_sha256, incoming, deadline):
    with tarfile.open(fileobj=incoming, mode="r|") as archive:
        iterator = iter(archive)
        first = next(iterator)
        if first.name != "root-input.json" or not first.isfile() or first.size > 2 << 20:
            common.fail("first member must be the bounded root input")
        raw = archive.extractfile(first).read()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            common.fail("Mac root input hash differs")
        request = common.parsed(raw)
        job = request["job_id"]
        if (request.get("schema_version") != 1 or JOB.fullmatch(job) is None
                or request.get("device") != "local-mbp" or request["operator_stage_sha256"] != common.sha(__file__)
                or type(request["min_free_bytes"]) is not int or request["min_free_bytes"] <= 0):
            common.fail("fresh Mac root job/stager identity required")
        broker_path, policy_path = common.protected(BROKER_CONFIG), common.protected(POLICY)
        if common.sha(broker_path) != request["broker_config_sha256"] or common.sha(policy_path) != request["policy_sha256"]:
            common.fail("existing Mac root configuration/policy differs")
        broker, policy = common.parsed(broker_path.read_bytes()), common.parsed(policy_path.read_bytes())
        control_identity(policy)
        physical = machine_identity()
        if physical != policy["machine_identity"]:
            common.fail("actual physical Mac identity differs")
        recovered = ensure_broker(broker, request["broker_config_sha256"])
        stop = Path(broker["stop"])
        if stop.exists() or stop.is_symlink():
            common.fail("existing Mac STOP is present")
        fd = os.open(broker["lock"], os.O_RDONLY | os.O_NOFOLLOW)
        try:
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode) or [held.st_dev, held.st_ino] != broker["lock_dev_inode"]:
                common.fail("original Mac lease inode differs")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker_idle(broker)
            for path in (IROOT, SROOT, MANIFESTS, BASE / "jobs"):
                common.protected(path, directory=True)
            staging, source = IROOT / ("operator-stage-" + job), SROOT / ("operator-" + job)
            target_config, target_manifest = IROOT / (job + ".json"), MANIFESTS / (job + ".json")
            for path in (staging, source, target_config, target_manifest, BASE / "jobs" / job):
                if path.exists() or path.is_symlink():
                    common.fail("fresh Mac job path already exists")
            if shutil.disk_usage(SROOT).free < request["min_free_bytes"]:
                common.fail("reviewed Mac free-space reserve unavailable")
            staging.mkdir(mode=0o700)
            common.write(staging / "root-input.json", raw)
            files, seen, total = request["files"], set(), 0
            for item in iterator:
                common.remaining(deadline)
                name = common.relative(item.name)
                if (name not in files or name in seen or not item.isfile() or "/" in name
                        or item.size != files[name]["bytes"] or not 0 <= item.size <= 128 << 20):
                    common.fail("unexpected/duplicate Mac stage member")
                total += item.size
                if total > common.LIMIT:
                    common.fail("Mac stage payload exceeds bound")
                target = staging / name
                with archive.extractfile(item) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                target.chmod(0o400)
                if common.sha(target) != files[name]["sha256"]:
                    common.fail("Mac stage member hash differs")
                seen.add(name)
            if seen != set(files):
                common.fail("Mac stage input is missing")
            config, manifest = (common.parsed((staging / name).read_bytes()) for name in ("config.json", "manifest.json"))
            sides = common.source_sides(config)
            maps = {side: common.parsed((staging / ("source-map.json" if side == "candidate" else "base-source-map.json")).read_bytes()) for side in sides}
            expected_sources = {side: {"revision": config["sides"][side]["revision"], "tree": maps[side]["tree"]} for side in sides}
            if (config["manifest_path"] != str(target_manifest) or config["manifest_sha256"] != common.sha(staging / "manifest.json")
                    or manifest["job_id"] != job or manifest["source_revisions"] != {side: config["sides"][side]["revision"] for side in sides}
                    or config["revision"] != request["candidate"] or maps["candidate"]["revision"] != request["candidate"]
                    or maps["candidate"]["tree"] != request["source_tree"] or request.get("sources") != expected_sources):
                common.fail("Mac exact source/config/manifest binding differs")
            policy_check(policy, request, config, manifest, broker)
            if common.sha(common.protected(CONTROL / "bench/prompts/prompt-512.txt")) != common.PERFORMANCE_PROTOCOL["prompt_sha256"]:
                common.fail("installed Mac fixed512 prompt differs")
            count = models_readonly(config, policy)
            source.mkdir(mode=0o755)
            extracted = {}
            for side in sides:
                common.remaining(deadline)
                target = source / side
                extracted[side] = dict(common.extract_source(staging / (side + "-source.tar.gz"), maps[side], target), **expected_sources[side])
                if sum(value["bytes"] for value in extracted.values()) > common.LIMIT:
                    common.fail("combined Mac source bytes exceed bound")
                for data in config["sides"][side]["models"].values():
                    template = data["template"]
                    path = Path(template["host_path"])
                    if (template["runtime_path"] != str(path) or target not in path.parents
                            or common.sha(common.protected(path)) != template["sha256"]):
                        common.fail("Mac chat template differs from the sealed source side")
            for model in config["sides"]["candidate"]["models"]:
                payloads = {}
                for side in sides:
                    data = config["sides"][side]["models"][model]
                    name = "settings-" + model + "-" + side + ".ssd"
                    payload = (staging / name).read_bytes()
                    template_sha = hashlib.sha256(settings_template(payload, job, side)).hexdigest()
                    if (common.sha(staging / name) != data["settings"]["sha256"]
                            or template_sha != policy["side_settings_template_sha256"][side][model]
                            or (side == "candidate" and template_sha != policy["models"][model]["settings_template_sha256"])):
                        common.fail("Mac settings payload/template differs from fixed root admission")
                    payloads[side] = payload
                if len(sides) == 2:
                    prefix = ("/Library/InferSSD-CI/jobs/" + job + "/frozen/").encode()
                    if payloads["base"].replace(prefix + b"base/", prefix + b"candidate/") != payloads["candidate"]:
                        common.fail("Mac side effective settings differ")
                for side, payload in payloads.items():
                    common.write(source / ("settings-" + model + "-" + side + ".ssd"), payload)
            source.chmod(0o555)
            worker_idle(broker)
            current = os.stat(broker["lock"], follow_symlinks=False)
            if (common.sha(broker_path) != request["broker_config_sha256"] or common.sha(policy_path) != request["policy_sha256"]
                    or [current.st_dev, current.st_ino] != broker["lock_dev_inode"] or stop.exists() or stop.is_symlink()):
                common.fail("Mac broker/policy/lease/STOP changed during staging")
            common.remaining(deadline)
            common.write(target_manifest, (staging / "manifest.json").read_bytes())
            common.write(target_config, (staging / "config.json").read_bytes())
            proof = dict(schema_version=1, status="passed", device="local-mbp", job_id=job, root_uid=0,
                candidate=request["candidate"], source_tree=request["source_tree"], source=extracted["candidate"], sources=extracted,
                root_input_sha256=expected_sha256, envelope_sha256=common.sha(staging / "envelope.json"),
                config_sha256=common.sha(target_config), manifest_sha256=common.sha(target_manifest),
                control_revision=config["control_revision"], control_manifest_sha256=config["control_manifest_sha256"],
                original_lock=[held.st_dev, held.st_ino], policy_sha256=request["policy_sha256"],
                operator_stage_sha256=request["operator_stage_sha256"], machine_identity=physical,
                broker_recovery=recovered, broker_config_unchanged=True, existing_coordination_unmodified=True,
                model_shards_stat_readonly_verified=count, performance_protocol=config["performance_protocol"],
                resource_enforcement="sampled-UMA-guard-not-hard-cgroup", finished_ns=time.time_ns())
            common.write(staging / "ROOT-STAGE-RECEIPT.json", (json.dumps(proof, indent=2) + "\n").encode())
            return proof
        finally:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    def expired(_number, _frame):
        raise TimeoutError("fixed Mac root staging deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(TIMEOUT)
    try:
        print(json.dumps(receive(args.expected_sha256), allow_nan=False))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
