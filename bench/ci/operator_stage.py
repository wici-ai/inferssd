#!/usr/bin/env python3
"""Root-only, once-only Linux operator staging. Never executes candidate code.

The SSH sender passes the reviewed first-member SHA out of band and sends a
plain tar stream. Existing broker config, model mounts, lease and STOP are not
modified. A failed fresh job remains evidence and cannot be reused.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time

BROOT = Path("/var/lib/inferssd-ci-broker")
IROOT = Path("/opt/inferssd-ci-integrations")
SROOT = Path("/var/lib/inferssd-ci-inputs")
CONTROL = Path("/opt/inferssd-ci-control/current")
BROKER_CONFIG = Path("/opt/inferssd-ci-broker/config.json")
POLICY = Path("/opt/inferssd-ci-control/operator-policy.json")
LIMIT = 512 << 20


def parsed(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            fail("nonfinite JSON")
        return value
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: fail("nonfinite JSON"), parse_float=number)


def fail(message):
    raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
            or PurePosixPath(name).is_absolute()
            or any(x in ("", ".", "..") for x in name.split("/"))):
        fail("unsafe archive/member path")
    return name


def protected(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        fail("absolute protected path required")
    for item in (path, *path.parents):
        value = item.lstat()
        if stat.S_ISLNK(value.st_mode) or value.st_uid != 0 or value.st_mode & 0o022:
            fail("unsafe protected path owner/mode")
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        fail("protected path type differs")
    return path


def write(path, raw, mode=0o444):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def extract_source(archive_path, source_map, target):
    """Complete Git blob set and bytes, without tar extract/link semantics."""
    records = source_map["files"]
    rows = {relative(row["path"]): row for row in records}
    if len(rows) != len(records) or any(row["mode"] not in ("100644", "100755") for row in records):
        fail("source map has duplicate/nonregular tracked files")
    target.mkdir(mode=0o755, exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in archive:
            name = relative(item.name.rstrip("/") if item.isdir() else item.name)
            if item.isdir():
                continue
            if (not item.isfile() or name not in rows or name in seen
                    or item.size < 0 or item.size > 128 << 20):
                fail("unsafe/extra/duplicate source archive member")
            total += item.size
            if total > LIMIT:
                fail("source bytes exceed staging bound")
            row = rows[name]
            if item.size != row["bytes"]:
                fail("source blob size differs")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha1(("blob " + str(item.size) + "\0").encode())
            with archive.extractfile(item) as incoming, destination.open("xb") as output:
                for block in iter(lambda: incoming.read(1 << 20), b""):
                    h.update(block)
                    output.write(block)
            if h.hexdigest() != row["blob"]:
                fail("source Git blob differs")
            os.chmod(destination, 0o555 if row["mode"] == "100755" else 0o444)
            seen.add(name)
    if seen != set(rows):
        fail("source archive omits tracked files")
    for directory, _, _ in os.walk(target, topdown=False):
        os.chmod(directory, 0o555)
    return dict(files=len(seen), bytes=total, root=str(target), root_owned_readonly=True)


def control_identity(config):
    manifest = protected(CONTROL / "control-source.json")
    if sha(manifest) != config["control_manifest_sha256"]:
        fail("installed control manifest differs")
    value = parsed(manifest.read_bytes())
    if value.get("revision") != config["control_revision"]:
        fail("installed control revision differs")
    for name, expected in value["files"].items():
        if sha(protected(CONTROL / relative(name))) != expected:
            fail("installed trusted helper differs from its manifest")


def model_identities(config):
    seen = set()
    models = config["sides"]["candidate"]["models"]
    for model, data in models.items():
        identity = data.get("model_identity", config.get("model_identity", {}).get(model))
        if not identity or not identity["shards"]:
            fail("complete model identity required")
        for row in identity["shards"]:
            path = Path(row["host_path"])
            if not path.is_absolute() or ".." in path.parts or path.is_symlink() or path in seen:
                fail("canonical distinct readonly model paths required")
            seen.add(path)
            actual = path.stat()
            expected = row.get("stat", row)
            if [actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns] != [expected[k] for k in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]:
                fail("existing model stat identity differs")
            if not stat.S_ISREG(actual.st_mode):
                fail("model shard is not regular")
            mounted = subprocess.check_output(["/usr/bin/findmnt", "-rn", "--target", str(path), "-o", "VFS-OPTIONS"], timeout=10).decode().strip()
            if "ro" not in mounted.split(","):
                fail("existing model view is not readonly")
    for name in ("model_admission", "model_mount_proof"):
        key = name + "_path"
        if key in config and sha(protected(config[key])) != config[name + "_sha256"]:
            fail("existing root model admission differs")
    return len(seen)


def policy_check(policy, request, config, manifest, broker):
    """Local root policy wins over all caller-supplied paths and declarations."""
    job = request["job_id"]
    if (policy.get("schema_version") != 1 or policy.get("purpose") != "operator-stage-policy"
            or policy["device"] != request["device"] or broker["device"] != policy["device"]
            or request["broker_config_sha256"] != policy["broker_config_sha256"]
            or broker["lock_identity"] != policy["lock_identity"]
            or config["control_revision"] != policy["control_revision"]
            or config["control_manifest_sha256"] != policy["control_manifest_sha256"]
            or fingerprint(config["host"]["resources"]) != policy["resources_sha256"]
            or request["min_free_bytes"] != policy["min_free_bytes"]):
        fail("request differs from local root device/resource/control policy")
    models = config["sides"]["candidate"]["models"]
    if not models or not set(models) <= set(policy["models"]):
        fail("model set is outside local root admission policy")
    mounts = {}
    for model, data in models.items():
        admitted = policy["models"][model]
        if (fingerprint(data["model_identity"]) != admitted["identity_sha256"]
                or data["settings"]["sha256"] != admitted["settings_sha256"]
                or data["settings"]["host_path"] != str(IROOT / (job + "-" + model + "-candidate-settings.ssd"))
                or data["settings"]["runtime_path"] != "/models/settings-" + model + "-candidate.ssd"):
            fail("model/settings declaration differs from local root admission")
        mounts[model] = admitted["mount"]
        mounts["settings-" + model + "-candidate.ssd"] = data["settings"]["host_path"]
    if (manifest["models"] != mounts or manifest["builds"] != {"candidate": "candidate"}
            or manifest["sources"] != {"candidate": str(SROOT / ("operator-" + job))}):
        fail("manifest host mapping is outside local root policy")
    expected_admission = policy.get("model_admission", {})
    actual_admission = {key: config[key] for key in ("model_admission_path", "model_admission_sha256",
        "model_mount_proof_path", "model_mount_proof_sha256", "prior_model_job_id") if key in config}
    if actual_admission != expected_admission:
        fail("model admission metadata path/hash differs from root policy")


def gpu_idle(broker, policy):
    expected = broker.get("gpu_uuid") or policy.get("gpu_uuid")
    if (not isinstance(expected, str) or not expected.startswith("GPU-")
            or policy.get("gpu_uuid") != expected):
        fail("nonempty reviewed physical GPU UUID required")
    actual = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], timeout=15).decode().splitlines()
    if actual != [expected]:
        fail("actual physical GPU UUID differs")
    compute = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], timeout=15).decode().strip()
    if compute:
        fail("external GPU compute present; leave it untouched")
    return dict(uuid=expected, compute_apps_empty=True)


def receive(expected_sha256, incoming=None):
    if os.geteuid() != 0:
        raise PermissionError("root operator staging required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        fail("reviewed stage input SHA required")
    incoming = incoming or sys.stdin.buffer
    with tarfile.open(fileobj=incoming, mode="r|") as archive:
        iterator = iter(archive)
        first = next(iterator)
        if first.name != "root-input.json" or not first.isfile() or first.size > 2 << 20:
            fail("first plain-tar member must be bounded root input")
        raw = archive.extractfile(first).read()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            fail("reviewed root input hash differs")
        request = parsed(raw)
        job = request["job_id"]
        if (request.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", job)
                or request["device"] not in ("cloud-5090", "dgx-spark")):
            fail("fresh admitted Linux job identity required")
        if sha(__file__) != request["operator_stage_sha256"]:
            fail("installed trusted root stager differs")
        if type(request["min_free_bytes"]) is not int or request["min_free_bytes"] <= 0:
            fail("positive reviewed free-space reserve required")
        broker_path = protected(BROKER_CONFIG)
        if sha(broker_path) != request["broker_config_sha256"]:
            fail("existing broker configuration differs")
        broker = parsed(broker_path.read_bytes())
        if broker["device"] != request["device"]:
            fail("physical device slot differs")
        policy_path = protected(POLICY)
        if sha(policy_path) != request["policy_sha256"]:
            fail("installed root admission policy differs")
        policy = parsed(policy_path.read_bytes())
        stop = Path(broker["stop_path"])
        if stop.exists() or stop.is_symlink():
            fail("existing STOP present")
        fd = os.open(broker["lock_path"], os.O_RDONLY | os.O_NOFOLLOW)
        try:
            lock = os.fstat(fd)
            if not stat.S_ISREG(lock.st_mode) or [lock.st_dev, lock.st_ino] != broker["lock_identity"]:
                fail("original lease inode differs")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu = gpu_idle(broker, policy)
            for root in (IROOT, SROOT, BROOT / "jobs"):
                protected(root, directory=True)
            staging = IROOT / ("operator-stage-" + job)
            source = SROOT / ("operator-" + job)
            jobdir, target_config = BROOT / "jobs" / job, IROOT / (job + ".json")
            for path in (staging, source, jobdir, target_config):
                if path.exists() or path.is_symlink():
                    fail("fresh job path already exists")
            if shutil.disk_usage(SROOT).free < request["min_free_bytes"]:
                fail("reviewed staging free-space reserve unavailable")
            staging.mkdir(mode=0o700)
            write(staging / "root-input.json", raw)
            files, seen, total = request["files"], set(), 0
            for item in iterator:
                name = relative(item.name)
                if (name not in files or name in seen or not item.isfile()
                        or "/" in name or item.size != files[name]["bytes"] or item.size > 128 << 20):
                    fail("unexpected/duplicate installation member")
                total += item.size
                if total > LIMIT:
                    fail("installation input exceeds bound")
                target = staging / name
                with archive.extractfile(item) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output, 1 << 20)
                os.chmod(target, 0o400)
                if sha(target) != files[name]["sha256"]:
                    fail("installation member hash differs")
                seen.add(name)
            if seen != set(files):
                fail("installation input missing")
            config, manifest, source_map = (parsed((staging / name).read_bytes()) for name in ("config.json", "manifest.json", "source-map.json"))
            if (config["manifest_path"] != str(jobdir / "manifest.json")
                    or config["manifest_sha256"] != sha(staging / "manifest.json")
                    or manifest["job_id"] != job or manifest["sources"] != {"candidate": str(source)}
                    or manifest["source_revisions"] != {"candidate": request["candidate"]}
                    or config["revision"] != request["candidate"] or config["device"] != request["device"]
                    or source_map["revision"] != request["candidate"] or source_map["tree"] != request["source_tree"]):
                fail("source/config/manifest binding differs")
            policy_check(policy, request, config, manifest, broker)
            control_identity(config)
            count = model_identities(config)
            extracted = extract_source(staging / "candidate-source.tar.gz", source_map, source)
            for model, data in config["sides"]["candidate"]["models"].items():
                settings = data["settings"]
                name = "settings-" + model + ".ssd"
                path = IROOT / (job + "-" + model + "-candidate-settings.ssd")
                if settings["host_path"] != str(path) or sha(staging / name) != settings["sha256"]:
                    fail("settings payload/path differs")
                write(path, (staging / name).read_bytes())
            jobdir.mkdir(mode=0o711)
            write(jobdir / "manifest.json", (staging / "manifest.json").read_bytes())
            write(target_config, (staging / "config.json").read_bytes())
            final_lock = os.stat(broker["lock_path"], follow_symlinks=False)
            if (sha(broker_path) != request["broker_config_sha256"] or stop.exists() or stop.is_symlink()
                    or [final_lock.st_dev, final_lock.st_ino] != broker["lock_identity"]):
                fail("broker config or STOP changed during staging")
            proof = dict(schema_version=1, status="passed", job_id=job, device=request["device"],
                root_uid=0, candidate=request["candidate"], root_input_sha256=expected_sha256,
                envelope_sha256=sha(staging / "envelope.json"), config_sha256=sha(target_config),
                manifest_sha256=sha(jobdir / "manifest.json"), control_revision=config["control_revision"],
                control_manifest_sha256=config["control_manifest_sha256"], original_lock=[lock.st_dev, lock.st_ino],
                source=extracted, model_shards_stat_readonly_verified=count,
                operator_stage_sha256=request["operator_stage_sha256"], source_tree=request["source_tree"],
                policy_sha256=request["policy_sha256"],
                gpu=gpu,
                broker_config_unchanged=True, existing_coordination_unmodified=True, finished_ns=time.time_ns())
            write(staging / "ROOT-STAGE-RECEIPT.json", (json.dumps(proof, indent=2) + "\n").encode())
            return proof
        finally:
            os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(receive(args.expected_sha256), indent=2))
