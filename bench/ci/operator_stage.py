#!/usr/bin/env python3
"""Once-only Linux operator staging with a fixed Spark root socket bridge.

The SSH sender passes the reviewed first-member SHA out of band and sends a
plain tar stream. Existing broker config, model mounts, lease and STOP are not
modified. A failed fresh job remains evidence and cannot be reused.
Non-root Spark callers forward only the same hash and tar to the root service.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

BROOT = Path("/var/lib/inferssd-ci-broker")
IROOT = Path("/opt/inferssd-ci-integrations")
SROOT = Path("/var/lib/inferssd-ci-inputs")
CONTROL = Path("/opt/inferssd-ci-control/current")
BROKER_CONFIG = Path("/opt/inferssd-ci-broker/config.json")
POLICY = Path("/opt/inferssd-ci-control/operator-policy.json")
LIMIT = 512 << 20
BRIDGE_SOCKET = Path("/run/inferssd-ci-stage.sock")
BRIDGE_TIMEOUT = 300
WIRE_LIMIT = LIMIT + (16 << 20)
RESPONSE_LIMIT = 1 << 20


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


PERFORMANCE_MODE = "functional-numerical-performance"
# A PR runs candidate, base, candidate, candidate; a baseline job measures one
# main commit (built once, as the candidate side) four times.
PERFORMANCE_PROTOCOL = dict(tier="512", prompt_sha256="17ac26b4a3bc13d6c51dd10b62eb54931651d48e41ca4071c587a931a662e283",
                            output_tokens=256, purpose="regression-gate",
                            order=["candidate", "base", "candidate", "candidate"])
BASELINE_PROTOCOL = dict(PERFORMANCE_PROTOCOL, purpose="baseline", order=["candidate"] * 4)


def protocol_admitted(config):
    """The fixed gate protocol, or the baseline protocol for one main commit."""
    value = config.get("performance_protocol")
    if not isinstance(value, dict) or type(value.get("output_tokens")) is not int:
        return False
    if value == PERFORMANCE_PROTOCOL:
        return True
    return (value == BASELINE_PROTOCOL
            and len({config["sides"][side]["revision"] for side in ("base", "candidate")}) == 1)


def sealed_machine(config, host_path, runtime_path):
    """A PR job's machine characteristics as bytes to stage read-only, or None (a baseline job profiles)."""
    value = config.get("machine_characteristics")
    if value is None:
        return None
    fields = {"sha256", "text", "job_id", "commit", "host_path", "runtime_path"}
    if (not isinstance(value, dict) or set(value) != fields or not isinstance(value["text"], str)
            or value["host_path"] != host_path or value["runtime_path"] != runtime_path
            or config.get("performance_protocol") == BASELINE_PROTOCOL
            or hashlib.sha256(value["text"].encode()).hexdigest() != value["sha256"]):
        fail("sealed machine characteristics differ from their fixed path or hash, or are given to a baseline job")
    return value["text"].encode()


def source_sides(config):
    mode = config.get("validation_mode", "functional-numerical")
    if mode not in ("functional-numerical", PERFORMANCE_MODE):
        fail("unsupported root operator validation mode")
    if mode == PERFORMANCE_MODE and config.get("performance_protocol") == BASELINE_PROTOCOL:
        return ("candidate",)
    return (("base", "candidate") if mode == PERFORMANCE_MODE and any(
        data.get("benchmarks") for data in config["sides"]["candidate"]["models"].values()) else ("candidate",))


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
    sides = source_sides(config)
    if config.get("validation_mode") == PERFORMANCE_MODE:
        if (not protocol_admitted(config)
                or policy.get("performance_protocol") != PERFORMANCE_PROTOCOL
                or set(policy.get("side_settings_sha256", {})) != {"base", "candidate"}):
            fail("fixed performance protocol/side settings are outside root policy")
    models = config["sides"]["candidate"]["models"]
    if not models or not set(models) <= set(policy["models"]):
        fail("model set is outside local root admission policy")
    mounts = {}
    for side in sides:
        side_models = config["sides"][side]["models"]
        if set(side_models) != set(models):
            fail("base/candidate model sets differ")
        for model, data in side_models.items():
            admitted = policy["models"][model]
            expected_settings = (policy["side_settings_sha256"][side][model]
                if config.get("validation_mode") == PERFORMANCE_MODE else admitted["settings_sha256"])
            if (fingerprint(data["model_identity"]) != admitted["identity_sha256"]
                    or data["settings"]["sha256"] != expected_settings
                    or (side == "candidate" and expected_settings != admitted["settings_sha256"])
                    or data["settings"]["host_path"] != str(IROOT / (job + "-" + model + "-" + side + "-settings.ssd"))
                    or data["settings"]["runtime_path"] != "/models/settings-" + model + "-" + side + ".ssd"):
                fail("model/settings declaration differs from local root admission")
            mounts[model] = admitted["mount"]
            mounts["settings-" + model + "-" + side + ".ssd"] = data["settings"]["host_path"]
    if sealed_machine(config, str(IROOT / (job + "-machine.json")), "/models/machine.json") is not None:
        mounts["machine.json"] = str(IROOT / (job + "-machine.json"))
    source_root = SROOT / ("operator-" + job)
    sources = {side: str(source_root / side) if config.get("validation_mode") == PERFORMANCE_MODE else str(source_root) for side in sides}
    if (manifest["models"] != mounts or manifest["builds"] != {side: side for side in sides}
            or manifest["sources"] != sources):
        fail("manifest host mapping is outside local root policy")
    expected_admission = policy.get("model_admission", {})
    actual_admission = {key: config[key] for key in ("model_admission_path", "model_admission_sha256",
        "model_mount_proof_path", "model_mount_proof_sha256", "prior_model_job_id") if key in config}
    if actual_admission != expected_admission:
        fail("model admission metadata path/hash differs from root policy")


def paired_settings(base, candidate):
    return base.replace(b"/build/base/", b"/build/candidate/").replace(
        b"/source/base/", b"/source/candidate/") == candidate


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
            config, manifest = (parsed((staging / name).read_bytes()) for name in ("config.json", "manifest.json"))
            sides = source_sides(config)
            sources = {side: str(source / side) if config.get("validation_mode") == PERFORMANCE_MODE else str(source) for side in sides}
            maps = {side: parsed((staging / ("source-map.json" if side == "candidate" else "base-source-map.json")).read_bytes()) for side in sides}
            expected_sources = {side: {"revision": config["sides"][side]["revision"], "tree": maps[side]["tree"]} for side in sides}
            if (config["manifest_path"] != str(jobdir / "manifest.json")
                    or config["manifest_sha256"] != sha(staging / "manifest.json")
                    or manifest["job_id"] != job or manifest["sources"] != sources
                    or manifest["source_revisions"] != {side: config["sides"][side]["revision"] for side in sides}
                    or config["revision"] != request["candidate"] or config["device"] != request["device"]
                    or maps["candidate"]["revision"] != request["candidate"] or maps["candidate"]["tree"] != request["source_tree"]
                    or any(maps[side]["revision"] != config["sides"][side]["revision"] for side in sides)
                    or (config.get("validation_mode") == PERFORMANCE_MODE and request.get("sources") != expected_sources)):
                fail("source/config/manifest binding differs")
            policy_check(policy, request, config, manifest, broker)
            control_identity(config)
            if config.get("validation_mode") == PERFORMANCE_MODE and sha(protected(CONTROL / "bench/prompts/prompt-512.txt")) != PERFORMANCE_PROTOCOL["prompt_sha256"]:
                fail("installed trusted fixed512 prompt differs")
            count = model_identities(config)
            if config.get("validation_mode") == PERFORMANCE_MODE:
                source.mkdir(mode=0o755, exist_ok=False)
            extracted_sides = {}
            for side in sides:
                target = Path(sources[side])
                extracted_sides[side] = dict(extract_source(staging / (side + "-source.tar.gz"), maps[side], target),
                    revision=maps[side]["revision"], tree=maps[side]["tree"])
                if sum(row["bytes"] for row in extracted_sides.values()) > LIMIT:
                    fail("combined source bytes exceed staging bound")
                if config.get("validation_mode") == PERFORMANCE_MODE:
                    for model, data in config["sides"][side]["models"].items():
                        template = data["template"]
                        prefix = "/source/" + side + "/"
                        if not template["runtime_path"].startswith(prefix):
                            fail("chat template source side differs")
                        expected = target / relative(template["runtime_path"][len(prefix):])
                        if template["host_path"] != str(expected) or sha(protected(expected)) != template["sha256"]:
                            fail("chat template differs from exact sealed source")
            if config.get("validation_mode") == PERFORMANCE_MODE:
                os.chmod(source, 0o555)
            extracted = extracted_sides["candidate"]
            for model in config["sides"]["candidate"]["models"]:
                payloads = {}
                for side in sides:
                    data = config["sides"][side]["models"][model]
                    settings = data["settings"]
                    name = "settings-" + model + ("-" + side if config.get("validation_mode") == PERFORMANCE_MODE else "") + ".ssd"
                    path = IROOT / (job + "-" + model + "-" + side + "-settings.ssd")
                    if settings["host_path"] != str(path) or sha(staging / name) != settings["sha256"]:
                        fail("settings payload/path differs")
                    payloads[side] = (staging / name).read_bytes()
                if len(sides) == 2 and not paired_settings(payloads["base"], payloads["candidate"]):
                    fail("base/candidate effective settings differ")
                for side, payload in payloads.items():
                    write(IROOT / (job + "-" + model + "-" + side + "-settings.ssd"), payload)
            machine = sealed_machine(config, str(IROOT / (job + "-machine.json")), "/models/machine.json")
            if machine is not None:
                write(IROOT / (job + "-machine.json"), machine)
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
            if config.get("validation_mode") == PERFORMANCE_MODE:
                proof["sources"] = extracted_sides
                proof["performance_protocol"] = config["performance_protocol"]
            write(staging / "ROOT-STAGE-RECEIPT.json", (json.dumps(proof, indent=2) + "\n").encode())
            return proof
        finally:
            os.close(fd)


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("fixed staging bridge deadline exceeded")
    return value


def peer_uid(connection):
    if not hasattr(socket, "SO_PEERCRED"):
        fail("Linux SO_PEERCRED is required")
    return struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]


def input_chunks(incoming, deadline):
    try:
        descriptor = incoming.fileno()
    except (OSError, AttributeError):
        descriptor = None
    while True:
        if descriptor is not None:
            if not select.select([descriptor], [], [], remaining(deadline))[0]:
                raise TimeoutError("staging stdin deadline exceeded")
            block = os.read(descriptor, 1 << 20)
        else:
            remaining(deadline)
            block = incoming.read(1 << 20)
        if not block:
            return
        yield block


def client(expected_sha256, incoming=None):
    """Forward the fixed root-stage input; no caller-supplied command or path."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        fail("reviewed stage input SHA required")
    protected(BRIDGE_SOCKET.parent, directory=True)
    info = BRIDGE_SOCKET.lstat()
    if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != 0
            or stat.S_IMODE(info.st_mode) != 0o660):
        fail("fixed bridge socket owner/type/mode differs")
    deadline = time.monotonic() + BRIDGE_TIMEOUT
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(min(10, remaining(deadline)))
        connection.connect(str(BRIDGE_SOCKET))
        if peer_uid(connection) != 0:
            fail("staging bridge peer is not root")
        connection.settimeout(remaining(deadline))
        connection.sendall((expected_sha256 + "\n").encode())
        total = 0
        for block in input_chunks(incoming or sys.stdin.buffer, deadline):
            total += len(block)
            if total > WIRE_LIMIT:
                fail("complete staging wire exceeds bound")
            connection.settimeout(remaining(deadline))
            connection.sendall(block)
        connection.shutdown(socket.SHUT_WR)
        response = bytearray()
        while True:
            connection.settimeout(remaining(deadline))
            block = connection.recv(min(65536, RESPONSE_LIMIT + 1 - len(response)))
            if not block:
                break
            response.extend(block)
            if len(response) > RESPONSE_LIMIT:
                fail("root staging response exceeds bound")
    value = parsed(bytes(response))
    if (not isinstance(value, dict) or value.get("status") != "passed"
            or value.get("root_uid") != 0 or value.get("root_input_sha256") != expected_sha256):
        detail = value.get("error", "invalid root receipt") if isinstance(value, dict) else "invalid root receipt"
        fail("root staging bridge refused: " + str(detail)[:2048])
    return value


def root_bridge(authorized_uid, connection=None):
    """One Accept=yes connection; root receive/policy/lease remain unchanged."""
    if os.geteuid() != 0:
        raise PermissionError("root bridge mode requires root")
    broker = parsed(protected(BROKER_CONFIG).read_bytes())
    if (type(authorized_uid) is not int or authorized_uid <= 0 or broker.get("device") != "dgx-spark"
            or broker.get("coordination_owner", {}).get("uid") != authorized_uid):
        fail("bridge peer UID differs from protected Spark coordination owner")
    if connection is None:
        connection = socket.socket(fileno=os.dup(0))
    deadline = time.monotonic() + BRIDGE_TIMEOUT
    with connection:
        try:
            if connection.family != socket.AF_UNIX or peer_uid(connection) != authorized_uid:
                fail("root staging bridge peer UID refused")
            header_deadline = min(deadline, time.monotonic() + 10)
            header = bytearray()
            while len(header) < 65:
                connection.settimeout(remaining(header_deadline))
                block = connection.recv(65 - len(header))
                if not block:
                    fail("truncated staging hash header")
                header.extend(block)
            if not re.fullmatch(rb"[0-9a-f]{64}\n", header):
                fail("fixed 64hex newline header required")
            # Bound the complete wire including tar padding before root receive
            # publishes anything. Anonymous spool closes on failure/process death.
            with tempfile.TemporaryFile() as incoming:
                total = 0
                while True:
                    connection.settimeout(remaining(deadline))
                    block = connection.recv(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > WIRE_LIMIT:
                        fail("complete staging wire exceeds bound")
                    incoming.write(block)
                incoming.seek(0)
                proof = receive(header[:-1].decode("ascii"), incoming)
            response = json.dumps(proof, indent=2).encode() + b"\n"
            if len(response) > RESPONSE_LIMIT:
                fail("root staging response exceeds bound")
        except Exception as error:
            response = json.dumps(dict(status="failed", error=type(error).__name__ + ": " + str(error)[:2048])).encode() + b"\n"
        connection.settimeout(remaining(deadline))
        connection.sendall(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expected-sha256")
    mode.add_argument("--root-bridge", action="store_true")
    parser.add_argument("--peer-uid", type=int)
    args = parser.parse_args()
    if args.root_bridge:
        if args.peer_uid is None:
            parser.error("root bridge requires the fixed peer UID")
        root_bridge(args.peer_uid)
    elif args.peer_uid is not None:
        parser.error("peer UID belongs only to the root service mode")
    elif os.geteuid() == 0:
        print(json.dumps(receive(args.expected_sha256), indent=2))
    else:
        try:
            print(json.dumps(client(args.expected_sha256), indent=2))
        except Exception as error:
            print("fixed staging client: " + type(error).__name__ + ": " + str(error)[:2048], file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
