"""Use existing root-admitted Spark RO views without changing protected files.

GGUF bytes remain UID1001-owned. Only root-sealed admission metadata, exact
current view identities/mounts and this job's two real worker probes authorize
that exception. No payload copy, chown, payload hash or broker extension occurs.
"""

from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time

from ci.contract import HASH, digest, load, save
from ci.staged_build import sealed_file
from ci.staged_checkpoint import file_identity, model_mount
from ci.staged_smoke import collect_log

KEYS = ("model_admission_path", "model_admission_sha256", "model_mount_proof_path",
        "model_mount_proof_sha256", "prior_model_job_id")
STAT = ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
OWNER = ("uid", "gid", "mode")
TIMEOUT = 120


def absolute(value):
    if (not isinstance(value, str) or not value.startswith("/") or "\0" in value
            or str(PurePosixPath(value)) != value or ".." in PurePosixPath(value).parts):
        raise ValueError("model admission requires canonical absolute paths")
    return value


def enabled(config):
    present = set(KEYS) & set(config)
    if not present:
        return False
    if (present != set(KEYS) or config.get("device") != "dgx-spark"
            or config["paths"]["platform"] != "linux"
            or not isinstance(config["prior_model_job_id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", config["prior_model_job_id"]) is None):
        raise ValueError("complete explicit Spark readonly model admission required")
    for key in ("model_admission_path", "model_mount_proof_path"):
        absolute(config[key])
    for key in ("model_admission_sha256", "model_mount_proof_sha256"):
        if not isinstance(config[key], str) or HASH.fullmatch(config[key]) is None:
            raise ValueError("root model metadata needs an exact SHA256")
    return True


def binding(config):
    return {key: config[key] for key in KEYS}


def parsed(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate model proof JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("nonfinite model proof JSON")))


def documents(config, *, root=None):
    """Read small sealed metadata on device, or its hash-checked retained copy."""
    rows = []
    for key, name in (("model_admission", "admission.json"), ("model_mount_proof", "root-proof.json")):
        if root is None:
            evidence, raw = sealed_file(config[key + "_path"], contents=True, limit=1 << 20)
            actual = evidence["sha256"]
        else:
            path = Path(root) / "model-admission" / name
            if path.is_symlink():
                raise ValueError("retained root model metadata cannot be a symlink")
            raw = path.read_bytes()
            if len(raw) > 1 << 20:
                raise ValueError("root model metadata exceeds its bound")
            actual = digest(path)
        if actual != config[key + "_sha256"]:
            raise ValueError("root-sealed model metadata hash differs")
        rows.append((parsed(raw), raw))
    admission, proof = (row[0] for row in rows)
    if (admission.get("status") != "full-payload-verified-awaiting-root-readonly-exposure"
            or admission.get("job_id") != config["prior_model_job_id"]
            or proof.get("status") != "readonly-host-exposure-pass"
            or type(proof.get("root_uid")) is not int or proof["root_uid"] != 0
            or proof.get("job_id") != config["prior_model_job_id"]
            or proof.get("admission_sha256") != config["model_admission_sha256"]
            or proof.get("source_uid_preserved") != 1001
            or proof.get("source_mode_preserved") != "0o664"
            or proof.get("payload_copy_or_chown") is not False):
        raise ValueError("original root model admission/proof binding differs")
    shards = [row for model in admission["models"].values() for row in model["shards"]]
    hosts = [absolute(row["host_path"]) for row in shards]
    if (type(admission.get("source_files")) is not int or admission["source_files"] != len(shards)
            or not shards or any(not model["shards"] for model in admission["models"].values())
            or len({row["runtime_path"] for row in shards}) != len(shards) or len(set(hosts)) != len(hosts)
            or sorted(proof["verified_host_paths"]) != sorted(hosts)):
        raise ValueError("root proof must cover the complete admitted shard set")
    exposed = {row["host"]: row for row in proof["source_and_view_identities"]}
    if (len(proof["source_and_view_identities"]) != len(shards)
            or len(exposed) != len(shards) or set(exposed) != set(hosts)):
        raise ValueError("root source/view proof set differs")
    for row in shards:
        for key in ("source_path", "runtime_path"):
            absolute(row[key])
        expected = {key: row[key] for key in (*STAT, *OWNER)}
        if (any(type(row[key]) is not int or row[key] < 0 for key in STAT)
                or row["uid"] != 1001 or row["gid"] != 1001 or row["mode"] != "0o664"
                or not isinstance(row["sha256"], str) or HASH.fullmatch(row["sha256"]) is None
                or row.get("historical_sha256") != row["sha256"]):
            raise ValueError("original full-payload hash/stat/owner declaration differs")
        view = exposed[row["host_path"]]
        if (view["source"] != row["source_path"] or view["identity"] != expected or view["view"] != expected
                or "ro" not in view["mount_options"].split(",")):
            raise ValueError("original source and readonly view identities differ")
    lease = admission["original_lock"]
    if (not isinstance(lease, list) or len(lease) != 2
            or any(type(x) is not int or x < 0 for x in lease)):
        raise ValueError("original model admission lease is missing")
    return admission, proof, rows


def view_identity(path):
    """The root controls directories; the RO leaf deliberately retains UID1001."""
    path = Path(absolute(path))
    for parent in path.parents:
        value = parent.lstat()
        if (not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode)
                or value.st_uid != 0 or value.st_mode & 0o022):
            raise ValueError("model view directories must be root-owned and nonwritable")
    value = path.lstat()
    if (not stat.S_ISREG(value.st_mode) or value.st_uid != 1001 or value.st_gid != 1001
            or stat.S_IMODE(value.st_mode) != 0o664):
        raise ValueError("admitted model view type/owner/mode changed")
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                uid=value.st_uid, gid=value.st_gid, mode=oct(stat.S_IMODE(value.st_mode)))


def readonly_mount(path):
    result = subprocess.run(["/usr/bin/findmnt", "-rn", "--mountpoint", absolute(path),
                             "-o", "VFS-OPTIONS"], capture_output=True, text=True, check=True, timeout=10)
    options = result.stdout.strip()
    if "\n" in options or "ro" not in options.split(","):
        raise ValueError("model view is not the exact readonly per-file bind")
    return options


def checked_model(config, manifest, model, data, admission=None, *, live=True):
    identity = data["model_identity"]
    if not enabled(config):
        for row in identity["shards"]:
            if file_identity(row["host_path"]) != row["stat"]:
                raise ValueError("root-bound readonly model shard changed")
        return identity
    if admission is None:
        admission, _, _ = documents(config)
    expected = {row["runtime_path"]: row for row in admission["models"][model]["shards"]}
    shards = identity["shards"]
    if len(shards) != len(expected) or {row["runtime_path"] for row in shards} != set(expected):
        raise ValueError("selected model must use its exact admitted shard set")
    for row in shards:
        bound = expected[row["runtime_path"]]
        absolute(row["host_path"]); absolute(row["runtime_path"])
        if (row["host_path"] != bound["host_path"] or not row["host_path"].endswith(".gguf")
                or not model_mount(manifest, row["host_path"], row["runtime_path"])
                or set(row["stat"]) != set(STAT) or any(type(row["stat"][key]) is not int for key in STAT)
                or row["stat"] != {key: bound[key] for key in STAT}
                or row.get("sha256") != bound["sha256"]):
            raise ValueError("selected shard differs from root admission or readonly mount mapping")
        if live:
            if view_identity(row["host_path"]) != {key: bound[key] for key in (*STAT, *OWNER)}:
                raise ValueError("admitted readonly model view identity changed")
            readonly_mount(row["host_path"])
    return dict(identity, readonly_admission=binding(config))


def selected_inputs(config, manifest, admission, *, live):
    selected = {}
    for spec in config["sides"].values():
        for model, data in spec["models"].items():
            value = checked_model(config, manifest, model, data, admission, live=live)
            if model in selected and selected[model] != value:
                raise ValueError("base/candidate admitted model identities differ")
            selected[model] = value
    return selected


def prepare(config, manifest, output):
    if not enabled(config):
        return
    probe_declarations(config, manifest)
    admission, _, rows = documents(config)
    selected = selected_inputs(config, manifest, admission, live=True)
    target = Path(output) / "model-admission"
    target.mkdir()
    for (_, raw), name in zip(rows, ("admission.json", "root-proof.json")):
        with (target / name).open("xb") as stream:
            stream.write(raw)
    save(target / "inputs-before.json", selected)


def probe_spec(config, manifest, when):
    if when not in ("before", "after"):
        raise ValueError("unknown readonly probe phase")
    paths = sorted({absolute(row["runtime_path"]) for spec in config["sides"].values()
                    for data in spec["models"].values() for row in data["model_identity"]["shards"]})
    if not manifest["sources"] or not set(manifest["sources"]) <= {"base", "candidate"}:
        raise ValueError("readonly probe requires the exact selected source sides")
    sources = ["/source/" + side + "/engine/include/llama.h" for side in sorted(manifest["sources"])]
    code = """import os,errno,json;from pathlib import Path
 def denied(p,flags):
  try:fd=os.open(p,flags,0o600)
  except OSError as e:assert e.errno in (errno.EACCES,errno.EPERM,errno.EROFS),str(e)
  else:os.close(fd);raise AssertionError('unexpected write access: '+p)
 paths=PATHS
 for p in paths:
  assert Path(p).open('rb').read(4)==b'GGUF'
  denied(p,os.O_RDWR)
 for p in SOURCES:denied(p,os.O_RDWR)
 for p in sorted({str(Path(p).parent)for p in paths}):denied(p,os.O_RDWR|os.O_TMPFILE)
 print(json.dumps({'marker':'MODEL_READONLY_PASS','model_files':len(paths),'source_files':len(SOURCES),'model_parents':len({str(Path(p).parent)for p in paths})}))
""".replace("\n ", "\n").replace("PATHS", repr(paths)).replace("SOURCES", repr(sources))
    cwd = config["sides"]["candidate"]["paths"]["cwd"]
    return dict(stage_id="model-readonly-" + when, kind="build" if when == "before" else "test",
                argv=["/usr/bin/python3", "-I", "-c", code], environment={},
                cwd=cwd if when == "before" else str(PurePosixPath(cwd).parent), timeout_s=TIMEOUT)


def probe_declarations(config, manifest):
    for when in ("before", "after"):
        spec = probe_spec(config, manifest, when)
        expected = dict(kind=spec["kind"], side="candidate", executables=[spec["argv"][0]],
                        cwd=spec["cwd"], gpu=False)
        if manifest.get("stages", {}).get(spec["stage_id"]) != expected:
            raise ValueError("root manifest must authorize the exact readonly probe stage")


def expected_counts(config, manifest):
    paths = {row["runtime_path"] for spec in config["sides"].values()
             for data in spec["models"].values() for row in data["model_identity"]["shards"]}
    return dict(marker="MODEL_READONLY_PASS", model_files=len(paths),
                source_files=len(manifest["sources"]),
                model_parents=len({str(PurePosixPath(path).parent) for path in paths}))


def validate_probe(config, manifest, when, started, status, stdout, lease, controller_uid):
    spec = probe_spec(config, manifest, when)
    keys = ("stage_id", "handle", "worker_pid", "worker_uid", "birth_identity", "scope_identity")
    if (type(controller_uid) is not int or controller_uid <= 0
            or started.get("stage_id") != spec["stage_id"] or any(status.get(k) != started[k] for k in keys)
            or type(started.get("worker_uid")) is not int or started["worker_uid"] in (0, 1001, controller_uid)
            or type(started.get("worker_pid")) is not int or started["worker_pid"] <= 1
            or not isinstance(started.get("handle"), str)
            or not all(started.get(k) for k in ("handle", "birth_identity", "scope_identity"))
            or type(status.get("returncode")) is not int or status["returncode"] != 0 or status.get("running") is not False
            or status.get("cleanup_confirmed") is not True):
        raise ValueError("current worker readonly probe identity/exit/cleanup differs")
    receipt = status["receipt"]
    if (receipt.get("passed") is not True or "error" in receipt or "cleanup_error" in receipt):
        raise ValueError("readonly probe producer recorded failure despite zero-exit cleanup")
    scope = started["scope_identity"]
    if not isinstance(scope, str) or not scope.startswith("/"):
        raise ValueError("readonly probe requires its actual Linux resource scope")
    unit = PurePosixPath(scope).name
    if (not unit or receipt.get("unit") != unit or receipt.get("cleanup", {}).get("unit") != unit
            or receipt.get("guardian", {}).get("unit") != unit
            or receipt.get("guardian", {}).get("cleanup", {}).get("unit") != unit):
        raise ValueError("readonly probe guardian/cleanup unit differs from its current scope")
    if (any(receipt.get(k) != started[k] for k in keys) or receipt.get("lease") != lease
            or receipt.get("argv") != spec["argv"]
            or type(receipt.get("returncode")) is not int or receipt["returncode"] != 0
            or receipt.get("cleanup_confirmed") is not True
            or receipt.get("cleanup", {}).get("processes_gone") is not True
            or type(receipt.get("guardian", {}).get("returncode")) is not int
            or receipt["guardian"]["returncode"] != 0
            or receipt.get("guardian", {}).get("cleanup_confirmed") is not True):
        raise ValueError("readonly probe lacks its actual lease/guardian receipt")
    counts = parsed(stdout)
    if (counts != expected_counts(config, manifest)
            or any(type(counts[key]) is not int for key in ("model_files", "source_files", "model_parents"))):
        raise ValueError("worker probe did not cover the selected model/source set")


def probe(session, config, manifest, when, output):
    if not enabled(config):
        return
    admission, _, _ = documents(config)
    target = Path(output) / "model-admission"
    if when == "after":
        save(target / "inputs-after.json", selected_inputs(config, manifest, admission, live=True))
    folder = target / when
    folder.mkdir()
    spec = probe_spec(config, manifest, when)
    started_ns = time.monotonic_ns()
    stage = session.start(**dict(stage_id=spec["stage_id"], kind=spec["kind"], argv=spec["argv"],
                                 environment=spec["environment"], cwd=spec["cwd"], timeout_s=spec["timeout_s"]))
    save(folder / "started.json", asdict(stage))
    completed = session.wait(stage, deadline=time.monotonic() + TIMEOUT)
    save(folder / "stage.json", completed)
    collect_log(session, stage, folder)
    lease = dict(zip(("device", "inode"), admission["original_lock"]))
    validate_probe(config, manifest, when, asdict(stage), completed, (folder / "stdout.log").read_bytes(), lease, session.controller_uid)
    save(folder / "probe.json", dict(job_id=session.job_id, phase=when, controller_uid=session.controller_uid,
         manifest_sha256=config["manifest_sha256"], started_ns=started_ns, finished_ns=time.monotonic_ns(),
         stage_sha256=digest(folder / "stage.json"), started_sha256=digest(folder / "started.json"),
         stdout_sha256=digest(folder / "stdout.log")))


def gate(root, config, manifest):
    probe_declarations(config, manifest)
    root = Path(root)
    admission, _, _ = documents(config, root=root)
    expected = selected_inputs(config, manifest, admission, live=False)
    target = root / "model-admission"
    for when in ("before", "after"):
        if load(target / ("inputs-" + when + ".json")) != expected:
            raise ValueError("readonly model identities changed across the selected job")
    opened, finished = load(root / "job-opened.json"), load(root / "job-finished.json")
    lease = dict(zip(("device", "inode"), admission["original_lock"]))
    if (opened.get("opened") is not True or opened.get("job_id") != manifest["job_id"] or finished.get("job_id") != manifest["job_id"]
            or opened.get("manifest_sha256") != config["manifest_sha256"]
            or finished.get("manifest_sha256") != config["manifest_sha256"]
            or opened.get("lease") != lease or finished.get("lease") != lease
            or finished.get("cleanup_confirmed") is not True or finished.get("errors") != []):
        raise ValueError("readonly proof belongs to another job/manifest/lease or lacks cleanup")
    phases = {}
    identities = []
    for when in ("before", "after"):
        folder = target / when
        evidence, started, status = (load(folder / name) for name in ("probe.json", "started.json", "stage.json"))
        if (evidence.get("job_id") != manifest["job_id"] or evidence.get("phase") != when
                or evidence.get("manifest_sha256") != config["manifest_sha256"]
                or any(evidence.get(key + "_sha256") != digest(folder / name) for key, name in
                       (("stage", "stage.json"), ("started", "started.json"), ("stdout", "stdout.log")))
                or type(evidence.get("started_ns")) is not int or type(evidence.get("finished_ns")) is not int
                or not 0 < evidence["started_ns"] <= evidence["finished_ns"]):
            raise ValueError("current readonly probe evidence is missing or changed")
        validate_probe(config, manifest, when, started, status, (folder / "stdout.log").read_bytes(), lease, evidence["controller_uid"])
        if finished["stages"].get(started["stage_id"]) != status["receipt"]:
            raise ValueError("readonly probe is not an actual stage of the current finished job")
        identities.append((started["handle"], (started["worker_pid"], started["birth_identity"]), started["scope_identity"]))
        phases[when] = dict(evidence, probe_sha256=digest(folder / "probe.json"))
    if (any(a == b for a, b in zip(identities[0], identities[1]))
            or phases["before"]["controller_uid"] != phases["after"]["controller_uid"]
            or phases["before"]["finished_ns"] >= phases["after"]["started_ns"]):
        raise ValueError("before/after readonly probes reused a worker or reversed order")
    return dict(schema_version=1, status="passed", job_id=manifest["job_id"],
                binding=binding(config), phases=phases)


def finalize(root, config, manifest):
    if enabled(config):
        save(Path(root) / "model-admission" / "gate.json", gate(root, config, manifest))
        return digest(Path(root) / "model-admission" / "gate.json")
    return None


def retained(root, config, manifest, report):
    if enabled(config):
        path = Path(root) / "model-admission" / "gate.json"
        if report.get("model_admission_sha256") != digest(path) or load(path) != gate(root, config, manifest):
            raise ValueError("current readonly model admission gate is missing or differs")
