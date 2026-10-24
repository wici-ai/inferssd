"""Temporary-filesystem root-staging protocol tests, without privilege or devices."""

import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import operator_prepare as prepare, operator_stage as stage
from ci.contract import load
from test_ci_operator_prepare import prepared_fixture, source as _source_fixture

git_source = _source_fixture


def packet(request, files, extra=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        members = [("root-input.json", (json.dumps(request) + "\n").encode())]
        members.extend((name, Path(row["local_path"]).read_bytes()) for name, row in files.items())
        if extra:
            members.append(extra)
        for name, raw in members:
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    return output.getvalue(), hashlib.sha256(members[0][1]).hexdigest()


@pytest.fixture
def staged(tmp_path, git_source, monkeypatch):
    args, api = prepared_fixture(tmp_path, git_source, devices=["cloud-5090"])
    prepare.prepare(args, api)
    root = tmp_path / "fake-host"
    root.mkdir()
    for name in ("inputs", "integrations", "broker", "control"):
        (root / name).mkdir()
    (root / "broker/jobs").mkdir()
    lock = root / "lock"
    lock.write_bytes(b"")
    original = lock.stat()
    constants = dict(BROOT=root / "broker", IROOT=root / "integrations", SROOT=root / "inputs",
                     CONTROL=root / "control", BROKER_CONFIG=root / "broker-config.json", POLICY=root / "policy.json")
    for key, value in constants.items():
        monkeypatch.setattr(stage, key, value)
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)
    # Only this synthetic test bypasses real root ownership. Actual receive()
    # calls the protected() chain checks on the installed filesystem.
    monkeypatch.setattr(stage, "protected", lambda p, **kw: Path(p))
    monkeypatch.setattr(stage, "control_identity", lambda config: None)
    monkeypatch.setattr(stage, "model_identities", lambda config: 2)
    monkeypatch.setattr(stage, "gpu_idle", lambda broker, policy: dict(uuid="GPU-synthetic", compute_apps_empty=True))
    job = "1" * 32
    config = load(args.output / "cloud-5090/config.json")
    manifest = load(args.output / "cloud-5090/manifest.json")
    policy = load(args.output / "cloud-5090/policy.review.json")
    request = load(args.output / "cloud-5090/root-input.json")
    data = load(args.output / "rootstage-inputs.json")["devices"]["cloud-5090"]
    files = copy.deepcopy(data["files"])
    broker = dict(device="cloud-5090", lock_identity=[original.st_dev, original.st_ino], lock_path=str(lock), stop_path=str(root / "STOP"))
    constants["BROKER_CONFIG"].write_text(json.dumps(broker))
    config["manifest_path"] = str(constants["BROOT"] / "jobs" / job / "manifest.json")
    manifest["sources"] = {"candidate": str(constants["SROOT"] / ("operator-" + job))}
    for model, data in config["sides"]["candidate"]["models"].items():
        data["settings"]["host_path"] = str(constants["IROOT"] / (job + "-" + model + "-candidate-settings.ssd"))
        manifest["models"]["settings-" + model + "-candidate.ssd"] = data["settings"]["host_path"]
    for name, value in (("manifest.json", manifest), ("config.json", config)):
        path = root / name
        if name == "config.json": value["manifest_sha256"] = prepare.digest(root / "manifest.json")
        path.write_text(json.dumps(value))
        files[name] = dict(local_path=str(path), bytes=path.stat().st_size, sha256=prepare.digest(path))
    request["files"] = {name: {k: value[k] for k in ("bytes", "sha256")} for name, value in files.items()}
    policy["broker_config_sha256"] = request["broker_config_sha256"] = prepare.digest(constants["BROKER_CONFIG"])
    policy["lock_identity"] = broker["lock_identity"]
    constants["POLICY"].write_text(json.dumps(policy))
    request["policy_sha256"] = prepare.digest(constants["POLICY"])
    return root, request, files, constants


def test_plain_packet_seals_exact_git_bytes_and_preserves_lease_config(staged):
    root, request, files, constants = staged
    before = (root / "lock").stat()
    broker_before = constants["BROKER_CONFIG"].read_bytes()
    raw, expected = packet(request, files)
    proof = stage.receive(expected, io.BytesIO(raw))
    assert proof["status"] == "passed" and proof["root_uid"] == 0
    assert constants["BROKER_CONFIG"].read_bytes() == broker_before
    after = (root / "lock").stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert not (root / "STOP").exists()
    source = Path(proof["source"]["root"])
    assert (source / "new.cpp").read_bytes() == b"old\n"
    assert stat.S_IMODE((source / "new.cpp").stat().st_mode) == 0o444
    assert stat.S_IMODE(source.stat().st_mode) == 0o555
    with pytest.raises(ValueError, match="fresh job"):
        stage.receive(expected, io.BytesIO(raw))


@pytest.mark.parametrize("kind", ["extra", "duplicate", "wrong-hash"])
def test_bad_member_cannot_publish_config(staged, kind):
    root, request, files, constants = staged
    extra = None
    if kind == "extra": extra = ("../escape", b"bad")
    elif kind == "duplicate": extra = ("config.json", b"duplicate")
    else: request["files"]["config.json"]["sha256"] = "0" * 64
    raw, expected = packet(request, files, extra)
    with pytest.raises(ValueError):
        stage.receive(expected, io.BytesIO(raw))
    assert not (constants["IROOT"] / (request["job_id"] + ".json")).exists()
    assert not (root / "escape").exists()


def test_caller_cannot_replace_local_root_policy(staged):
    root, request, files, constants = staged
    request["policy_sha256"] = "0" * 64
    raw, expected = packet(request, files)
    with pytest.raises(ValueError, match="root admission policy"):
        stage.receive(expected, io.BytesIO(raw))
    assert not list(constants["IROOT"].iterdir())


def test_gzip_transport_is_rejected_before_installation(staged):
    import gzip
    _, request, files, constants = staged
    raw, expected = packet(request, files)
    with pytest.raises(tarfile.ReadError):
        stage.receive(expected, io.BytesIO(gzip.compress(raw)))
    assert not list(constants["IROOT"].iterdir())


def test_source_link_and_missing_blob_rejected(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo("source.cpp")
        item.type, item.linkname = tarfile.SYMTYPE, "/etc/shadow"
        stream.addfile(item)
    source_map = dict(files=[dict(path="source.cpp", mode="100644", bytes=1, blob="0" * 40)])
    with pytest.raises(ValueError):
        stage.extract_source(archive, source_map, tmp_path / "output")


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_root_json_rejects_nonunique_or_nonfinite(raw):
    with pytest.raises(ValueError):
        stage.parsed(raw)


@pytest.mark.parametrize("mode", ["active", "wrong-gpu", "missing-uuid", "idle"])
def test_gpu_preflight_is_read_only_and_never_kills_external_work(monkeypatch, mode):
    calls = []
    def query(argv, timeout):
        calls.append(argv)
        if "--query-gpu=uuid" in argv:
            return b"GPU-other\n" if mode == "wrong-gpu" else b"GPU-synthetic\n"
        return b"1234, GPU-synthetic\n" if mode == "active" else b""
    monkeypatch.setattr(stage.subprocess, "check_output", query)
    policy = dict(gpu_uuid="" if mode == "missing-uuid" else "GPU-synthetic")
    if mode == "idle":
        assert stage.gpu_idle({}, policy)["compute_apps_empty"] is True
    else:
        with pytest.raises(ValueError):
            stage.gpu_idle({}, policy)
    assert all(argv[0] == "/usr/bin/nvidia-smi" and "--format=csv,noheader" in argv for argv in calls)
