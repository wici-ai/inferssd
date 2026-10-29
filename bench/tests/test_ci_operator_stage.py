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
from test_ci_operator_prepare import prepared_fixture, changed_identity, source as _source_fixture

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


@pytest.fixture(params=[prepare.LEGACY_MODE, prepare.MODE, "control-only", "calibration"])
def staged(tmp_path, git_source, monkeypatch, request):
    mode = prepare.LEGACY_MODE if request.param == prepare.LEGACY_MODE else prepare.MODE
    if request.param == "calibration":
        head = git_source[1]["head"]
        git_source = (git_source[0], dict(base=head, head=head, candidate=head))
    args, api = prepared_fixture(tmp_path, git_source, devices=["cloud-5090"], mode=mode)
    if request.param == "control-only":
        changed_identity(args, "bench/ci/example.py")
    if request.param == "calibration":
        args.calibration_pairs = 4
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
    sides = stage.source_sides(config)
    manifest["sources"] = {side: str(constants["SROOT"] / ("operator-" + job) / side) if mode == prepare.MODE
        else str(constants["SROOT"] / ("operator-" + job)) for side in sides}
    for side in sides:
        for model, data in config["sides"][side]["models"].items():
            data["settings"]["host_path"] = str(constants["IROOT"] / (job + "-" + model + "-" + side + "-settings.ssd"))
            manifest["models"]["settings-" + model + "-" + side + ".ssd"] = data["settings"]["host_path"]
            data["template"]["host_path"] = manifest["sources"][side] + "/chat.jinja"
    if mode == prepare.MODE:
        import final_chat
        path = constants["CONTROL"] / "bench/prompts/prompt-512.txt"
        path.parent.mkdir(parents=True)
        path.write_bytes(final_chat.prompt_path("512").read_bytes())
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
    leaf = source / ("new.cpp" if (source / "new.cpp").exists() else "old.cpp")
    assert leaf.read_bytes() == b"old\n"
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o444
    assert stat.S_IMODE(source.stat().st_mode) == 0o555
    if "sources" in request:
        assert proof["performance_protocol"] == load(root / "config.json")["performance_protocol"]
        assert set(proof["sources"]) == set(request["sources"])
        if "base" in proof["sources"]:
            calibration = request["sources"]["base"] == request["sources"]["candidate"]
            assert calibration == (proof["performance_protocol"] != stage.PERFORMANCE_PROTOCOL)
            # A calibration seals the same commit twice, as two separate sides.
            present, absent = ("new.cpp", "old.cpp") if calibration else ("old.cpp", "new.cpp")
            assert Path(proof["sources"]["base"]["root"], present).read_bytes() == b"old\n"
            assert not Path(proof["sources"]["base"]["root"], absent).exists()
            assert proof["sources"]["base"]["root"] != proof["sources"]["candidate"]["root"]
            for model in ("muse", "deepseek"):
                left, right = ((constants["IROOT"] / (request["job_id"] + "-" + model + "-" + side + "-settings.ssd")).read_bytes() for side in ("base", "candidate"))
                assert left != right and stage.paired_settings(left, right)
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


@pytest.mark.parametrize("staged", [prepare.MODE], indirect=True)
@pytest.mark.parametrize("mutation", ["base-revision", "base-tree", "source-map-omitted", "settings-policy", "prompt", "template"])
def test_double_source_root_admission_refuses_mixed_inputs(staged, mutation):
    root, request, files, constants = staged
    assert "sources" in request
    if mutation == "base-revision": request["sources"]["base"]["revision"] = request["candidate"]
    elif mutation == "base-tree": request["sources"]["base"]["tree"] = "0" * 40
    elif mutation == "source-map-omitted":
        del files["base-source-map.json"]
        del request["files"]["base-source-map.json"]
    elif mutation == "settings-policy":
        policy = load(constants["POLICY"])
        policy["side_settings_sha256"]["base"]["muse"] = "0" * 64
        constants["POLICY"].write_text(json.dumps(policy))
        request["policy_sha256"] = prepare.digest(constants["POLICY"])
    elif mutation == "prompt": (constants["CONTROL"] / "bench/prompts/prompt-512.txt").write_text("wrong prompt")
    else:
        config = load(Path(files["config.json"]["local_path"]))
        config["sides"]["base"]["models"]["muse"]["template"]["sha256"] = "0" * 64
        path = Path(files["config.json"]["local_path"])
        path.write_text(json.dumps(config))
        files["config.json"].update(bytes=path.stat().st_size, sha256=prepare.digest(path))
        request["files"]["config.json"].update(bytes=path.stat().st_size, sha256=prepare.digest(path))
    raw, expected = packet(request, files)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.receive(expected, io.BytesIO(raw))
    assert not (constants["IROOT"] / (request["job_id"] + ".json")).exists()


@pytest.mark.parametrize("case,admitted", [
    ("gate", True), ("calibration", True), ("calibration-two-revisions", False), ("calibration-too-many", False),
    ("calibration-bool", False), ("single-sample", False), ("gate-other-pairs", False), ("other-order", False),
    ("missing", False)])
def test_root_admits_only_the_gate_protocol_or_its_one_revision_calibration(case, admitted):
    sides = {side: dict(revision="c" * 40) for side in ("base", "candidate")}
    protocol = dict(stage.PERFORMANCE_PROTOCOL)
    if case.startswith("calibration"):
        protocol.update(samples_per_side=4, purpose="noise-calibration")
    if case == "calibration-two-revisions": sides["base"]["revision"] = "a" * 40
    elif case == "calibration-too-many": protocol["samples_per_side"] = 5
    elif case == "calibration-bool": protocol["samples_per_side"] = True
    elif case == "single-sample": protocol.update(samples_per_side=1)
    elif case == "gate-other-pairs": protocol["samples_per_side"] = 3
    elif case == "other-order": protocol["order"] = "ABAB"
    config = dict(sides=sides, performance_protocol=None if case == "missing" else protocol)
    assert stage.protocol_admitted(config) is admitted
    # The trusted controller derives the same protocol values.
    from ci.selection_contract import performance_protocol
    assert stage.PERFORMANCE_PROTOCOL == performance_protocol()
