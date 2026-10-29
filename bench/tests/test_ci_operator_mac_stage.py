"""Mac operator protocol contracts; no SSH, broker, Metal or model execution."""

import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import operator_prepare as prepare, operator_mac_stage as mac
from ci.contract import load
from test_ci_operator_prepare import prepared_fixture, source as source_fixture
from test_ci_operator_stage import packet

git_source = source_fixture
MACHINE = dict(hardware_uuid_sha256="a" * 64, hardware_model="Mac17,6", cpu_brand="Apple M5 Max")
RESOURCES = dict(memory_budget_bytes=40 << 30, host_reserve_bytes=3 << 30, max_processes=128)


def mac_prepared_fixture(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, devices=["local-mbp"], mode=prepare.MODE)
    inputs = load(args.inputs)
    declared = copy.deepcopy(inputs["devices"]["cloud-5090"])
    config = load(tmp_path / declared["template"])
    manifest = load(tmp_path / declared["manifest"])
    old_job = manifest["job_id"]
    job_root = "/Library/InferSSD-CI/jobs/" + old_job
    config.update(device="local-mbp", numerical=dict(config["numerical"], backend="MTL0"),
                  host=dict(resources=RESOURCES, os_page_cache_treatment="same reviewed Mac cache treatment"))
    assets = {}
    for side, spec in config["sides"].items():
        spec["paths"] = dict(platform="darwin", job_root=job_root, source="/Library/InferSSD-CI/inputs/old/" + side,
                             build=job_root + "/work/" + side, frozen=job_root + "/frozen/" + side,
                             cwd=job_root + "/work/" + side)
        for model, data in spec["models"].items():
            data["model_identity"] = copy.deepcopy(config["model_identity"][model])
            data["model_identity"]["shards"][0].update(host_path="/Library/InferSSD-CI/models/" + model + "/model.gguf",
                                                      runtime_path="/Library/InferSSD-CI/models/" + model + "/model.gguf")
            source_path = spec["paths"]["source"] + "/chat.jinja"
            data["template"].update(host_path=source_path, runtime_path=source_path)
            settings_path = "/Library/InferSSD-CI/inputs/old/settings-" + model + "-" + side + ".ssd"
            name = "mac-" + model + "-" + side + ".ssd"
            raw = ('inferssd-settings 1\nexpert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/' + side +
                   '/assets/' + model + '-alias.tsv"]\n').encode()
            (tmp_path / name).write_bytes(raw)
            assets["settings-" + model + "-" + side + ".ssd"] = name
            data["settings"].update(host_path=settings_path, runtime_path=settings_path, sha256=prepare.digest(tmp_path / name))
            data["smoke"].update(cwd=spec["paths"]["cwd"], environment={},
                os_page_cache_treatment=config["host"]["os_page_cache_treatment"])
            data["benchmarks"] = {"512": copy.deepcopy(data["smoke"])}
    config["paths"] = config["sides"]["candidate"]["paths"]
    manifest["models"] = {model: "/Library/InferSSD-CI/models/" + model for model in ("muse", "deepseek")}
    declared.update(job_id="3" * 32, template="mac-template.json", manifest="mac-manifest.json", rootstage_files=assets,
                    machine_identity=MACHINE, broker_device="mac-m5-max")
    declared.pop("gpu_uuid")
    declared["ssh"] = dict(argv=["ssh", "-T", "-a", "operator@example.invalid"], sudo_password_required=False,
        root_argv=[prepare.MAC_PYTHON, "-I", "-B", prepare.device_layout("local-mbp", "0" * 32)["stager"]])
    for name, value in [(declared["template"], config), (declared["manifest"], manifest)]:
        (tmp_path / name).write_text(json.dumps(value))
    inputs["devices"]["local-mbp"] = declared
    args.inputs.write_text(json.dumps(inputs))
    return args, api


def test_mac_prepare_seals_two_sources_and_fixed_template_policy(tmp_path, git_source):
    args, api = mac_prepared_fixture(tmp_path, git_source)
    envelope = prepare.prepare(args, api)
    assert envelope["execution_devices"] == ["local-mbp"]
    folder = args.output / "local-mbp"
    config, manifest, policy, request = (load(folder / name) for name in
        ("config.json", "manifest.json", "policy.review.json", "root-input.json"))
    job = envelope["jobs"]["local-mbp"]["job_id"]
    layout = prepare.device_layout("local-mbp", job)
    assert config["manifest_path"] == "/Library/InferSSD-CI/private/job-manifests/" + job + ".json"
    assert manifest["input_roots"] == [layout["source"]]
    assert manifest["selection_lock_identity"] == [1, 2]
    assert manifest["selection_resource_policy"] == RESOURCES
    assert "gpu_uuid" not in policy and policy["machine_identity"] == MACHINE
    assert "side_settings_sha256" not in policy
    assert request["operator_stage_sha256"] == prepare.digest(Path(mac.__file__))
    for side in ("base", "candidate"):
        assert manifest["sources"][side] == layout["source"] + "/" + side
        for model, data in config["sides"][side]["models"].items():
            name = "settings-" + model + "-" + side + ".ssd"
            raw = (folder / name).read_bytes()
            assert b"{{job_id}}" not in raw
            assert hashlib.sha256(raw).hexdigest() == data["settings"]["sha256"] == request["files"][name]["sha256"]
            canonical = mac.settings_template(raw, job, side)
            assert hashlib.sha256(canonical).hexdigest() == policy["side_settings_template_sha256"][side][model]
            assert data["settings"]["runtime_path"] == data["settings"]["host_path"] == layout["source"] + "/" + name
            assert data["template"]["runtime_path"] == data["template"]["host_path"] == manifest["sources"][side] + "/chat.jinja"
    assert prepare.paired_settings((folder / "settings-muse-base.ssd").read_bytes(),
        (folder / "settings-muse-candidate.ssd").read_bytes(), device="local-mbp", job_id=job)
    # The Mac worker reads the sealed characteristics in its read-only input root; nothing is mounted.
    machine = config["machine_characteristics"]
    assert machine["host_path"] == machine["runtime_path"] == layout["source"] + "/machine.json"
    assert envelope["machine_characteristics_sha256"] == {"local-mbp": machine["sha256"]}
    assert "machine.json" not in manifest["models"]


@pytest.mark.parametrize("raw", [
    b'expert.source = ["/elsewhere/{{job_id}}/x"]\n',
    b'expert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/candidate/../x"]\n',
    b'expert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/base/x"]\n',
    b'place.costs = {"/Library/InferSSD-CI/jobs/{{job_id}}/frozen/candidate/x":1}\n',
    b'expert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/candidate/x"]\nother = "{{job_id}}"\n',
    b'expert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/candidate/x", "y"]\n',
])
def test_settings_placeholder_only_one_exact_expert_source(raw):
    with pytest.raises(ValueError):
        mac.settings_payload(raw, "3" * 32, "candidate")


def test_mac_transport_only_frozen_python_and_stager():
    value = dict(argv=["ssh", "-T", "-a", "host"], sudo_password_required=False,
        root_argv=[prepare.MAC_PYTHON, "-I", "-B", prepare.device_layout("local-mbp", "0" * 32)["stager"]])
    assert prepare.transport_profile(value, device="local-mbp")[1] == value["root_argv"]
    for argv in [["/usr/bin/python3", *value["root_argv"][1:]], value["root_argv"][:-1] + ["/tmp/helper.py"],
                 ["/bin/sh"], [prepare.MAC_PYTHON, "-I", value["root_argv"][-1]]]:
        with pytest.raises(ValueError):
            prepare.transport_profile(dict(value, root_argv=argv), device="local-mbp")
    for fields in (dict(sudo_password_required=True), dict(sudo_mode="nopasswd")):
        with pytest.raises(ValueError, match="forced SSH entry"):
            prepare.transport_profile(dict(value, **fields), device="local-mbp")


@pytest.fixture
def admitted(tmp_path, git_source, monkeypatch):
    args, api = mac_prepared_fixture(tmp_path, git_source)
    prepare.prepare(args, api)
    folder = args.output / "local-mbp"
    config, manifest, policy, request = (load(folder / name) for name in
        ("config.json", "manifest.json", "policy.review.json", "root-input.json"))
    files = copy.deepcopy(load(args.output / "rootstage-inputs.json")["devices"]["local-mbp"]["files"])
    fake = tmp_path / "host"
    for name in ("integrations", "inputs", "jobs", "private/job-manifests", "controller/current/bench/prompts", "control"):
        (fake / name).mkdir(parents=True, exist_ok=True)
    constants = dict(BASE=fake, CONTROL=fake / "controller/current", BROKER_CONFIG=fake / "control/config.json",
        POLICY=fake / "control/operator-policy.json", IROOT=fake / "integrations", SROOT=fake / "inputs", MANIFESTS=fake / "private/job-manifests")
    for name, value in constants.items():
        monkeypatch.setattr(mac, name, value)
    monkeypatch.setattr(mac.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mac.sys, "platform", "darwin")
    monkeypatch.setattr(mac.common, "protected", lambda path, **kwargs: Path(path))
    monkeypatch.setattr(mac, "control_identity", lambda policy: None)
    monkeypatch.setattr(mac, "machine_identity", lambda: MACHINE)
    monkeypatch.setattr(mac, "worker_idle", lambda broker: None)
    monkeypatch.setattr(mac, "models_readonly", lambda config, policy: 2)
    monkeypatch.setattr(mac, "ensure_broker", lambda broker, expected: dict(status="ready", config_sha256=expected))
    lock = fake / "lock"
    lock.write_bytes(b"")
    st = lock.stat()
    job = request["job_id"]
    broker = dict(device="mac-m5-max", worker_uid=62002, runner_uid=62001, lock=str(lock),
        lock_dev_inode=[st.st_dev, st.st_ino], stop=str(fake / "STOP"), executor=RESOURCES)
    constants["BROKER_CONFIG"].write_text(json.dumps(broker))
    policy["broker_config_sha256"] = request["broker_config_sha256"] = prepare.digest(constants["BROKER_CONFIG"])
    policy["lock_identity"] = broker["lock_dev_inode"]
    config["manifest_path"] = str(constants["MANIFESTS"] / (job + ".json"))
    source = constants["SROOT"] / ("operator-" + job)
    manifest["input_roots"] = [str(source)]
    manifest["selection_lock_identity"] = broker["lock_dev_inode"]
    for side in ("base", "candidate"):
        manifest["sources"][side] = str(source / side)
        for model, data in config["sides"][side]["models"].items():
            path = str(source / ("settings-" + model + "-" + side + ".ssd"))
            data["settings"].update(host_path=path, runtime_path=path)
            manifest["models"]["settings-" + model + "-" + side + ".ssd"] = path
            data["template"].update(host_path=str(source / side / "chat.jinja"), runtime_path=str(source / side / "chat.jinja"))
    machine = str(source / "machine.json")
    config["machine_characteristics"].update(host_path=machine, runtime_path=machine)
    for name, value in (("manifest.json", manifest), ("config.json", config)):
        if name == "config.json":
            value["manifest_sha256"] = prepare.digest(fake / "manifest.json")
        path = fake / name
        path.write_text(json.dumps(value))
        files[name] = dict(local_path=str(path), bytes=path.stat().st_size, sha256=prepare.digest(path))
    request["files"] = {name: {k: v[k] for k in ("bytes", "sha256")} for name, v in files.items()}
    constants["POLICY"].write_text(json.dumps(policy))
    request["policy_sha256"] = prepare.digest(constants["POLICY"])
    import final_chat
    (constants["CONTROL"] / "bench/prompts/prompt-512.txt").write_bytes(final_chat.prompt_path("512").read_bytes())
    return fake, request, files, constants


def test_mac_receive_seals_without_precreating_worker_job(admitted):
    fake, request, files, constants = admitted
    constants["MANIFESTS"].chmod(0o700)
    before = (fake / "lock").stat()
    raw, expected = packet(request, files)
    proof = mac.receive(expected, io.BytesIO(raw))
    job = request["job_id"]
    assert proof["status"] == "passed" and proof["root_uid"] == 0 and proof["machine_identity"] == MACHINE
    assert proof["resource_enforcement"] == "sampled-UMA-guard-not-hard-cgroup"
    assert not (fake / "jobs" / job).exists()
    assert (constants["MANIFESTS"] / (job + ".json")).is_file()
    private = constants["MANIFESTS"] / (job + ".json")
    mirror = constants["SROOT"] / ("operator-" + job) / "manifest.json"
    assert mirror.read_bytes() == private.read_bytes()
    assert stat.S_IMODE(mirror.stat().st_mode) == 0o444
    assert stat.S_IMODE(mirror.parent.stat().st_mode) == 0o555
    assert stat.S_IMODE(constants["MANIFESTS"].stat().st_mode) == 0o700
    assert proof["runner_manifest"] == dict(path=str(mirror), sha256=prepare.digest(mirror),
        broker_path=str(private), broker_sha256=prepare.digest(private), root_owned_readonly=True)
    assert (constants["IROOT"] / (job + ".json")).is_file()
    assert (fake / "lock").stat().st_ino == before.st_ino
    assert not (fake / "STOP").exists()
    assert Path(proof["sources"]["base"]["root"], "old.cpp").read_bytes() == b"old\n"
    assert Path(proof["sources"]["candidate"]["root"], "new.cpp").read_bytes() == b"old\n"
    assert stat.S_IMODE(Path(proof["source"]["root"]).stat().st_mode) == 0o555
    # The PR job's sealed characteristics, read-only in its input root, where the worker reads them.
    machine = mirror.parent / "machine.json"
    assert machine.read_text() == load(fake / "config.json")["machine_characteristics"]["text"]
    assert stat.S_IMODE(machine.stat().st_mode) == 0o444
    with pytest.raises(ValueError, match="fresh Mac job"):
        mac.receive(expected, io.BytesIO(raw))


@pytest.mark.parametrize("damage", ["text", "path"])
def test_mac_root_stages_only_exact_sealed_characteristics(admitted, damage):
    fake, request, files, constants = admitted
    path = Path(files["config.json"]["local_path"])
    config = load(path)
    machine = config["machine_characteristics"]
    if damage == "text": machine["text"] += " "
    else: machine["runtime_path"] = "/Library/InferSSD-CI/jobs/" + request["job_id"] + "/work/machine/machine.json"
    path.write_text(json.dumps(config))
    for row in (files["config.json"], request["files"]["config.json"]):
        row.update(bytes=path.stat().st_size, sha256=prepare.digest(path))
    raw, expected = packet(request, files)
    with pytest.raises(ValueError, match="sealed machine characteristics"):
        mac.receive(expected, io.BytesIO(raw))
    assert not (constants["SROOT"] / ("operator-" + request["job_id"]) / "machine.json").exists()
    assert not (constants["IROOT"] / (request["job_id"] + ".json")).exists()


def test_slow_mac_upload_leaves_full_processing_budget(monkeypatch):
    clock, alarms, progress = [100.0], [], {}
    spool = io.BytesIO()
    monkeypatch.setattr(mac.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mac.sys, "platform", "darwin")
    monkeypatch.setattr(mac.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mac.tempfile, "TemporaryFile", lambda: spool)
    def slow_input(incoming, deadline):
        assert deadline == 100.0 + mac.TRANSFER_TIMEOUT
        clock[0] += mac.TRANSFER_TIMEOUT - 1
        mac.common.remaining(deadline)
        yield b"complete wire"
    def seal(expected, incoming, deadline):
        assert incoming.read() == b"complete wire"
        assert deadline == clock[0] + mac.PROCESSING_TIMEOUT
        clock[0] += mac.PROCESSING_TIMEOUT - 1
        mac.common.remaining(deadline)
        return {"status": "passed"}
    monkeypatch.setattr(mac.common, "input_chunks", slow_input)
    monkeypatch.setattr(mac, "receive_packet", seal)
    assert mac.receive("a" * 64, io.BytesIO(), progress=progress, arm_deadline=alarms.append) == {"status": "passed"}
    assert alarms == [mac.TRANSFER_TIMEOUT, mac.PROCESSING_TIMEOUT]
    assert progress == dict(phase="processing", received_bytes=len(b"complete wire"))
    assert spool.closed


def test_expired_mac_upload_closes_anonymous_spool_without_sealing(monkeypatch):
    clock, progress = [100.0], {}
    spool = io.BytesIO()
    monkeypatch.setattr(mac.os, "geteuid", lambda: 0)
    monkeypatch.setattr(mac.sys, "platform", "darwin")
    monkeypatch.setattr(mac.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mac.tempfile, "TemporaryFile", lambda: spool)
    def expired_input(incoming, deadline):
        yield b"partial"
        clock[0] = deadline
        mac.common.remaining(deadline)
    monkeypatch.setattr(mac.common, "input_chunks", expired_input)
    monkeypatch.setattr(mac, "receive_packet", lambda *args: pytest.fail("expired input cannot seal or publish"))
    with pytest.raises(TimeoutError):
        mac.receive("a" * 64, io.BytesIO(), progress=progress)
    assert spool.closed
    assert progress == dict(phase="transfer", received_bytes=len(b"partial"))


@pytest.mark.parametrize("phase,budget", [("transfer", mac.TRANSFER_TIMEOUT), ("processing", mac.PROCESSING_TIMEOUT)])
def test_root_timeout_reports_phase_and_bytes_and_disarms(monkeypatch, capsys, phase, budget):
    handlers, alarms = {}, []
    monkeypatch.setattr(mac.sys, "argv", ["operator_mac_stage.py", "--expected-sha256", "a" * 64])
    monkeypatch.setattr(mac.signal, "signal", lambda number, handler: handlers.update({number: handler}))
    monkeypatch.setattr(mac.signal, "alarm", alarms.append)
    def expire(expected, *, progress, arm_deadline):
        progress.update(phase=phase, received_bytes=12345)
        arm_deadline(budget)
        handlers[mac.signal.SIGALRM](mac.signal.SIGALRM, None)
    monkeypatch.setattr(mac, "receive", expire)
    with pytest.raises(SystemExit) as error:
        mac.main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert not output.out
    detail = json.loads(output.err)
    assert detail["status"] == "failed" and detail["phase"] == phase and detail["received_bytes"] == 12345
    assert detail["error"] == "TimeoutError: fixed Mac root staging " + phase + " deadline exceeded"
    assert detail["elapsed_seconds"] >= 0
    assert alarms == [budget, 0]


def test_mac_receive_refuses_changed_runner_copy_before_publishing(admitted, monkeypatch):
    _, request, files, constants = admitted
    original = mac.common.write
    mirror = constants["SROOT"] / ("operator-" + request["job_id"]) / "manifest.json"
    def write(path, raw, *args, **kwargs):
        return original(path, raw + b" " if path == mirror else raw, *args, **kwargs)
    monkeypatch.setattr(mac.common, "write", write)
    raw, expected = packet(request, files)
    with pytest.raises(ValueError, match="runner manifest copy"):
        mac.receive(expected, io.BytesIO(raw))
    assert not (constants["IROOT"] / (request["job_id"] + ".json")).exists()
    assert not (constants["MANIFESTS"] / (request["job_id"] + ".json")).exists()


@pytest.mark.parametrize("mutation", ["machine", "lease", "stop", "resources", "protocol", "source-tree", "template-policy", "member", "busy"])
def test_mac_root_admission_rejects_without_publishing(admitted, monkeypatch, mutation):
    fake, request, files, constants = admitted
    if mutation == "machine":
        monkeypatch.setattr(mac, "machine_identity", lambda: dict(MACHINE, hardware_model="other"))
    elif mutation == "stop":
        (fake / "STOP").write_bytes(b"")
    elif mutation == "busy":
        monkeypatch.setattr(mac, "worker_idle", lambda broker: mac.common.fail("worker busy"))
    elif mutation == "lease":
        broker = load(constants["BROKER_CONFIG"])
        broker["lock_dev_inode"] = [0, 0]
        constants["BROKER_CONFIG"].write_text(json.dumps(broker))
        request["broker_config_sha256"] = prepare.digest(constants["BROKER_CONFIG"])
    elif mutation in ["resources", "protocol"]:
        config = load(Path(files["config.json"]["local_path"]))
        if mutation == "resources":
            config["host"]["resources"]["memory_budget_bytes"] += 1
        else:
            config["performance_protocol"]["samples_per_side"] = True
        path = Path(files["config.json"]["local_path"])
        path.write_text(json.dumps(config))
        files["config.json"].update(bytes=path.stat().st_size, sha256=prepare.digest(path))
        request["files"]["config.json"].update(bytes=path.stat().st_size, sha256=prepare.digest(path))
    elif mutation == "source-tree":
        request["sources"]["base"]["tree"] = "0" * 40
    elif mutation == "template-policy":
        policy = load(constants["POLICY"])
        policy["side_settings_template_sha256"]["base"]["muse"] = "0" * 64
        constants["POLICY"].write_text(json.dumps(policy))
        request["policy_sha256"] = prepare.digest(constants["POLICY"])
    else:
        request["files"]["config.json"]["sha256"] = "0" * 64
    raw, expected = packet(request, files)
    with pytest.raises((ValueError, KeyError)):
        mac.receive(expected, io.BytesIO(raw))
    assert not (constants["IROOT"] / (request["job_id"] + ".json")).exists()
    assert not (constants["MANIFESTS"] / (request["job_id"] + ".json")).exists()


@pytest.mark.parametrize("mutation", [None, "writable", "changed-stat", "outside-root", "hardlink", "empty"])
def test_root_readonly_models_use_existing_stat_without_payload_hash(tmp_path, monkeypatch, mutation):
    root = tmp_path / "root"
    model = root / "models/muse"
    model.mkdir(parents=True)
    path = model / "model.gguf"
    path.write_bytes(b"no payload hashing in this path")
    path.chmod(0o444)
    st = path.stat()
    row = dict(host_path=str(path), stat=dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                                           mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns))
    data = dict(model_identity=dict(shards=[row]))
    config = dict(sides=dict(candidate=dict(models=dict(muse=data))))
    policy = dict(models=dict(muse=dict(mount=str(model))))
    monkeypatch.setattr(mac, "BASE", root)
    monkeypatch.setattr(mac.common, "protected", lambda value, **kw: Path(value))
    monkeypatch.setattr(mac.common, "sha", lambda _: pytest.fail("model payload hash is not allowed"))
    if mutation == "writable":
        path.chmod(0o644)
    elif mutation == "changed-stat":
        row["stat"]["bytes"] += 1
    elif mutation == "outside-root":
        policy["models"]["muse"]["mount"] = str(root / "other")
    elif mutation == "hardlink":
        import os
        os.link(path, model / "alias.gguf")
    elif mutation == "empty":
        data["model_identity"]["shards"] = []
    if mutation:
        with pytest.raises(ValueError):
            mac.models_readonly(config, policy)
    else:
        assert mac.models_readonly(config, policy) == 1


@pytest.mark.parametrize("processes", [b"0 11\n501 21\n62001 31\n", b"62002 41\n", b"truncated\n"])
def test_worker_busy_or_unobservable_is_not_cleaned(monkeypatch, processes):
    calls = []
    def inventory(argv, timeout):
        calls.append(argv)
        return processes
    monkeypatch.setattr(mac.subprocess, "check_output", inventory)
    broker = dict(worker_uid=62002, runner_uid=62001)
    if b"62002" in processes or b"truncated" in processes:
        with pytest.raises(ValueError):
            mac.worker_idle(broker)
    else:
        mac.worker_idle(broker)
    assert calls == [["/bin/ps", "-axo", "uid=,pid="]]


def test_settings_cannot_accept_another_current_job_or_side():
    raw = b'expert.source = ["/Library/InferSSD-CI/jobs/{{job_id}}/frozen/base/assets/muse.tsv"]\n'
    payload = mac.settings_payload(raw, "3" * 32, "base")
    assert mac.settings_template(payload, "3" * 32, "base") == raw
    for job, side in [("4" * 32, "base"), ("3" * 32, "candidate")]:
        with pytest.raises(ValueError):
            mac.settings_template(payload, job, side)


def test_mac_control_only_keeps_full_policy_but_stages_candidate_only(tmp_path, git_source):
    from test_ci_operator_prepare import changed_identity
    args, api = mac_prepared_fixture(tmp_path, git_source)
    changed_identity(args, "bench/ci/operator_note.py")
    envelope = prepare.prepare(args, api)
    folder = args.output / "local-mbp"
    manifest = load(folder / "manifest.json")
    policy = load(folder / "policy.review.json")
    request = load(folder / "root-input.json")
    assert set(manifest["sources"]) == set(request["sources"]) == {"candidate"}
    assert "base-source.tar.gz" not in request["files"]
    assert not any("-base.ssd" in name for name in request["files"])
    assert set(policy["side_settings_template_sha256"]) == {"base", "candidate"}
    assert envelope["validation_mode"] == prepare.MODE


def test_recovery_precedes_lease_and_sealing_holds_original_inode(admitted, monkeypatch):
    import fcntl
    import os
    fake, request, files, _ = admitted
    order = []
    def ensure(broker, expected):
        with (fake / "lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        order.append("recovered")
        return dict(status="ready", config_sha256=expected)
    original = mac.common.extract_source
    def extract(*args):
        fd = os.open(fake / "lock", os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        order.append("sealed")
        return original(*args)
    monkeypatch.setattr(mac, "ensure_broker", ensure)
    monkeypatch.setattr(mac.common, "extract_source", extract)
    raw, expected = packet(request, files)
    mac.receive(expected, io.BytesIO(raw))
    assert order == ["recovered", "sealed", "sealed"]


@pytest.mark.parametrize("mutation", ["machine", "lock", "hard-cgroup", "recovery", "mirror-path", "mirror-sha", "mirror-private-path", "mirror-private-sha", "mirror-writable", "mirror-missing"])
def test_mac_sender_rejects_wrong_physical_or_recovery_receipt(tmp_path, git_source, mutation):
    from test_ci_operator_prepare import successful_receipt
    args, api = mac_prepared_fixture(tmp_path, git_source)
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    request = load(args.output / "local-mbp/root-input.json")
    proof = successful_receipt(envelope, inputs, "local-mbp")
    proof.update(sources={side:dict(value,root_owned_readonly=True) for side,value in request['sources'].items()},
        machine_identity=MACHINE, original_lock=[1,2], resource_enforcement='sampled-UMA-guard-not-hard-cgroup',
        broker_recovery=dict(status='ready',config_sha256=request['broker_config_sha256']))
    job = proof["job_id"]
    proof["runner_manifest"] = dict(path="/Library/InferSSD-CI/inputs/operator-" + job + "/manifest.json",
        sha256=proof["manifest_sha256"], broker_path="/Library/InferSSD-CI/private/job-manifests/" + job + ".json",
        broker_sha256=proof["manifest_sha256"], root_owned_readonly=True)
    prepare.validate_root_receipt(envelope, inputs, "local-mbp", proof)
    if mutation == "machine":proof['machine_identity'] = dict(MACHINE,hardware_model='other')
    elif mutation == "lock":proof['original_lock'] = [1,3]
    elif mutation == "hard-cgroup":proof['resource_enforcement'] = 'hard-cgroup'
    elif mutation == "recovery":proof['broker_recovery']['config_sha256'] = '0'*64
    elif mutation == "mirror-path":proof['runner_manifest']['path'] = '/tmp/manifest.json'
    elif mutation == "mirror-sha":proof['runner_manifest']['sha256'] = '0'*64
    elif mutation == "mirror-private-path":proof['runner_manifest']['broker_path'] = '/tmp/private.json'
    elif mutation == "mirror-private-sha":proof['runner_manifest']['broker_sha256'] = '0'*64
    elif mutation == "mirror-writable":proof['runner_manifest']['root_owned_readonly'] = False
    else:proof.pop('runner_manifest')
    with pytest.raises(ValueError,match='actual Mac'):
        prepare.validate_root_receipt(envelope, inputs, "local-mbp", proof)
