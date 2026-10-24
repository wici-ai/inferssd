"""Local operator preparation contracts; no GitHub/device/model requests."""

import json
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import operator_prepare as prepare
from ci.contract import load

POLICY = load(Path(__file__).parents[1] / "ci/policy.json")


def command(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    command(root, "init", "-q")
    command(root, "config", "user.name", "Synthetic operator")
    command(root, "config", "user.email", "operator@example.invalid")
    (root / "old.cpp").write_text("old\n")
    command(root, "add", ".")
    command(root, "commit", "-qm", "base")
    base = command(root, "rev-parse", "HEAD")
    (root / "old.cpp").rename(root / "new.cpp")
    command(root, "add", "-A")
    command(root, "commit", "-qm", "head")
    head = command(root, "rev-parse", "HEAD")
    return root, dict(base=base, head=head, candidate=head)


class API:
    def __init__(self, source=None):
        self.source = source
        self.calls = []
        self.permission = "admin"
        self.private = True

    def call(self, path, data=None):
        self.calls.append((path, data))
        if path == "user":
            return dict(login="Operator", id=42)
        if path.endswith("/permission"):
            return dict(permission=self.permission)
        if path == "repos/wici-ai/inferssd-device-ci":
            return dict(private=self.private, default_branch="main")
        if path == "repos/wici-ai/inferssd-device-ci/branches/main":
            return dict(commit=dict(sha="d" * 40))
        assert path.startswith("repos/wici-ai/inferssd/git/commits/")
        sha = path.rsplit("/", 1)[1]
        return dict(sha=sha, tree=dict(sha=command(self.source, "rev-parse", sha + "^{tree}")),
                    parents=[dict(sha=p) for p in command(self.source, "show", "-s", "--format=%P", sha).split()])


def test_complete_diff_keeps_both_sides_of_rename(source):
    root, identity = source
    frozen, paths, diff = prepare.source_identity(root, identity, API(root), POLICY["repository"])
    assert paths == ["new.cpp", "old.cpp"]
    assert b"deleted file" in diff and b"new file" in diff
    assert frozen["commits"]["head"]["revision"] == identity["head"]
    assert command(root, "status", "--porcelain") == ""
    assert not command(root, "for-each-ref", "refs/inferssd-ci")


def test_remote_object_mismatch_rejected(source):
    root, identity = source
    class Wrong(API):
        def call(self, path, data=None):
            value = super().call(path, data)
            value["tree"]["sha"] = "e" * 40
            return value
    with pytest.raises(ValueError, match="authenticated source"):
        prepare.source_identity(root, identity, Wrong(root), POLICY["repository"])


def test_only_exact_merge_or_head_candidate_allowed(source):
    root, identity = source
    tree = command(root, "rev-parse", identity["head"] + "^{tree}")
    candidate = command(root, "commit-tree", tree, "-p", identity["base"], "-p", identity["head"], "-m", "candidate")
    prepare.source_identity(root, dict(identity, candidate=candidate), API(root), POLICY["repository"])
    wrong = command(root, "commit-tree", tree, "-p", identity["head"], "-m", "unrelated candidate")
    with pytest.raises(ValueError, match="exact base/head merge"):
        prepare.source_identity(root, dict(identity, candidate=wrong), API(root), POLICY["repository"])


@pytest.mark.parametrize("permission", ["read", "triage", None])
def test_existing_account_must_write_both_repositories(permission):
    api = API()
    api.permission = permission
    with pytest.raises(ValueError, match="write permission"):
        prepare.access(api, POLICY)


def test_account_and_private_main_are_explicit():
    actor, sha = prepare.access(API(), POLICY)
    assert actor["login"] == "Operator" and sha == "d" * 40
    api = API()
    api.private = False
    with pytest.raises(ValueError, match="private main"):
        prepare.access(api, POLICY)


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a//b", "a\\b", ""])
def test_reviewed_files_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        prepare.input_file(tmp_path, name)


def test_reviewed_file_symlink_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    (tmp_path / "alias.json").symlink_to(real)
    with pytest.raises(ValueError, match="regular reviewed"):
        prepare.input_file(tmp_path, "alias.json")
    (tmp_path / "alias-dir").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        prepare.input_file(tmp_path, "alias-dir/real.json")


def test_job_substitution_is_data_only():
    original = dict(path="/stage/{{job_id}}/input", argv=["$(do-not-execute)", "{{job_id}}"])
    frozen = prepare.substitute(original, "f" * 32)
    assert frozen["argv"] == ["$(do-not-execute)", "f" * 32]
    assert original["path"] == "/stage/{{job_id}}/input"


def test_gh_api_uses_existing_cli_no_token_export(monkeypatch):
    seen = []
    def execute(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b'{"login":"Operator"}', b"")
    monkeypatch.setattr(prepare.subprocess, "run", execute)
    assert prepare.Gh("/existing/ghc").call("user")["login"] == "Operator"
    assert seen[0][0] == ["/existing/ghc", "api", "--hostname", "github.com", "user"]
    assert "env" not in seen[0][1]


def test_gh_errors_never_copy_auth_diagnostics(monkeypatch):
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 1, b"", b"synthetic-sensitive-diagnostic"))
    with pytest.raises(RuntimeError) as caught:
        prepare.Gh().call("user")
    assert "synthetic-sensitive" not in str(caught.value)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_api_json_rejects_ambiguous_evidence(raw):
    with pytest.raises(ValueError):
        prepare.parsed(raw)


def prepared_fixture(tmp_path, source, *, devices=None):
    from test_ci_selection import template
    root, identity = source
    impact = prepare.plan(["old.cpp", "new.cpp"], POLICY, **identity)
    profiles = {}
    old_job = "f" * 32
    for index, device in enumerate(prepare.DEVICES):
        config = template(impact, device)
        config.update(model_identity={m: dict(shards=[dict(host_path="/admitted/" + m + "/model.gguf", runtime_path="/models/" + m + "/model.gguf",
                      device=1, inode=2, bytes=3, mtime_ns=4, ctime_ns=5, historical_sha256="8" * 64)])
                      for m in ("deepseek", "muse")}, broker_config_sha256="9" * 64,
                      host=dict(resources=dict(memory_max=64 << 30)))
        assets = {}
        for side in config["sides"].values():
            for model, data in side["models"].items():
                data["benchmarks"] = {}
                data["template"] = dict(host_path="/old/source/template", runtime_path="/source/template", sha256="e" * 64)
                data["settings"] = dict(host_path="/opt/inferssd-ci-integrations/" + old_job + "-" + model + "-candidate-settings.ssd",
                                        runtime_path="/models/settings-" + model + "-candidate.ssd", sha256="0" * 64)
        for model in ("deepseek", "muse"):
            name = device + "-" + model + ".ssd"
            (tmp_path / name).write_text("[ssd]\n")
            config["sides"]["candidate"]["models"][model]["settings"]["sha256"] = prepare.digest(tmp_path / name)
            assets["settings-" + model + ".ssd"] = name
        mounts = {m: "/admitted/" + m for m in ("deepseek", "muse")}
        mounts.update({"settings-" + m + "-candidate.ssd": config["sides"]["candidate"]["models"][m]["settings"]["host_path"]
                       for m in ("deepseek", "muse")})
        manifest = dict(job_id=old_job, sources=dict(candidate="/old/source"), source_revisions=dict(candidate=identity["candidate"]), models=mounts)
        for name, value in ((device + "-template.json", config), (device + "-manifest.json", manifest)):
            (tmp_path / name).write_text(json.dumps(value))
        profiles[device] = dict(job_id=str(index + 1) * 32, template=device + "-template.json", manifest=device + "-manifest.json",
            control_revision="d" * 40, control_manifest_sha256="e" * 64, rootstage_files=assets, min_free_bytes=1, lock_identity=[1,2], gpu_uuid="GPU-synthetic",
            ssh=dict(argv=["ssh", "-T", "-a", "operator@example.invalid"], sudo_password_required=False,
                     root_argv=["/usr/bin/python3", "-I", "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]))
    (tmp_path / "inputs.json").write_text(json.dumps(dict(schema_version=1, controller_workflow_sha="d" * 40, devices=profiles)))
    args = SimpleNamespace(source=root, inputs=tmp_path / "inputs.json", output=tmp_path / "prepared",
        policy=Path(__file__).parents[1] / "ci/policy.json", gh="never-used", execution_devices=list(devices or prepare.DEVICES), **identity)
    return args, API(root)


def test_prepare_keeps_full_impact_but_seals_only_linux_functional_work(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    envelope = prepare.prepare(args, api)
    assert envelope["policy_sha256"] == prepare.fingerprint(POLICY)
    assert envelope["prepared_by"]["id"] == 42
    assert len(envelope["selection"]["cells"]) == 6
    assert all(cell["required"] and cell["performance"]["required"] for cell in envelope["selection"]["cells"])
    assert envelope["execution_devices"] == list(prepare.DEVICES)
    for device in prepare.DEVICES:
        manifest = load(args.output / device / "manifest.json")
        assert set(manifest["sources"]) == {"candidate"}
        assert not any("benchmark" in stage or "preflight" in stage for stage in manifest["stages"])
        assert {"candidate-smoke-deepseek", "candidate-smoke-muse"} <= set(manifest["stages"])
    frozen, _ = prepare.verified(args.output)
    assert frozen == envelope
    assert all(data is None for _, data in api.calls)


def test_single_device_scope_retains_mac_and_other_linux_requirements(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"])
    envelope = prepare.prepare(args, api)
    assert set(envelope["jobs"]) == {"cloud-5090"}
    assert all(cell["required"] for cell in envelope["selection"]["cells"])
    assert not (args.output / "dgx-spark").exists()


def successful_receipt(envelope, inputs, device):
    job, data = envelope["jobs"][device], inputs["devices"][device]
    return dict(status="passed", root_uid=0, device=device, job_id=job["job_id"], candidate=envelope["candidate"],
        root_input_sha256=data["root_input_sha256"], envelope_sha256=inputs["envelope_sha256"],
        config_sha256=job["config_sha256"], manifest_sha256=data["manifest_sha256"],
        control_revision=job["control_revision"], control_manifest_sha256=job["control_manifest_sha256"],
        broker_config_unchanged=True, existing_coordination_unmodified=True, source=dict(root_owned_readonly=True), policy_sha256=data["policy_sha256"])


def test_staging_requires_exact_root_receipt_and_refuses_reissue(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    args.prepared = args.output
    called = []
    def sender(device, data, password=None):
        called.append(device)
        return successful_receipt(envelope, inputs, device)
    prepare.stage(args, sender)
    assert called == list(prepare.DEVICES)
    with pytest.raises(FileExistsError):
        prepare.stage(args, sender)
    assert called == list(prepare.DEVICES)


def test_first_stage_failure_does_not_dispatch_or_continue(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    args.prepared = args.output
    called = []
    def sender(device, data, password=None):
        called.append(device)
        return dict(successful_receipt(envelope, inputs, device), config_sha256="0" * 64)
    with pytest.raises(ValueError, match="actual root staging"):
        prepare.stage(args, sender)
    assert called == [prepare.DEVICES[0]]
    assert not (args.output / "dispatch-attempt.json").exists()


def test_dispatch_once_only_exact_private_workflow_and_run(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    args.prepared = args.output
    prepare.stage(args, lambda device, data, password=None: successful_receipt(envelope, inputs, device))
    class Dispatch(API):
        def call(self, path, data=None):
            if path.endswith("/actions/workflows/device-performance.yml"):
                return dict(id=123, path=prepare.WORKFLOW, state="active")
            if path.endswith("/dispatches"):
                self.calls.append((path, data))
                assert set(data["inputs"]) == {"prepared_envelope"}
                assert json.loads(data["inputs"]["prepared_envelope"]) == envelope
                return dict(workflow_run_id=456)
            if path.endswith("/actions/runs/456"):
                return dict(id=456, workflow_id=123, path=prepare.WORKFLOW, event="workflow_dispatch", head_branch="main",
                            head_sha="d" * 40, run_attempt=1, actor=dict(id=42))
            return super().call(path, data)
    api = Dispatch()
    assert prepare.dispatch(args, api)["id"] == 456
    with pytest.raises(FileExistsError):
        prepare.dispatch(args, api)
    assert len([row for row in api.calls if row[1] is not None]) == 1


def test_prepared_input_tamper_blocks_stage_before_transport(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    prepare.prepare(args, api)
    (args.output / "cloud-5090/config.json").write_text("{}")
    args.prepared = args.output
    with pytest.raises(ValueError, match="prepared source/config"):
        prepare.stage(args, lambda *a, **kw: pytest.fail("transport must not run"))


def test_ssh_transport_requires_no_pty_no_forwarding_fixed_root_helper():
    value = dict(argv=["ssh", "-T", "-a", "host"], sudo_password_required=False,
                 root_argv=["/usr/bin/python3", "-I", "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"])
    prepare.transport_profile(value)
    for bad in (dict(value, argv=["ssh", "-t", "host"]), dict(value, root_argv=["/bin/sh"]),
                dict(value, argv=["ssh", "-T", "-A", "host"])):
        with pytest.raises(ValueError):
            prepare.transport_profile(bad)


def test_source_export_includes_export_ignored_file_as_git_data(tmp_path, source):
    root, identity = source
    (root / ".gitattributes").write_text("new.cpp export-ignore\n")
    command(root, "add", ".")
    command(root, "commit", "-qm", "attributes")
    output = tmp_path / "export"
    output.mkdir()
    value = prepare.export_source(root, command(root, "rev-parse", "HEAD"), output)
    assert "new.cpp" in {row["path"] for row in value["files"]}
    from ci.operator_stage import extract_source
    proof = extract_source(output / "candidate-source.tar.gz", value, tmp_path / "restored")
    assert proof["root_owned_readonly"] and (tmp_path / "restored/new.cpp").read_text() == "old\n"


@pytest.mark.parametrize("mutation", ["extra-mapping", "model-path", "settings-path", "resource", "broker", "admission"])
def test_local_root_policy_blocks_unreviewed_paths_and_resources(tmp_path, source, mutation):
    from ci import operator_stage
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"])
    prepare.prepare(args, api)
    config = load(args.output / "cloud-5090/config.json")
    manifest = load(args.output / "cloud-5090/manifest.json")
    policy = load(args.output / "cloud-5090/policy.review.json")
    request = load(args.output / "cloud-5090/root-input.json")
    broker = dict(device="cloud-5090", lock_identity=[1,2])
    operator_stage.policy_check(policy, request, config, manifest, broker)
    if mutation == "extra-mapping": manifest["models"]["shadow"] = "/etc/shadow"
    elif mutation == "model-path": config["sides"]["candidate"]["models"]["muse"]["model_identity"]["shards"][0]["host_path"] = "/var/lib/unreviewed-runner/.credentials"
    elif mutation == "settings-path": config["sides"]["candidate"]["models"]["muse"]["settings"]["host_path"] = "/etc/shadow"
    elif mutation == "resource": config["host"]["resources"]["memory_max"] += 1
    elif mutation == "broker": request["broker_config_sha256"] = "0" * 64
    else: config["model_admission_path"] = "/etc/shadow"
    with pytest.raises(ValueError):
        operator_stage.policy_check(policy, request, config, manifest, broker)


def test_nopasswd_profile_is_limited_to_installed_stage_helper():
    root_argv = ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]
    value = dict(argv=["ssh", "-T", "-a", "host"], sudo_password_required=False, sudo_mode="nopasswd", root_argv=root_argv)
    prepare.transport_profile(value)
    with pytest.raises(ValueError):
        prepare.transport_profile(dict(value, root_argv=["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "/tmp/payload.py"]))


def test_sender_uses_plain_tar_after_memory_only_sudo_line(tmp_path, monkeypatch):
    import tarfile
    secret = "synthetic-memory-only-password"
    request = tmp_path / "root-input.json"
    request.write_text("{}")
    payload = tmp_path / "settings.ssd"
    payload.write_text("safe")
    data = dict(root_input_path=str(request), root_input_sha256=prepare.digest(request),
                files={"settings-muse.ssd": dict(local_path=str(payload), bytes=4, sha256=prepare.digest(payload))},
                ssh=dict(argv=["ssh", "-T", "-a", "host"], sudo_password_required=True,
                         root_argv=["/usr/bin/sudo", "-S", "-p", "", "/usr/bin/python3", "-I",
                                    "/opt/inferssd-ci-control/current/bench/ci/operator_stage.py"]))
    captured = {}
    class Pipe(io.BytesIO):
        def close(self): self.flush()
    class Process:
        def __init__(self, argv, **kwargs):
            captured.update(argv=argv, options=kwargs)
            self.pipe = self.stdin = Pipe()
            self.returncode = 0
        def communicate(self, timeout):
            captured["wire"] = self.pipe.getvalue()
            return b'{"device":"cloud-5090"}', None
        def poll(self): return 0
    monkeypatch.setenv("GH_TOKEN", "synthetic-github-token")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-github-token")
    monkeypatch.setattr(prepare.subprocess, "Popen", Process)
    prepare.send_stage("cloud-5090", data, password=secret)
    assert secret not in repr(captured["argv"])
    assert "GH_TOKEN" not in captured["options"]["env"] and "GITHUB_TOKEN" not in captured["options"]["env"]
    line, raw = captured["wire"].split(b"\n", 1)
    assert line.decode() == secret and secret.encode() not in raw
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        assert archive.getnames() == ["root-input.json", "settings-muse.ssd"]


def test_ambiguous_dispatch_remains_claimed_and_is_never_reposted(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"])
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    args.prepared = args.output
    prepare.stage(args, lambda device, data, password=None: successful_receipt(envelope, inputs, device))
    class Ambiguous(API):
        posts = 0
        def call(self, path, data=None):
            if path.endswith("/actions/workflows/device-performance.yml"):
                return dict(id=123, path=prepare.WORKFLOW, state="active")
            if path.endswith("/dispatches"):
                self.posts += 1
                return None
            return super().call(path, data)
    api = Ambiguous()
    with pytest.raises(ValueError, match="exact run ID"):
        prepare.dispatch(args, api)
    with pytest.raises(FileExistsError):
        prepare.dispatch(args, api)
    assert api.posts == 1 and (args.output / "dispatch-attempt.json").is_file()
