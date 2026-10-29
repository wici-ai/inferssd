"""Local operator preparation contracts; no GitHub/device/model requests."""

import json
import io
import copy
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
    (root / "chat.jinja").write_text("{{ messages }}\n")
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


@pytest.mark.parametrize("phase,kind", [
    ("audit-open", "unsafe-path"), ("audit-append", "storage-exhausted"),
    ("diagnostic-write", "io-failed"), ("client-read", "reap-timeout"),
])
def test_gh_fixed_local_failure_metadata_is_visible_without_auth_output(monkeypatch, phase, kind):
    safe = dict(schema_version=1, phase=phase, kind=kind, errno=28)
    stderr = b"gh-ci-read-retry-v3: local-failure=" + json.dumps(safe).encode() + b"\n"
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 1, b"not retained", stderr))
    with pytest.raises(RuntimeError, match=phase + "/" + kind) as caught:
        prepare.Gh().call("user")
    assert "exit 1" in str(caught.value) and "not retained" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    {"schema_version": 1, "phase": "token-sensitive", "kind": "io-failed", "errno": 28},
    {"schema_version": 1, "phase": [], "kind": "io-failed", "errno": 28},
    {"schema_version": 1, "phase": "audit-open", "kind": "io-failed", "errno": "token-sensitive"},
    {"schema_version": 1, "phase": "audit-open", "kind": "io-failed", "errno": 28, "secret": "token-sensitive"},
    [], None, "token-sensitive",
])
def test_gh_malformed_or_unrecognized_failure_never_copies_diagnostics(monkeypatch, payload):
    stderr = b"gh-ci-read-retry-v3: local-failure=" + json.dumps(payload).encode()
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 1, b"", stderr))
    with pytest.raises(RuntimeError) as caught:
        prepare.Gh().call("user")
    assert "local " not in str(caught.value) and "token-sensitive" not in str(caught.value)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}'])
def test_api_json_rejects_ambiguous_evidence(raw):
    with pytest.raises(ValueError):
        prepare.parsed(raw)


def prepared_fixture(tmp_path, source, *, devices=None, mode=prepare.LEGACY_MODE):
    from test_ci_selection import template
    root, identity = source
    impact = prepare.plan(["old.cpp", "new.cpp"], POLICY, **identity)
    profiles = {}
    old_job = "f" * 32
    for index, device in enumerate(prepare.LINUX_DEVICES):
        config = template(impact, device)
        config.update(model_identity={m: dict(shards=[dict(host_path="/admitted/" + m + "/model.gguf", runtime_path="/models/" + m + "/model.gguf",
                      device=1, inode=2, bytes=3, mtime_ns=4, ctime_ns=5, historical_sha256="8" * 64)])
                      for m in ("deepseek", "muse")}, broker_config_sha256="9" * 64,
                      host=dict(resources=dict(memory_max=64 << 30), os_page_cache_treatment="same reviewed cache treatment; not a cold-cache claim"))
        assets = {}
        for side_name, side in config["sides"].items():
            for model, data in side["models"].items():
                data["benchmarks"] = {}
                data["template"] = dict(host_path="/old/source/chat.jinja", runtime_path="/source/" + side_name + "/chat.jinja", sha256=prepare.digest(root / "chat.jinja"))
                data["settings"] = dict(host_path="/opt/inferssd-ci-integrations/" + old_job + "-" + model + "-candidate-settings.ssd",
                                        runtime_path="/models/settings-" + model + "-candidate.ssd", sha256="0" * 64)
        for model in ("deepseek", "muse"):
            name = device + "-" + model + ".ssd"
            (tmp_path / name).write_text("[ssd]\n")
            config["sides"]["candidate"]["models"][model]["settings"]["sha256"] = prepare.digest(tmp_path / name)
            assets["settings-" + model + ".ssd"] = name
        if mode == prepare.MODE:
            assets = {}
            for side_name, side in config["sides"].items():
                for model, data in side["models"].items():
                    name = device + "-" + model + "-" + side_name + ".ssd"
                    (tmp_path / name).write_text('inferssd-settings 1\nexpert.source = ["/build/' + side_name + '/assets/' + model + '-alias.tsv"]\nvram.budget = 0\n')
                    data["settings"].update(host_path="/opt/inferssd-ci-integrations/" + old_job + "-" + model + "-" + side_name + "-settings.ssd",
                        runtime_path="/models/settings-" + model + "-" + side_name + ".ssd", sha256=prepare.digest(tmp_path / name))
                    assets["settings-" + model + "-" + side_name + ".ssd"] = name
                    data["smoke"].update(revision=identity[side_name], argv=["/build/" + side_name + "/bin/llama-server", "--ssd-settings", data["settings"]["runtime_path"]],
                        environment={}, cwd="/scratch", port=18080, timeout_s=120, request_timeout_s=60,
                        os_page_cache_treatment=config["host"]["os_page_cache_treatment"])
                    data["benchmarks"] = {"512": dict(copy.deepcopy(data["smoke"]), preflight_argv=copy.deepcopy(data["smoke"]["argv"]))}
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
        policy=Path(__file__).parents[1] / "ci/policy.json", gh="never-used", execution_devices=list(devices or prepare.LINUX_DEVICES), validation_mode=mode, **identity)
    return args, API(root)


def test_prepare_keeps_full_impact_but_seals_only_linux_functional_work(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    envelope = prepare.prepare(args, api)
    assert envelope["policy_sha256"] == prepare.fingerprint(POLICY)
    assert envelope["prepared_by"]["id"] == 42
    assert len(envelope["selection"]["cells"]) == 6
    assert all(cell["required"] and cell["performance"]["required"] for cell in envelope["selection"]["cells"])
    assert envelope["execution_devices"] == list(prepare.LINUX_DEVICES)
    for device in prepare.LINUX_DEVICES:
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
    assert called == list(prepare.LINUX_DEVICES)
    with pytest.raises(FileExistsError):
        prepare.stage(args, sender)
    assert called == list(prepare.LINUX_DEVICES)


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
    assert called == [prepare.LINUX_DEVICES[0]]
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


@pytest.mark.parametrize("early_exit", [False, True])
def test_sender_retains_root_failure_even_when_upload_pipe_breaks(tmp_path, monkeypatch, early_exit):
    import sys
    request = tmp_path / "root-input.json"
    request.write_text("{}")
    payload = tmp_path / "source.tar.gz"
    payload.write_bytes(b"x" * (1 << 20))
    script = ("import sys; " + ("" if early_exit else "sys.stdin.buffer.read(); ")
              + "sys.stderr.write('root sealing deadline exceeded\\n'); sys.exit(7)")
    monkeypatch.setattr(prepare, "transport_profile", lambda *a, **kw: ([sys.executable, "-c", script], []))
    data = dict(root_input_path=str(request), root_input_sha256=prepare.digest(request),
                ssh=dict(sudo_password_required=False), files={"source.tar.gz": dict(
                    local_path=str(payload), bytes=payload.stat().st_size, sha256=prepare.digest(payload))})
    with pytest.raises(RuntimeError, match="exit 7.*root sealing deadline exceeded"):
        prepare.send_stage("local-mbp", data)
    saved = tmp_path / "staging-stderr.log"
    assert saved.read_text() == "root sealing deadline exceeded\n"
    assert saved.stat().st_mode & 0o777 == 0o600


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


def test_default_mode_is_performance_and_two_sources_are_exact(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, mode=prepare.MODE)
    del args.validation_mode  # The public default, including the watcher API.
    envelope = prepare.prepare(args, api)
    assert envelope["validation_mode"] == prepare.MODE
    for device in prepare.LINUX_DEVICES:
        config = load(args.output / device / "config.json")
        manifest = load(args.output / device / "manifest.json")
        request = load(args.output / device / "root-input.json")
        assert manifest["source_revisions"] == {side: getattr(args, side) for side in ("base", "candidate")}
        assert set(manifest["sources"]) == set(manifest["builds"]) == {"base", "candidate"}
        for side in ("base", "candidate"):
            assert request["sources"][side]["revision"] == getattr(args, side)
            assert manifest["sources"][side].endswith("/" + side)
            assert all(set(data["benchmarks"]) == {"512"} for data in config["sides"][side]["models"].values())
        # One base run and three PR runs per model, each with its own stage.
        assert {name for name in manifest["stages"] if name.startswith(("base-muse-r", "candidate-muse-r"))} == {
            "base-muse-r1-benchmark-512", *("candidate-muse-r" + run + "-benchmark-512" for run in "123")}
        assert config["performance_protocol"] == prepare.performance_protocol()
        assert {"base-source-map.json", "base-source.tar.gz", "source-map.json", "candidate-source.tar.gz"} <= set(request["files"])
    assert load(args.output / "base-source-map.json")["revision"] == args.base
    assert load(args.output / "source-map.json")["revision"] == args.candidate
    prepare.verified(args.output)


def test_baseline_job_builds_one_commit_once_on_every_device_and_model_with_an_empty_diff(tmp_path, source):
    from ci import operator_stage
    root, identity = source
    head = identity["head"]
    args, api = prepared_fixture(tmp_path, (root, dict(base=head, head=head, candidate=head)), mode=prepare.MODE)
    args.baseline = True
    envelope = prepare.prepare(args, api)
    plan = envelope["selection"]
    assert plan["purpose"] == "baseline-history" and plan["changed_paths"] == []
    assert {plan[key] for key in ("base", "head", "candidate")} == {head}
    assert all(cell["required"] and cell["performance"]["tiers"] == ["512"] for cell in plan["cells"])
    assert envelope["execution_devices"] == list(prepare.LINUX_DEVICES)
    for device in prepare.LINUX_DEVICES:
        config = load(args.output / device / "config.json")
        manifest = load(args.output / device / "manifest.json")
        request = load(args.output / device / "root-input.json")
        assert config["performance_protocol"] == prepare.performance_protocol("baseline")
        assert manifest["source_revisions"] == {"candidate": head} and set(manifest["builds"]) == {"candidate"}
        assert {side: value["revision"] for side, value in request["sources"].items()} == {"candidate": head}
        assert {name for name in manifest["stages"] if "-benchmark-" in name} == {
            "candidate-" + model + "-r" + str(index) + "-benchmark-512"
            for model in ("deepseek", "muse") for index in range(1, 5)}
        assert not any(name.startswith("base-") for name in manifest["stages"])
        # The installed root policy pins the gate protocol and admits the baseline form of one commit.
        assert load(args.output / device / "policy.review.json")["performance_protocol"] == prepare.performance_protocol()
        assert operator_stage.protocol_admitted(config)
    assert not (args.output / "base-source-map.json").exists()
    assert load(args.output / "source-map.json")["revision"] == head
    prepare.verified(args.output)


def history_store(tmp_path, records):
    from ci import history
    root = tmp_path / history.DIRECTORY
    for record in records:
        history.write(root, record)
    return root


def test_device_history_is_sealed_into_every_config_and_named_by_the_envelope(tmp_path, source):
    from test_ci_performance_series import record
    root, identity = source
    stored = [record(identity["base"], device="cloud-5090"), record("e" * 40, device="cloud-5090", model="deepseek"),
              record(identity["base"], device="local-mbp")]
    history_store(tmp_path, stored)
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"], mode=prepare.MODE)
    envelope = prepare.prepare(args, api)
    sealed = load(args.output / "cloud-5090/config.json")["baseline_history"]
    # Only the executing devices' records, in the store's verified order.
    assert sorted((item["model"], item["commit"]) for item in sealed) == [("deepseek", "e" * 40), ("muse", identity["base"])]
    assert envelope["baseline_history_sha256"] == prepare.fingerprint(sealed)
    # The root stager and its installed policy never see or pin the history.
    assert "baseline_history" not in json.dumps(load(args.output / "cloud-5090/policy.review.json"))
    (tmp_path / "empty").mkdir()
    empty, api = prepared_fixture(tmp_path / "empty", source, devices=["dgx-spark"], mode=prepare.MODE)
    assert prepare.prepare(empty, api)["baseline_history_sha256"] == prepare.fingerprint([])


def test_tampered_history_file_refuses_preparation(tmp_path, source):
    from test_ci_performance_series import record
    root, identity = source
    store = history_store(tmp_path, [record(identity["base"])])
    path = next(store.rglob("*.json"))
    value = json.loads(path.read_text())
    value["runs"][0]["metrics"]["decode_tps"] = 99.0
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    args, api = prepared_fixture(tmp_path, source, mode=prepare.MODE)
    with pytest.raises(ValueError, match="logs differ"):
        prepare.prepare(args, api)
    assert not args.output.exists()


def cache_fixture(tmp_path, source, flag=True, device="cloud-5090"):
    args, api = prepared_fixture(tmp_path, source, devices=[device], mode=prepare.MODE)
    inputs = load(args.inputs)
    inputs["devices"][device]["base_build_cache"] = flag
    template_path = tmp_path / inputs["devices"][device]["template"]
    config = load(template_path)
    for side in config["sides"].values():
        side["build"]["toolchain"] = {"CMAKE_CXX_COMPILER": "/usr/bin/g++-12"}
    template_path.write_text(json.dumps(config))
    args.inputs.write_text(json.dumps(inputs))
    return args, api


def test_base_build_cache_stores_a_baseline_build_and_offers_it_to_a_pr_on_that_base(tmp_path, source):
    from ci.selection_contract import build_cache_key
    from test_ci_performance_series import record
    root, identity = source
    base = identity["base"]
    # Baseline job of the base commit: the broker keeps its candidate build under the exact key.
    args, api = cache_fixture(tmp_path, (root, dict(base=base, head=base, candidate=base)))
    args.baseline = True
    prepare.prepare(args, api)
    config = load(args.output / "cloud-5090/config.json")
    manifest = load(args.output / "cloud-5090/manifest.json")
    key = build_cache_key(config, "candidate")
    assert config["cache_build_as"] == key and manifest["cache_as"] == {"candidate": key}
    assert "prebuilt_offer" not in manifest and "base_build" not in config
    # A PR on that base gets the offer named by the base commit's record.
    built = record(base)
    built["build"] = dict(built["build"], cache_key=key, freeze_manifest_sha256="7" * 64)
    (tmp_path / "pr").mkdir()
    args, api = cache_fixture(tmp_path / "pr", source)
    history_store(tmp_path / "pr", [built])
    prepare.prepare(args, api)
    config = load(args.output / "cloud-5090/config.json")
    manifest = load(args.output / "cloud-5090/manifest.json")
    assert config["base_build"] == dict(key=key, source_job=built["job_id"], freeze_manifest_sha256="7" * 64)
    assert manifest["prebuilt_offer"] == {"base": dict(key=key, manifest_sha256="7" * 64)}
    assert "cache_as" not in manifest


@pytest.mark.parametrize("case", ["no-record", "other-key", "flag-off"])
def test_no_cache_offer_without_a_matching_base_build(tmp_path, source, case):
    from test_ci_performance_series import record
    root, identity = source
    args, api = cache_fixture(tmp_path, source, flag=case != "flag-off")
    if case != "no-record":
        history_store(tmp_path, [record(identity["base"])])  # cache key "3" * 64 is not this build's key
    prepare.prepare(args, api)
    config = load(args.output / "cloud-5090/config.json")
    assert "base_build" not in config and "prebuilt_offer" not in load(args.output / "cloud-5090/manifest.json")


@pytest.mark.parametrize("damage", ["not-bool", "mac"])
def test_cache_flag_is_an_explicit_boolean_for_a_linux_device(tmp_path, source, damage):
    args, api = cache_fixture(tmp_path, source, flag="yes" if damage == "not-bool" else True)
    if damage == "mac":
        inputs = load(args.inputs)
        inputs["devices"]["local-mbp"] = dict(inputs["devices"]["cloud-5090"])
        args.inputs.write_text(json.dumps(inputs))
        args.execution_devices = ["local-mbp"]
    with pytest.raises(ValueError, match="base_build_cache must be a boolean"):
        prepare.prepare(args, api)


@pytest.mark.parametrize("damage", ["different-revisions", "legacy-mode", "not-bool"])
def test_baseline_job_is_explicit_and_only_for_one_commit(tmp_path, source, damage):
    root, identity = source
    head = identity["head"]
    same = dict(base=head, head=head, candidate=head)
    args, api = prepared_fixture(tmp_path, (root, identity if damage == "different-revisions" else same),
                                 mode=prepare.LEGACY_MODE if damage == "legacy-mode" else prepare.MODE)
    args.baseline = "yes" if damage == "not-bool" else True
    with pytest.raises(ValueError):
        prepare.prepare(args, api)
    assert not args.output.exists()


@pytest.mark.parametrize("mutation", ["missing-base", "different-budget", "model-identity", "payload-hash"])
def test_performance_preparation_refuses_unpaired_inputs(tmp_path, source, mutation):
    args, api = prepared_fixture(tmp_path, source, mode=prepare.MODE)
    inputs = load(args.inputs)
    row = inputs["devices"]["cloud-5090"]
    template_path = tmp_path / row["template"]
    config = load(template_path)
    if mutation == "missing-base": del row["rootstage_files"]["settings-muse-base.ssd"]
    elif mutation == "different-budget":
        name = row["rootstage_files"]["settings-muse-base.ssd"]
        path = tmp_path / name
        path.write_text(path.read_text().replace("vram.budget = 0", "vram.budget = 1"))
        config["sides"]["base"]["models"]["muse"]["settings"]["sha256"] = prepare.digest(path)
    elif mutation == "model-identity": config["sides"]["base"]["models"]["muse"]["model_identity"] = {"shards": []}
    else: config["sides"]["base"]["models"]["muse"]["settings"]["sha256"] = "0" * 64
    template_path.write_text(json.dumps(config))
    args.inputs.write_text(json.dumps(inputs))
    with pytest.raises(ValueError): prepare.prepare(args, api)
    assert not args.output.exists()


def test_performance_receipt_cannot_omit_or_mix_a_source(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"], mode=prepare.MODE)
    envelope = prepare.prepare(args, api)
    inputs = load(args.output / "rootstage-inputs.json")
    device = "cloud-5090"
    receipt = successful_receipt(envelope, inputs, device)
    request = load(args.output / device / "root-input.json")
    with pytest.raises(ValueError, match="source sides"):
        prepare.validate_root_receipt(envelope, inputs, device, receipt)
    receipt["sources"] = {side: dict(value, root_owned_readonly=True) for side, value in request["sources"].items()}
    prepare.validate_root_receipt(envelope, inputs, device, receipt)
    receipt["sources"]["base"]["revision"] = args.candidate
    with pytest.raises(ValueError, match="source sides"):
        prepare.validate_root_receipt(envelope, inputs, device, receipt)


def changed_identity(args, path):
    command(args.source, "checkout", "-q", args.base)
    file = args.source / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("changed\n")
    command(args.source, "add", ".")
    command(args.source, "commit", "-qm", "targeted change")
    args.head = args.candidate = command(args.source, "rev-parse", "HEAD")


def test_control_only_new_mode_does_not_stage_base_or_timed_arms(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"], mode=prepare.MODE)
    changed_identity(args, "bench/ci/example.py")
    prepare.prepare(args, api)
    manifest = load(args.output / "cloud-5090/manifest.json")
    request = load(args.output / "cloud-5090/root-input.json")
    assert set(manifest["sources"]) == set(request["sources"]) == {"candidate"}
    assert not any("benchmark" in name or "preflight" in name for name in manifest["stages"])
    assert not (args.output / "base-source.tar.gz").exists()


def test_subset_job_still_binds_full_fixed_root_policy(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, mode=prepare.MODE)
    changed_identity(args, "engine/ggml/src/ggml-cuda/ssd/mxfp4-compute-adapter.cpp")
    prepare.prepare(args, api)
    for device in prepare.LINUX_DEVICES:
        config = load(args.output / device / "config.json")
        policy = load(args.output / device / "policy.review.json")
        assert set(config["sides"]["candidate"]["models"]) == {"deepseek"}
        assert set(policy["models"]) == {"deepseek", "muse"}
        assert set(policy["side_settings_sha256"]["base"]) == {"deepseek", "muse"}
        assert not any("muse" in name for name in load(args.output / device / "root-input.json")["files"])


def test_no_enabled_affected_linux_device_refuses_before_output(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source, mode=prepare.MODE)
    changed_identity(args, "engine/ggml/src/ggml-metal/example.m")
    with pytest.raises(ValueError, match="no enabled device"):
        prepare.prepare(args, api)
    assert not args.output.exists()


@pytest.mark.parametrize("damage", [None, "collision", "missing-argv", "other-flags"])
def test_performance_preflight_and_timed_private_writer_paths(tmp_path, source, damage):
    args, api = prepared_fixture(tmp_path, source, devices=["cloud-5090"], mode=prepare.MODE)
    path = tmp_path / "cloud-5090-template.json"
    config = load(path)
    for side, spec in config["sides"].items():
        for model, data in spec["models"].items():
            before = "/scratch/" + side + "/" + model + "-preflight.jsonl"
            after = "/scratch/" + side + "/" + model + "-benchmark-512.jsonl"
            data["smoke"]["argv"] += ["--ssd-telemetry", before]
            timed = data["benchmarks"]["512"]
            timed["preflight_argv"] = copy.deepcopy(data["smoke"]["argv"])
            timed["argv"] = [*data["smoke"]["argv"][:-1], after]
            if damage == "collision": timed["argv"][-1] = before
            elif damage == "missing-argv": del timed["preflight_argv"]
            elif damage == "other-flags": timed["preflight_argv"] += ["--different-budget"]
    path.write_text(json.dumps(config))
    if damage:
        with pytest.raises(ValueError): prepare.prepare(args, api)
        assert not args.output.exists()
    else:
        prepare.prepare(args, api)
        actual = load(args.output / "cloud-5090/config.json")
        for side, spec in actual["sides"].items():
            for model, data in spec["models"].items():
                timed = data["benchmarks"]["512"]
                assert timed["preflight_argv"] == data["smoke"]["argv"]
                assert timed["argv"][-1] != timed["preflight_argv"][-1]


def test_bad_platform_profile_fails_during_preparation(tmp_path, source):
    args, api = prepared_fixture(tmp_path, source)
    path = tmp_path / "cloud-5090-template.json"
    config = json.loads(path.read_text())
    config["sides"]["candidate"]["build"]["cmake_flags"] = ["-DLLAMA_USE_SYSTEM_GGML=ON", "-DLLAMA_SSD_STORE=ON"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="LLAMA_SSD_STORE requires the bundled ggml"):
        prepare.prepare(args, api)
    assert not (args.output / "prepared.json").exists()
