"""Typed preparation/ownership contracts; synthetic metadata is not GPU evidence."""

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_engine_runtime import Effects, Owner, policy
from test_mac_alias import fixture as gguf_fixture

from inferssd.cli import main
from inferssd.configuration.derive import SelectionConflict, selection
from inferssd.configuration.settings import parse
from inferssd.engine import runtime
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.typed import LaunchSpec, reject_legacy_environment, resolve_typed
from inferssd.launch.types import LaunchInput
from inferssd.loop.typed_controller import TypedController
from inferssd.source import mac_alias


def setup(tmp_path, platform="linux", automatic=False):
    root = tmp_path / "repo"
    template = root / "engine/models/templates/muse-spark.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("template")
    binary = root / "bin/server"
    binary.parent.mkdir()
    binary.write_text("must never execute")
    binary.chmod(0o755)
    model = root / "model.gguf"
    tensors = [
        (f"blk.0.ffn_{role}_exps.weight", (32, 32, 256), 8, i * 278528)
        for i, role in enumerate(("up", "gate", "down"))
    ]
    gguf_fixture(model, tensors=tensors, payload=3 * 278528)
    settings = root / "input.ssd"
    source = root / "provided.tsv"
    source.write_text("explicit source fixture; resolver does not validate payloads")
    text = "inferssd-settings 1\ncpu.threads=4 # measured CPU count\n"
    if not automatic:
        text += "expert.source=" + json.dumps([str(source)]) + " # provided index\n"
    settings.write_text(text)
    spec = LaunchSpec(str(model), 8080, str(settings), (("cpu.threads", "2"),), str(binary))
    host = {
        "platform_system": "Linux" if platform == "linux" else "Darwin",
        "platform_machine": "arm64",
        "euid": 0,
        "cgroup_v2": True,
        "gpu_available": True,
        "gpu_processes": "",
    }
    inputs = LaunchInput(str(root), (("PATH", "/bin"), ("PRIVATE_TOKEN", "never-persist")))
    out = tmp_path / "run"
    out.mkdir()
    seen = []

    def indexer(command):
        assert command[:4] == ["metadata-python", "-I", "-m", "inferssd.source.mac_alias"]
        seen.append(command)
        assert mac_alias.main(command[4:]) == 0

    adapter = RecordedArtifacts(
        inputs, host, run=indexer, replace=lambda a, b: Path(a).replace(b), python="metadata-python"
    )
    return inputs, spec, host, adapter, out, seen


def resolve(inputs, spec, adapter, out, platform="linux", configured=None):
    return prepare(
        inputs,
        resolver=partial(
            resolve_typed,
            spec=spec,
            platform=platform,
            directory=str(out),
            memory=4 << 30,
            configured=configured,
        ),
        observe=adapter,
        emit=lambda _: None,
    )


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_settings_file_and_cli_provenance_reach_native_argv(tmp_path, platform):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    result = resolve(inputs, spec, adapter, out, platform)
    assert result.code == 0 and not result.assigned_environment
    assert "--ssd-settings" in result.argv and "--ssd-telemetry" in result.argv
    assert result.argv[result.argv.index("--ssd-cpu-threads") + 1] == "2"
    body = (out / "settings.ssd").read_text()
    assert "expert.source" in body and "provided index" in body
    assert "cpu.threads" not in body  # actual CLI wins, registry defaults stay defaults
    native = parse(body, str(out / "settings.ssd"), overrides=spec.overrides, platform=platform)
    assert native.at("cpu.threads").origin == "cli"
    assert native.at("vram.budget").origin == "default"
    assert native.at("expert.source").origin == "file"
    assert not any("GGML_KIMI" in x for x in result.argv)
    assert ("MemoryMax=4294967296" in result.scope) == (platform == "linux")
    assert not (out / "native-events.jsonl").exists()  # only the native writer owns creation
    with pytest.raises(FileExistsError):
        resolve(inputs, spec, adapter, out, platform)


@pytest.mark.parametrize(
    "platform,present,chosen",
    [
        ("linux", ["build", "build-linux-cpu", "build-linux-cuda"], "build-linux-cuda"),
        ("linux", ["build", "build-linux-cpu"], "build-linux-cpu"),
        ("linux", ["build", "build-mac"], "build"),
        ("darwin", ["build", "build-mac", "build-macos-metal"], "build-macos-metal"),
        ("darwin", ["build", "build-mac"], "build-mac"),
        ("darwin", ["build", "build-linux-cpu"], "build"),
    ],
)
def test_repo_adapter_prefers_preset_build_directories(tmp_path, platform, present, chosen):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    for name in present:
        binary = Path(inputs.repo) / name / "bin/llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("must never execute")
        binary.chmod(0o755)
    result = resolve(inputs, replace(spec, binary=None), adapter, out, platform)
    assert result.code == 0
    assert str((Path(inputs.repo) / chosen / "bin/llama-server").resolve()) in result.argv


@pytest.mark.parametrize(
    "platform,first", [("linux", "build-linux-cuda"), ("darwin", "build-macos-metal")]
)
def test_repo_adapter_reports_the_preset_build_when_none_exists(tmp_path, platform, first):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    missing = "missing server executable: .*/" + first + "/bin/llama-server"
    with pytest.raises(ValueError, match=missing):
        resolve(inputs, replace(spec, binary=None), adapter, out, platform)


def test_auto_alias_uses_real_strict_header_indexer(tmp_path):
    inputs, spec, _, adapter, out, calls = setup(tmp_path, automatic=True)
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0 and len(calls) == 1
    value = parse((out / "settings.ssd").read_text(), str(out / "settings.ssd"))
    alias = Path(value.at("expert.source").value[0])
    rows = alias.read_text().splitlines()
    assert len(rows) == 1 + 3 * 256 and "source_path" in rows[0]
    assert str(Path(spec.model)) in rows[-1]
    assert "source=generated" in value.at("expert.source").receipt


@pytest.mark.parametrize(
    "name", ["GGML_KIMI_CPU_SINK_THREADS", "TASK3_PREFETCH", "SETTINGS", "TRACE_ENV"]
)
def test_retired_inputs_fail_by_name_without_value(name):
    with pytest.raises(ValueError) as error:
        reject_legacy_environment({name: "PRIVATE-OLD-CONTROL"})
    assert name in str(error.value) and "PRIVATE-OLD-CONTROL" not in str(error.value)


@pytest.mark.parametrize(
    "extra",
    [
        ("-mwrong",),
        ("--model=wrong",),
        ("--port=8",),
        ("--ssd-cpu-threads", "8"),
        ("-np2",),
        ("--threads=9",),
        ("--ubatch-size", "64"),
    ],
)
def test_cannot_override_index_model_or_owned_listener(tmp_path, extra):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit typed"):
        resolve(replace(inputs, extra=extra), spec, adapter, out)
    assert not (out / "settings.ssd").exists()


@pytest.mark.parametrize(
    "fact,value,message",
    [
        ("systemd", False, "systemd is required"),
        ("cgroup_v2", False, "cgroup v2"),
        ("gpu_available", False, "nvidia-smi"),
        ("gpu_processes", "99 foreign", "compute process"),
    ],
)
def test_resource_preparation_refuses_before_materialized_settings(tmp_path, fact, value, message):
    inputs, spec, host, _, out, _ = setup(tmp_path)
    host[fact] = value
    adapter = RecordedArtifacts(
        inputs,
        host,
        run=lambda _: pytest.fail("no index"),
        replace=lambda *_: None,
        python="unused",
    )
    with pytest.raises(ValueError, match=message):
        resolve(inputs, spec, adapter, out)
    assert not (out / "settings.ssd").exists()


def test_runtime_uses_typed_resolution_and_keeps_owned_process_port(tmp_path, monkeypatch):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    effects = Effects()
    owner = Owner(effects)
    captured = {}
    monkeypatch.setattr(runtime, "LiveArtifacts", lambda *a, **kw: adapter)
    monkeypatch.setattr(runtime, "NativeEffects", lambda: effects)

    def owned(result, directory, execution_policy, *, reservation):
        assert execution_policy.platform == "linux"
        assert reservation is effects.reservation
        captured.update(result=result, directory=directory)
        return owner

    # Keep the real Runtime→Engine wiring, replacing only OS ownership effects.
    monkeypatch.setattr(runtime, "OwnedProcess", owned)
    monkeypatch.setattr(effects, "reservation", object(), raising=False)
    value = runtime.Runtime(
        inputs, policy(tmp_path, "linux"), host=object(), emit=lambda _: None, configuration=spec
    )
    engine = value.engine(
        str(out),
        8080,
        4,
        "all",
        "",
        [],
        configured=parse(Path(spec.settings_file).read_text(), spec.settings_file),
    )
    assert engine.ok and captured["result"].argv[1:3] == ("-m", spec.model)
    env = dict(captured["result"].effective_environment)
    assert "INFERSSD_DEBUG" not in env  # controller observes JSONL without debug logging
    assert "GGML_KIMI_CPU_SINK_TRACE" not in env and "LLAMA_SERVER_PORT" not in env
    engine.stop()
    assert effects.events[-1] == ("busy", 8080)


def test_busy_port_refuses_before_claim_or_owner(tmp_path, monkeypatch):
    effects = Effects(busy=True)
    monkeypatch.setattr(runtime, "NativeEffects", lambda: effects)
    value = runtime.Runtime(
        LaunchInput("/unused", ()),
        policy(tmp_path),
        host=object(),
        emit=lambda _: None,
        configuration=LaunchSpec("/unused", 8080),
    )
    with pytest.raises(ValueError, match="already in use"):
        value.engine(str(tmp_path / "run"), 8080, 0, "static", "", [])
    assert effects.events == [("busy", 8080)]


def test_measured_selection_changes_actual_typed_values_without_env_bridge():
    model = SimpleNamespace(
        P={"split": {"pf_k": 3, "tier_slot_bytes": 1024, "tier_base_bytes": 8192}},
        caps={"split": 8},
        ref="split",
        dev={
            "default_threads": 4,
            "coresets": {"g1": {"cpus": [0, 1, 2]}, "g2": {"cpus": [3, 4]}},
            "lanes": {"cpu": {"fixed_us": 12, "per_route_us": 3}, "copy_gbps": 7},
        },
        PF={"memory": {}},
    )
    base = parse()
    chosen = selection(model, 4, "split", base, lambda: pytest.fail("measured count exists"))
    assert chosen.at("cpu.threads").value == 4
    assert chosen.at("cpu.compute_cpus").value == (0, 1, 3)
    assert chosen.at("cpu.io_cpus").value == (2,)
    assert chosen.at("cpu.decode_participants").value == 2
    assert chosen.at("vram.max_slots").value == 4
    assert chosen.at("vram.budget").value == 4096
    assert chosen.at("prefetch.depth").value == 3
    assert "source=planner" in chosen.at("cpu.threads").receipt
    assert base.at("cpu.threads").value == 2


def test_typed_controller_never_calls_machine_env():
    snapshot = parse()
    model = SimpleNamespace(env=lambda *_: pytest.fail("historical env path"))
    controller = object.__new__(TypedController)
    captured = {}
    controller.m, controller.out, controller.port = model, "/run", 8080
    controller.insts, controller.args = [], []
    controller._configuration = lambda *_: snapshot
    controller._typed_engine = lambda *args: (
        captured.update(args=args) or SimpleNamespace(ok=True, start_s=0)
    )
    controller.note = lambda record: captured.update(record=record)
    controller._start(4, "all")
    assert captured["args"][-1] is snapshot and "settings" in captured["record"]
    assert "env" not in captured["record"]


def test_typed_command_prepares_native_file_without_executing_or_recording_secrets(
    tmp_path, monkeypatch
):
    inputs, spec, host, _, out, _ = setup(tmp_path)
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path, "linux"))))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(host))
    receipt = tmp_path / "resolution.json"
    monkeypatch.setattr("os.environ", dict(inputs.environment))
    assert (
        main(
            [
                "launch",
                "--dry-run",
                "--platform",
                "linux",
                "--repo",
                inputs.repo,
                "--model",
                spec.model,
                "--binary",
                spec.binary,
                "--settings",
                spec.settings_file,
                "--set",
                "cpu.threads=2",
                "--runtime-policy",
                str(contract),
                "--host-facts",
                str(facts),
                "--output-dir",
                str(out),
                "--result",
                str(receipt),
            ]
        )
        == 0
    )
    assert (out / "settings.ssd").exists() and (out / ".inferssd-owner").exists()
    assert not (out / "runtime.json").exists()
    assert "never-persist" not in receipt.read_text()
    assert "--ssd-cpu-threads" in json.loads(receipt.read_text())["argv"]


def test_omitted_output_directory_creates_a_new_run_directory(tmp_path, monkeypatch, capsys):
    inputs, spec, host, _, _, _ = setup(tmp_path)
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path, "linux"))))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(host))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr("os.environ", dict(inputs.environment))
    command = ["launch", "--dry-run", "--platform", "linux", "--model", spec.model]
    command += ["--binary", spec.binary, "--repo", inputs.repo, "--settings", spec.settings_file]
    command += ["--runtime-policy", str(contract), "--host-facts", str(facts)]
    assert main(command) == 0 and main(command) == 0
    runs = sorted((work / "runs").iterdir())
    assert len(runs) == 2 and all((run / "settings.ssd").exists() for run in runs)
    assert str(runs[0]) in capsys.readouterr().err


def test_historical_entry_cannot_execute(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["launch-recorded", "--execute", "--platform", "linux", "--repo", str(tmp_path)])
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_measured_selection_respects_explicit_cli_and_preserves_matching_receipt():
    model = SimpleNamespace(
        P={"all": {}},
        caps={"all": 8},
        ref="all",
        dev={"default_threads": 4},
        PF={"memory": {}},
    )
    matching = parse(overrides=[("cpu.threads", "4")])
    chosen = selection(model, 4, "all", matching, lambda: 4)
    assert chosen.at("cpu.threads") is matching.at("cpu.threads")
    conflicting = parse(overrides=[("cpu.threads", "2")])
    with pytest.raises(SelectionConflict, match="cpu.threads"):
        selection(model, 4, "all", conflicting, lambda: 4)


def test_controller_filters_conflicting_measurements_before_start(tmp_path, monkeypatch):
    captured = {}
    model = SimpleNamespace(
        P={"all": {}, "pin": {}},
        caps={"all": 8, "pin": 8},
        ref="all",
        configs=[(4, "all"), (4, "pin")],
        dev={"default_threads": 4, "coresets": {"g1": {"cpus": [0, 1, 2]}, "g2": {"cpus": [3, 4]}}},
        PF={"memory": {}},
    )

    def load(self, *a, **kw):
        self.__dict__.update(model.__dict__)

    monkeypatch.setattr("inferssd.model.machine.Machine.__init__", load)
    monkeypatch.setattr(runtime, "TypedController", lambda *a, **kw: captured.update(kw) or kw)
    value = runtime.Runtime(
        LaunchInput("/fixture/repo", ()),
        policy(tmp_path, "linux"),
        host=object(),
        emit=lambda _: None,
        configuration=LaunchSpec("/fixture/model.gguf", 8080, overrides=(("cpu.threads", "4"),)),
    )
    value.controller(
        "/recorded", "/output", 8080, (4, "all"), [], host_threads=lambda: 4, stdout=None
    )
    selected = captured["machine_factory"]("/recorded")
    assert selected.configs == [(4, "all")]
    assert selected.settings(4, "all").at("cpu.threads").value == 4
    assert selected.ubatch("all") == 512
    assert selected.route("all") is None
    value.controller(
        "/recorded", "/output", 8080, (4, "pin"), [], host_threads=lambda: 4, stdout=None
    )
    with pytest.raises(ValueError, match="start configuration"):
        captured["machine_factory"]("/recorded")


def test_owned_snapshot_does_not_reread_mutated_settings_file(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    configured = parse(Path(spec.settings_file).read_text(), spec.settings_file)
    Path(spec.settings_file).unlink()
    assert resolve(inputs, spec, adapter, out, configured=configured).code == 0


def test_controller_rejects_unconfigured_history_persistence(tmp_path):
    value = runtime.Runtime(
        LaunchInput("/unused", ()),
        policy(tmp_path, "linux"),
        host=object(),
        emit=lambda _: None,
        configuration=LaunchSpec("/unused", 8080),
    )
    with pytest.raises(ValueError, match="prompt persistence"):
        value.controller(
            "/unused",
            "/unused",
            8080,
            (4, "all"),
            [],
            host_threads=lambda: 4,
            stdout=None,
            keeps_history=True,
        )


@pytest.mark.parametrize("threads", [4, 8])
def test_mac_measured_prefill_threads_are_explicit_and_preserved(tmp_path, threads):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, "darwin")
    spec = replace(spec, overrides=(("cpu.threads", str(threads)),), batch_threads=12)
    result = resolve(inputs, spec, adapter, out, "darwin")
    assert result.argv[result.argv.index("-t") + 1] == str(threads)
    assert result.argv[result.argv.index("-tb") + 1] == "12"
    assert json.loads(result.outputs[-1].text)["launch"]["batch_threads"] == 12


@pytest.mark.parametrize("batch_threads", [0, -1, 257, True])
def test_bad_explicit_batch_threads_refused_before_native_file(tmp_path, batch_threads):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="batch_threads"):
        resolve(inputs, replace(spec, batch_threads=batch_threads), adapter, out)
    assert not (out / "settings.ssd").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ("--fit=on",),
        ("--chat-template-file", "other"),
        ("-ngl0",),
        ("--cpu-moe",),
        ("--no-host",),
        ("--tensor-split=1,0",),
        ("-otbad",),
    ],
)
def test_extra_cannot_replace_selected_native_placement(tmp_path, extra):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit typed"):
        resolve(replace(inputs, extra=extra), spec, adapter, out)


@pytest.mark.parametrize("explicit,expected", [(False, 1), (True, 0)])
def test_deepseek_profile_materializes_model_policy_and_preserves_explicit_values(
    tmp_path, explicit, expected
):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    inputs = replace(inputs, profile="dsv")
    (Path(inputs.repo) / "engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja").write_text(
        "template"
    )
    if explicit:
        spec = replace(
            spec,
            overrides=spec.overrides + (("model.engram_prefetch", "0"), ("model.hc_fused", "0")),
        )
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0
    native = parse((out / "settings.ssd").read_text(), overrides=spec.overrides)
    assert native.at("model.engram_prefetch").value == expected
    assert native.at("model.hc_fused").value == expected
