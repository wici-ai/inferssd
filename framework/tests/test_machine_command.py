# SPDX-License-Identifier: Apache-2.0
"""The profile command: measure, save, summarize and show, with a fake host and probes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from test_decode_profiling import setup as gpu_machine
from test_machine_identity import FakeHost, linux_tree, nvme_key, write
from test_machine_probes import FakeRunner, replies
from test_machine_profile import GIB, Clock, shards
from test_strategy_inputs import ENGINE_FIXTURE, Q4_K, Q8_0, engine_facts, moe
from test_windows_platform import REGISTRY, MultiGroupKernel, WindowsHost

from inferssd.app import catalog
from inferssd.cli import main
from inferssd.launch.resources import installed_resources
from inferssd.machine import command
from inferssd.machine.probes import Completed
from inferssd.machine.store import exclusive, save
from inferssd.platform.windows.system import Volume, Win32
from inferssd.strategy.inputs import RuntimeFacts, facts_from_engine
from inferssd.strategy.kernels import INPUT_BLOCK, VEC_DOT_TYPE, WEIGHT_BLOCK
from inferssd.strategy.routing import cache_slot_bytes
from inferssd.strategy.weights import dense_weights


def parse(argv):
    parser = argparse.ArgumentParser(prog="inferssd")
    command.add_command(parser.add_subparsers(dest="command"))
    return parser.parse_args(["profile", *argv])


def machine(tmp_path):
    root = tmp_path / "root"
    commands = linux_tree(root)
    models = tmp_path / "models"
    files = shards(models, [2 * GIB, 2 * GIB])
    host = FakeHost(root, commands=commands, devices={os.path.realpath(models): (259, 2)})
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    for name in ("inferssd-ssdprobe", "inferssd-cpuprobe", "inferssd-memprobe"):
        (build / "bin" / name).write_text("")
    return host, files, build


RUNTIME = RuntimeFacts((0, 1, 2, 3), (), 24 * GIB)


def execute(argv, host, runner, tmp_path, environ=None, *, runtime=RUNTIME):
    out, progress = [], []
    code = command.run(
        parse(argv),
        host=host,
        runner=runner,
        environ={"XDG_STATE_HOME": str(tmp_path / "state"), **(environ or {})},
        home=tmp_path / "home",
        clock=Clock(),
        progress=progress.append,
        emit=out.append,
        runtime=lambda: runtime,
    )
    return code, out, progress


def resource_descriptor(tmp_path, build):
    descriptor = tmp_path / "engine-resources.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": str(build / "bin/inferssd-server"),
                "capabilities": {"settings": "inferssd-settings 1", "telemetry": 1},
                "templates": {"muse": str(build / "chat.jinja")},
                "probes": {
                    name: str(build / "bin" / executable)
                    for name, executable in (
                        ("cpu", "inferssd-cpuprobe"),
                        ("memory", "inferssd-memprobe"),
                        ("ssd", "inferssd-ssdprobe"),
                    )
                },
            }
        )
    )
    return descriptor


def test_profile_uses_the_installed_wrappers_resources(tmp_path):
    host, files, build = machine(tmp_path)
    descriptor = resource_descriptor(tmp_path, build)
    runner = FakeRunner(replies())
    code, _, _ = execute(
        ["--model", files[0]],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(descriptor)},
    )
    assert code == 0
    assert {argv[0] for argv, _ in runner.calls} == {
        str(build / "bin" / name)
        for name in ("inferssd-cpuprobe", "inferssd-memprobe", "inferssd-ssdprobe")
    }
    saved = json.loads((tmp_path / "state/inferssd/machine.json").read_text())
    assert all(section["status"] == "ok" for section in saved["sections"].values())


@pytest.mark.parametrize("option", ["--resources", "--build"])
def test_explicit_probe_location_overrides_installed_resources(tmp_path, option):
    host, _, build = machine(tmp_path)
    location = resource_descriptor(tmp_path, build) if option == "--resources" else build
    runner = FakeRunner(replies())
    code, _, _ = execute(
        [option, str(location)],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
    )
    assert code == 0 and runner.calls
    assert all(argv[0].startswith(str(build / "bin")) for argv, _ in runner.calls)


@pytest.mark.parametrize("environ", [{}, {"INFERSSD_RESOURCES": ""}])
def test_profile_uses_the_current_release_without_the_variable(tmp_path, environ):
    host, _, build = machine(tmp_path)
    current = tmp_path / "home/.local/share/inferssd/current"
    current.mkdir(parents=True)
    resource_descriptor(tmp_path, build).replace(current / "engine-resources.json")
    runner = FakeRunner(replies())
    code, _, _ = execute([], host, runner, tmp_path, environ=environ)
    assert code == 0 and runner.calls
    assert all(argv[0].startswith(str(build / "bin")) for argv, _ in runner.calls)


@pytest.mark.parametrize("environ", [{}, {"INFERSSD_RESOURCES": ""}])
def test_profile_does_not_look_for_probes_on_path(tmp_path, monkeypatch, environ):
    host, _, build = machine(tmp_path)
    for probe in (build / "bin").iterdir():
        probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(build / "bin"))
    runner = FakeRunner({})
    with pytest.raises(ValueError, match="engine not installed"):
        execute([], host, runner, tmp_path, environ=environ)
    assert not runner.calls
    assert not (tmp_path / "state/inferssd/machine.json").exists()


def test_installed_resources_prefer_the_variable_over_the_current_release(tmp_path):
    data = "AppData/Local/inferssd" if sys.platform == "win32" else ".local/share/inferssd"
    current = tmp_path / data / "current/engine-resources.json"
    assert installed_resources({}, tmp_path) == current
    assert installed_resources({"INFERSSD_RESOURCES": ""}, tmp_path) == current
    assert installed_resources({"INFERSSD_RESOURCES": "/a.json"}, tmp_path) == Path("/a.json")
    assert installed_resources({"INFERSSD_HOME": "/data"}, tmp_path) == Path(
        "/data/current/engine-resources.json"
    )


def test_show_ignores_installed_resources_and_does_not_measure(tmp_path):
    host, _, build = machine(tmp_path)
    code, expected, _ = execute(["--build", str(build)], host, FakeRunner(replies()), tmp_path)
    assert code == 0
    saved = tmp_path / "state/inferssd/machine.json"
    before = saved.read_bytes()
    runner = FakeRunner({})
    code, shown, _ = execute(
        ["--show"],
        host,
        runner,
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
    )
    assert code == 0 and shown == expected and not runner.calls
    assert saved.read_bytes() == before


def test_missing_installed_descriptor_is_not_silently_ignored(tmp_path):
    host, _, _ = machine(tmp_path)
    runner = FakeRunner({})
    with pytest.raises(ValueError, match="engine not installed"):
        execute(
            [],
            host,
            runner,
            tmp_path,
            environ={"INFERSSD_RESOURCES": str(tmp_path / "missing.json")},
        )
    assert not runner.calls
    assert not (tmp_path / "state/inferssd/machine.json").exists()


def test_profile_is_listed(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0 and "measure this machine" in text
    with pytest.raises(SystemExit) as error:
        main(["profile", "--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0
    for option in ("--model PATH", "--resources", "--build", "--reprofile", "--output", "--show"):
        assert option in text


def test_profile_saves_to_the_state_directory_and_summarizes(tmp_path):
    host, files, build = machine(tmp_path)
    runner = FakeRunner(replies())
    code, out, progress = execute(
        ["--model", files[0], "--build", str(build)], host, runner, tmp_path
    )
    saved = tmp_path / "state/inferssd/machine.json"
    assert code == 0 and saved.exists()
    assert runner.calls[0][0][0] == str(build / "bin/inferssd-cpuprobe")
    assert out[0].startswith(f"machine profile {saved} (written ")
    assert out[1] == (
        "identity: linux x86_64, Example CPU 9000, 4 logical CPUs, 31.2 GiB RAM, "
        "GPUs: NVIDIA GeForce RTX 5090 (CUDA)"
    )
    names = [line.split(":", 1)[0] for line in out[2:]]
    assert names == ["cpu", "memory", "compute", "gpus", "storage", "estimate memory"] + [
        "estimate"
    ] * len(catalog.MODELS)
    assert out[7] == "estimate memory: 24.0 GiB (available now; --memory-gib sets another amount)"
    # This machine measured Q4_K alone: every expert model is priced with its other formats taken
    # from Q4_K's rate, and the model without routed experts by its dense weights from RAM (its GPU
    # measured no matrix product).
    for model, line in zip(catalog.MODELS, out[8:]):
        assert line.startswith(f"estimate: {model.repo}:{model.quantization} ")
        assert " tok/s" in line and "none:" not in line
        if model.facts.experts is None:
            assert "GB of dense weights from RAM at " in line
        else:
            assert " ms a token: dense " in line
    assert out[2].endswith("4 logical CPUs (4 uniform); ISA AVX2 FMA F16C")
    assert "DRAM read 11.18 GiB/s with 1 thread" in out[3]
    assert out[4].endswith("1 formats at 4 threads; skipped MXFP4 (no CPU kernel)")
    assert "NVIDIA GeForce RTX 5090 31.8 GiB, host to device 22.35 GiB/s" in out[5]
    assert out[6].startswith(f"storage:{nvme_key()}: ok (")
    assert out[6].endswith(
        "Example NVMe 2TB (nvme, io_uring), 2 files; best 4.75 GiB/s at "
        "4 MiB QD 4; knee QD 4 at 1 MiB, QD 4 at 4 MiB"
    )
    assert progress[0] == "reading machine identity"

    # A second run finds nothing to measure and leaves the file alone.
    before = saved.read_bytes()
    runner = FakeRunner({})
    code, again, _ = execute(["--model", files[0], "--build", str(build)], host, runner, tmp_path)
    assert code == 0 and runner.calls == [] and saved.read_bytes() == before and again == out

    code, shown, _ = execute(["--show", "--model", files[0]], host, FakeRunner({}), tmp_path)
    assert code == 0 and shown == out
    code, shown, _ = execute(["--show"], host, FakeRunner({}), tmp_path)
    assert code == 0 and shown == out[:7]


def test_failed_sections_set_the_exit_status(tmp_path):
    host, files, _ = machine(tmp_path)
    output = tmp_path / "custom.json"
    code, out, _ = execute(
        ["--output", str(output), "--build", str(tmp_path / "none")], host, FakeRunner({}), tmp_path
    )
    assert code == 1 and output.exists()
    assert out[-2:] == ["2 section(s) partial: cpu, memory", "2 section(s) failed: compute, gpus"]
    assert any("inferssd-cpuprobe not found" in line for line in out)


def test_unreadable_saved_profile_is_replaced(tmp_path):
    host, _, build = machine(tmp_path)
    output = tmp_path / "machine.json"
    output.write_text(json.dumps({"schema": "inferssd-machine/0"}))
    code, _, progress = execute(
        ["--output", str(output), "--build", str(build)], host, FakeRunner(replies()), tmp_path
    )
    assert code == 0 and any(
        line.startswith("ignoring the unreadable saved profile") for line in progress
    )
    assert json.loads(output.read_text())["schema"] == "inferssd-machine/1"


def test_memory_gib_sets_the_memory_the_estimates_plan_with(tmp_path):
    host, files, build = machine(tmp_path)
    execute(["--model", files[0], "--build", str(build)], host, FakeRunner(replies()), tmp_path)
    show = ["--show", "--model", files[0]]
    busy = replace(RUNTIME, available_bytes=2 * GIB)

    def shown(argv, runtime):
        code, out, _ = execute(show + argv, host, FakeRunner({}), tmp_path, runtime=runtime)
        assert code == 0
        return [line for line in out if line.startswith("estimate")]

    now = shown([], busy)
    assert now[0] == "estimate memory: 2.0 GiB (available now; --memory-gib sets another amount)"
    asked = shown(["--memory-gib", "16"], busy)
    assert asked[0] == "estimate memory: 16.0 GiB (--memory-gib)"
    # The estimates are those of a host with that much available, not those of this moment.
    assert asked[1:] == shown([], replace(RUNTIME, available_bytes=16 * GIB))[1:]
    assert asked[1:] != now[1:]
    # A smaller amount still lowers it, and no amount exceeds the installed memory.
    assert shown(["--memory-gib", "1"], busy)[0] == "estimate memory: 1.0 GiB (--memory-gib)"
    assert shown(["--memory-gib", "64"], busy)[0] == (
        "estimate memory: 31.2 GiB (--memory-gib asked for 64.0 GiB, more than a process can have "
        "here)"
    )
    code, out, _ = execute(
        [*show, "--json", "--memory-gib", "16"], host, FakeRunner({}), tmp_path, runtime=busy
    )
    planned = {
        row["components"].get("memory_budget_bytes") for row in json.loads(out[0])["estimates"]
    }
    assert code == 0 and planned - {None} == {16.0 * GIB}


def test_memory_gib_stays_within_the_cgroup_limit(tmp_path):
    host, files, build = machine(tmp_path)
    execute(["--model", files[0], "--build", str(build)], host, FakeRunner(replies()), tmp_path)
    # The installed memory is 31.2 GiB, but no launch in this cgroup gets more than 8.
    write(host.root, "/proc/self/cgroup", "0::/box.scope\n")
    write(host.root, "/sys/fs/cgroup/memory.max", "max\n")
    write(host.root, "/sys/fs/cgroup/box.scope/memory.max", f"{8 * GIB}\n")

    def shown(argv, runtime):
        argv = ["--show", "--model", files[0], *argv]
        code, out, _ = execute(argv, host, FakeRunner({}), tmp_path, runtime=runtime)
        assert code == 0
        return [line for line in out if line.startswith("estimate")]

    asked = shown(["--memory-gib", "16"], replace(RUNTIME, available_bytes=2 * GIB))
    assert asked[0] == (
        "estimate memory: 8.0 GiB (--memory-gib asked for 16.0 GiB, more than a process can have "
        "here)"
    )
    assert asked[1:] == shown([], replace(RUNTIME, available_bytes=8 * GIB))[1:]
    assert shown(["--memory-gib", "4"], RUNTIME)[0] == "estimate memory: 4.0 GiB (--memory-gib)"


def test_show_measures_nothing(tmp_path):
    host, files, _ = machine(tmp_path)
    with pytest.raises(ValueError, match="no machine profile at .*; run inferssd profile first"):
        execute(["--show"], host, FakeRunner({}), tmp_path)
    with pytest.raises(ValueError, match="--show measures nothing"):
        execute(["--show", "--reprofile"], host, FakeRunner({}), tmp_path)


def test_json_prints_one_object_with_the_machine_and_the_estimates(tmp_path):
    host, files, build = machine(tmp_path)
    code, out, progress = execute(
        ["--model", files[0], "--build", str(build), "--json"],
        host,
        FakeRunner(replies()),
        tmp_path,
    )
    (line,) = out
    value = json.loads(line)
    assert code == 0 and progress[0] == "reading machine identity"
    assert value["profile"] == str(tmp_path / "state/inferssd/machine.json")
    assert value["machine"] == {
        "platform": "linux",
        "cpu": "Example CPU 9000",
        "logical_cpus": 4,
        "memory_bytes": 32768000 * 1024,
        "memory_read_Bps": 3.6e10,
        "gpus": ["NVIDIA GeForce RTX 5090"],
        "models_drive": {"model": "Example NVMe 2TB", "read_Bps": 5.1e9},
    }
    assert [(row["model"], row["quantization"]) for row in value["estimates"]] == [
        (model.repo, model.quantization) for model in catalog.MODELS
    ]
    # Q4_K alone was measured: the other formats take its rate, so every model has an estimate.
    assert all(row["decode_tokens_per_second"] > 0 for row in value["estimates"])
    for row in value["estimates"]:
        assert row["estimator_version"] == 2
        assert row["scenarios"][row["reference_scenario"]] == row["decode_tokens_per_second"]
    # Without --model the estimates have no drive, and the machine no models drive.
    descriptor = tmp_path / "unreadable-engine.json"
    descriptor.write_text("invalid descriptor")
    code, out, _ = execute(
        ["--show", "--json"],
        host,
        FakeRunner({}),
        tmp_path,
        environ={"INFERSSD_RESOURCES": str(descriptor)},
    )
    value = json.loads(out[0])
    assert value["machine"]["models_drive"] is None and {
        row["detail"] for row in value["estimates"]
    } == {"pass --model with the directory that holds the models"}


def test_cli_entry_point(tmp_path, monkeypatch, capsys):
    host, files, build = machine(tmp_path)
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    monkeypatch.setattr(command, "run_process", FakeRunner(replies()))
    output = tmp_path / "machine.json"
    assert (
        main(["profile", "--model", files[0], "--build", str(build), "--output", str(output)]) == 0
    )
    assert main(["profile", "--show", "--output", str(output)]) == 0
    assert capsys.readouterr().out.count("\nstorage:linux:nvme:Example NVMe 2TB:") == 2
    with pytest.raises(SystemExit) as error:
        main(["profile", "--show", "--output", str(tmp_path / "absent.json")])
    assert error.value.code == 2 and "no machine profile" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["profile", "--build", str(build), "--resources", str(output)])
    assert error.value.code == 2 and "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    "process_defaults,reserved_elsewhere,expected_max",
    [((), False, 128), ((257,), True, 127), ((257, 321), True, 1)],
)
def test_desktop_profile_and_saved_report_on_two_windows_processor_groups(
    tmp_path, monkeypatch, capsys, process_defaults, reserved_elsewhere, expected_max
):
    models = tmp_path / "models"
    models.mkdir()
    kernel = MultiGroupKernel()
    kernel.default_sets = process_defaults
    if reserved_elsewhere:
        kernel.flags[1] = 0b010
    windows = Win32(kernel, lambda key, name: REGISTRY.get((key, name)), last_error=lambda: 122)
    monkeypatch.setattr(windows, "volume", lambda path: Volume("C:\\", "NTFS", ()))
    host = WindowsHost(tmp_path, windows)
    host.devices[str(models)] = (0, 1)
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    monkeypatch.setattr("inferssd.strategy.inputs.NativeHost", lambda: host)
    monkeypatch.setattr("inferssd.machine.profile.TEST_FILE_BYTES", 4096)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    descriptor = resource_descriptor(tmp_path, tmp_path / "build")
    monkeypatch.setenv("INFERSSD_RESOURCES", str(descriptor))
    simulated = replies()
    probe_maxima = []

    def run_probe(argv, *_):
        name = Path(argv[0]).name
        if name == "inferssd-cpuprobe":
            counts = argv[argv.index("--threads") + 1]
            probe_maxima.append(max(map(int, counts.split(","))))
        return simulated[name]

    monkeypatch.setattr(command, "run_process", run_probe)

    # Measure and read back what the desktop setup shows with its actual CLI arguments.
    for flags in ([], ["--show"]):
        assert main(["profile", "--model", str(models), *flags, "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["machine"]["logical_cpus"] == 128
        assert report["machine"]["memory_bytes"] > 0 and report["machine"]["cpu"]
        assert report["estimates"]
    saved = json.loads(Path(report["profile"]).read_text())
    assert saved["sections"]["cpu"]["status"] == "ok"
    assert len(saved["cpu"]["logical"]) == 128
    assert probe_maxima == [expected_max]


def test_cli_estimates_an_unlisted_split_gguf_and_can_replay_its_exported_facts(
    tmp_path, monkeypatch, capsys
):
    host, files, build = machine(tmp_path)
    raw = engine_facts(
        [
            ("token_embd.weight", (256, 256), Q8_0),
            ("output.weight", (256, 256), Q8_0),
            ("blk.0.ffn_up.weight", (256, 256), Q4_K),
        ],
        {"general.architecture": "unlisted-test-decoder"},
        path=files[0],
    )
    raw["files"] = files
    raw["tensors"][-1]["file"] = 1
    runner = FakeRunner(replies(**{"inferssd-server": Completed(0, json.dumps(raw))}))
    monkeypatch.setattr(
        "inferssd.cli.run_profile",
        partial(
            command.run,
            host=host,
            runner=runner,
            runtime=lambda: RUNTIME,
        ),
    )
    saved = tmp_path / "machine.json"
    args = ["profile", "--output", str(saved), "--json"]
    target = ["--estimate-gguf", files[0], "--ple", files[1], "--build", str(build)]
    assert main([*args, *target]) == 0
    report = json.loads(capsys.readouterr().out)
    assert runner.calls[0][0] == [
        str(build / "bin/inferssd-server"),
        "--ssd-model-facts",
        files[0],
        "--shard",
        files[1],
    ]
    assert report["machine"]["models_drive"]["model"] == "Example NVMe 2TB"
    (row,) = report["estimates"]
    assert row["model"] == files[0] and row["quantization"] == "Q4_K"
    assert row["decode_tokens_per_second"] == row["scenarios"][row["reference_scenario"]] > 0
    assert row["components"]["dense_read_bytes"] == sum(t["bytes"] for t in raw["tensors"][1:])
    original_profile = saved.read_bytes()
    runner.calls.clear()
    assert main([*args, "--show", *target]) == 0
    assert json.loads(capsys.readouterr().out)["estimates"] == [row]
    assert len(runner.calls) == 1 and "--ssd-model-facts" in runner.calls[0][0]
    exported = tmp_path / "facts.json"
    exported.write_text(json.dumps(raw))
    runner.calls.clear()
    assert main([*args, "--show", "--estimate-facts", str(exported)]) == 0
    assert json.loads(capsys.readouterr().out)["estimates"] == [row]
    assert saved.read_bytes() == original_profile and not runner.calls


@pytest.mark.parametrize(
    ("data", "message"),
    [(None, "absent.json"), ("bad json", "Expecting value"), ('{"version":1}', "model.")],
)
def test_cli_diagnoses_missing_and_invalid_exported_facts(tmp_path, capsys, data, message):
    path = tmp_path / "absent.json"
    if data is not None:
        path.write_text(data)
    with pytest.raises(SystemExit) as error:
        main(["profile", "--show", "--estimate-facts", str(path)])
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_cli_reports_active_and_stored_bytes_for_native_graph_moe(tmp_path):
    host, paths, build = machine(tmp_path)
    raw = engine_facts(
        [
            ("blk.0.ffn_gate_up_exps.weight", (256, 512, 8), Q4_K),
            ("blk.0.ffn_down_exps.weight", (256, 256, 8), Q4_K),
        ],
        {
            "general.architecture": "test-moe",
            "test-moe.expert_count": 8,
            "test-moe.expert_used_count": 2,
        },
        path=paths[0],
    )
    assert raw["experts"] is None
    export = tmp_path / "graph-facts.json"
    export.write_text(json.dumps(raw))
    code, out, _ = execute(
        ["--build", str(build), "--estimate-facts", str(export), "--json"],
        host,
        FakeRunner(replies()),
        tmp_path,
    )
    (result,) = json.loads(out[0])["estimates"]
    assert code == 0 and result["decode_tokens_per_second"] > 0
    full = sum(t["bytes"] for t in raw["tensors"])
    assert result["components"]["graph_expert_store_bytes"] == full
    assert result["components"]["dense_read_bytes"] == full // 4


def test_routed_estimate_reserves_vram_for_other_compute_processes(tmp_path):
    host, _, _ = machine(tmp_path)
    profile, runtime = gpu_machine()
    profile["compute"]["cpu"] = []
    saved = tmp_path / "machine.json"
    save(profile, saved)
    tensors, kv = moe(
        "qwen35moe",
        layers=2,
        experts=256,
        used=2,
        latent=1024,
        hidden=2048,
        gate=Q4_K,
        up=Q4_K,
        down=Q4_K,
    )
    raw = engine_facts(tensors, kv)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(raw))
    experts = facts_from_engine(raw).experts
    _, slot_bytes = cache_slot_bytes(experts)
    full = dense_weights(raw).active_bytes + slot_bytes * experts.layers * experts.count
    args = ["--show", "--output", str(saved), "--estimate-facts", str(facts), "--json"]

    def report(free, processes):
        gpu = replace(runtime.gpus[0], free_bytes=free, processes=processes)
        code, out, _ = execute(
            args, host, FakeRunner({}), tmp_path, runtime=replace(runtime, gpus=(gpu,))
        )
        assert code == 0
        return json.loads(out[0])["estimates"][0]

    idle = report(full, 0)
    assert idle["decode_tokens_per_second"] > 0 and idle["components"]["gpu_hit_fraction"] == 1
    busy = report(full, 1)
    assert busy["decode_tokens_per_second"] is None
    assert "no CPU compute measurement of Q4_K" in busy["detail"]
    restored = report(full + GIB, 1)
    assert restored["decode_tokens_per_second"] == idle["decode_tokens_per_second"]
    assert restored["components"]["gpu_hit_fraction"] == 1


@pytest.mark.parametrize(
    ("architecture", "extra_tensor", "refusal"),
    [
        ("qwen3moe", None, "no model profile for the GGUF architecture qwen3moe"),
        ("gemma4", None, "no routed SSD executor"),
        ("qwen35moe", "ffn_up_exps.scale", "unsupported per-expert scales"),
        ("qwen4exp", "ffn_gate_up_exps.weight", "fused gate/up"),
        ("qwen35moe", None, ""),
    ],
)
def test_explicit_target_requires_a_supported_ssd_path(
    tmp_path, architecture, extra_tensor, refusal
):
    host, paths, build = machine(tmp_path)
    model = json.loads(ENGINE_FIXTURE.read_text())
    model["architecture"] = architecture
    if extra_tensor:
        model["tensors"].append(dict(model["tensors"][-1], name="blk.0." + extra_tensor))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(model))
    code, out, _ = execute(
        ["--build", str(build), "--model", paths[0], "--estimate-facts", str(facts), "--json"],
        host,
        FakeRunner(replies()),
        tmp_path,
    )
    (row,) = json.loads(out[0])["estimates"]
    assert code == 0
    if refusal:
        assert row["decode_tokens_per_second"] is None
        assert refusal in row["detail"]
        assert not row["scenarios"]
    else:
        assert row["decode_tokens_per_second"] == row["scenarios"]["rank_serial"] > 0


@pytest.mark.parametrize(
    ("architecture", "kv", "tensor", "refusal"),
    [
        ("qwen35", {}, "blk.0.ffn_up.weight", None),
        ("bert", {}, "blk.0.ffn_up.weight", "encoder"),
        ("gemma-embedding", {}, "blk.0.ffn_up.weight", "encoder"),
        ("llama-embed", {}, "blk.0.ffn_up.weight", "encoder"),
        ("dream", {}, "blk.0.ffn_up.weight", "autoregressive"),
        ("pangu-embedded", {}, "blk.0.ffn_up.weight", None),
        (
            "bert",
            {"bert.attention.causal": False},
            "blk.0.ffn_up.weight",
            "the bert model is an encoder",
        ),
        ("t5encoder", {}, "enc.blk.0.ffn_up.weight", "the t5encoder model is an encoder"),
        # No decoder layers to read facts from: the reason still reaches the report.
        ("clip", {}, "v.blk.0.ffn_up.weight", "does not run autoregressive text decode"),
    ],
)
def test_a_dense_decoder_needs_no_profile_and_an_encoder_has_no_estimate(
    tmp_path, architecture, kv, tensor, refusal
):
    host, paths, build = machine(tmp_path)
    raw = engine_facts(
        [("token_embd.weight", (256, 256), Q8_0), (tensor, (256, 256), Q4_K)],
        {"general.architecture": architecture, **kv},
        path=paths[0],
    )
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(raw))
    code, out, _ = execute(
        ["--build", str(build), "--model", paths[0], "--estimate-facts", str(facts), "--json"],
        host,
        FakeRunner(replies()),
        tmp_path,
    )
    (row,) = json.loads(out[0])["estimates"]
    assert code == 0
    if refusal:
        assert row["decode_tokens_per_second"] is None and refusal in row["detail"]
    else:
        assert row["decode_tokens_per_second"] > 0


@pytest.mark.parametrize(
    ("up", "gate", "down", "refusal"),
    [
        ("f32", "f32", "f32", "cannot compute F32 experts"),
        ("q8_0", "q4_K", "q4_K", "up=Q8_0 and gate=Q4_K"),
        ("q4_K", "q4_K", "q6_K", None),
        ("iq4_nl", "q4_0", "q8_0", None),
    ],
)
def test_routed_formats_are_those_the_cpu_expert_executor_computes(up, gate, down, refusal):
    raw = json.loads(ENGINE_FIXTURE.read_text())
    raw["architecture"] = "qwen35moe"
    layout = raw["experts"]
    layout["latent"] = layout["hidden"] = 256
    for roles in layout["formats"]:
        roles["up"]["type"], roles["gate"]["type"], roles["down"]["type"] = up, gate, down
    reason = command.launch_refusal(raw)
    assert reason is None if refusal is None else refusal in reason


@pytest.mark.parametrize(("weight", "width"), [("q1_0", 64), ("q2_0", 32), ("nvfp4", 32)])
def test_expert_rows_must_contain_whole_weight_blocks(weight, width):
    raw = json.loads(ENGINE_FIXTURE.read_text())
    raw["architecture"] = "qwen35moe"
    raw["experts"]["hidden"] = width
    for fmt in raw["experts"]["formats"]:
        fmt["down"]["type"] = weight
    assert f"cannot compute {weight.upper()} experts" in (command.launch_refusal(raw) or "")


def test_expert_input_types_follow_the_cpu_type_traits():
    ggml = ENGINE_FIXTURE.parents[4] / "engine/ggml/src"
    source = (ggml / "ggml-cpu/ggml-cpu.c").read_text()
    table = source[source.index("type_traits_cpu[GGML_TYPE_COUNT] = {") :]
    table = table[table.index("{") + 1 : table.index("\n};")]
    found = {}
    quantizers = set()
    for name, block in re.findall(r"\[GGML_TYPE_(\w+)\]\s*=\s*\{(.*?)\n    \}", table, re.S):
        input_type = re.search(r"\.vec_dot_type\s*=\s*GGML_TYPE_(\w+)", block)
        if re.search(r"\.from_float\s*=", block):
            quantizers.add(name)
        if re.search(r"\.vec_dot\s*=", block) and input_type:
            found[name] = input_type.group(1)
    assert found == VEC_DOT_TYPE
    macros = dict(
        re.findall(r"^#define (QK\w*)\s+(\d+)\b", (ggml / "ggml-common.h").read_text(), re.M)
    )
    blocks = {
        name: int(macros.get(size, size))
        for name, size in re.findall(
            r"\[GGML_TYPE_(\w+)\]\s*=\s*\{[^}]*?\.blck_size\s*=\s*(\w+)",
            (ggml / "ggml.c").read_text(),
        )
    }
    inputs = (set(found.values()) & quantizers) - {"F32"}
    assert INPUT_BLOCK == {name: blocks[name] for name in inputs}
    assert WEIGHT_BLOCK == {
        weight: blocks[weight]
        for weight, activation in found.items()
        if activation in inputs and blocks[activation] % blocks[weight]
    }


def test_estimates_use_the_gpu_backends_of_the_engine_they_are_for(tmp_path, monkeypatch):
    host, paths, build = machine(tmp_path)
    seen = []
    estimates = command.estimates

    def spy(value, drive_key, platform, runtime, **kwargs):
        seen.append(runtime().gpu_backends)
        return estimates(value, drive_key, platform, runtime, **kwargs)

    monkeypatch.setattr(command, "estimates", spy)
    # A CPU-only build, as its CMake cache names it.
    (build / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=OFF\nGGML_VULKAN:BOOL=OFF\n")
    assert (
        execute(
            ["--build", str(build), "--model", paths[0], "--json"],
            host,
            FakeRunner(replies()),
            tmp_path,
        )[0]
        == 0
    )
    # The resource descriptor's capabilities.
    descriptor = resource_descriptor(tmp_path, build)
    value = json.loads(descriptor.read_text())
    value["capabilities"]["gpu_backends"] = ["CUDA"]
    descriptor.write_text(json.dumps(value))
    assert (
        execute(
            ["--resources", str(descriptor), "--model", paths[0], "--json"],
            host,
            FakeRunner(replies()),
            tmp_path,
        )[0]
        == 0
    )
    assert seen == [(), ("CUDA",)]


@pytest.mark.parametrize(
    ("entry", "advertised", "cuda_cache", "gpu"),
    [
        ("gguf", None, "OFF", False),
        ("gguf", [], "ON", False),
        ("facts", [], "ON", False),
        ("facts-build", None, "OFF", False),
        ("installed", [], "ON", False),
        ("installed", None, "ON", True),
        ("offline", None, None, True),
    ],
)
def test_show_estimates_use_the_selected_engine_and_work_offline(
    tmp_path, entry, advertised, cuda_cache, gpu
):
    host, paths, build = machine(tmp_path)
    profile, runtime = gpu_machine()
    saved = tmp_path / "machine.json"
    save(profile, saved)
    original = saved.read_bytes()
    descriptor = resource_descriptor(tmp_path, build)
    data = json.loads(descriptor.read_text())
    if advertised is not None:
        data["capabilities"]["gpu_backends"] = advertised
        descriptor.write_text(json.dumps(data))
    if cuda_cache:
        (build / "CMakeCache.txt").write_text(f"GGML_CUDA:BOOL={cuda_cache}\n")
    raw = engine_facts(
        [("token_embd.weight", (256, 256), Q8_0), ("blk.0.ffn_up.weight", (256, 256), Q4_K)],
        {"general.architecture": "qwen35"},
        path=paths[0],
    )
    exported = tmp_path / "facts.json"
    exported.write_text(json.dumps(raw))
    target = (
        ["--estimate-gguf", paths[0]] if entry == "gguf" else ["--estimate-facts", str(exported)]
    )
    selection = (
        ["--build", str(build)]
        if entry == "facts-build"
        else ["--resources", str(descriptor)]
        if entry in ("gguf", "facts")
        else []
    )
    environ = {"INFERSSD_RESOURCES": str(descriptor)} if entry == "installed" else {}
    runner = FakeRunner({"inferssd-server": Completed(0, json.dumps(raw))})
    code, out, _ = execute(
        ["--show", "--output", str(saved), "--json", *selection, *target],
        host,
        runner,
        tmp_path,
        environ,
        runtime=runtime,
    )
    (row,) = json.loads(out[0])["estimates"]
    assert code == 0 and row["decode_tokens_per_second"] > 0
    assert row["reference_scenario"] == ("dense_pipeline" if gpu else "dense_host_serial")
    assert saved.read_bytes() == original and len(runner.calls) == int(entry == "gguf")


@pytest.mark.parametrize("selection", ["installed", "explicit", "offline"])
def test_show_catalog_uses_the_selected_engine_when_available(tmp_path, selection):
    host, paths, build = machine(tmp_path)
    profile, runtime = gpu_machine()
    saved = tmp_path / "machine.json"
    save(profile, saved)
    directory = Path(paths[0]).parent
    args = ["--show", "--model", str(directory), "--output", str(saved), "--json"]
    environ = {}
    if selection != "offline":
        descriptor = resource_descriptor(tmp_path, build)
        data = json.loads(descriptor.read_text())
        data["capabilities"]["gpu_backends"] = []
        descriptor.write_text(json.dumps(data))
        if selection == "installed":
            environ["INFERSSD_RESOURCES"] = str(descriptor)
        else:
            args += ["--resources", str(descriptor)]
    runner = FakeRunner({})
    code, out, _ = execute(args, host, runner, tmp_path, environ, runtime=runtime)
    rows = {row["model"]: row for row in json.loads(out[0])["estimates"]}
    dense = rows["google/gemma-4-E4B-it-qat-q4_0-gguf"]
    routed = rows["unsloth/Qwen3.6-35B-A3B-GGUF"]
    gpu = selection == "offline"
    assert code == 0 and not runner.calls
    assert dense["reference_scenario"] == ("dense_pipeline" if gpu else "dense_host_serial")
    assert routed["components"]["gpu_execution_fraction"] == (1.0 if gpu else 0.0)


def test_cli_rejects_missing_gguf_and_invalid_shard_combinations(tmp_path, monkeypatch, capsys):
    _, _, build = machine(tmp_path)
    runner = FakeRunner({"inferssd-server": Completed(1, "", "error: file does not exist")})
    monkeypatch.setattr(command, "run_process", runner)
    with pytest.raises(SystemExit) as error:
        main(["profile", "--estimate-gguf", "missing.gguf", "--build", str(build)])
    assert error.value.code == 2 and "missing.gguf: file does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["profile", "--show", "--estimate-facts", "export.json", "--shard", "extra.gguf"])
    assert (
        error.value.code == 2
        and "--shard/--ple requires --estimate-gguf" in capsys.readouterr().err
    )
    with pytest.raises(SystemExit) as error:
        main(["profile", "--estimate-facts", "export.json", "--estimate-gguf", "model.gguf"])
    assert error.value.code == 2 and "not allowed with argument" in capsys.readouterr().err


def test_cli_entry_uses_installer_environment(tmp_path, monkeypatch):
    host, files, build = machine(tmp_path)
    descriptor = resource_descriptor(tmp_path, build)
    runner = FakeRunner(replies())
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    monkeypatch.setattr(command, "run_process", runner)
    monkeypatch.setenv("INFERSSD_RESOURCES", str(descriptor))
    output = tmp_path / "machine.json"
    assert main(["profile", "--model", files[0], "--output", str(output)]) == 0
    assert all(
        section["status"] == "ok" for section in json.loads(output.read_text())["sections"].values()
    )
    assert runner.calls[0][0][0] == str(build / "bin/inferssd-cpuprobe")


@pytest.mark.skipif(os.name == "nt", reason="flock")
def test_a_running_profile_holds_the_lock(tmp_path, monkeypatch, capsys):
    import fcntl

    host, _, build = machine(tmp_path)
    monkeypatch.setattr(command, "NativeHost", lambda: host)
    output = tmp_path / "machine.json"
    with exclusive(output):
        pass
    holder = os.open(tmp_path / "machine.json.lock", os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            main(["profile", "--build", str(build), "--output", str(output)])
    finally:
        os.close(holder)
    assert error.value.code == 2 and "another inferssd profile holds" in capsys.readouterr().err
