"""Formal command, ordered preparation, and no implicit execution."""

import json
import os
import subprocess
import sys

import pytest
from observe_launch_F import header

from ssdash.cli import main
from ssdash.launch.recorded import RecordedArtifacts
from ssdash.launch.types import Fact, LaunchInput


def test_installed_command_synthetic_assets(tmp_path):
    d = tmp_path.resolve()
    model = d / "model.gguf"
    header(model)
    bin_dir = d / "bin"
    bin_dir.mkdir()
    sentinel = d / "FORBIDDEN"
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\ntouch " + str(sentinel) + "\nexit 95\n")
    binary.chmod(0o755)
    (bin_dir / "libkimi-no-mmap-prefetch.so").touch()
    (bin_dir / "libkimi-vram-tier.so").touch()
    conf = d / "model.conf"
    conf.write_text("NAME=synthetic\nSAMPLING=--temp 1.0 --top-k 9\n")
    host = d / "host.json"
    host.write_text(
        json.dumps(
            dict(
                filesystem="ext4",
                gpu_available=False,
                ss_available=False,
                has_uring=False,
                cgroup_v2=True,
                memory_kib=[16777216, 14680064],
                euid=0,
            )
        )
    )
    result = d / "launch.json"
    command = [
        sys.executable,
        "-m",
        "ssdash",
        "launch-recorded",
        "--dry-run",
        "--platform",
        "linux",
        "--repo",
        str(d),
        "--host-facts",
        str(host),
        "--model-conf",
        str(conf),
        "--model",
        str(model),
        "--artifact-dir",
        str(d / "index"),
        "--result",
        str(result),
        "--",
        "--seed",
        "17",
    ]
    env = dict(
        HOME=str(d),
        PATH=os.defpath,
        SERVER_BIN=str(binary),
        CPUSET="0-1",
        MEMORY_MAX="10737418240",
        LAUNCH_SCOPE="synthetic.scope",
    )
    # In the source-only test this preserves its explicit import path; installed
    # wheel validation deliberately runs with PYTHONPATH unset.
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    data = json.loads(result.read_text())
    assert data["argv"][-2:] == ["--seed", "17"]
    assert data["scope"][:2] == ["systemd-run", "--scope"]
    assert dict(data["assigned_environment"])["GGML_KIMI_CPU_SINK_THREADS"] == "2"
    assert (d / "index/experts.tsv").is_file()
    assert not sentinel.exists()
    assert "SCOPE --unit=synthetic.scope\n" in completed.stdout
    assert "nvidia-smi not found" in completed.stderr


def test_command_requires_explicit_dry_run_and_host_facts(capsys):
    with pytest.raises(SystemExit) as failure:
        main(["launch-recorded", "--platform", "linux", "--repo", "/does-not-exist"])
    assert failure.value.code == 2
    assert "--dry-run" in capsys.readouterr().err


def test_recorded_adapter_never_queries_host(monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("unexpected process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()),
        {"gpu_available": False},
        run=forbidden,
        replace=forbidden,
        python="unused",
    )
    assert adapter(Fact("gpu_available")).value is False
    with pytest.raises(ValueError, match="Missing recorded host fact: physical_memory"):
        adapter(Fact("physical_memory"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("gpu_available", "false"),
        ("has_uring", "false"),
        ("ss_available", 0),
        ("euid", "0"),
        ("euid", False),
        ("memory_kib", [1]),
        ("memory_kib", [True, 2]),
        ("memory_kib", ["1", 2]),
        ("memory_kib", {}),
        ("gpu_memory", 16384),
        ("filesystem", None),
        ("gpu_processes", []),
        ("physical_memory", True),
        ("physical_memory", "not an integer"),
        ("disk_free", 1.5),
        ("default_threads", False),
    ],
)
def test_requested_host_fact_type_is_explicit(name, value):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()), {name: value}, run=forbidden, replace=forbidden, python="unused"
    )
    with pytest.raises(ValueError, match="Recorded host fact " + name + ":"):
        adapter(Fact(name))


def test_unused_malformed_host_fact_is_not_read(tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()),
        {"gpu_available": "false", "euid": "0", "platform_system": "Darwin"},
        run=forbidden,
        replace=forbidden,
        python="unused",
    )
    assert adapter(Fact("is_file", (str(tmp_path / "missing"),))).value is False
    assert adapter(Fact("platform_system")).value == "Darwin"


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("gpu_memory", "[N/A]", "[N/A]"),
        ("euid", 0, 0),
        ("memory_kib", [16, 14], [16, 14]),
        ("default_threads", 8, "8"),
        ("default_threads", "8", "8"),
        ("physical_memory", "34359738368\n", "34359738368\n"),
        ("disk_free", 64, "64"),
    ],
)
def test_recorded_host_valid_representations(name, value, expected):
    def forbidden(*a, **kw):
        raise AssertionError("no effect")

    adapter = RecordedArtifacts(
        LaunchInput("/repo", ()), {name: value}, run=forbidden, replace=forbidden, python="unused"
    )
    assert adapter(Fact(name)).value == expected
