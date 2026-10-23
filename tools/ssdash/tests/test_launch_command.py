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
        "launch",
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
        main(["launch", "--platform", "linux", "--repo", "/does-not-exist"])
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
