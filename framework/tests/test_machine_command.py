# SPDX-License-Identifier: Apache-2.0
"""The profile command: measure, save, summarize and show, with a fake host and probes."""

from __future__ import annotations

import argparse
import json
import os

import pytest
from test_machine_identity import FakeHost, linux_tree, nvme_key
from test_machine_probes import FakeRunner, replies
from test_machine_profile import GIB, Clock, shards

from inferssd.cli import main
from inferssd.machine import command
from inferssd.machine.store import exclusive


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


def execute(argv, host, runner, tmp_path):
    out, progress = [], []
    code = command.run(
        parse(argv),
        host=host,
        runner=runner,
        environ={"XDG_STATE_HOME": str(tmp_path / "state")},
        home=tmp_path / "home",
        clock=Clock(),
        progress=progress.append,
        emit=out.append,
    )
    return code, out, progress


def test_profile_stays_available_when_unlisted(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    text = capsys.readouterr().out
    assert error.value.code == 0 and "measure this machine" not in text
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
    assert out[1].startswith(
        "identity: linux x86_64, Example CPU 9000, 4 logical CPUs, 31.2 GiB RAM"
    )
    names = [line.split(":", 1)[0] for line in out[2:]]
    assert names == ["cpu", "memory", "compute", "gpus", "storage"]
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

    code, shown, _ = execute(["--show"], host, FakeRunner({}), tmp_path)
    assert code == 0 and shown == out


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


def test_show_measures_nothing(tmp_path):
    host, files, _ = machine(tmp_path)
    with pytest.raises(ValueError, match="no machine profile at .*; run inferssd profile first"):
        execute(["--show"], host, FakeRunner({}), tmp_path)
    with pytest.raises(ValueError, match="--show measures nothing"):
        execute(["--show", "--model", files[0]], host, FakeRunner({}), tmp_path)


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
