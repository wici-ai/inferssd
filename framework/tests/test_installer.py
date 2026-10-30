# SPDX-License-Identifier: Apache-2.0
"""Exercise the piped installer with a synthetic toolchain, never compiling a model."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "scripts/install.sh"


@pytest.fixture
def installation(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    helper = tools / "helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "uname":
    print(os.environ.get("TEST_SYSTEM", "Darwin") if args == ["-s"] else "arm64")
elif name == "git":
    root = pathlib.Path(args[-1])
    (root / "framework").mkdir(parents=True)
    (root / "scripts").mkdir()
elif name == "cmake":
    with open(os.environ["CMAKE_LOG"], "a") as log:
        log.write(" ".join(args) + "\n")
    if os.environ.get("FAIL_BUILD"):
        sys.exit(1)
elif name in ("python3", "python"):
    if args[:2] == ["-m", "venv"]:
        bin = pathlib.Path(args[2]) / "bin"
        bin.mkdir(parents=True)
        (bin / "python").symlink_to(os.path.realpath(sys.argv[0]))
        app = bin / "inferssd"
        app.write_text("#!/bin/sh\nprintf '%s\\n' \"$INFERSSD_RESOURCES\"\n")
        app.chmod(0o755)
    elif args[:2] == ["-m", "pip"]:
        pass
    elif args and args[0] == "scripts/engine-resources.py":
        pathlib.Path(args[args.index("--output") + 1]).write_text("{}")
    else:
        sys.exit(subprocess.call([sys.executable, *args]))
"""
    )
    helper.chmod(0o755)
    for tool in ("uname", "git", "cc", "c++", "xcrun", "python3", "cmake", "nvcc"):
        (tools / tool).symlink_to(helper)
    home = tmp_path / "install with spaces"
    bindir = tmp_path / "bin with spaces"
    env = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        INFERSSD_HOME=str(home),
        INFERSSD_BIN_DIR=str(bindir),
        CMAKE_LOG=str(tmp_path / "cmake.log"),
    )
    return home, bindir, env


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_piped_install_and_upgrade(installation, system):
    home, bindir, env = installation
    env["TEST_SYSTEM"] = system
    result = subprocess.run(
        ["sh"], input=INSTALLER.read_text(), text=True, env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    app = bindir / "inferssd"
    previous = app.read_text()
    descriptor = subprocess.check_output([str(app), "serve", "--help"], env=env, text=True).strip()
    assert Path(descriptor).is_file()
    assert home.joinpath("current/engine-resources.json").is_file()
    assert "export PATH=" in result.stdout
    log = Path(env["CMAKE_LOG"]).read_text()
    assert ("inferssd-memprobe" in log) == (system == "Linux")
    result = subprocess.run(
        ["sh"],
        input=INSTALLER.read_text(),
        text=True,
        env=dict(env, FAIL_BUILD="1"),
        capture_output=True,
    )
    assert result.returncode != 0
    assert app.read_text() == previous
    assert Path(descriptor).is_file()
    assert len(list((home / "releases").iterdir())) == 1


def test_install_help_needs_no_toolchain():
    result = subprocess.run(["sh", str(INSTALLER), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "INFERSSD_REF" in result.stdout
