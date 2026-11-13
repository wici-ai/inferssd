# SPDX-License-Identifier: Apache-2.0
"""Reserved artifact names are occupied even when symlink targets are missing."""

import os
from pathlib import Path

import pytest

from inferssd.platform import run_directory


def symlink(path, target):
    try:
        path.symlink_to(target)
    except OSError as error:
        if os.name != "nt":
            raise
        pytest.skip(f"this host cannot create a symbolic link: {error}")


@pytest.mark.parametrize("name", ["server.log", "settings.ssd", "STOP"])
def test_existing_dangling_artifact_prevents_claim(tmp_path, name):
    directory = tmp_path / "run"
    directory.mkdir()
    target = tmp_path / "missing-target"
    artifact = directory / name
    symlink(artifact, target)
    assert artifact.is_symlink() and not artifact.exists()

    with pytest.raises(FileExistsError, match="artifact already exists"):
        run_directory.reserve(str(directory))

    assert artifact.is_symlink()
    assert not target.exists()
    assert not (directory / ".inferssd-owner").exists()


def test_artifact_symlink_cycle_prevents_claim(tmp_path):
    artifact = tmp_path / "server.log"
    symlink(artifact, artifact.name)
    with pytest.raises(FileExistsError, match="artifact already exists"):
        run_directory.reserve(str(tmp_path))
    assert artifact.is_symlink()


def test_dangling_artifact_appearing_during_claim_is_rechecked(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    target = tmp_path / "missing-target"
    symlink(tmp_path / "symlink-check", target)
    real_open = run_directory.os.open

    def open_and_publish_artifact(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if Path(path).name == ".inferssd-owner":
            try:
                (directory / "server.log").symlink_to(target)
            except BaseException:
                os.close(fd)
                raise
        return fd

    monkeypatch.setattr(run_directory.os, "open", open_and_publish_artifact)
    with pytest.raises(FileExistsError, match="artifact appeared while claiming"):
        run_directory.reserve(str(directory))
    assert (directory / "server.log").is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_regular_artifact_still_prevents_claim(tmp_path, kind):
    artifact = tmp_path / "server.log"
    if kind == "file":
        artifact.write_text("existing log", encoding="utf-8")
    else:
        artifact.mkdir()
    with pytest.raises(FileExistsError, match="artifact already exists"):
        run_directory.reserve(str(tmp_path))
    if kind == "file":
        assert artifact.read_text(encoding="utf-8") == "existing log"
    else:
        assert artifact.is_dir()


def test_unreserved_entry_does_not_block_an_empty_artifact_directory(tmp_path):
    note = tmp_path / "notes-link"
    symlink(note, "missing-notes")
    claim = run_directory.reserve(str(tmp_path))
    claim.verify()
    assert note.is_symlink()
