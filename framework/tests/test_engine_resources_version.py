"""scripts/engine-resources.py names a build's version from the clone's release tags."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in server is a shell script"
)
IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def git(repository, *arguments):
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        capture_output=True,
        env={**os.environ, **IDENTITY},
    )


@pytest.fixture
def clone(tmp_path):
    engine = tmp_path / "engine"
    (engine / "models/templates").mkdir(parents=True)
    for name in ("deepseek-ai-DeepSeek-V4.1.jinja", "muse-spark.jinja"):
        (engine / "models/templates" / name).write_text("")
    git(engine, "init", "-q")
    git(engine, "commit", "-q", "--allow-empty", "-m", "first")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    server = build / "bin/inferssd-server"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    (build / "CMakeCache.txt").write_text("")
    return engine, build


def resources(clone, **environ):
    engine, build = clone
    output = build / "resources.json"
    inherited = {k: v for k, v in os.environ.items() if k != "INFERSSD_RELEASE_TAG"}
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/engine-resources.py"),
            "--build",
            str(build),
            "--output",
            str(output),
            "--engine",
            str(engine),
        ],
        env={**inherited, **environ},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return result
    value = json.loads(output.read_text())
    return value["version"], value["tag"]


def test_a_tagged_commit_is_its_release(clone):
    git(clone[0], "tag", "v0.1.5")
    assert resources(clone) == ("0.1.5", "v0.1.5")


def test_a_later_commit_is_the_next_patch_in_development(clone):
    git(clone[0], "tag", "v0.1.5")
    for message in ("second", "third", "fourth"):
        git(clone[0], "commit", "-q", "--allow-empty", "-m", message)
    assert resources(clone) == ("0.1.6.dev3", None)


def test_the_release_workflow_names_the_tag_it_is_about_to_create(clone):
    git(clone[0], "tag", "v0.1.5")
    git(clone[0], "commit", "-q", "--allow-empty", "-m", "second")
    assert resources(clone, INFERSSD_RELEASE_TAG="v0.2.0") == ("0.2.0", "v0.2.0")
    refused = resources(clone, INFERSSD_RELEASE_TAG="0.2.0")
    assert refused.returncode != 0 and "vX.Y.Z" in refused.stderr


def test_without_a_release_tag_there_is_no_version(clone):
    git(clone[0], "tag", "nightly")
    assert resources(clone) == (None, None)
