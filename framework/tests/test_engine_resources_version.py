"""scripts/engine-resources.py names a build's version, <yymmdd>.<build>, from its history."""

import json
import os
import shutil
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
# 14:42 in UTC+8 on 2026-11-19 is 06:42 UTC, the same day.
DAY = "2026-11-19T14:42:55+08:00"


def git(repository, *arguments, date=DAY):
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
        env={**os.environ, **IDENTITY, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
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


def commits(clone, count, date=DAY):
    for number in range(count):
        git(clone[0], "commit", "-q", "--allow-empty", "-m", f"change {number}", date=date)


def test_every_build_is_named_by_its_day_and_build_number(clone):
    commits(clone, 2)
    assert resources(clone) == ("261119.3", None)


def test_the_day_is_the_commits_in_utc(clone):
    commits(clone, 1, date="2026-11-19T23:30:00-05:00")
    assert resources(clone) == ("261120.2", None)


def test_a_tagged_commit_is_its_release(clone):
    git(clone[0], "tag", "261119.1")
    assert resources(clone) == ("261119.1", "261119.1")
    # A tag of the old form, or of another build, does not make it one.
    commits(clone, 1)
    git(clone[0], "tag", "v0.1.5")
    git(clone[0], "tag", "261119.1-rc")
    assert resources(clone) == ("261119.2", None)


def test_the_release_workflow_names_the_tag_it_is_about_to_create(clone):
    commits(clone, 1)
    assert resources(clone, INFERSSD_RELEASE_TAG="261119.2") == ("261119.2", "261119.2")
    refused = resources(clone, INFERSSD_RELEASE_TAG="261119.1")
    assert refused.returncode != 0 and "not this build's version" in refused.stderr


def test_without_git_metadata_there_is_no_version(clone, tmp_path):
    shutil.rmtree(clone[0] / ".git")
    assert resources(clone, GIT_CEILING_DIRECTORIES=str(tmp_path)) == (None, None)
