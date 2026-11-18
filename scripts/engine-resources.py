#!/usr/bin/env python3
"""Describe one built engine for an installed framework without source discovery."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# The framework beside this script reads a build's GPU backends, whether or not it is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework" / "src"))
from inferssd.launch.resources import cmake_gpu_backends  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parents[1] / "engine")
args = parser.parse_args()
platform = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
executable = ".exe" if platform == "windows" else ""
binary = (args.build / ("bin/inferssd-server" + executable)).resolve(strict=True)
engine = args.engine.resolve(strict=True)
templates = {
    "dsv": (engine / "models/templates/deepseek-ai-DeepSeek-V4.1.jinja").resolve(strict=True),
    "muse": (engine / "models/templates/muse-spark.jinja").resolve(strict=True),
}
# Only inspect the schema; never start inference or probe a device.
subprocess.run([str(binary), "--ssd-schema"], check=True, stdout=subprocess.DEVNULL)
# A source archive has no Git metadata to name the version, the revision, the build number (the
# commit count, which needs the history the installer's blobless clone has), the day or the branch.


def git(*arguments: str) -> str:
    return subprocess.run(["git", "-C", str(engine), *arguments], check=True,
                          capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout.strip()


TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def released() -> tuple[Optional[str], Optional[str]]:
    """The engine's version and, for a release, its tag.

    A release is the version its tag names: the Release workflow passes the tag it is about to
    create in INFERSSD_RELEASE_TAG, and a clone of a released commit has the tag itself. Any other
    build is the next patch's development version, counted from the latest tag before it as
    setuptools-scm counts: 0.1.6.dev3 three commits after v0.1.5. Without a tag, none.
    """
    tag = os.environ.get("INFERSSD_RELEASE_TAG", "")
    if tag:
        if not TAG.fullmatch(tag):
            sys.exit(f"engine-resources: INFERSSD_RELEASE_TAG {tag!r} is not a vX.Y.Z tag")
        return tag[1:], tag
    try:
        described = git("describe", "--tags", "--long", "--match", "v[0-9]*.[0-9]*.[0-9]*", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None, None
    match = re.fullmatch(r"(v[^-]+)-([0-9]+)-g[0-9a-f]+", described)
    found = TAG.fullmatch(match[1]) if match else None
    if not match or not found:
        return None, None
    if match[2] == "0":
        return match[1][1:], match[1]
    major, minor, patch = (int(part) for part in found.groups())
    return f"{major}.{minor}.{patch + 1}.dev{match[2]}", None


try:
    revision = git("rev-parse", "HEAD") or "unknown"
    build: Optional[int] = int(git("rev-list", "--count", "HEAD"))
    date: Optional[str] = git("show", "-s", "--format=%cd", "--date=format-local:%Y-%m-%d", "HEAD")
    branch: Optional[str] = git("rev-parse", "--abbrev-ref", "HEAD")
except (OSError, subprocess.CalledProcessError, ValueError):
    revision, build, date, branch = "unknown", None, None, None
version, tag = released()
value = {
    "schema": "inferssd-engine 1", "binary": str(binary), "revision": revision,
    "version": version, "tag": tag, "build": build, "date": date, "branch": branch,
    "capabilities": {
        "settings": "inferssd-settings 1", "telemetry": 1, "platform": platform,
        "gpu_backends": list(cmake_gpu_backends(
            (args.build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace"))),
    },
    "templates": {name: str(path) for name, path in templates.items()},
    "probes": {name: str(path.resolve()) for name, path in {
        "io": args.build / ("bin/inferssd-ioreplay" + executable),
        "memory": args.build / ("bin/inferssd-memprobe" + executable),
        "ssd": args.build / ("bin/inferssd-ssdprobe" + executable),
        "cpu": args.build / ("bin/inferssd-cpuprobe" + executable),
    }.items() if path.is_file()},
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(value, indent=2) + "\n")
