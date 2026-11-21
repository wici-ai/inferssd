#!/usr/bin/env python3
"""Describe one built engine for an installed framework without source discovery."""
import argparse
import json
import os
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
# The standalone --ssd-schema query exits before loading compute backends.
# Keep it standalone: mixed argument lists use normal backend-aware parsing.
subprocess.run([str(binary), "--ssd-schema"], check=True, stdout=subprocess.DEVNULL)
# A source archive has no Git metadata to name the version, the revision, the build number (the
# commit count, which needs the history the installer's blobless clone has), the day or the branch.


def git(*arguments: str) -> str:
    return subprocess.run(["git", "-C", str(engine), *arguments], check=True,
                          capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout.strip()


try:
    revision = git("rev-parse", "HEAD") or "unknown"
    build: Optional[int] = int(git("rev-list", "--count", "HEAD"))
    date: Optional[str] = git("show", "-s", "--format=%cd", "--date=format-local:%Y-%m-%d", "HEAD")
    branch: Optional[str] = git("rev-parse", "--abbrev-ref", "HEAD")
except (OSError, subprocess.CalledProcessError, ValueError):
    revision, build, date, branch = "unknown", None, None, None
# The version is <yymmdd>.<build>: the commit's day (UTC) and the commit count, 261119.300.
version: Optional[str] = f"{date[2:4]}{date[5:7]}{date[8:10]}.{build}" if build and date else None


def released() -> Optional[str]:
    """A release's tag, which is its version: the Release workflow passes the tag it is about to
    create in INFERSSD_RELEASE_TAG, and a clone of a released commit has the tag itself. Any other
    build has none."""
    tag = os.environ.get("INFERSSD_RELEASE_TAG", "")
    if tag:
        if tag != version:
            sys.exit(f"engine-resources: INFERSSD_RELEASE_TAG {tag!r} is not this build's version, {version!r}")
        return tag
    if not version:
        return None
    try:
        tags = git("tag", "--points-at", "HEAD").split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return version if version in tags else None


tag = released()
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
