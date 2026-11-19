"""What `inferssd --version` says: the version, and the build this engine was made from.

scripts/engine-resources.py reads them from the clone at install time. The version is
<yymmdd>.<build>, 261119.300: the day the commit's and the build number the branch's commit count.
A release is the version its tag names, the same 261119.300. With the commit, the version names the
build, <yymmdd>.<build>.<commit>. Whether an engine is one the app works with is the app's own
call, by this build number. The GPU backends the build was made with come from the same file, so a
caller can tell the Windows packages, one per backend, apart.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Optional

from inferssd import __version__
from inferssd.launch.resources import GPU_BACKENDS, installed_resources
from inferssd.launch.typed import PROFILES


def label(build: Optional[int], date: Optional[str], revision: Optional[str]) -> Optional[str]:
    """<yymmdd>.<build>.<commit>, or None without a build number, a day and a commit."""
    if not build or not date or not revision or revision == "unknown" or len(date) < 10:
        return None
    return f"{date[2:4]}{date[5:7]}{date[8:10]}.{build}.{revision[:7]}"


VERSION = re.compile(r"[0-9]{6}\.[1-9][0-9]*")


def report(environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """The version and, when the installer recorded them, the release's tag (None for any other
    build), the build, commit, day, branch and the GPU backends the build has
    (`capabilities.gpu_backends`, empty for a CPU-only build). Without a recorded version, the
    framework's own stands."""
    value: dict[str, Any] = {
        "version": __version__,
        "tag": None,
        "build": None,
        "revision": None,
        "date": None,
        "branch": None,
        "label": None,
        "capabilities": {"model_profiles": list(PROFILES)},
    }
    try:
        data = json.loads(installed_resources(environ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return value
    if not isinstance(data, dict):
        return value
    build = data.get("build")
    revision = data.get("revision")
    version = data.get("version")
    if isinstance(version, str) and version:
        value["version"] = version
    tag = data.get("tag")
    if isinstance(tag, str) and VERSION.fullmatch(tag) and tag == value["version"]:
        value["tag"] = tag
    for key, kind in (("build", int), ("revision", str), ("date", str), ("branch", str)):
        item = data.get(key)
        if isinstance(item, kind) and not isinstance(item, bool) and item != "unknown":
            value[key] = item
    capabilities = data.get("capabilities")
    backends = capabilities.get("gpu_backends") if isinstance(capabilities, dict) else None
    if isinstance(backends, list) and all(
        isinstance(item, str) and item in GPU_BACKENDS for item in backends
    ):
        value["capabilities"]["gpu_backends"] = list(backends)
    value["label"] = label(
        build if isinstance(build, int) else None,
        value["date"],
        revision if isinstance(revision, str) else None,
    )
    return value


def line(value: Mapping[str, Any]) -> str:
    """The line for people: `inferssd 261119.300 (1a2b3c4)`, with the commit when there is one."""
    text = f"inferssd {value['version']}"
    return f"{text} ({value['revision'][:7]})" if value.get("revision") else text
