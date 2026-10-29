"""Explicit deployment resources; paths are relative to the descriptor, never cwd."""

from __future__ import annotations

import json
import ntpath
import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferssd.configuration.settings import PLATFORMS

# A checkout's build directories, searched in this order: the presets first,
# then the conventional ad hoc ones.
BUILDS = {
    "linux": ("build-linux-cuda", "build-linux-cpu", "build"),
    "darwin": ("build-macos-metal", "build-mac", "build"),
    "windows": ("build-x64-windows-llvm-release", "build"),
}
# The machine probes by the name a resource descriptor's probes map gives them.
PROBES = {"ssd": "inferssd-ssdprobe", "cpu": "inferssd-cpuprobe", "memory": "inferssd-memprobe"}


@dataclass(frozen=True)
class EngineResources:
    binary: str
    templates: tuple[tuple[str, str], ...]
    probes: tuple[tuple[str, str], ...] = ()
    platform: Optional[str] = None
    revision: str = "unknown"

    def template(self, profile: str) -> str:
        try:
            return dict(self.templates)[profile]
        except KeyError as error:
            raise ValueError("engine resources have no template for " + profile) from error


def read_resources(path: Path) -> EngineResources:
    path = path.resolve(strict=True)
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "inferssd-engine 1":
        raise ValueError("expected inferssd-engine 1 resource descriptor")
    if set(data) - {"schema", "binary", "templates", "probes", "capabilities", "revision"}:
        raise ValueError("unknown engine resource field")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("settings") != "inferssd-settings 1":
        raise ValueError("engine resources must declare inferssd-settings 1")
    if capabilities.get("telemetry") != 1:
        raise ValueError("engine resources must declare telemetry version 1")
    platform = capabilities.get("platform")
    if platform is not None and platform not in PLATFORMS:
        raise ValueError("unsupported engine resource platform")
    revision = data.get("revision", "unknown")
    if not isinstance(revision, str) or not revision:
        raise ValueError("engine revision must be nonempty text")

    def filename(value: object) -> str:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("engine resource paths must be nonempty strings")
        return str((path.parent / value).resolve())

    def paths(name: str, required: bool) -> tuple[tuple[str, str], ...]:
        values = data.get(name, {})
        if not isinstance(values, dict) or required and not values:
            raise ValueError(name + " must be a nonempty path map")
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError(name + " must have nonempty string keys")
        return tuple((key, filename(value)) for key, value in values.items())

    return EngineResources(
        filename(data.get("binary")),
        paths("templates", True),
        paths("probes", False),
        platform,
        revision,
    )


def executable(platform: str, name: str) -> str:
    return name + ".exe" if platform == "windows" else name


def server_binary(
    platform: str,
    binary: Optional[str],
    resources: Optional[EngineResources],
    repo: str,
    is_file: Callable[[str], bool],
) -> Optional[str]:
    """The server executable: BINARY, else the descriptor's, else the first REPO build with one.

    Without one in any build, the first build's path is named, so the error
    names where it was expected. None when there is nowhere to look.
    """
    path = binary or (resources.binary if resources else None)
    if path or not repo:
        return path
    join = ntpath.join if platform == "windows" else posixpath.join
    server = executable(platform, "inferssd-server")
    candidates = [join(repo, name, "bin", server) for name in BUILDS[platform]]
    return next((candidate for candidate in candidates if is_file(candidate)), candidates[0])


def probe_paths(
    platform: str,
    binary: Optional[str],
    resources: Optional[EngineResources],
    repo: str,
    is_file: Callable[[str], bool],
) -> dict[str, str]:
    """Each probe found: next to the server BINARY, else in the descriptor, else in a REPO build.

    The repo's builds are searched in the order the server is. A probe found
    nowhere is left out.
    """
    join = ntpath.join if platform == "windows" else posixpath.join
    dirname = ntpath.dirname if platform == "windows" else posixpath.dirname
    named = dict(resources.probes) if resources else {}
    found = {}
    for name, program in PROBES.items():
        filename = executable(platform, program)
        candidates = [join(dirname(binary), filename)] if binary else []
        candidates += [named[name]] if name in named else []
        if repo:
            candidates += [join(repo, build, "bin", filename) for build in BUILDS[platform]]
        path = next((candidate for candidate in candidates if is_file(candidate)), None)
        if path is not None:
            found[name] = path
    return found
