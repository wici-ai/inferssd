"""Explicit deployment resources; paths are relative to the descriptor, never cwd."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from inferssd.configuration.settings import PLATFORMS


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
