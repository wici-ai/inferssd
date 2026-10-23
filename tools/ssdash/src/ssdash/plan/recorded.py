"""Only recorded artifacts are read; host defaults must be supplied explicitly."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ssdash.plan.settings_types import HostThreads


class DirectoryArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_json(self, name: str) -> object:
        with open(self.root / name) as stream:
            value: object = json.load(stream)
        return value

    def exists(self, name: str) -> bool:
        return os.path.exists(self.root / name)


def write_json(path: str, value: object) -> None:
    # Preserve plan.py's direct write and indent/no-final-newline format. Atomic
    # publication would be a separate behavior change, not an extraction detail.
    with open(path, "w") as stream:
        json.dump(value, stream, indent=1)


def explicit_host_threads(value: Optional[int]) -> HostThreads:
    def fallback() -> int:
        if value is None:
            raise ValueError(
                "split placement needs recorded default_threads or explicit --host-threads"
            )
        return value

    return fallback
