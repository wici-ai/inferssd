"""Explicit filesystem adapter for existing probe reports; never invokes probes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional

from ssdash.probe.device_profile import ExpertMetadata, Reports


def load_reports(
    directory: Path,
    experts: Path,
    *,
    read_text: Callable[[Path], str] = Path.read_text,
    find: Callable[[Path, str], Sequence[Path]] = lambda path, pattern: list(path.glob(pattern)),
) -> tuple[Reports, ExpertMetadata]:
    """Supply ordered report bytes separately from the pure aggregation core."""

    def optional(path: Path) -> Optional[str]:
        try:
            return read_text(path)
        except OSError:
            return None

    def if_present(path: Path) -> Optional[str]:
        try:
            return read_text(path)
        except FileNotFoundError:
            return None

    queue = tuple(
        (str(p), optional(p))
        for pattern in ("reads-qd*.json", "fio-qd*.json")
        for p in find(directory, pattern)
    )
    files = if_present(directory / "files.txt")
    reads = []
    for line in files.splitlines() if files is not None else []:
        index, _ = line.split(maxsplit=1)
        for prefix in ("reads", "fio"):
            name = f"{prefix}-file{index}.json"
            reads.append((name, optional(directory / name)))
    return Reports(
        read_text(directory / "cpus.txt"),
        read_text(directory / "meminfo.txt"),
        read_text(directory / "memprobe.txt"),
        read_text(directory / "coresets.txt"),
        queue,
        files,
        tuple(reads),
        if_present(directory / "lanes.txt"),
    ), ExpertMetadata.from_json(read_text(experts / "experts.json"))
