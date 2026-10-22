"""Explicit filesystem adapter for existing probe reports; never invokes probes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional

from ssdash.probe.device_profile import ExpertMetadata, Reports, read_report


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
        except (OSError, UnicodeDecodeError):
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
        name = f"reads-file{index}.json"
        primary = optional(directory / name)
        reads.append((name, primary))
        # The original `read_report(reads) or read_report(fio)` never reads a
        # fallback after a truthy primary, including a zero-bandwidth record.
        if not read_report(primary):
            name = f"fio-file{index}.json"
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
