"""File operations for recorded legacy inputs, with no I/O during construction."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from typing import Optional, TextIO, cast

from inferssd.telemetry.types import ArmFiles


class LocalArmFiles:
    def open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO:
        return cast(TextIO, open(path, mode, errors=errors))

    def gzip_open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO:
        return cast(TextIO, gzip.open(path, mode, errors=errors))


LOCAL_FILES = LocalArmFiles()


def readable(path: str, files: ArmFiles) -> bool:
    # This is deliberately an open/close probe, not os.path.exists. The legacy
    # reader chooses the compressed source by readability before decoding it.
    try:
        files.open(path).close()
        return True
    except OSError:
        return False


def lines(path: str, files: ArmFiles) -> Iterator[str]:
    op = files.gzip_open if path.endswith(".gz") else files.open
    with op(path, "rt", errors="replace") as stream:
        yield from stream
