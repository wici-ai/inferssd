"""Caller-owned legacy arm cache; one session replaces one isolated F module."""

from __future__ import annotations

from ssdash.loop.ports import ObservationFiles
from ssdash.telemetry.legacy import ArmParser
from ssdash.telemetry.recorded import read_arm
from ssdash.telemetry.types import ArmResult


class ObservationSession:
    def __init__(self, *, files: ObservationFiles) -> None:
        self.files = files
        self.arms: dict[tuple[str, int, int, int], ArmResult] = {}
        self.live: dict[str, tuple[ArmParser, int]] = {}

    def read_arm_cached(self, d: str, skip: int) -> ArmResult:
        """read_arm without reading a log twice: a stopped server's log.gz is parsed once; a live
        server's plain log is read on from where the last call stopped.
        """
        gz = f"{d}/server.log.gz"
        if self.files.exists(gz):
            st = self.files.stat(gz)
            key = (d, skip, st.st_size, st.st_mtime_ns)
            if key not in self.arms:
                self.arms[key] = read_arm(d, skip, files=self.files)
            return self.arms[key]
        path = f"{d}/server.log"
        parser, offset = self.live.get(path, (None, 0))
        if parser is None:
            parser = ArmParser()
        with self.files.open_binary(path) as f:
            f.seek(offset)
            data = f.read()
        cut = data.rfind(b"\n") + 1  # whole lines only; a partial last line waits
        for line in data[:cut].decode("utf-8", "replace").splitlines(True):
            parser.feed(line)
        self.live[path] = (parser, offset + cut)
        return read_arm(d, skip, parser, files=self.files)
