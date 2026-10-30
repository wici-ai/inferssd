"""Production model routes bound to the settings actually passed to Engines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from inferssd.configuration.derive import Selection, SelectionConflict, selection
from inferssd.configuration.settings import Snapshot
from inferssd.launch.types import Threads
from inferssd.model.machine import Machine
from inferssd.model.machine_types import ArtifactReader


class TypedMachine(Machine):
    def __init__(
        self,
        artifacts: ArtifactReader,
        *,
        base: Snapshot,
        threads: Threads,
        native_ubatch: int,
        host_threads: Callable[[], Optional[int]],
    ) -> None:
        super().__init__(artifacts, host_threads=host_threads)
        self._selections: dict[tuple[int, str], Selection] = {}
        self._native_ubatches: dict[str, int] = {}
        for tier, placement in self.configs:
            suffix = placement.partition("@")[2]
            ubatch = int(suffix) if suffix else native_ubatch
            try:
                chosen = selection(
                    self, tier, placement, base, host_threads, native_ubatch=ubatch, threads=threads
                )
            except SelectionConflict:
                continue
            self._selections[tier, placement] = chosen
            self._native_ubatches[placement] = ubatch
        self.configs = list(self._selections)

    def selected(self, tier: int, placement: str) -> Selection:
        return self._selections[tier, placement]

    def route(self, p: str) -> Optional[float]:
        chosen = next(value for (_, name), value in self._selections.items() if name == p)
        maximum = chosen.settings.at("prefill.cpu_max_tokens").value
        assert isinstance(maximum, int)
        return None if maximum == (1 << 32) - 1 else float(maximum)

    def ubatch(self, p: str) -> Optional[int]:
        return self._native_ubatches[p]
