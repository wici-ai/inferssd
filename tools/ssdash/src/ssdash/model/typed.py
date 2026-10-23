"""Production model routes bound to the settings actually passed to Engines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from ssdash.configuration.derive import SelectionConflict, selection
from ssdash.configuration.settings import Snapshot
from ssdash.model.machine import Machine
from ssdash.model.machine_types import ArtifactReader


class TypedMachine(Machine):
    def __init__(
        self,
        artifacts: ArtifactReader,
        *,
        base: Snapshot,
        native_ubatch: int,
        host_threads: Callable[[], Optional[int]],
    ) -> None:
        super().__init__(artifacts, host_threads=host_threads)
        self._settings: dict[tuple[int, str], Snapshot] = {}
        self._native_ubatches: dict[str, int] = {}
        for tier, placement in self.configs:
            suffix = placement.partition("@")[2]
            ubatch = int(suffix) if suffix else native_ubatch
            try:
                snapshot = selection(
                    self, tier, placement, base, host_threads, native_ubatch=ubatch
                )
            except SelectionConflict:
                continue
            self._settings[tier, placement] = snapshot
            self._native_ubatches[placement] = ubatch
        self.configs = list(self._settings)

    def settings(self, tier: int, placement: str) -> Snapshot:
        return self._settings[tier, placement]

    def route(self, p: str) -> Optional[float]:
        snapshot = next(value for (_, name), value in self._settings.items() if name == p)
        maximum = snapshot.at("prefill.cpu_max_tokens").value
        assert isinstance(maximum, int)
        return None if maximum == (1 << 32) - 1 else float(maximum)

    def ubatch(self, p: str) -> Optional[int]:
        return self._native_ubatches[p]
