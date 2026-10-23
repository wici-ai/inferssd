"""Production launch hook for the retained controller decision algorithm."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from inferssd.configuration.settings import Snapshot
from inferssd.loop.controller import Controller
from inferssd.loop.ports import Engine
from inferssd.model.machine import Machine


class TypedController(Controller):
    def __init__(
        self,
        *args: Any,
        configuration: Callable[[Machine, int, str], Snapshot],
        typed_engine: Callable[[str, int, int, str, Sequence[str], Snapshot], Engine],
        keeps_history: bool = False,
        **kwargs: Any,
    ) -> None:
        self._configuration, self._typed_engine = configuration, typed_engine
        super().__init__(*args, **kwargs)
        self.keeps_history = keeps_history

    def _start(self, tier: int, cpus: str) -> None:
        # The historical Machine.env/Engine env-string port is never called.
        selected = self._configuration(cast(Machine, self.m), tier, cpus)
        directory = f"{self.out}/inst{len(self.insts)}"
        self.engine = self._typed_engine(directory, self.port, tier, cpus, self.args, selected)
        self.insts.append(directory)
        self.note(
            {
                "event": "start",
                "inst": len(self.insts) - 1,
                "tier": tier,
                "cpus": cpus,
                "settings": selected.document(),
                "ok": self.engine.ok,
                "start_s": round(self.engine.start_s, 1),
            }
        )
        if not self.engine.ok:
            self.engine.stop()
            raise RuntimeError(
                f"typed server did not come up in {tier},{cpus}; log: {directory}/server.log.gz"
            )
