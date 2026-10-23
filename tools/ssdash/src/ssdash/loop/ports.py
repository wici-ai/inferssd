"""Consumed interfaces only: no process, HTTP, filesystem or host defaults.

Legacy records remain unvalidated mapping views. Implementing these ports does
not confer ownership of native processes; a later execution adapter must do so.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import BinaryIO, Optional, Protocol, TextIO

from ssdash.model.machine_types import Feedback, Prediction
from ssdash.model.prefill_model import PrefillModel
from ssdash.model.routing_replay import Replay, Routing, Rows
from ssdash.telemetry.types import ArmFiles, Record

Segments = list[list[list[list[int]]]]
Point = tuple[float, float, str]


class FileStatus(Protocol):
    @property
    def st_size(self) -> int: ...

    @property
    def st_mtime_ns(self) -> int: ...


class ObservationFiles(ArmFiles, Protocol):
    def exists(self, path: str) -> bool: ...

    def stat(self, path: str) -> FileStatus: ...

    def open_binary(self, path: str) -> BinaryIO: ...


class Effects(Protocol):
    """Operations called at the original Controller sites, without eager effects."""

    @property
    def stdout(self) -> TextIO: ...

    def open(self, path: str, mode: str = "r") -> TextIO: ...

    def makedirs(self, path: str, *, exist_ok: bool = False) -> None: ...

    def getenv(self, name: str, default: str) -> str: ...

    def clock(self) -> float: ...


class Engine(Protocol):
    """The synchronous controller-facing handle; no implementation is provided."""

    dir: str
    port: int
    tier: int
    cpus: str
    envs: str
    ok: bool
    start_s: float

    def alive(self) -> bool: ...

    def stop(self) -> None: ...


class Model(Protocol):
    slope: float

    @property
    def P(self) -> Mapping[str, object]: ...

    @property
    def PF(self) -> PrefillModel: ...

    @property
    def routing(self) -> Routing: ...

    @property
    def configs(self) -> Sequence[tuple[int, str]]: ...

    def env(self, tier: int, p: str) -> str: ...

    def route(self, p: str) -> Optional[float]: ...

    def ubatch(self, p: str) -> Optional[int]: ...

    def price(self, rows: Rows, p: str) -> tuple[float, float]: ...

    def replay(self, key: Hashable, segs: Routing, tier: int) -> Replay: ...

    def sink(
        self, segs: Routing, tier: int, p: str, key: str = "conversation"
    ) -> tuple[float, float, float]: ...

    def predict(
        self,
        segs: Routing,
        tier: int,
        p: str,
        ctx: float,
        prompts: Iterable[int],
        decode: int,
        w0: int,
        w1: int,
        cold: bool,
        fb: Feedback,
        key: str = "conversation",
    ) -> list[Prediction]: ...


MachineFactory = Callable[[str], Model]
EngineFactory = Callable[[str, int, int, str, str, list[str]], Engine]
Post = Callable[[int, str, Record], object]
