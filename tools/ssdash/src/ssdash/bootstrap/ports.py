"""Legacy HTTP/file protocol views; no network, engine or host defaults."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Protocol, TextIO

from ssdash.telemetry.types import Record

if TYPE_CHECKING:
    from urllib.request import Request

History = list[Record]
OpenText = Callable[..., TextIO]
Emit = Callable[..., None]


class Response(Protocol):
    def __enter__(self) -> Response: ...
    def __exit__(
        self,
        kind: Optional[type[BaseException]],
        value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def read(self) -> bytes: ...


class Transport(Protocol):
    def __call__(self, request: Request, *, timeout: int) -> Response: ...


class Turn(Protocol):
    def __call__(
        self,
        port: int,
        history: History,
        max_tokens: int,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> Record: ...


class FileDevice(Protocol):
    @property
    def st_dev(self) -> int: ...


Post = Callable[[int, str, Record], Record]
