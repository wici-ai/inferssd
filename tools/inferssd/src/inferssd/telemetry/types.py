"""Typing views of legacy observations; these do not validate or copy input data."""

from __future__ import annotations

from typing import Any, Optional, Protocol, TextIO, TypedDict

# Legacy JSON fields are heterogeneous and retain their original permissive
# behavior. Any here is a view of that unversioned boundary, not a schema claim.
Record = dict[str, Any]
TokenKey = tuple[tuple[int, int], int]
TraceLayer = tuple[list[int], list[bool], list[bool]]


class ArmFiles(Protocol):
    def open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO: ...

    def gzip_open(self, path: str, mode: str = "r", *, errors: Optional[str] = None) -> TextIO: ...


class ArmResult(TypedDict):
    cache_load: Optional[Record]
    new_slot_bytes: float
    new_slot_windows: list[float]
    load_s: Optional[float]
    segments: list[list[dict[int, TraceLayer]]]
    groups: list[list[TokenKey]]
    window: list[TokenKey]
    lprof: dict[tuple[tuple[int, int], int, int], tuple[float, float]]
    prefill: list[Record]
    trace: dict[TokenKey, dict[int, TraceLayer]]
    prof: dict[TokenKey, list[float]]
    admission: list[Record]
    threads: Optional[Record]
    host: Optional[Record]
    tier_slots_max: Optional[float]
    tier_slots_first: Optional[int]
    tier_bytes: Optional[tuple[float, Optional[float]]]
    mrc: Any
    tier_min_count: Optional[float]
    lio: dict[TokenKey, list[float]]
    client: list[Record]
    pf_graphs: list[Record]
    memory: Record
    checkpoint_align: Optional[int]
    ledger: Optional[Record]
    prefetch: Optional[Record]
    ram_slots: Optional[int]
    decode_place: Optional[Record]
    host_register: Optional[Record]
    prefill_engine: Optional[Record]
