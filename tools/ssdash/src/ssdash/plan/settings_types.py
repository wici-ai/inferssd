"""Typing views for F's settings generator; these do not validate or normalize data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, Optional, Protocol, TypedDict, Union

from ssdash.model.prefill_calibration import CalibratedModel, PrefillPoint, ReadRate
from ssdash.model.routing_replay import Replay, Routing


class Setting(NamedTuple):
    name: str
    value: Union[int, float, str]
    source: str
    input: str
    pending: bool = False


class CoreSet(TypedDict):
    cpus: list[int]


class LaneFit(TypedDict):
    fixed_us: float
    per_route_us: float


class ProbeLanes(TypedDict, total=False):
    format: str
    latent: int
    hidden: int
    threads: int
    cpu: LaneFit
    gpu: LaneFit
    gpu_host: LaneFit
    copy_gbps: Optional[float]


class GPU(TypedDict, total=False):
    integrated: bool
    h2d_pinned_gbs: Optional[float]
    h2d_pageable_gbs: Optional[float]


class Device(TypedDict, total=False):
    coresets: Mapping[str, CoreSet]
    default_threads: Optional[Union[int, float, str]]
    lanes: Optional[ProbeLanes]
    gpu: Optional[GPU]
    ssd: Optional[Mapping[str, ReadRate]]
    ssd_knee_qd: Optional[int]
    bytes_per_role: Optional[Mapping[str, float]]


class Prefetch(TypedDict, total=False):
    predicted_r: Sequence[int]
    novel_r: Sequence[int]
    novel_routed_r: Sequence[int]
    windows: int
    layers: int
    readers: int
    window_us_p50: float
    window_us_mean: Optional[float]
    issued: Optional[int]
    reader_busy_us: Optional[float]
    spill_us: Optional[float]
    spills: Optional[int]
    started_r: Sequence[int]
    spill_us_r: Optional[Sequence[float]]


class Ledger(TypedDict, total=False):
    unified: bool
    margin_updates: int
    margin: float


class BootstrapPoint(PrefillPoint, total=False):
    layers_costed: float


class Placement(TypedDict, total=False):
    tier_slot_bytes: Optional[float]
    tier_base_bytes: Optional[float]
    prefill: Optional[Sequence[BootstrapPoint]]
    layer_beta: Optional[Sequence[float]]
    drive_us_per_mb: Optional[float]
    drive_other_bytes: Optional[float]
    pf_k: Optional[int]
    ram_slots: Optional[int]
    prefetch: Optional[Prefetch]
    checkpoint_align: Optional[int]
    ledger: Optional[Ledger]


class Workload(TypedDict):
    prompts: Sequence[int]
    decode: int


class KnownWorkload(Workload, total=False):
    known: bool


class Insitu(TypedDict):
    skip: int


class MachineView(Protocol):
    caps: Mapping[str, int]
    P: Mapping[str, Placement]
    ref: str
    dev: Device
    PF: CalibratedModel
    routing: Routing
    mc: float
    ins: Insitu

    def replay(self, key: tuple[str, int], segs: Routing, tier: int) -> Replay: ...


# Called only after an otherwise valid split placement lacks a truthy recorded
# default_threads. It may return None or raise, preserving the old failure path.
HostThreads = Callable[[], Optional[int]]
Spill = Union[float, list[Optional[float]], tuple[Optional[float], ...]]
