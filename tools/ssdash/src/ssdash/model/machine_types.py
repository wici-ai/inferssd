"""Typing views of legacy model inputs; no validation, copying or normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Protocol, TypedDict

from ssdash.model.prefill_calibration import CalibratedModel
from ssdash.model.routing_replay import Routing
from ssdash.plan.settings_types import KnownWorkload
from ssdash.plan.settings_types import Placement as SettingsPlacement


class ArtifactReader(Protocol):
    def read_json(self, name: str) -> object: ...

    def exists(self, name: str) -> bool: ...


class OutsideFit(TypedDict):
    b_ms_per_ktok: float
    positions: Sequence[float]
    mean_ms: float
    mean_position: float


class Measured(TypedDict):
    decode_tps_window: float
    hits: Mapping[str, float]


class Placement(SettingsPlacement):
    tier_slots: int
    admission_min_count: Optional[float]
    t_out_fit: OutsideFit
    layer_table: Mapping[str, Sequence[float]]
    measured: Measured


class Insitu(TypedDict):
    placements: Mapping[str, Placement]
    reference: str
    pf_k: int
    skip: int
    topk: int


class ModelArtifact(TypedDict):
    routing: Routing


class SavedDerived(TypedDict, total=False):
    pending: Optional[Iterable[str]]


class SavedPlan(TypedDict, total=False):
    derived: Optional[SavedDerived]
    workload: Optional[KnownWorkload]


class Feedback(TypedDict):
    k_cpu: Mapping[str, float]
    out0: Mapping[str, float]
    k_drive: float
    k_bytes: float
    pf_cpu: Mapping[str, float]
    pf_drive: float


class Turn(TypedDict):
    prefill_s: float
    decode_tps: float


class Prediction(Turn):
    hit_v: float


class PlanRow(TypedDict):
    tier: int
    cpus: str
    total_s: float
    prefill_s: float
    decode_tps: float
    hit_v: float
    turns: list[Turn]
    env: str


class Rejected(TypedDict):
    tier: int
    cpus: str
    total_s: float
    why: list[str]


class Calibration(TypedDict):
    decode_tps_model: float
    decode_tps_measured: float
    hit_v_model: float
    hit_v_measured: float


class PlannedWorkload(KnownWorkload):
    window: str


class DerivedPlan(TypedDict):
    pending: Optional[Iterable[str]]
    plan: list[dict[str, object]]


class PlanArtifact(TypedDict):
    reference: str
    budget_tier_slots: int
    admission_min_count: float
    t_out_slope_ms_per_ktok: float
    prefill_model: CalibratedModel
    workload: PlannedWorkload
    plan: PlanRow
    other: Optional[PlanRow]
    table: list[PlanRow]
    rejected: list[Rejected]
    calibration: Mapping[str, Calibration]
    derived: DerivedPlan
    plan_seconds: float
