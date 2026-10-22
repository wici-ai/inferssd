"""Versioned immutable device observations; no engine settings are defined here."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Union, cast


class ContractError(ValueError):
    """An artifact is outside this version's explicit input contract."""


def mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{field} must be an object with string keys")
    return cast("Mapping[str, object]", value)


def numeric(value: object, field: str) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ContractError(f"{field} exceeds the finite numeric range") from error
    if not finite:
        raise ContractError(f"{field} must be finite")
    return value


def number(value: object, field: str) -> float:
    return float(numeric(value, field))


def integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer")
    return value


def sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be an array")
    return cast("list[object]", value)


def string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string")
    return value


def optional_number(value: object, field: str) -> Optional[float]:
    return None if value is None else number(value, field)


@dataclass(frozen=True)
class CPU:
    index: int
    capacity: float
    max_khz: float


@dataclass(frozen=True)
class Memory:
    total: float
    available: float


@dataclass(frozen=True)
class GPU:
    integrated: Optional[bool]
    read_gbs: Optional[float]
    h2d_pinned_gbs: Optional[float]
    h2d_pageable_gbs: Optional[float]


@dataclass(frozen=True)
class SSDDepth:
    depth: int
    gibs: float
    lat_ms: float
    gibs_runs: tuple[float, ...]


@dataclass(frozen=True)
class CoreSet:
    name: str
    cpus: tuple[int, ...]


@dataclass(frozen=True)
class LaneFit:
    fixed_us: float
    per_route_us: float
    at150_us: float
    load: str


LaneValue = Union[LaneFit, str, int, float]


@dataclass(frozen=True)
class Lanes:
    # Preserve the original first-insertion order: a later busy sample replaces
    # its lane without moving that lane after the common shape metadata.
    fields: tuple[tuple[str, LaneValue], ...]

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in self.fields:
            result[key] = (
                {
                    "fixed_us": value.fixed_us,
                    "per_route_us": value.per_route_us,
                    "at150_us": value.at150_us,
                    "load": value.load,
                }
                if isinstance(value, LaneFit)
                else value
            )
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Lanes:
        fields: list[tuple[str, LaneValue]] = []
        for key, raw in data.items():
            if key in ("cpu", "gpu", "gpu_host"):
                row = mapping(raw, key)
                fields.append(
                    (
                        key,
                        LaneFit(
                            number(row["fixed_us"], key),
                            number(row["per_route_us"], key),
                            number(row["at150_us"], key),
                            string(row["load"], key),
                        ),
                    )
                )
            elif key in ("latent", "hidden", "threads"):
                fields.append((key, integer(raw, key)))
            elif key == "format":
                fields.append((key, string(raw, key)))
            elif key == "copy_gbps":
                fields.append((key, number(raw, key)))
            else:
                raise ContractError(f"unknown lane field: {key}")
        return cls(tuple(fields))


@dataclass(frozen=True)
class DeviceProfile:
    schema: int
    expert_read_bytes: Optional[int]
    bytes_per_role: Optional[tuple[tuple[str, int], ...]]
    cpus: tuple[CPU, ...]
    mem_gib: Memory
    gpu: GPU
    ssd: tuple[SSDDepth, ...]
    ssd_knee_qd: Optional[int]
    expert_files: tuple[tuple[str, float], ...]
    slow_files: tuple[str, ...]
    coresets: tuple[CoreSet, ...]
    lanes: Optional[Lanes]

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 2:
            raise ContractError(f"unsupported device schema: {self.schema!r}")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "expert_read_bytes": self.expert_read_bytes,
            "bytes_per_role": None if self.bytes_per_role is None else dict(self.bytes_per_role),
            "cpus": {
                str(c.index): {"capacity": c.capacity, "max_khz": c.max_khz} for c in self.cpus
            },
            "mem_gib": {"total": self.mem_gib.total, "available": self.mem_gib.available},
            "gpu": {
                "integrated": self.gpu.integrated,
                "read_gbs": self.gpu.read_gbs,
                "h2d_pinned_gbs": self.gpu.h2d_pinned_gbs,
                "h2d_pageable_gbs": self.gpu.h2d_pageable_gbs,
            },
            "ssd": {
                str(s.depth): {"gibs": s.gibs, "lat_ms": s.lat_ms, "gibs_runs": list(s.gibs_runs)}
                for s in self.ssd
            },
        }
        if self.ssd_knee_qd is not None:
            result["ssd_knee_qd"] = self.ssd_knee_qd
        result.update(
            expert_files=dict(self.expert_files),
            slow_files=list(self.slow_files),
            coresets={c.name: {"cpus": list(c.cpus)} for c in self.coresets},
            lanes=None if self.lanes is None else self.lanes.to_mapping(),
        )
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=1, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> DeviceProfile:
        return cls.from_mapping(mapping(json.loads(text), "device"))

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> DeviceProfile:
        if type(data.get("schema")) is not int or data.get("schema") != 2:
            raise ContractError(f"unsupported device schema: {data.get('schema')!r}")
        try:
            cpus = []
            for key, raw in mapping(data["cpus"], "cpus").items():
                row = mapping(raw, "cpu")
                cpus.append(
                    CPU(
                        int(key),
                        number(row["capacity"], "capacity"),
                        number(row["max_khz"], "max_khz"),
                    )
                )
            mem = mapping(data["mem_gib"], "mem_gib")
            gpu = mapping(data["gpu"], "gpu")
            integrated = gpu["integrated"]
            if integrated is not None and not isinstance(integrated, bool):
                raise ContractError("gpu.integrated must be boolean or null")
            ssd = []
            for key, raw in mapping(data["ssd"], "ssd").items():
                row = mapping(raw, "ssd depth")
                ssd.append(
                    SSDDepth(
                        int(key),
                        number(row["gibs"], "gibs"),
                        number(row["lat_ms"], "lat_ms"),
                        tuple(
                            number(n, "gibs_runs") for n in sequence(row["gibs_runs"], "gibs_runs")
                        ),
                    )
                )
            roles = data["bytes_per_role"]
            read_bytes = data["expert_read_bytes"]
            knee = data.get("ssd_knee_qd")
            coresets = tuple(
                CoreSet(
                    key,
                    tuple(
                        integer(c, "cpu index")
                        for c in sequence(mapping(raw, "coreset")["cpus"], "cpus")
                    ),
                )
                for key, raw in mapping(data["coresets"], "coresets").items()
            )
            lanes = data["lanes"]
            return cls(
                2,
                None if read_bytes is None else integer(read_bytes, "expert_read_bytes"),
                None
                if roles is None
                else tuple(
                    (key, integer(n, "bytes_per_role"))
                    for key, n in mapping(roles, "bytes_per_role").items()
                ),
                tuple(cpus),
                Memory(number(mem["total"], "total"), number(mem["available"], "available")),
                GPU(
                    integrated,
                    optional_number(gpu["read_gbs"], "read_gbs"),
                    optional_number(gpu["h2d_pinned_gbs"], "h2d_pinned_gbs"),
                    optional_number(gpu["h2d_pageable_gbs"], "h2d_pageable_gbs"),
                ),
                tuple(ssd),
                None if knee is None else integer(knee, "ssd_knee_qd"),
                tuple(
                    (key, number(n, "file rate"))
                    for key, n in mapping(data["expert_files"], "expert_files").items()
                ),
                tuple(string(s, "slow file") for s in sequence(data["slow_files"], "slow_files")),
                coresets,
                None if lanes is None else Lanes.from_mapping(mapping(lanes, "lanes")),
            )
        except KeyError as error:
            raise ContractError(f"missing device field: {error.args[0]}") from error
