"""Explicit execution policy, separate from model or planner calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class Policy:
    platform: str
    lock_path: str
    stop_path: str
    memory_budget_bytes: int
    host_reserve_bytes: int
    minimum_memlock_bytes: int
    max_compressor_growth_bytes: int
    max_swap_growth_bytes: int
    runtime_seconds: int
    cleanup_seconds: int

    @classmethod
    def read(cls, data: Mapping[str, object]) -> Policy:
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise ValueError("runtime policy fields differ: " + str(sorted(set(data) ^ expected)))
        for name in ("platform", "lock_path", "stop_path"):
            if not isinstance(data[name], str) or not data[name]:
                raise ValueError("runtime policy " + name + " must be nonempty text")
        if data["platform"] not in ("linux", "darwin"):
            raise ValueError("runtime policy platform must be linux or darwin")
        numeric = expected - {"platform", "lock_path", "stop_path"}
        for name in numeric:
            value = data[name]
            minimum = (
                1 if name in ("memory_budget_bytes", "runtime_seconds", "cleanup_seconds") else 0
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"runtime policy {name} must be an integer >= {minimum}")
        return cls(
            platform=str(data["platform"]),
            lock_path=str(data["lock_path"]),
            stop_path=str(data["stop_path"]),
            memory_budget_bytes=cast(int, data["memory_budget_bytes"]),
            host_reserve_bytes=cast(int, data["host_reserve_bytes"]),
            minimum_memlock_bytes=cast(int, data["minimum_memlock_bytes"]),
            max_compressor_growth_bytes=cast(int, data["max_compressor_growth_bytes"]),
            max_swap_growth_bytes=cast(int, data["max_swap_growth_bytes"]),
            runtime_seconds=cast(int, data["runtime_seconds"]),
            cleanup_seconds=cast(int, data["cleanup_seconds"]),
        )
