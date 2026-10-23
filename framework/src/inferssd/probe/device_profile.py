"""Pure extraction of F's device profile arithmetic from recorded observations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, TypedDict, cast

from inferssd.contracts import DeviceProfile, integer, mapping, numeric


@dataclass(frozen=True)
class Reports:
    cpus: str
    meminfo: str
    memprobe: str
    coresets: str
    # Preserve the loader's reads-* then fio-* order and each list's order.
    queue_reads: tuple[tuple[str, Optional[str]], ...]
    files: Optional[str]
    file_reads: tuple[tuple[str, Optional[str]], ...]
    lanes: Optional[str]


@dataclass(frozen=True)
class ExpertMetadata:
    expert_read_bytes: Optional[int]
    bytes_per_role: Optional[tuple[tuple[str, int], ...]]

    @classmethod
    def from_json(cls, text: str) -> ExpertMetadata:
        data = mapping(json.loads(text), "experts")
        size = data.get("expert_read_bytes")
        roles = data.get("bytes_per_role")
        return cls(
            None if size is None else integer(size, "expert_read_bytes"),
            None
            if roles is None
            else tuple(
                (key, integer(value, "bytes_per_role"))
                for key, value in mapping(roles, "bytes_per_role").items()
            ),
        )


class SSDRow(TypedDict):
    gibs: float
    lat_ms: float
    gibs_runs: list[float]


def read_report(text: Optional[str]) -> Optional[Mapping[str, object]]:
    if text is None:
        return None
    # Match the original parser's missing/malformed JSON policy. Structural
    # TypeError must propagate rather than masquerading as a missing observation.
    try:
        data: object = json.loads(text[text.index("{") :])
        if not isinstance(data, dict):
            raise TypeError("read report must be an object")
        jobs: object = data["jobs"]
        if isinstance(jobs, dict):
            job: object = jobs[0]  # JSON keys are strings: legacy skips KeyError.
        elif isinstance(jobs, (list, str)):
            job = jobs[0]
        else:
            raise TypeError("jobs must be indexable")
        if not isinstance(job, dict):
            raise TypeError("job must be an object")
        row: object = job["read"]
    except (ValueError, KeyError, IndexError):
        return None
    if not row:
        return None
    if not isinstance(row, dict):
        raise TypeError("nonempty read observation must be an object")
    return cast(Mapping[str, object], row)


def lanes(text: Optional[str]) -> Optional[dict[str, object]]:
    if text is None:
        return None
    out: dict[str, object] = {}
    for line in text.splitlines(keepends=True):
        if line.startswith("[kimi_vram_group_fit] ") and "fixed_us=" in line:
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            key = "cpu" if kv["lane"] == "cpu" else "gpu_host" if kv["source"] == "host" else "gpu"
            out[key] = {
                "fixed_us": max(0.0, float(kv["fixed_us"])),
                "per_route_us": max(0.0, float(kv["per_route_us"])),
                "at150_us": float(kv["at150_us"]),
                "load": kv["load"],
            }
            out.update(
                format=kv["format"],
                latent=int(kv["latent"]),
                hidden=int(kv["hidden"]),
                threads=int(kv["threads"]),
            )
        elif line.startswith("[kimi_vram_group_copy] "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            busy = float(kv.get("busy_gbps", 0) or 0)
            out["copy_gbps"] = busy if busy > 0 else float(kv["idle_gbps"])
    return out if "gpu" in out or "cpu" in out else None


def aggregate(reports: Reports, experts: ExpertMetadata) -> DeviceProfile:
    """Compute device schema 2 without consulting files, environment, or the host."""
    dev: dict[str, object] = {
        "schema": 2,
        "expert_read_bytes": experts.expert_read_bytes,
        "bytes_per_role": None if experts.bytes_per_role is None else dict(experts.bytes_per_role),
    }
    cpus = {}
    for line in reports.cpus.splitlines():
        fields = line.split()
        cpus[str(int(fields[0]))] = {"capacity": float(fields[1]), "max_khz": float(fields[2])}
    dev["cpus"] = cpus
    mem = {}
    for line in reports.meminfo.splitlines():
        key, value = line.split(":")
        mem[key] = int(value.split()[0]) / 2**20
    dev["mem_gib"] = {
        "total": round(mem["MemTotal"], 2),
        "available": round(mem["MemAvailable"], 2),
    }
    match = re.search(r"integrated=(\d)", reports.memprobe)
    gpu: dict[str, object] = {"integrated": bool(int(match.group(1))) if match else None}
    for key, pattern in (
        ("read_gbs", r"gpu_read_bw=([\d.]+)"),
        ("h2d_pinned_gbs", r"pinned=([\d.]+)"),
        ("h2d_pageable_gbs", r"pageable=([\d.]+)"),
    ):
        match = re.search(pattern, reports.memprobe)
        gpu[key] = float(match.group(1)) if match else None
    dev["gpu"] = gpu
    runs: dict[int, list[tuple[float, float]]] = {}
    for name, text in reports.queue_reads:
        read = read_report(text)
        if read and numeric(read["bw_bytes"], "bw_bytes") > 0:
            match = re.search(r"qd(\d+)", name)
            if match is None:
                raise ValueError(f"queue read report has no qd: {name}")
            runs.setdefault(int(match.group(1)), []).append(
                (
                    numeric(read["bw_bytes"], "bw_bytes") / 2**30,
                    numeric(mapping(read["lat_ns"], "lat_ns")["mean"], "mean") / 1e6,
                )
            )
    ssd: dict[int, SSDRow] = {
        depth: {
            "gibs": round(sum(g for g, _ in rows) / len(rows), 3),
            "lat_ms": round(sum(t for _, t in rows) / len(rows), 3),
            "gibs_runs": sorted(round(g, 3) for g, _ in rows),
        }
        for depth, rows in sorted(runs.items())
    }
    dev["ssd"] = {str(depth): value for depth, value in ssd.items()}
    if ssd:
        fastest = max(ssd, key=lambda depth: ssd[depth]["gibs"])
        dev["ssd_knee_qd"] = min(
            depth for depth in ssd if ssd[depth]["gibs_runs"][-1] >= ssd[fastest]["gibs_runs"][0]
        )
    reads = dict(reports.file_reads)
    files = {}
    for line in reports.files.splitlines() if reports.files is not None else []:
        index, path = line.split(maxsplit=1)
        read = read_report(reads.get(f"reads-file{index}.json")) or read_report(
            reads.get(f"fio-file{index}.json")
        )
        if read:
            files[path.strip()] = round(numeric(read["bw_bytes"], "bw_bytes") / 2**30, 3)
    dev["expert_files"] = files
    median = sorted(files.values())[len(files) // 2] if files else 0
    dev["slow_files"] = [path for path, rate in files.items() if rate < 0.5 * median]
    dev["coresets"] = {
        name: {"cpus": [int(c) for c in text.split(",")]}
        for name, text in (line.split() for line in reports.coresets.splitlines())
    }
    dev["lanes"] = lanes(reports.lanes)
    return DeviceProfile.from_mapping(dev)
