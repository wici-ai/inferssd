"""Behavior-preserving reader for frozen legacy records, not a versioned engine API. State
and result aliases intentionally remain mutable. File operations are explicit ports and
occur only when result() is called, in the original order."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Optional, cast

from inferssd.model.prefill_workload import OFFLOAD_MIN
from inferssd.telemetry.files import readable
from inferssd.telemetry.types import ArmFiles, ArmResult, Record, TokenKey, TraceLayer

COST_TERMS = (
    "cpu_fixed_us",
    "cpu_route_us",
    "gpu_fixed_us",
    "gpu_route_us",
    "gpu_host_fixed_us",
    "gpu_host_route_us",
    "link_gbps",
    "pageable_gbps",
    "drive_gbps",
    "drive_fixed_us",
    "gpu_token_us",
)


def _kv(line: str, key: str) -> Optional[float]:
    m = re.search(f"\\b{key}=([-\\d.]+)", line)
    return float(m.group(1)) if m else None


class ArmParser:
    """The state read_arm builds from a server log, fed line by line, so a live server's log
    can be read in steps."""

    def __init__(self) -> None:
        self.trace: defaultdict[TokenKey, dict[int, TraceLayer]]
        self.prof: defaultdict[TokenKey, list[float]]
        self.lprof: dict[tuple[tuple[int, int], int, int], tuple[float, float]]
        self.lio: defaultdict[TokenKey, list[float]]
        self.last_decode: Optional[TokenKey]
        self.steps: dict[str, tuple[Optional[tuple[int, int, int]], int]]
        self.adm: list[Record]
        self.threads: Optional[Record]
        self.host: Optional[Record]
        self.slots: list[float]
        self.mrc: Any
        self.tier_min_count: Optional[float]
        self.tier_bytes: Optional[tuple[float, Optional[float]]]
        self.tier_first: Optional[int]
        self.pf_layers: list[Record]
        self.pf_graphs: list[Record]
        self.memory: Record
        self.reserve: Optional[Record]
        self.sink_max: Optional[float]
        self.unrouted: list[list[Any]]
        self.checkpoint_align: Optional[int]
        self.ledger: Optional[Record]
        self.prefetch: Optional[Record]
        self.ram_slots: Optional[int]
        self.cache_load: Optional[Record]
        self.new_slot: defaultdict[TokenKey, float]
        self.ram_slot_bytes: Optional[float]
        self.ram_costed: Optional[bool]
        self.decode_place: Optional[Record]
        self.host_register: Optional[Record]
        self.pf_cost: Optional[Record]
        self.prefill_engine: Optional[Record]
        self.prefill_ring: dict[str, int]
        self.trace = defaultdict(dict)
        self.prof = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
        self.lprof = {}
        self.lio = defaultdict(lambda: [0.0, 0.0])
        self.last_decode = None
        self.steps = {}
        self.adm, self.threads, self.host, self.slots, self.mrc = ([], None, None, [], None)
        self.tier_min_count = None
        self.tier_bytes = None
        self.tier_first = None
        self.tier_grown = False
        self.pf_layers = []
        self.pf_graphs = []
        self.memory = {}
        self.reserve = None
        self.sink_max = None
        self.unrouted = []
        self.checkpoint_align = None
        self.ledger = None
        self.prefetch = None
        self.ram_slots = None
        self.cache_load = None
        self.new_slot_bytes = 0.0
        self.new_slot = defaultdict(float)
        self.ram_slot_bytes = self.ram_costed = None
        self.decode_place = None
        self.host_register = None
        self.pf_cost = None
        self.prefill_engine = None
        self.prefill_ring = {}

    def decode_key(self, kind: str, r: Record) -> TokenKey:
        """The key of the decode token of trace or profile record r. The engine numbers a
        token by its position in an epoch, and a request whose prompt does not extend the
        previous prompt and reply (the server cuts its cache back to what they share)
        decodes positions of the epoch again: a layer at a position not past the last
        one's starts a run of its own, so that the two requests' tokens are not merged."""
        e, t, layer = (r["epoch"], r["token"], r["layer"])
        last, run = self.steps.get(kind, (None, 0))
        if last and last[0] == e and (t < last[1] or (t == last[1] and layer <= last[2])):
            run += 1
        self.steps[kind] = ((e, t, layer), run)
        return ((e, run), t)

    def feed(self, line: str) -> None:
        k: Any
        i = line.find("[kimi_")
        if i < 0:
            self._feed_other(line)
            return
        line = line[i:]
        if line.startswith("[kimi_cpu_sink_trace] "):
            if '"phase":"decode"' not in line[:120]:
                return
            r = json.loads(line[22:])
            if r.get("phase") == "decode" and r.get("event") == "layer":
                self.trace[self.decode_key("trace", r)][r["layer"]] = (
                    r["selected_experts"],
                    r["vram_hit"],
                    r["cache_hit"],
                )
        elif line.startswith("[kimi_cpu_sink_profile] "):
            head = line[24:].split(',"workers"')[0].rstrip()
            try:
                r = json.loads(head if head.endswith("}") else head + "}")
            except json.JSONDecodeError:
                return
            if r.get("phase") == "prefill" and r.get("event") == "layer" and self.pf_graphs:
                self.pf_graphs[-1]["cpu_s"] += r["compute_wall_us"] / 1000000.0
            if r.get("phase") == "decode" and r.get("event") == "layer":
                k = self.decode_key("profile", r)
                p = self.prof[k]
                p[0] += r["total_us"]
                p[1] += r["wait_io_us"]
                p[2] += r["compute_wall_us"]
                p[3] += r["other_wall_us"]
                p[4] += 1
                self.lprof[(*k, r["layer"])] = (r["total_us"], r["compute_wall_us"])
                self.last_decode = k
            else:
                self.last_decode = None
        elif line.startswith("[kimi_cpu_sink_layer] ") and self.last_decode:
            d = self.lio[self.last_decode]
            d[0] += _kv(line, "io_us") or 0.0
            d[1] += _kv(line, "bytes") or 0.0
            self.new_slot_bytes += _kv(line, "cache_new_slot_bytes") or 0.0
            self.new_slot[self.last_decode] += _kv(line, "cache_new_slot_bytes") or 0.0
            self.last_decode = None
        elif line.startswith("[kimi_vram_admission]"):
            self.adm.append(
                {k: _kv(line, k) for k in ("min_count", "upload_us", "miss_us", "hit_us")}
            )
        elif line.startswith("[kimi_cpu_sink_threads] decode_participants="):
            m = re.search(
                r"decode_participants=(\d+) us_per_expert=([\d.]+) "
                r"\((\d+) participants: ([\d.]+)\)",
                line,
            )
            if m:
                self.threads = {
                    "participants": int(m.group(1)),
                    "us_per_expert": float(m.group(2)),
                    "other": int(m.group(3)),
                    "other_us_per_expert": float(m.group(4)),
                }
        elif line.startswith("[kimi_cpu_sink_threads] participants="):
            m = re.search(
                "participants=(\\d+) fast_cluster_cpus=(\\d+) decode_candidates=(\\d+),(\\d+)", line
            )
            if m and self.threads is None:
                self.threads = {
                    "participants": int(m.group(3)) if m.group(3) == m.group(4) else None,
                    "candidates": [int(m.group(3)), int(m.group(4))],
                }
        elif line.startswith("[kimi_vram_host]") and "resident_us_per_expert" in line:
            self.host = {
                "resident_us": _kv(line, "resident_us_per_expert"),
                "host_us": _kv(line, "host_us_per_expert"),
            }
            for k in (
                "in_place",
                "jobs",
                "copies",
                "reserved_copies",
                "refused",
                "copy_us_per_mb",
                "resident_call_us",
                "resident_expert_us",
                "host_call_us",
                "host_expert_us",
            ):
                if _kv(line, k) is not None:
                    self.host[k] = _kv(line, k)
        elif line.startswith("[kimi_decode_place] reason="):
            self.decode_place = {
                k: _kv(line, k)
                for k in (
                    "layers",
                    "tokens",
                    "gpu_experts",
                    "gpu_per_token",
                    "planned",
                    "candidates",
                    "refused",
                    "wanted",
                    "error_p50_pct",
                    "error_p90_pct",
                    "predicted_ms",
                    "measured_ms",
                    "cpu_only_ms",
                    "cpu_expert_us",
                    "early",
                    "fixed_us",
                    "decode_read_gbs",
                    "cpu_alone_us",
                    "cpu_shared_us",
                    "cpu_moved",
                    "pending",
                    "pending_planned",
                )
            }
        elif line.startswith("[kimi_host_register] reason=destroy"):
            self.host_register = {
                k: _kv(line, k)
                for k in (
                    "registered",
                    "failures",
                    "unregistered",
                    "signals",
                    "wanted",
                    "us_mean",
                    "us_max",
                    "us_total",
                )
            }
        elif line.startswith("[kimi_vram_tier] reason=allocated") and self.tier_bytes is None:
            c, n, b = (_kv(line, "capacity_bytes"), _kv(line, "slots"), _kv(line, "budget_bytes"))
            if c and n:
                self.tier_bytes = (c / n, b)
            if n and (not self.tier_grown):
                self.tier_first = int(n)
            self.tier_min_count = _kv(line, "min_count")
        elif line.startswith("[kimi_vram_elastic]") and "slots=" in line:
            v = _kv(line, "slots")
            if v is not None:
                self.slots.append(v)
            if line.startswith("[kimi_vram_elastic] event=decode_grow") and (not self.tier_grown):
                g = _kv(line, "after_slots")
                if g is not None:
                    self.tier_first, self.tier_grown = (int(g), True)
        elif line.startswith("[kimi_host_ledger] reason=destroy"):
            self.ledger = {k: _kv(line, k) for k in ("unified", "margin", "margin_updates")}
        elif line.startswith("[kimi_cpu_sink_prefetch] enabled"):
            self.prefetch = dict(
                self.prefetch or {},
                topk=int(cast(float, _kv(line, "topk"))),
                readers=int(cast(float, _kv(line, "readers"))),
            )
        elif line.startswith("[kimi_cpu_sink_prefetch] reason="):
            r = {
                k: _kv(line, k)
                for k in (
                    "topk",
                    "layers",
                    "issued",
                    "windows",
                    "window_us_p50",
                    "window_us_mean",
                    "reader_bytes",
                    "reader_busy_us",
                    "width",
                    "skip_vram",
                    "vram_late",
                    "skip_vram_lost",
                    "spill_us",
                    "spills",
                )
            }
            for k in (
                "predicted_r",
                "novel_r",
                "novel_routed_r",
                "issued_r",
                "used_r",
                "started_r",
                "spill_us_r",
            ):
                m = re.search(f"\\b{k}=([\\d,]+)", line)
                r[k] = [int(x) for x in m.group(1).split(",")] if m else None
            self.prefetch = dict(
                self.prefetch or {}, **{k: v for k, v in r.items() if v is not None}
            )
        elif line.startswith("[kimi_cpu_sink_cache] reason=load "):
            self.cache_load = {
                k: _kv(line, k) for k in ("slots", "slot_bytes", "committed_bytes", "us")
            }
        elif line.startswith("[kimi_cpu_sink_cache] auto_budget=") and self.ram_slot_bytes is None:
            self.ram_slot_bytes = _kv(line, "slot_bytes") or 0.0
            if self.ram_slot_bytes:
                self.ram_slots = int(cast(float, _kv(line, "auto_budget")) // self.ram_slot_bytes)
        elif (
            line.startswith("[kimi_cpu_sink_cache] auto_capacity_bytes=")
            and self.ram_slot_bytes
            and (not self.ram_costed)
        ):
            self.ram_slots, self.ram_costed = (
                int(cast(float, _kv(line, "auto_capacity_bytes")) // self.ram_slot_bytes),
                True,
            )
        elif line.startswith("[kimi_cpu_sink_mrc] "):
            self.mrc = json.loads(line[20:])
        elif line.startswith(
            ("[kimi_cpu_prefill_graph] event=activate ", "[kimi_cpu_prefill_graph] event=fallback ")
        ):
            self.pf_graphs.append(
                {
                    "n_tokens": int(_kv(line, "requested_tokens") or 0),
                    "ubatch": int(_kv(line, "configured_max") or 0),
                    "cpu_s": 0.0,
                    "sink_bytes": 0.0,
                    "graph_bytes": 0.0,
                    "graph_read_us": 0.0,
                    "graph_passes": 0,
                    "graph_tokens": 0,
                }
            )
        elif line.startswith(
            ("[kimi_cpu_prefill_cost] calibrated=", "[kimi_cpu_prefill_mxfp4_cost] calibrated=")
        ):
            self.pf_cost = {
                k: _kv(line, k) for k in COST_TERMS + ("measured_us", "predicted_us", "tokens")
            }
        elif line.startswith("[kimi_vram_prefill_engine] format="):
            e = {
                k: _kv(line, k)
                for k in (
                    "in_place",
                    "route_bytes",
                    "token_bytes",
                    "output_bytes",
                    "fixed_bytes",
                    "bytes_per_token",
                    "reserved_rows",
                )
            }
            m = re.search("\\bformat=(\\S+) engine=(\\S+)", line)
            self.prefill_engine = dict(
                e, format=m.group(1) if m else None, engine=m.group(2) if m else None
            )
        elif line.startswith("[kimi_cpu_prefill_schedule] ") and " ring=" in line:
            self.prefill_ring = {
                k: int(cast(float, _kv(line, k)))
                for k in ("ring", "ring_locked")
                if _kv(line, k) is not None
            }
        elif line.startswith("[kimi_cpu_prefill_layer] "):
            self.pf_layers.append(
                {
                    k: _kv(line, k)
                    for k in (
                        "layer",
                        "n_tokens",
                        "unique_experts",
                        "routed_slots",
                        "gpu_experts",
                        "gpu_ssd_transient_experts",
                        "ssd_experts",
                        "ssd_bytes",
                    )
                }
            )
            self.pf_layers[-1]["cost"], self.pf_cost = (self.pf_cost, None)
            if self.pf_graphs:
                self.pf_graphs[-1]["sink_bytes"] += self.pf_layers[-1]["ssd_bytes"] or 0.0
        elif line.startswith("[kimi_prefill_route] "):
            self._reserve(line)

    def _reserve(self, line: str) -> None:
        """Each graph reserve: [prefill_reserve_outputs] n_tokens=N, then the device's and
        the host's compute buffers; the graph route's (it copies the experts to the
        device) when N exceeds the sink's threshold, the max_sink_tokens of the engine's
        route lines ('all': every pass the sink's), which the reserve at load precedes and
        every request's follows, and the GPU takes the experts' product at N tokens
        (OFFLOAD_MIN)."""
        if line.startswith("[kimi_prefill_route] "):
            m = re.search("\\bmax_sink_tokens=(\\w+)", line)
            if m:
                self.sink_max = float("inf") if m.group(1) == "all" else int(m.group(1))
                for x in self.unrouted:
                    x[3] = self._graph_pass(x[0])
                self.unrouted = []
            return
        if line.startswith("[prefill_reserve_outputs] ") and _kv(line, "n_tokens") is not None:
            self.reserve = {"n_tokens": int(cast(float, _kv(line, "n_tokens")))}
            return
        r = self.reserve
        if r is None:
            return
        m = re.search("(\\S+) compute buffer size = +([\\d.]+) MiB", line)
        if m:
            r["device_mib" if m.group(1) == self.memory.get("device") else "host_mib"] = float(
                m.group(2)
            )
        if "graph splits = " in line:
            n = r["n_tokens"]
            x = [
                n,
                r.get("device_mib", 0.0),
                r.get("host_mib", 0.0),
                None if self.sink_max is None else self._graph_pass(n),
            ]
            seen = self.memory.setdefault("reserves", [])
            if x[:3] not in [y[:3] for y in seen]:
                seen.append(x)
                seen.sort(key=lambda y: y[:3])
                if x[3] is None:
                    self.unrouted.append(x)
            self.reserve = None

    def _graph_pass(self, n: float) -> int:
        return int(n > cast(float, self.sink_max) and n >= OFFLOAD_MIN)

    def _feed_other(self, line: str) -> None:
        self._reserve(line)
        if "context checkpoints are taken after each prompt batch" in line:
            m = re.search("at multiples of (\\d+) tokens", line)
            self.checkpoint_align = int(m.group(1)) if m else None
        elif line.startswith("[moe_prefill_copy] "):
            if self.pf_graphs:
                g = self.pf_graphs[-1]
                g["graph_bytes"] += _kv(line, "bytes") or 0.0
                g["graph_read_us"] += _kv(line, "read_us") or 0.0
                g["graph_passes"] += int(_kv(line, "passes") or 0)
                g["graph_tokens"] += int(_kv(line, "tokens") or 0)
        elif self.memory.get("compute") == "device":
            m = re.search("(\\S+) compute buffer size = +([\\d.]+) MiB", line)
            if m and m.group(1) != self.memory.get("device"):
                self.memory["host_compute_mib"] = float(m.group(2))
            self.memory["compute"] = True
        elif "compute" not in self.memory:
            m = re.search("using device (\\S+) .* - (\\d+) MiB free", line)
            if m and "free_mib" not in self.memory:
                self.memory.update(device=m.group(1), free_mib=int(m.group(2)))
            m = re.search("(\\S+) (model|KV|compute) buffer size = +([\\d.]+) MiB", line)
            if m and m.group(1) == self.memory.get("device"):
                key = {"model": "dense_mib", "KV": "kv_mib", "compute": "compute_mib"}[m.group(2)]
                self.memory[key] = self.memory.get(key, 0.0) + float(m.group(3))
                if key == "compute_mib":
                    self.memory["compute"] = "device"
            m = re.search("n_ubatch += (\\d+)", line)
            if m:
                self.memory["n_ubatch"] = int(m.group(1))

    def result(self, run_dir: str, skip: int = 128, *, files: ArmFiles) -> ArmResult:
        ks = sorted(self.trace)
        groups: list[list[TokenKey]] = []
        for k in ks:
            if groups and groups[-1][-1][0] == k[0] and (k[1] == groups[-1][-1][1] + 1):
                groups[-1].append(k)
            else:
                groups.append([k])
        segments = [[self.trace[k] for k in g] for g in groups]
        window = [k for g in groups for k in g[skip:]]
        clients = (
            [json.loads(x) for x in files.open(f"{run_dir}/client.jsonl")]
            if readable(f"{run_dir}/client.jsonl", files)
            else []
        )
        meta = (
            files.open(f"{run_dir}/meta.txt").read()
            if readable(f"{run_dir}/meta.txt", files)
            else ""
        )
        load_s = re.search("\\bload_s=([\\d.]+)", meta)
        new_slot = [0.0, 0.0]
        firsts: dict[tuple[int, int], int] = {}
        for run, t in sorted(self.new_slot):
            firsts.setdefault(run, t)
            new_slot[t - firsts[run] >= 128] += self.new_slot[run, t]
        passes: list[list[Record]] = []
        for r in self.pf_layers:
            if not passes or r["layer"] <= passes[-1][-1]["layer"]:
                passes.append([])
            passes[-1].append(r)
        prefill = []
        for ps in passes:
            if ps[0]["n_tokens"] < 64:
                continue
            prefill.append(
                {
                    "n_tokens": int(ps[0]["n_tokens"]),
                    "layers": len(ps),
                    "unique": sum(r["unique_experts"] for r in ps),
                    "tier": sum(r["gpu_experts"] for r in ps),
                    "routes": sum(r["routed_slots"] for r in ps),
                    "ssd_bytes": sum(r["ssd_bytes"] for r in ps),
                    "unique_per_layer": [int(r["unique_experts"]) for r in ps],
                }
            )
            costed = [
                r["cost"] for r in ps if r.get("cost") and r["cost"].get("measured_us") is not None
            ]
            if self.prefill_engine and costed:
                prefill[-1]["lane"] = {
                    k: v for k, v in costed[-1].items() if k in COST_TERMS and v is not None
                }
                prefill[-1]["layers_s"] = sum(c["measured_us"] for c in costed) / 1000000.0
                prefill[-1]["layers_costed"] = len(costed)
        return {
            "cache_load": self.cache_load,
            "new_slot_bytes": self.new_slot_bytes,
            "new_slot_windows": new_slot,
            "load_s": float(load_s.group(1)) if load_s else None,
            "segments": segments,
            "groups": groups,
            "window": window,
            "lprof": self.lprof,
            "prefill": prefill,
            "trace": self.trace,
            "prof": self.prof,
            "admission": self.adm,
            "threads": self.threads,
            "host": self.host,
            "tier_slots_max": max(self.slots) if self.slots else None,
            "tier_slots_first": self.tier_first,
            "tier_bytes": self.tier_bytes,
            "mrc": self.mrc,
            "tier_min_count": self.tier_min_count,
            "lio": dict(self.lio),
            "client": clients,
            "pf_graphs": [g for g in self.pf_graphs if g["n_tokens"] >= 64],
            "memory": {k: v for k, v in self.memory.items() if k != "compute"},
            "checkpoint_align": self.checkpoint_align,
            "ledger": self.ledger,
            "prefetch": self.prefetch,
            "ram_slots": self.ram_slots,
            "decode_place": self.decode_place,
            "host_register": self.host_register,
            "prefill_engine": dict(self.prefill_engine, **self.prefill_ring)
            if self.prefill_engine
            else None,
        }


def hits(arm: ArmResult, keys: Optional[Iterable[TokenKey]] = None) -> dict[str, float]:
    v = r = m = 0
    for k in keys if keys is not None else arm["window"]:
        for sel, vh, ch in arm["trace"][k].values():
            for a, b in zip(vh, ch):
                if a:
                    v += 1
                elif b:
                    r += 1
                else:
                    m += 1
    n = v + r + m or 1
    return {"v": v / n, "r": r / n, "m": m / n, "experts": n}


def window_rate(tok_s: Sequence[Optional[float]], first: int, last: int) -> Optional[float]:
    """Tokens per second between decode tokens `first` and `last` (1-based, inclusive ends as
    milestones)."""
    if not tok_s or len(tok_s) < last or tok_s[first - 1] is None or (tok_s[last - 1] is None):
        return None
    dt = cast(float, tok_s[last - 1]) - cast(float, tok_s[first - 1])
    return (last - first) / dt if dt > 0 else None


def median(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.median(cast(list[float], xs)) if xs else None
