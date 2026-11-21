# SPDX-License-Identifier: Apache-2.0
"""What a strategy is derived from besides the machine characteristics.

``ModelFacts`` come from the engine's facts of the model being launched
(inferssd.engine.model): what the engine would load, before it loads anything.
``RuntimeFacts`` are read when the launch starts, because they change while the
machine runs: the CPUs this process may use, the GPUs with the expert VRAM tier
it can use and their free memory, the memory available now, and the user's
memory budget.
Readers take an injectable host and never write anything.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast

from inferssd.configuration.settings import cpus as cpu_list
from inferssd.contracts import ContractError
from inferssd.machine import identity
from inferssd.machine.characteristics import (
    COUNT,
    POSITIVE,
    TEXT,
    array,
    nullable,
    record,
)

ROLES = ("gate", "up", "down")


@dataclass(frozen=True)
class Role:
    """One routed-expert role as the engine lays it out.

    A role is a matrix of ``rows`` output features by ``cols`` reduction
    columns per expert; ``formats`` gives each ggml type it takes, the layers
    in it and one expert's bytes of the role in it, the most common type first.
    """

    name: str
    formats: tuple[tuple[str, int, int], ...]
    rows: int
    cols: int

    @property
    def format(self) -> str:
        return self.formats[0][0]

    def slice_bytes(self, format: Optional[str] = None) -> int:
        """One expert's bytes of this role in FORMAT (default: the most common one)."""
        name = format or self.format
        for item, _, size in self.formats:
            if item == name:
                return size
        raise KeyError(f"routed {self.name} experts take no {name}")

    def bits_per_weight(self, format: Optional[str] = None) -> float:
        return 8 * self.slice_bytes(format) / (self.rows * self.cols)


@dataclass(frozen=True)
class Experts:
    """A model's routed experts as the engine lays them out."""

    layers: int  # layers with routed experts
    count: int  # routed experts per layer
    used: Optional[int]  # routed experts per token
    gate: Role
    up: Role
    down: Role
    expert_bytes: int  # the largest expert, all three roles
    expert_read_bytes: int  # the most one expert's three role reads (whole 4 KiB pages) take
    format_bytes: tuple[tuple[str, int], ...]  # routed bytes per ggml type, the most first

    @property
    def roles(self) -> tuple[Role, Role, Role]:
        return self.gate, self.up, self.down

    @property
    def routed_bytes(self) -> int:
        """Every routed expert's bytes, of every type."""
        return sum(size for _, size in self.format_bytes)

    @property
    def mean_read_bytes(self) -> float:
        """What the engine reads for one routed expert on average.

        Layers may quantize their experts to different types, so the largest expert overstates
        the rest: this is the mean expert's bytes, widened to whole pages as the largest one's
        three role reads are. It equals expert_read_bytes when every layer has one type.
        """
        mean = self.routed_bytes / (self.layers * self.count)
        return mean + self.expert_read_bytes - self.expert_bytes

    @property
    def dominant_format(self) -> str:
        """The ggml type that holds the most routed-expert bytes."""
        return self.format_bytes[0][0]


@dataclass(frozen=True)
class ModelFacts:
    """Model facts from the engine; nothing here depends on the machine."""

    paths: tuple[str, ...]
    architecture: Optional[str]
    dense_bytes: int  # every tensor that is not a routed expert
    # The layers' tensors (blk.N.*) that are not routed experts: bytes per ggml type, the most
    # first. The lookup tables outside the layers (token and per-layer embeddings) are not in it.
    layer_format_bytes: tuple[tuple[str, int], ...]
    experts: Optional[Experts]  # None for a model without routed experts

    @property
    def cpu_format(self) -> str:
        """The ggml type the CPU computes the most of in a token: the routed experts' dominant
        type, or the layers' for a model without routed experts."""
        return self.experts.dominant_format if self.experts else self.layer_format_bytes[0][0]


def _ranked(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def routed_tensor(layer: int, role: str) -> str:
    return f"blk.{layer}.ffn_{role}_exps.weight"


def facts_from_engine(model: Mapping[str, Any]) -> ModelFacts:
    """ModelFacts of MODEL, the engine's facts of a model (inferssd.engine.model.validate)."""
    experts = model["experts"]
    routed: set[str] = set()
    found: Optional[Experts] = None
    if experts is not None:
        formats, layers = experts["formats"], experts["layers"]
        counts: dict[str, Counter[str]] = {role: Counter() for role in ROLES}
        sizes: dict[str, dict[str, int]] = {role: {} for role in ROLES}
        per_format: Counter[str] = Counter()
        for _, index in layers:
            for role in ROLES:
                # ggml's type names, as the GGUF and every receipt spell them: Q4_K, MXFP4, IQ4_XS
                name, size = formats[index][role]["type"].upper(), formats[index][role]["bytes"]
                counts[role][name] += 1
                sizes[role][name] = size
                per_format[name] += size * experts["count"]
        shapes = {
            "gate": (experts["hidden"], experts["latent"]),
            "up": (experts["hidden"], experts["latent"]),
            "down": (experts["latent"], experts["hidden"]),
        }
        roles = {
            role: Role(
                role,
                tuple((name, count, sizes[role][name]) for name, count in _ranked(counts[role])),
                *shapes[role],
            )
            for role in ROLES
        }
        routed = {routed_tensor(layer, role) for layer, _ in layers for role in ROLES}
        found = Experts(
            layers=len(layers),
            count=experts["count"],
            used=experts["used"] or None,
            gate=roles["gate"],
            up=roles["up"],
            down=roles["down"],
            expert_bytes=experts["expert_bytes"],
            expert_read_bytes=experts["expert_read_bytes"],
            format_bytes=_ranked(per_format),
        )
    dense = [t for t in model["tensors"] if t["name"] not in routed]
    per_type: Counter[str] = Counter()
    for t in dense:
        if t["name"].startswith("blk."):
            per_type[t["type"].upper()] += t["bytes"]
    if not per_type:
        raise ValueError(f"{model['files'][0]}: the model has no layer tensors")
    return ModelFacts(
        paths=tuple(model["files"]),
        architecture=model["architecture"],
        dense_bytes=sum(t["bytes"] for t in dense),
        layer_format_bytes=_ranked(per_type),
        experts=found,
    )


def _formats(value: object, field: str) -> None:
    """A nonempty list of [ggml type name, layers, one expert's bytes] triples."""
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a nonempty array")
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], str):
            raise ContractError(f"{field}[{index}] must be [a ggml type name, layers, bytes]")
        POSITIVE(item[1], f"{field}[{index}][1]")
        POSITIVE(item[2], f"{field}[{index}][2]")


def _type_bytes(value: object, field: str) -> None:
    """A nonempty list of [ggml type name, positive bytes] pairs."""
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a nonempty array")
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
            raise ContractError(f"{field}[{index}] must be [a ggml type name, bytes]")
        POSITIVE(item[1], f"{field}[{index}][1]")


ROLE_RECORD = record({"formats": _formats, "rows": POSITIVE, "cols": POSITIVE})
EXPERTS_RECORD = record(
    {
        "layers": POSITIVE,
        "count": POSITIVE,
        "used": nullable(POSITIVE),
        "roles": record({role: ROLE_RECORD for role in ROLES}),
        "expert_bytes": POSITIVE,
        "expert_read_bytes": POSITIVE,
        "format_bytes": _type_bytes,
    }
)
MODEL_RECORD = record(
    {
        "paths": array(TEXT, minimum=1),
        "architecture": nullable(TEXT),
        "dense_bytes": COUNT,
        "layer_format_bytes": _type_bytes,
        "experts": nullable(EXPERTS_RECORD),
    }
)


def model_facts_json(facts: ModelFacts) -> dict[str, object]:
    """FACTS as a JSON object; model_facts_from reads it back."""
    experts = facts.experts
    return {
        "paths": list(facts.paths),
        "architecture": facts.architecture,
        "dense_bytes": facts.dense_bytes,
        "layer_format_bytes": [list(item) for item in facts.layer_format_bytes],
        "experts": None
        if experts is None
        else {
            "layers": experts.layers,
            "count": experts.count,
            "used": experts.used,
            "roles": {
                role.name: {
                    "formats": [list(item) for item in role.formats],
                    "rows": role.rows,
                    "cols": role.cols,
                }
                for role in experts.roles
            },
            "expert_bytes": experts.expert_bytes,
            "expert_read_bytes": experts.expert_read_bytes,
            "format_bytes": [list(item) for item in experts.format_bytes],
        },
    }


def model_facts_from(value: object) -> ModelFacts:
    """ModelFacts from their JSON object; errors name the offending field."""
    MODEL_RECORD(value, "model_facts")
    item = cast(dict[str, Any], value)
    experts = cast(Optional[dict[str, Any]], item["experts"])
    found: Optional[Experts] = None
    if experts is not None:
        roles = cast(dict[str, dict[str, Any]], experts["roles"])
        built = {
            name: Role(
                name,
                tuple((format, layers, size) for format, layers, size in roles[name]["formats"]),
                roles[name]["rows"],
                roles[name]["cols"],
            )
            for name in ROLES
        }
        found = Experts(
            layers=experts["layers"],
            count=experts["count"],
            used=experts["used"],
            gate=built["gate"],
            up=built["up"],
            down=built["down"],
            expert_bytes=experts["expert_bytes"],
            expert_read_bytes=experts["expert_read_bytes"],
            format_bytes=tuple((name, size) for name, size in experts["format_bytes"]),
        )
    return ModelFacts(
        paths=tuple(item["paths"]),
        architecture=item["architecture"],
        dense_bytes=item["dense_bytes"],
        layer_format_bytes=tuple((name, size) for name, size in item["layer_format_bytes"]),
        experts=found,
    )


def model_drive_key(paths: Sequence[str], host: Optional[identity.Host] = None) -> Optional[str]:
    """The storage key of the drive holding most of the model's bytes, or None when unknown."""
    live = host or identity.NativeHost()
    sizes: Counter[str] = Counter()
    keys: dict[tuple[int, int], str] = {}
    try:
        for path in paths:
            for name, size in live.regular_files(path):
                device = live.device(name)
                if device not in keys:
                    keys[device] = identity.drive_key(live, name)
                sizes[keys[device]] += size
    except (OSError, RuntimeError, ValueError):
        return None
    if not sizes:
        return None
    return max(sizes.items(), key=lambda item: item[1])[0]


@dataclass(frozen=True)
class GPUState:
    """A GPU of a backend with the expert VRAM tier (identity.TIER_BACKENDS) the launch can use,
    as it is now."""

    uuid: str
    free_bytes: Optional[int]  # None where the driver reports no dedicated memory
    processes: int  # compute processes running on it
    backend: str = "CUDA"  # the backend that runs it: CUDA, or ROCm for an AMD GPU


@dataclass(frozen=True)
class RuntimeFacts:
    """Facts read at launch; never taken from the characteristics file."""

    cpus: tuple[int, ...]  # CPUs this process may run on
    gpus: Optional[tuple[GPUState, ...]]  # usable GPUs of the tier backends; None when unreadable
    # The launcher's view of the memory available now: MemAvailable within the
    # room of every bounded cgroup (file cache counted as reclaimable), on
    # macOS what an allocation takes before the kernel compresses anything, and
    # on Windows the available physical memory. None when it cannot be read.
    # The engine sizes its caches from its own reading at run time.
    available_bytes: Optional[int]
    memory_budget_bytes: Optional[int] = None  # the user's memory budget, when given
    # The GPU backends of the engine the launch starts, as its build names them; None when
    # unknown, and then taken to be the backends of the machine's GPUs (engine_gpu_backends).
    # Set by the launch, not recorded with the facts above.
    gpu_backends: Optional[tuple[str, ...]] = None


class Host(identity.Host, Protocol):
    def affinity(self) -> Optional[tuple[int, ...]]: ...
    def environment(self) -> Mapping[str, str]: ...


# A record made before GPUs carried their backend came from nvidia-smi: CUDA's.
GPU_RECORD = record(
    {"uuid": TEXT, "free_bytes": nullable(COUNT), "processes": COUNT}, {"backend": TEXT}
)
RUNTIME_RECORD = record(
    {
        "cpus": array(COUNT, minimum=1),
        "gpus": nullable(array(GPU_RECORD)),
        "available_bytes": nullable(COUNT),
    },
    {"memory_budget_bytes": nullable(POSITIVE)},
)


def runtime_facts_json(facts: RuntimeFacts) -> dict[str, object]:
    """FACTS as a JSON object; runtime_facts_from reads it back."""
    return {
        "cpus": list(facts.cpus),
        "gpus": None
        if facts.gpus is None
        else [
            {
                "uuid": gpu.uuid,
                "free_bytes": gpu.free_bytes,
                "processes": gpu.processes,
                "backend": gpu.backend,
            }
            for gpu in facts.gpus
        ],
        "available_bytes": facts.available_bytes,
        "memory_budget_bytes": facts.memory_budget_bytes,
    }


def runtime_facts_from(value: object) -> RuntimeFacts:
    """RuntimeFacts from their JSON object; errors name the offending field."""
    RUNTIME_RECORD(value, "runtime_facts")
    item = cast(dict[str, object], value)
    cpus = cast(list[int], item["cpus"])
    if len(set(cpus)) != len(cpus):
        raise ContractError("runtime_facts.cpus must not repeat a CPU")
    gpus = cast(Optional[list[dict[str, object]]], item["gpus"])
    return RuntimeFacts(
        tuple(cpus),
        None
        if gpus is None
        else tuple(
            GPUState(
                cast(str, gpu["uuid"]),
                cast(Optional[int], gpu["free_bytes"]),
                cast(int, gpu["processes"]),
                cast(str, gpu.get("backend", "CUDA")),
            )
            for gpu in gpus
        ),
        cast(Optional[int], item["available_bytes"]),
        cast(Optional[int], item.get("memory_budget_bytes")),
    )


class NativeHost(identity.NativeHost):
    """This host; ENVIRONMENT, the launch's (default: this process's), selects the GPUs."""

    def __init__(self, environment: Optional[Mapping[str, str]] = None) -> None:
        self.env = environment

    def affinity(self) -> Optional[tuple[int, ...]]:
        if not hasattr(os, "sched_getaffinity"):
            return None
        return tuple(sorted(os.sched_getaffinity(0)))

    def environment(self) -> Mapping[str, str]:
        return os.environ if self.env is None else self.env


def visible_gpus(listed: Sequence[tuple[int, str]], value: Optional[str]) -> list[tuple[int, str]]:
    """The (index, UUID) pairs a visible-devices list VALUE leaves, in its order.

    As CUDA and ROCm read CUDA_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES, the list
    ends at its first entry that names no GPU or one again. Numbers are taken as
    the indices of LISTED: nvidia-smi's follow the PCI bus, and CUDA numbers
    devices fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so on a machine
    with different GPUs a UUID is the unambiguous choice.
    """
    if value is None:
        return list(listed)
    chosen: list[tuple[int, str]] = []
    for token in (item.strip() for item in value.split(",")):
        if token.isdigit():
            found = [gpu for gpu in listed if gpu[0] == int(token)]
        elif token.startswith("GPU-"):
            found = [gpu for gpu in listed if gpu[1].startswith(token)]
        else:
            found = []
        if len(found) != 1 or found[0] in chosen:
            break
        chosen.append(found[0])
    return chosen


def _gpus(host: Host) -> Optional[tuple[GPUState, ...]]:
    """The tier backends' GPUs the launch can use: NVIDIA's (nvidia-smi) and AMD's on ROCm
    (amdkfd); None when NVIDIA's cannot be read."""
    nvidia = _nvidia(host)
    return None if nvidia is None else nvidia + _rocm(host)


# amdkfd lists each process that uses an AMD GPU, with its memory on every GPU it opened. It tears
# an entry down after the process exits, so an entry counts only while its process runs.
KFD_PROCESSES = "/sys/class/kfd/kfd/proc"


def _rocm(host: Host) -> tuple[GPUState, ...]:
    """The GPUs ROCm runs (identity.amd_compute) that the launch's environment leaves visible."""
    listed = identity.amd_compute(host)
    if not listed:
        return ()
    busy: Counter[str] = Counter()
    try:
        processes = host.listdir(KFD_PROCESSES)
    except OSError:
        processes = []
    for process in processes:
        if not host.exists(f"/proc/{process}"):
            continue
        for gpu in listed:
            if gpu.gpu_id and host.exists(f"{KFD_PROCESSES}/{process}/vram_{gpu.gpu_id}"):
                busy[gpu.uuid] += 1
    # ROCm applies ROCR_VISIBLE_DEVICES to amdkfd's order, then HIP_VISIBLE_DEVICES (or
    # CUDA_VISIBLE_DEVICES, which HIP reads for it) to the devices that leaves.
    environment = host.environment()
    chosen = visible_gpus(
        [(index, gpu.uuid) for index, gpu in enumerate(listed)],
        environment.get("ROCR_VISIBLE_DEVICES"),
    )
    chosen = visible_gpus(
        [(index, uuid) for index, (_, uuid) in enumerate(chosen)],
        environment.get("HIP_VISIBLE_DEVICES", environment.get("CUDA_VISIBLE_DEVICES")),
    )
    address = {gpu.uuid: gpu.address for gpu in listed}
    states = []
    for _, uuid in chosen:
        base = f"/sys/bus/pci/devices/{address[uuid]}/mem_info_vram_"
        total, used = identity.read(host, base + "total"), identity.read(host, base + "used")
        free = (
            int(total) - int(used)
            if total and used and total.isdigit() and used.isdigit()
            else None
        )
        states.append(GPUState(uuid, free, busy[uuid], "ROCm"))
    return tuple(states)


def _nvidia(host: Host) -> Optional[tuple[GPUState, ...]]:
    if not host.which("nvidia-smi"):
        return ()
    listed = host.command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if listed.returncode:
        if "No devices were found" in listed.stdout + listed.stderr:
            return ()
        return None
    rows: list[tuple[int, str, Optional[int]]] = []
    for line in listed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        free = int(fields[2]) * 1048576 if fields[2].isdigit() else None
        rows.append((int(fields[0]), fields[1], free))
    apps = host.command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    )
    if apps.returncode:
        return None
    busy = Counter(line.split(",")[0].strip() for line in apps.stdout.splitlines() if line.strip())
    memory = {uuid: value for _, uuid, value in rows}
    chosen = visible_gpus(
        [(index, uuid) for index, uuid, _ in rows],
        host.environment().get("CUDA_VISIBLE_DEVICES"),
    )
    return tuple(GPUState(uuid, memory[uuid], busy[uuid]) for _, uuid in chosen)


def _cgroup_room(host: Host) -> tuple[bool, Optional[int]]:
    """Whether the cgroup v2 limits could be read, and the room the tightest one leaves.

    At each ancestor with a finite memory.max the room is that limit less
    memory.current, where the file cache (active_file and inactive_file in
    memory.stat) counts as reclaimable, as the engine's host ledger counts it.
    A bounded level without memory.current or memory.stat gives no reading.
    """
    try:
        line = next(
            (row for row in host.text("/proc/self/cgroup").splitlines() if row.startswith("0::")),
            None,
        )
    except OSError:
        return True, None
    if line is None:
        return True, None
    path = line[3:].strip().rstrip("/")
    room: Optional[int] = None
    while True:
        base = "/sys/fs/cgroup" + path
        limit = identity.read(host, base + "/memory.max")
        if limit and limit.isdigit():
            used = identity.read(host, base + "/memory.current")
            stat = identity.read(host, base + "/memory.stat")
            fields = dict(row.split(None, 1) for row in (stat or "").splitlines() if " " in row)
            file = [fields.get(name, "").strip() for name in ("active_file", "inactive_file")]
            if not used or not used.isdigit() or not all(value.isdigit() for value in file):
                return False, None
            anonymous = max(0, int(used) - sum(int(value) for value in file))
            value = max(0, int(limit) - anonymous)
            room = value if room is None else min(room, value)
        if not path:
            return True, room
        path = path.rsplit("/", 1)[0]


def _available(host: Host, system: str) -> Optional[int]:
    if system == "linux":
        try:
            value: Optional[int] = identity.meminfo_bytes(host, "MemAvailable")
        except (OSError, ValueError):
            value = None
        readable, room = _cgroup_room(host)
        if not readable:
            return None
        if room is not None:
            value = room if value is None else min(value, room)
        return value
    if system == "darwin":
        # As the engine's ledger measures it (ledger-native.cpp): what an allocation takes
        # before the kernel compresses anything, by its own live thresholds. That is the
        # free pages above the free target, the file-backed pages above the file cache
        # minimum, and the purgeable and reusable pages.
        names = (
            "vm.pagesize",
            "vm.page_free_count",
            "vm.vm_page_free_target",
            "vm.page_pageable_external_count",
            "vm.vm_page_filecache_min",
            "vm.page_purgeable_count",
            "vm.page_reusable_count",
        )
        try:
            page, free, target, file, minimum, purgeable, reusable = (
                identity.sysctl_int(host, n) for n in names
            )
        except RuntimeError:
            return None
        return (max(0, free - target) + max(0, file - minimum) + purgeable + reusable) * page
    if system == "windows":
        try:
            return host.windows().memory().available_bytes
        except OSError:
            return None
    return None


def read_runtime(
    *, memory_budget_bytes: Optional[int] = None, host: Optional[Host] = None
) -> RuntimeFacts:
    """Read this machine's runtime facts now."""
    live: Host = host or NativeHost()
    system = live.platform()
    if system == "linux":
        allowed = live.affinity()
        cpus = allowed if allowed is not None else identity.online_cpus(live)
        gpus = _gpus(live)
    elif system == "darwin":
        # macOS has no affinity: every logical CPU is usable.
        cpus = cpu_list(f"0-{identity.sysctl_int(live, 'hw.logicalcpu') - 1}")
        gpus = ()
    elif system == "windows":
        # Group 0 logical processor indices, as the engine and its CPU masks number them.
        cpus = live.windows().process_cpus()
        gpus = _gpus(live)
    else:
        raise identity.UnsupportedPlatform(f"runtime facts are not read on {system} yet")
    return RuntimeFacts(tuple(cpus), gpus, _available(live, system), memory_budget_bytes)
