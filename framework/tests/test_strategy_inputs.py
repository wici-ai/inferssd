# SPDX-License-Identifier: Apache-2.0
"""Strategy inputs: model facts from the engine's facts of a model, runtime facts from hosts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_machine_identity import (
    LINUX_MACHINE,
    ROCMINFO,
    FakeHost,
    amd_card,
    kfd,
    linux_tree,
    nvme_key,
    ok,
    write,
)

from inferssd.contracts import ContractError
from inferssd.engine import model as engine_model
from inferssd.machine.identity import Command
from inferssd.strategy.inputs import (
    GPUState,
    Role,
    RuntimeFacts,
    facts_from_engine,
    model_drive_key,
    model_facts_from,
    model_facts_json,
    read_runtime,
    visible_gpus,
)

MXFP4, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F32 = 39, 11, 12, 13, 14, 8, 0
# ggml type -> (its name, elements per block, bytes per block), as the GGUF format defines them
TYPES = {MXFP4: ("mxfp4", 32, 17), Q3_K: ("q3_K", 256, 110), Q4_K: ("q4_K", 256, 144)}
TYPES.update({Q5_K: ("q5_K", 256, 176), Q6_K: ("q6_K", 256, 210), Q8_0: ("q8_0", 32, 34)})
TYPES[F32] = ("f32", 1, 4)
PAGE = 4096
ROOT = Path(__file__).resolve().parents[2]
# The split toy model's facts as the engine prints them (engine/tests/ssd/test-model-facts.cpp).
ENGINE_FIXTURE = ROOT / "engine/tests/ssd/fixtures/model-facts.json"


def engine_facts(tensors, kv, *, path="/models/model.gguf", alignment=PAGE, data=PAGE):
    """The engine's facts (inferssd-server --ssd-model-facts) of a one-file model with metadata KV
    and TENSORS (name, shape, type) back to back from byte DATA, each at a multiple of ALIGNMENT.

    The routed experts are laid out as the engine lays them out: one format per distinct
    up, gate and down, each role read in whole pages."""
    rows, offset = [], data
    for name, shape, kind in tensors:
        type_name, block, size = TYPES[kind]
        count = 1
        for dimension in shape:
            count *= dimension
        nbytes = count // block * size
        rows.append(
            {"name": name, "type": type_name, "shape": list(shape), "file": 0}
            | {"offset": offset, "bytes": nbytes}
        )
        offset += -(-nbytes // alignment) * alignment
    by_name = {row["name"]: row for row in rows}
    routed = sorted(
        int(match.group(1))
        for row in rows
        if (match := re.fullmatch(r"blk\.(\d+)\.ffn_up_exps\.weight", row["name"]))
        and all(
            f"blk.{match.group(1)}.ffn_{role}_exps.weight" in by_name for role in ("gate", "down")
        )
    )
    experts = None
    if routed:
        formats, layers = [], []
        read = {"up": 0, "gate": 0, "down": 0}
        largest = most = 0
        for layer in routed:
            roles = {role: by_name[f"blk.{layer}.ffn_{role}_exps.weight"] for role in read}
            count = roles["up"]["shape"][2]
            format = {
                role: {"type": row["type"], "bytes": row["bytes"] // count}
                for role, row in roles.items()
            }
            if format not in formats:
                formats.append(format)
            layers.append([layer, formats.index(format)])
            largest = max(largest, sum(item["bytes"] for item in format.values()))
            for expert in range(count):
                total = 0
                for role, row in roles.items():
                    size = format[role]["bytes"]
                    start = row["offset"] + expert * size
                    bytes_read = -(-(start + size) // PAGE) * PAGE - start // PAGE * PAGE
                    read[role] = max(read[role], bytes_read)
                    total += bytes_read
                most = max(most, total)
        up = by_name[f"blk.{routed[0]}.ffn_up_exps.weight"]["shape"]
        experts = {
            "count": up[2],
            "used": kv.get(f"{kv.get('general.architecture')}.expert_used_count", 0),
            "latent": up[0],
            "hidden": up[1],
            "formats": formats,
            "layers": layers,
            "read_bytes": read,
            "expert_bytes": largest,
            "expert_read_bytes": most,
        }
    return engine_model.validate(
        {
            "version": 1,
            "architecture": kv.get("general.architecture", ""),
            "layers": 1
            + max(
                (int(m.group(1)) for r in rows if (m := re.match(r"blk\.(\d+)\.", r["name"]))),
                default=-1,
            ),
            "files": [path],
            "metadata": dict(kv),
            "tensors": rows,
            "experts": experts,
        }
    )


def facts_of(tensors, kv, **options):
    """ModelFacts of the synthetic model TENSORS and KV (engine_facts)."""
    return facts_from_engine(engine_facts(tensors, kv, **options))


def recorded_facts(tensors, kv, **options):
    """The recorded model_facts host fact of the synthetic model TENSORS and KV."""
    return model_facts_json(facts_of(tensors, kv, **options))


class EngineHost:
    """A launch host whose engine prints FACTS for any model (inferssd-server --ssd-model-facts)."""

    def __init__(self, facts):
        self.facts, self.asked = facts, []

    def command(self, argv, timeout=10):
        self.asked.append(list(argv))
        return Command(0, json.dumps(self.facts))


def moe(arch, *, layers, experts, used, latent, hidden, gate, up, down, first=0):
    """Tensors and KV of a mixture-of-experts model; GATE, UP, DOWN give the type per layer."""
    tensors = [("token_embd.weight", (latent, 1024), Q8_0), ("output_norm.weight", (latent,), F32)]
    for index, layer in enumerate(range(first, first + layers)):
        for role, kinds in (("gate", gate), ("up", up), ("down", down)):
            kind = kinds[index] if isinstance(kinds, list) else kinds
            shape = (hidden, latent, experts) if role == "down" else (latent, hidden, experts)
            tensors.append((f"blk.{layer}.ffn_{role}_exps.weight", shape, kind))
        tensors.append((f"blk.{layer}.attn_norm.weight", (latent,), F32))
    kv = {"general.architecture": arch, f"{arch}.expert_used_count": used}
    return tensors, kv


def dense(arch, *, layers=2, width=256, kind=Q4_K):
    """Tensors and KV of a model without routed experts: token embeddings, then layers of a norm
    and two KIND matrices."""
    tensors = [("token_embd.weight", (width, 64), Q8_0)]
    for layer in range(layers):
        tensors.append((f"blk.{layer}.attn_norm.weight", (width,), F32))
        tensors += [(f"blk.{layer}.{n}.weight", (width, width), kind) for n in ("attn_q", "ffn_up")]
    return tensors, {"general.architecture": arch}


def deepseek_like():
    """DeepSeek-V4.1-Flash's routed layout: 40 MoE layers after 3 dense ones, 384 MXFP4 experts."""
    return moe(
        "deepseek41",
        layers=40,
        experts=384,
        used=6,
        latent=5120,
        hidden=2304,
        gate=MXFP4,
        up=MXFP4,
        down=MXFP4,
        first=3,
    )


def muse_like():
    """Muse Spark's routed layout: 62 MoE layers, 256 experts, Q4_K gate/up and Q5_K down."""
    return moe(
        "muse-spark",
        layers=62,
        experts=256,
        used=8,
        latent=4096,
        hidden=4096,
        gate=Q4_K,
        up=Q4_K,
        down=Q5_K,
    )


def test_the_engines_facts_of_the_split_toy_model():
    # What the engine prints (its test keeps this file), read as the strategy reads it.
    facts = facts_from_engine(engine_model.validate(json.loads(ENGINE_FIXTURE.read_text())))
    assert facts.paths == ("<dir>/toy-00001-of-00002.gguf", "<dir>/toy-00002-of-00002.gguf")
    experts = facts.experts
    assert experts is not None
    assert (facts.architecture, experts.layers, experts.count, experts.used) == (
        "qwen3moe",
        2,
        4,
        2,
    )
    # Layer 0 is Q8_0 throughout; layer 1 has Q4_0 up and gate. The NextN block is not there.
    assert experts.up == Role("up", (("Q4_0", 1, 1152), ("Q8_0", 1, 2176)), 32, 64)
    assert experts.down.formats == (("Q8_0", 2, 2176),)
    assert (experts.down.rows, experts.down.cols) == (64, 32)
    assert experts.expert_bytes == 3 * 2176 and experts.expert_read_bytes >= experts.expert_bytes
    assert experts.format_bytes == (("Q8_0", 4 * 2176 * 4), ("Q4_0", 2 * 1152 * 4))
    assert facts.dense_bytes == 64 * 16 * 4 + 64 * 64 * 4
    # The layers' one other tensor is layer 0's attention query; the token embeddings are outside
    # the layers.
    assert facts.layer_format_bytes == (("F32", 64 * 64 * 4),)
    assert facts.cpu_format == "Q8_0"
    assert model_facts_from(model_facts_json(facts)) == facts


def test_deepseek_like_facts():
    facts = facts_of(*deepseek_like())
    assert facts.paths == ("/models/model.gguf",)
    experts = facts.experts
    assert experts is not None
    assert (facts.architecture, experts.layers, experts.count, experts.used) == (
        "deepseek41",
        40,
        384,
        6,
    )
    assert [(r.name, r.format, r.rows, r.cols) for r in experts.roles] == [
        ("gate", "MXFP4", 2304, 5120),
        ("up", "MXFP4", 2304, 5120),
        ("down", "MXFP4", 5120, 2304),
    ]
    role = 5120 * 2304 // 32 * 17
    assert [r.slice_bytes() for r in experts.roles] == [role] * 3 == [6266880] * 3
    assert [r.bits_per_weight() for r in experts.roles] == [4.25] * 3
    # Page-aligned tensors of whole-page slices read exactly their bytes.
    assert experts.expert_bytes == experts.expert_read_bytes == 3 * role
    assert facts.dense_bytes == 5120 * 1024 // 32 * 34 + 5120 * 4 + 40 * 5120 * 4
    assert experts.format_bytes == (("MXFP4", 40 * 3 * 384 * role),)
    assert experts.dominant_format == facts.cpu_format == "MXFP4"


def test_muse_like_facts_and_dominant_format():
    experts = facts_of(*muse_like()).experts
    assert experts is not None
    assert (experts.layers, experts.count, experts.used) == (62, 256, 8)
    assert [(r.format, r.slice_bytes()) for r in experts.roles] == [
        ("Q4_K", 9437184),
        ("Q4_K", 9437184),
        ("Q5_K", 11534336),
    ]
    assert experts.expert_read_bytes == 2 * 9437184 + 11534336
    # Gate and up together hold more bytes than down.
    assert experts.format_bytes[0] == ("Q4_K", 62 * 2 * 256 * 9437184)
    assert experts.dominant_format == "Q4_K"


def test_mixed_formats_rank_by_layers():
    downs = [Q5_K, Q6_K, Q6_K]
    tensors, kv = moe(
        "toy", layers=3, experts=2, used=1, latent=256, hidden=256, gate=Q4_K, up=Q4_K, down=downs
    )
    experts = facts_of(tensors, kv, alignment=32).experts
    assert experts is not None
    assert experts.down.formats == (("Q6_K", 2, 256 * 210), ("Q5_K", 1, 256 * 176))
    assert experts.down.format == "Q6_K"
    assert experts.down.slice_bytes() == 256 * 210
    assert experts.down.slice_bytes("Q5_K") == 256 * 176
    with pytest.raises(KeyError, match="take no Q8_0"):
        experts.down.slice_bytes("Q8_0")
    assert experts.used == 1
    # Slices of 36-54 KiB at 32-byte offsets each read whole pages around them.
    assert experts.expert_bytes == 2 * 256 * 144 + 256 * 210
    assert experts.expert_read_bytes > experts.expert_bytes
    assert experts.expert_read_bytes % PAGE == 0


def test_a_model_without_routed_experts_is_its_dense_weights():
    facts = facts_of(*dense("toy"))
    assert facts.experts is None
    assert facts.dense_bytes == 256 * 64 // 32 * 34 + 2 * (256 * 4 + 2 * 256 * 256 // 256 * 144)
    # The token embeddings are a lookup outside the layers: the CPU computes the layers' Q4_K.
    assert facts.layer_format_bytes == (("Q4_K", 2 * 2 * 256 * 144), ("F32", 2 * 256 * 4))
    assert facts.cpu_format == "Q4_K"
    assert model_facts_from(model_facts_json(facts)) == facts


def test_a_model_without_layers_is_refused():
    with pytest.raises(ValueError, match="no layer tensors"):
        facts_of([("w", (32, 32), Q8_0)], {"general.architecture": "toy"})


def test_the_engines_facts_are_checked():
    good = engine_facts(
        *moe(
            "toy",
            layers=1,
            experts=2,
            used=1,
            latent=256,
            hidden=256,
            gate=Q8_0,
            up=Q8_0,
            down=Q8_0,
        )
    )
    for change, field in (
        (lambda m: m.update(version=2), "model.version"),
        (lambda m: m["tensors"][0].update(file=1), "model.tensors[0].file"),
        (lambda m: m["experts"]["layers"][0].__setitem__(1, 1), "model.experts.layers[0]"),
        (lambda m: m["experts"].pop("read_bytes"), "model.experts.read_bytes"),
        (lambda m: m.update(files=[]), "model.files"),
    ):
        broken = json.loads(json.dumps(good))
        change(broken)
        with pytest.raises(ContractError, match=re.escape(field)):
            engine_model.validate(broken)


def test_the_engine_query_reports_what_the_engine_said():
    model = json.loads(ENGINE_FIXTURE.read_text())
    seen = []

    def run(argv, returncode=0, stdout=json.dumps(model), stderr=""):
        seen.append(list(argv))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    assert engine_model.query("/bin/server", "/m.gguf", run) == model
    assert seen == [["/bin/server", "--ssd-model-facts", "/m.gguf"]]
    said = "llama_model_loader: loading\nerror: /m.gguf: key not found\n"
    with pytest.raises(ValueError, match="cannot read the model /m.gguf: /m.gguf: key not found$"):
        engine_model.query("/bin/server", "/m.gguf", lambda argv: run(argv, 1, "", said))
    with pytest.raises(ValueError, match="facts of the model /m.gguf"):
        engine_model.query("/bin/server", "/m.gguf", lambda argv: run(argv, stdout="{}"))

    def missing(argv):
        raise FileNotFoundError(argv[0])

    with pytest.raises(ValueError, match="cannot read the model /m.gguf"):
        engine_model.query("/bin/server", "/m.gguf", missing)


def test_model_drive_key_follows_the_files(tmp_path):
    linux_tree(tmp_path / "root")
    model = tmp_path / "models" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x" * 100)
    host = FakeHost(tmp_path / "root", devices={str(tmp_path / "models"): (259, 2)})
    assert model_drive_key([str(model)], host) == nvme_key()
    assert model_drive_key([str(tmp_path / "missing.gguf")], host) is None


class RuntimeHost(FakeHost):
    def __init__(self, root, *, affinity=None, environment=None, **options):
        super().__init__(root, **options)
        self.allowed, self.env = affinity, dict(environment or {})

    def affinity(self):
        return self.allowed

    def environment(self):
        return self.env


SMI_GPUS = ("nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits")
SMI_APPS = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")


def linux_runtime(root, *, gpus="0, GPU-aaaa, 30000\n1, GPU-bbbb, [N/A]\n", apps=""):
    write(root, "/proc/meminfo", "MemTotal: 64000000 kB\nMemAvailable: 48000000 kB\n")
    write(root, "/proc/self/cgroup", "0::/user.slice/run.scope\n")
    write(root, "/sys/fs/cgroup/user.slice/memory.max", f"{40 << 30}\n")
    write(root, "/sys/fs/cgroup/user.slice/memory.current", f"{14 << 30}\n")
    stat = f"anon {9 << 30}\nfile {5 << 30}\nactive_file {3 << 30}\ninactive_file {1 << 30}\n"
    write(root, "/sys/fs/cgroup/user.slice/memory.stat", stat)
    write(root, "/sys/fs/cgroup/user.slice/run.scope/memory.max", "max\n")
    write(root, "/sys/fs/cgroup/user.slice/run.scope/memory.current", "4096\n")
    return {SMI_GPUS: ok(gpus), SMI_APPS: ok(apps)}


def test_linux_runtime_facts(tmp_path):
    commands = linux_runtime(tmp_path, apps="GPU-bbbb, 4242\n")
    host = RuntimeHost(tmp_path, affinity=(0, 2, 3), commands=commands)
    facts = read_runtime(memory_budget_bytes=1 << 34, host=host)
    assert facts == RuntimeFacts(
        (0, 2, 3),
        (GPUState("GPU-aaaa", 30000 << 20, 0), GPUState("GPU-bbbb", None, 1)),
        30 << 30,  # 40 GiB less 14 in use, 4 of them file cache: below MemAvailable
        1 << 34,
    )


def test_cgroup_room_counts_file_cache_and_needs_memory_stat(tmp_path):
    linux_runtime(tmp_path)
    # A deeper, tighter limit whose memory is mostly file cache.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.max", f"{8 << 30}\n")
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{9 << 30}\n")
    stat = f"active_file {6 << 30}\ninactive_file {2 << 30}\n"
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.stat", stat)
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 7 << 30
    # More file cache than memory.current: nothing counts as in use.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{1 << 30}\n")
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 8 << 30
    # Over the limit in anonymous memory: no room, never negative.
    write(tmp_path, "/sys/fs/cgroup/user.slice/run.scope/memory.current", f"{20 << 30}\n")
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes == 0
    # A bounded level without memory.stat gives no reading.
    (tmp_path / "sys/fs/cgroup/user.slice/run.scope/memory.stat").unlink()
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).available_bytes is None


def test_linux_runtime_respects_cuda_visible_devices(tmp_path):
    commands = linux_runtime(tmp_path)
    for value, expected in (("GPU-bb", ("GPU-bbbb",)), ("1,0", ("GPU-bbbb", "GPU-aaaa")), ("", ())):
        host = RuntimeHost(
            tmp_path,
            affinity=(0,),
            environment={"CUDA_VISIBLE_DEVICES": value},
            commands=commands,
        )
        gpus = read_runtime(host=host).gpus
        assert gpus is not None and tuple(gpu.uuid for gpu in gpus) == expected


@pytest.mark.skipif(sys.platform == "win32", reason=LINUX_MACHINE)  # sysfs names carry colons
def test_linux_runtime_reads_rocm_gpus_through_amdkfd(tmp_path):
    commands = linux_runtime(tmp_path, gpus="")
    amd_card(tmp_path)
    kfd(tmp_path, 2, 64, location=8 << 8, unique=0x1111, gpu_id=7)
    write(tmp_path, "/sys/bus/pci/devices/0000:08:00.0/mem_info_vram_total", f"{16 << 30}\n")
    write(tmp_path, "/sys/bus/pci/devices/0000:08:00.0/mem_info_vram_used", f"{1 << 30}\n")
    write(tmp_path, "/sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used", f"{2 << 30}\n")
    # amdkfd lists the processes that use a GPU, with their memory on each GPU they opened; an
    # entry outlives its process for a moment, and then counts for nothing.
    write(tmp_path, "/sys/class/kfd/kfd/proc/4242/vram_64106", "1048576\n")
    write(tmp_path, "/proc/4242/stat", "4242 (inferssd-server) S\n")
    write(tmp_path, "/sys/class/kfd/kfd/proc/4343/vram_64106", "1048576\n")
    # Without the ROCm runtime the GPUs run on Vulkan, which has no VRAM tier.
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=commands)).gpus == ()
    commands[ROCMINFO] = ok("")
    first, second = "GPU-b8dff1e274ce6c9f", "GPU-0000000000001111"
    gpus = read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=commands)).gpus
    assert gpus == (
        GPUState(first, 34208743424 - (2 << 30), 1, "ROCm"),
        GPUState(second, 15 << 30, 0, "ROCm"),
    )
    # ROCR_VISIBLE_DEVICES selects among amdkfd's GPUs, then HIP_VISIBLE_DEVICES (or
    # CUDA_VISIBLE_DEVICES, which HIP reads for it) among those it leaves.
    for environment, expected in (
        ({"ROCR_VISIBLE_DEVICES": "1"}, (second,)),
        ({"ROCR_VISIBLE_DEVICES": second}, (second,)),
        ({"ROCR_VISIBLE_DEVICES": "1,0", "HIP_VISIBLE_DEVICES": "1"}, (first,)),
        ({"CUDA_VISIBLE_DEVICES": "1"}, (second,)),
        ({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"}, (first,)),
        ({"ROCR_VISIBLE_DEVICES": ""}, ()),
    ):
        host = RuntimeHost(tmp_path, affinity=(0,), environment=environment, commands=commands)
        gpus = read_runtime(host=host).gpus
        assert gpus is not None and tuple(gpu.uuid for gpu in gpus) == expected, environment


def test_linux_runtime_without_or_with_a_failing_nvidia_smi(tmp_path):
    linux_runtime(tmp_path)
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,))).gpus == ()
    failing = {SMI_GPUS: Command(9, "", "Failed to initialize NVML: Driver/library mismatch")}
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=failing)).gpus is None
    absent = {SMI_GPUS: Command(6, "No devices were found\n")}
    assert read_runtime(host=RuntimeHost(tmp_path, affinity=(0,), commands=absent)).gpus == ()


def test_linux_runtime_without_affinity_uses_the_online_cpus(tmp_path):
    linux_runtime(tmp_path)
    write(tmp_path, "/sys/devices/system/cpu/online", "0-3\n")
    assert read_runtime(host=RuntimeHost(tmp_path)).cpus == (0, 1, 2, 3)


def test_darwin_runtime_facts(tmp_path):
    counters = {
        "hw.logicalcpu": 10,
        "vm.pagesize": 16384,
        "vm.page_free_count": 1000,
        "vm.vm_page_free_target": 400,
        "vm.page_pageable_external_count": 9000,
        "vm.vm_page_filecache_min": 4000,
        "vm.page_purgeable_count": 70,
        "vm.page_reusable_count": 30,
    }
    commands = {("sysctl", "-n", name): ok(f"{value}\n") for name, value in counters.items()}
    host = RuntimeHost(tmp_path, system="darwin", commands=commands)
    facts = read_runtime(host=host)
    # free above the free target, file pages above the file cache minimum, purgeable, reusable
    assert facts == RuntimeFacts(tuple(range(10)), (), (600 + 5000 + 70 + 30) * 16384, None)
    below = dict(commands)
    below[("sysctl", "-n", "vm.vm_page_free_target")] = ok("5000\n")
    below[("sysctl", "-n", "vm.vm_page_filecache_min")] = ok("12000\n")
    facts = read_runtime(host=RuntimeHost(tmp_path, system="darwin", commands=below))
    assert facts.available_bytes == (70 + 30) * 16384


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, [(0, "GPU-a1"), (1, "GPU-b2")]),
        ("", []),
        ("-1", []),
        ("1", [(1, "GPU-b2")]),
        ("GPU-b", [(1, "GPU-b2")]),
        ("GPU-", []),  # ambiguous: CUDA stops at it
        ("0,7,1", [(0, "GPU-a1")]),  # an unknown device ends the list
        ("1,1", [(1, "GPU-b2")]),
    ],
)
def test_visible_gpus(value, expected):
    assert visible_gpus([(0, "GPU-a1"), (1, "GPU-b2")], value) == expected
