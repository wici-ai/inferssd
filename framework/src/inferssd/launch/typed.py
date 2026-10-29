"""Production launch from typed settings; no legacy environment translation."""

from __future__ import annotations

import json
import ntpath
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, cast

from inferssd.configuration.settings import PLATFORMS, Setting, Snapshot, parse, render, uncomment
from inferssd.launch.darwin import MODEL_OPTIONS, PROFILES
from inferssd.launch.resources import EngineResources
from inferssd.launch.types import AwaitFact, Fact, LaunchInput, Reply, Resolution, State, Threads


@dataclass(frozen=True)
class LaunchSpec:
    model: str
    port: int
    settings_file: Optional[str] = None
    overrides: tuple[tuple[str, str], ...] = ()
    binary: Optional[str] = None
    artifact_directory: Optional[str] = None
    context: int = 8192
    batch: int = 512
    ubatch: int = 512
    threads: Threads = Threads()
    template: Optional[str] = None
    resources: Optional[EngineResources] = None


def reject_legacy_environment(environment: Mapping[str, str]) -> None:
    prefixes = ("GGML_KIMI_", "GGML_MOE_", "LLAMA_K3_", "LLAMA_DSV41_", "DS4_", "SSD_LLAMA_MAC_")
    launch_names = {
        "MODEL_CONF",
        "SETTINGS",
        "MODEL",
        "MODEL_DIR",
        "EXPERT_INDEX",
        "DENSE_HOMES",
        "EXTRA_ENV",
        "TRACE_ENV",
        "KNOBS_VIA",
        "MEMORY_MAX",
        "CPUSET",
        "THREADS",
        "UBATCH",
        "CONTEXT",
        "SERVER_BIN",
        "LAUNCH_SCOPE",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_HF_FILE",
        "LLAMA_ARG_DOCKER_REPO",
        "SSD_LLAMA_MODEL",
        "SSD_LLAMA_BUILD_DIR",
    }
    for name in environment:
        if name.startswith(prefixes) or re.match(r"TASK[0-9]+_", name) or name in launch_names:
            raise ValueError("retired environment variable " + name + "; use typed launch options")


def literal(item: Setting, platform: str) -> str:
    body, _ = uncomment(render(Snapshot((item,), platform)).split("\n")[1])
    return body.partition("=")[2].strip(" \t\r\n")


def threadpools(threads: Threads, platform: str) -> Threads:
    """Resolve the counts; a strict CPU set sizes the pools and bounds both counts."""
    for count in (threads.decode, threads.batch):
        if count is not None and (type(count) is not int or not 1 <= count <= 256):
            raise ValueError("thread counts must be integers in 1..256")
    cpus = threads.cpus
    # A server CPU mask holds GGML_MAX_N_THREADS bits.
    if len(set(cpus)) != len(cpus) or any(type(x) is not int or not 0 <= x < 512 for x in cpus):
        raise ValueError("strict CPUs must be unique CPU IDs in 0..511")
    if not cpus:
        return Threads(threads.decode, threads.batch or threads.decode)
    if platform == "darwin":
        raise ValueError("macOS cannot pin threads to CPUs")
    decode = threads.decode or len(cpus)
    batch = threads.batch or decode
    if max(decode, batch) > len(cpus):
        raise ValueError("thread counts exceed the strict CPUs")
    return Threads(decode, batch, tuple(sorted(cpus)))


def materialized_settings(value: Snapshot) -> str:
    # Preserve native CLI provenance and registry defaults. Only file/derived
    # settings belong in the generated file; original receipts stay intact.
    return render(
        replace(
            value, entries=tuple(x for x in value.entries if x.origin not in ("cli", "default"))
        )
    )


def _resolve(
    s: State,
    spec: LaunchSpec,
    platform: str,
    directory: str,
    memory: int,
    configured: Optional[Snapshot],
) -> None:
    reject_legacy_environment(s.env)
    if platform not in PLATFORMS:
        raise ValueError("unknown launch platform " + platform)
    if s.inputs.profile not in PROFILES:
        raise ValueError("unknown model profile; choose " + ", ".join(PROFILES))
    if s.inputs.mode != "server":
        raise ValueError("owned typed launch currently supports server mode")
    if not 1 <= spec.port <= 65535 or min(spec.context, spec.batch, spec.ubatch) < 1:
        raise ValueError("invalid port, context or batch size")
    threads = threadpools(spec.threads, platform)
    protected = {
        "-m",
        "--model",
        "--model-url",
        "-hf",
        "--hf-repo",
        "--host",
        "--port",
        "-np",
        "--parallel",
        "-c",
        "--ctx-size",
        "-b",
        "--batch-size",
        "-ub",
        "--ubatch-size",
        "-t",
        "--threads",
        "-tb",
        "--threads-batch",
        "-C",
        "--cpu-mask",
        "-Cr",
        "--cpu-range",
        "--cpu-strict",
        "-Cb",
        "--cpu-mask-batch",
        "-Crb",
        "--cpu-range-batch",
        "--cpu-strict-batch",
        "--fit",
        "--chat-template-file",
        "--chat-template",
        "--reasoning",
        "-ngl",
        "--gpu-layers",
        "--n-gpu-layers",
        "--cpu-moe",
        "-ncmoe",
        "--n-cpu-moe",
        "--no-host",
        "--op-offload",
        "--no-op-offload",
        "--repack",
        "--no-repack",
        "--mmap",
        "--no-mmap",
        "--defer-experts",
        "--no-defer-experts",
        "--warmup",
        "--no-warmup",
        "-fa",
        "--flash-attn",
        "--cache-ram",
        "--cache-disk",
        "--cache-disk-mib",
        "-ot",
        "--override-tensor",
        "-ts",
        "--tensor-split",
        "-sm",
        "--split-mode",
        "-mg",
        "--main-gpu",
    }
    protected.update(MODEL_OPTIONS)
    glued = (
        "-m",
        "-hf",
        "-np",
        "-t",
        "-C",
        "-b",
        "-c",
        "-ub",
        "-ngl",
        "-ncmoe",
        "-fa",
        "-ot",
        "-sm",
    )
    for x in s.inputs.extra:
        name = x.split("=", 1)[0]
        if name.startswith("--"):
            name = name.replace("_", "-")  # the server's long option spelling
        if name in protected or name.startswith(("--ssd-",) + glued):
            raise ValueError("model, listener and SSD settings use explicit typed launch options")
    expected = {"linux": "Linux", "darwin": "Darwin", "windows": "Windows"}[platform]
    if (
        s.text("platform_system") != expected
        or platform == "darwin"
        and s.text("platform_machine") != "arm64"
    ):
        raise ValueError("launch platform differs from the observed host")
    # Host paths follow the launch platform's conventions, wherever this resolver runs.
    os_path = ntpath if platform == "windows" else posixpath
    model = s.text("resolved_path", spec.model)
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", os_path.basename(model))
    paths = [model]
    if match:
        prefix, first, count = match.groups()
        if first != "00001" or not 1 <= int(count) <= 100000:
            raise ValueError("--model must name the first GGUF shard")
        paths = [
            os_path.join(os_path.dirname(model), f"{prefix}-{i:05d}-of-{count}.gguf")
            for i in range(1, int(count) + 1)
        ]
    for path in paths:
        if not s.value("is_file", path):
            raise ValueError("missing GGUF shard: " + path)
    resources = spec.resources
    if resources is not None and resources.platform not in (None, platform):
        raise ValueError("engine resource platform differs from the selected platform")
    binary_path = spec.binary or (resources.binary if resources else None)
    if not binary_path and s.inputs.repo:
        # Preset build directories first, then the conventional ad hoc ones.
        builds = {
            "linux": ("build-linux-cuda", "build-linux-cpu", "build"),
            "darwin": ("build-macos-metal", "build-mac", "build"),
            "windows": ("build-x64-windows-llvm-release", "build"),
        }[platform]
        server = "inferssd-server.exe" if platform == "windows" else "inferssd-server"
        candidates = [os_path.join(s.inputs.repo, name, "bin", server) for name in builds]
        binary_path = next((c for c in candidates if s.value("is_file", c)), candidates[0])
    if not binary_path:
        raise ValueError("provide --binary or an engine resource descriptor")
    binary = s.text("resolved_path", binary_path)
    if not s.value("is_file", binary) or not s.value("is_executable", binary):
        raise ValueError("missing server executable: " + binary)
    template_path = spec.template or (resources.template(s.inputs.profile) if resources else None)
    if not template_path and s.inputs.repo:
        bundled = PROFILES[s.inputs.profile][1]
        template_path = os_path.join(s.inputs.repo, "engine", "models", "templates", bundled)
    if not template_path:
        raise ValueError("provide --template or an engine resource descriptor")
    template = s.text("resolved_path", template_path)
    if not s.value("is_file", template):
        raise ValueError("missing chat template: " + template)
    if configured is None:
        text = s.text("read_text", spec.settings_file) if spec.settings_file else None
        value = parse(text, spec.settings_file or "", overrides=spec.overrides, platform=platform)
    else:
        value = configured
    if value.platform != platform:
        raise ValueError("typed snapshot platform differs from execution policy")
    if s.inputs.profile == "dsv":
        value = replace(
            value,
            entries=tuple(
                replace(
                    item,
                    value=1,
                    origin="file",
                    location="model profile dsv",
                    receipt=" source=model; profile=dsv",
                )
                if item.key in ("model.engram_prefetch", "model.hc_fused")
                and item.origin == "default"
                else item
                for item in value.entries
            ),
        )
    if not value.at("expert.source").value:
        if value.at("expert.source").origin != "default":
            raise ValueError("typed launch requires a nonempty explicit expert.source")
        artifacts = spec.artifact_directory or os_path.join(directory, "artifacts")
        s.ask("mkdir", artifacts)
        alias = s.text("gguf_alias", artifacts, *paths)
        selected = Setting(
            "expert.source",
            (alias,),
            "file",
            "strict GGUF header index",
            " source=generated; input=complete GGUF shard/header validation",
        )
        value = replace(
            value, entries=tuple(selected if x.key == selected.key else x for x in value.entries)
        )
    for path in cast(tuple[str, ...], value.at("expert.source").value):
        if not isinstance(path, str) or not s.value("is_file", path):
            raise ValueError("missing typed expert source: " + str(path))
    if platform == "linux":
        guide = "; see docs/quickstart.md#linux-requirements"
        if not s.value("systemd"):
            raise ValueError(
                "systemd is required for owned Linux execution: it is not the init system "
                "here, as in most containers" + guide
            )
        if not s.value("cgroup_v2"):
            raise ValueError(
                "cgroup v2 is required for owned Linux execution: "
                "/sys/fs/cgroup/cgroup.controllers is missing" + guide
            )
        uid = s.value("euid")
        if uid != 0:
            if not s.value("user_manager"):
                raise ValueError(
                    "no systemd user manager is available: start a login session, "
                    "run `loginctl enable-linger`, or launch as root" + guide
                )
            if not s.value("memory_controller", str(uid)):
                raise ValueError(
                    "the memory controller is not delegated to your systemd user manager: "
                    "add Delegate=memory to a user@.service drop-in, or launch as root" + guide
                )
        if value.at("vram.budget").value != 0:
            if not s.value("gpu_available"):
                raise ValueError("typed GPU execution requires nvidia-smi")
            if s.text("gpu_processes").strip():
                raise ValueError("the GPU already has a compute process")
    # The sink's I/O thread keeps its own CPUs; decode runs on the first decode threads.
    if set(cast(tuple[int, ...], value.at("cpu.io_cpus").value)) & set(threads.cpus):
        raise ValueError("cpu.io_cpus overlaps the strict compute CPUs")
    participants = cast(int, value.at("cpu.decode_participants").value)
    if threads.decode is not None and participants > threads.decode:
        raise ValueError("cpu.decode_participants exceeds the decode thread count")
    settings = os_path.join(directory, "settings.ssd")
    telemetry = os_path.join(directory, "native-events.jsonl")
    s.ask("write_exclusive", settings, materialized_settings(value))
    command = [binary, "-m", model, "--ssd-settings", settings, "--ssd-telemetry", telemetry]
    for item in value.entries:
        if item.origin == "cli":
            command += [
                "--ssd-" + item.key.replace(".", "-").replace("_", "-"),
                literal(item, platform),
            ]
    command += [
        "--mmap",
        "--no-repack",
        "--fit",
        "off",
        "--no-warmup",
        "-ngl",
        "99",
        "-c",
        str(spec.context),
        "-b",
        str(spec.batch),
        "-ub",
        str(spec.ubatch),
    ]
    if threads.decode is not None:
        command += ["-t", str(threads.decode)]
    if threads.batch is not None:
        command += ["-tb", str(threads.batch)]
    if threads.cpus:
        mask = hex(sum(1 << cpu for cpu in threads.cpus))
        command += ["-C", mask, "-Cb", mask, "--cpu-strict", "1", "--cpu-strict-batch", "1"]
    command += [
        "--chat-template-file",
        template,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "-np",
        "1",
        # The SSD executor reads the routed experts; their mapped pages stay unread.
        "--defer-experts",
    ]
    if platform == "linux":
        command += ["-fa", "on"]
        s.scope = (
            ["systemd-run"]
            + (["--user"] if uid != 0 else [])
            + [
                "--scope",
                "--quiet",
                "--unit=inferssd-pending.scope",
                "-p",
                "MemoryMax=" + str(memory),
                "-p",
                "MemorySwapMax=0",
            ]
        )
    # Elsewhere Flash Attention keeps the server's auto choice, on for each layer
    # whose device runs it (every CPU layer), and the settings' host.budget_mib,
    # passed as given, bounds the process in place of a scope.
    command += list(s.inputs.extra)
    s.argv = command
    s.emit(
        json.dumps(
            {
                "settings": settings,
                "telemetry": telemetry,
                "effective": value.document(),
                "launch": {
                    "decode_threads": threads.decode,
                    "batch_threads": threads.batch,
                    "cpus": list(threads.cpus),
                    "context": spec.context,
                    "batch": spec.batch,
                    "ubatch": spec.ubatch,
                },
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def resolve_typed(
    inputs: LaunchInput,
    facts: Mapping[Fact, Reply],
    *,
    spec: LaunchSpec,
    platform: str,
    directory: str,
    memory: int,
    configured: Optional[Snapshot] = None,
) -> Resolution:
    state = State(inputs, facts)
    try:
        _resolve(state, spec, platform, directory, memory, configured)
    except AwaitFact as missing:
        return state.finish(None, missing.fact)
    return state.finish(0)
