"""Production launch from typed settings; no legacy environment translation."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, cast

from inferssd.configuration.settings import Setting, Snapshot, parse, render, uncomment
from inferssd.launch.darwin import MODEL_OPTIONS, PROFILES
from inferssd.launch.types import AwaitFact, Fact, LaunchInput, Reply, Resolution, State


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
    batch_threads: Optional[int] = None


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
    if platform not in ("linux", "darwin") or s.inputs.profile not in PROFILES:
        raise ValueError("production launch requires linux/darwin and profile dsv/muse")
    if s.inputs.mode != "server":
        raise ValueError("owned typed launch currently supports server mode")
    if not 1 <= spec.port <= 65535 or min(spec.context, spec.batch, spec.ubatch) < 1:
        raise ValueError("invalid port, context or batch size")
    if spec.batch_threads is not None and (
        type(spec.batch_threads) is not int or not 1 <= spec.batch_threads <= 256
    ):
        raise ValueError("batch_threads must be an integer in 1..256")
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
    if any(
        x.split("=", 1)[0] in protected
        or x.startswith("--ssd-")
        or x.startswith(
            ("-m", "-hf", "-np", "-t", "-b", "-c", "-ub", "-ngl", "-ncmoe", "-fa", "-ot", "-sm")
        )
        and not x.startswith("--")
        for x in s.inputs.extra
    ):
        raise ValueError("model, listener and SSD settings use explicit typed launch options")
    expected = "Linux" if platform == "linux" else "Darwin"
    if (
        s.text("platform_system") != expected
        or platform == "darwin"
        and s.text("platform_machine") != "arm64"
    ):
        raise ValueError("launch platform differs from the observed host")
    model = s.text("resolved_path", spec.model)
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", posixpath.basename(model))
    paths = [model]
    if match:
        prefix, first, count = match.groups()
        if first != "00001" or not 1 <= int(count) <= 100000:
            raise ValueError("--model must name the first GGUF shard")
        paths = [
            posixpath.join(posixpath.dirname(model), f"{prefix}-{i:05d}-of-{count}.gguf")
            for i in range(1, int(count) + 1)
        ]
    for path in paths:
        if not s.value("is_file", path):
            raise ValueError("missing GGUF shard: " + path)
    binary = s.text(
        "resolved_path",
        spec.binary
        or s.inputs.repo
        + ("/build-mac" if platform == "darwin" else "/build")
        + "/bin/llama-server",
    )
    if not s.value("is_file", binary) or not s.value("is_executable", binary):
        raise ValueError("missing server executable: " + binary)
    template = s.inputs.repo + "/models/templates/" + PROFILES[s.inputs.profile][1]
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
        artifacts = spec.artifact_directory or directory + "/artifacts"
        s.ask("mkdir", artifacts)
        alias = s.text("mac_alias", s.inputs.profile, artifacts, *paths)
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
        if not s.value("cgroup_v2"):
            raise ValueError("cgroup v2 is required for owned Linux execution")
        uid = s.value("euid")
        if uid != 0:
            if not s.value("user_manager"):
                raise ValueError("no systemd user manager is available")
            if not s.value("memory_controller", str(uid)):
                raise ValueError("the memory controller is not delegated")
        if value.at("vram.budget").value != 0:
            if not s.value("gpu_available"):
                raise ValueError("typed GPU execution requires nvidia-smi")
            if s.text("gpu_processes").strip():
                raise ValueError("the GPU already has a compute process")
    settings = directory + "/settings.ssd"
    telemetry = directory + "/native-events.jsonl"
    s.ask("write_exclusive", settings, materialized_settings(value))
    threads = str(value.at("cpu.threads").value)
    batch_threads = str(spec.batch_threads) if spec.batch_threads is not None else threads
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
        "-t",
        threads,
        "-tb",
        batch_threads,
        "--chat-template-file",
        template,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "-np",
        "1",
    ]
    if platform == "linux":
        command += ["--defer-experts", "-fa", "on"]
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
    else:
        command += [
            "--no-host",
            "--cpu-moe",
            "--no-op-offload",
            "--reasoning",
            "off",
            "--cache-ram",
            "0",
        ]
        if s.inputs.profile == "dsv":
            command += ["-ot", r"blk\.[0-9]+\.engram_embd\.weight=CPU"]
    command += list(s.inputs.extra)
    s.argv = command
    s.emit(
        json.dumps(
            {
                "settings": settings,
                "telemetry": telemetry,
                "effective": value.document(),
                "launch": {
                    "decode_threads": int(threads),
                    "batch_threads": int(batch_threads),
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
