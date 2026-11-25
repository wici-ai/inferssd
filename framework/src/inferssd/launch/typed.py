"""Production launch from typed settings; no legacy environment translation."""

from __future__ import annotations

import ipaddress
import json
import ntpath
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Optional, cast

from inferssd.configuration.settings import (
    PLATFORMS,
    Setting,
    Snapshot,
    parse,
    render,
    schema,
    uncomment,
)
from inferssd.launch.resources import EngineResources, server_binary
from inferssd.launch.types import AwaitFact, Fact, LaunchInput, Reply, Resolution, State, Threads
from inferssd.machine.identity import TIER_BACKENDS
from inferssd.strategy.inputs import model_facts_from, runtime_facts_from
from inferssd.strategy.launch import Planned, plan

# The engine options that name a model: the launch names the model itself.
MODEL_OPTIONS = frozenset(
    ("-m", "--model", "-mu", "--model-url", "-dr", "--docker-repo")
    + ("-hf", "-hfr", "--hf-repo", "-hff", "--hf-file")
)

# Each model profile's chat template: a bundled template replaces the GGUF's,
# and None keeps the template in the GGUF. The expert count comes from the model.
# A model with routed experts needs a profile, for the engine executes routed experts
# by architecture; a model without them runs without one.
PROFILES: dict[str, Optional[str]] = {
    "dsv": "deepseek-ai-DeepSeek-V4.1.jinja",
    "muse": "muse-spark.jinja",
    "qwen4exp": None,  # Qwen3.8-Flash-Next
    "qwen35moe": None,  # Qwen3.6-35B-A3B; its GGUF carries the model's sampling
    "gemma4": None,  # Gemma 4
}
# The profile of each GGUF general.architecture; --profile overrides it.
ARCHITECTURES = {
    "deepseek41": "dsv",
    "muse-spark": "muse",
    "qwen4exp": "qwen4exp",
    "qwen35moe": "qwen35moe",
    "gemma4": "gemma4",
}
# The server's default sampling for each profile, from the model's own
# generation config where the GGUF carries none; a request's values win.
SAMPLING: dict[str, tuple[str, ...]] = {
    "dsv": ("--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0"),
}
# The address of this machine only, where the server listens by default.
LOOPBACK = "127.0.0.1"


@dataclass(frozen=True)
class LaunchSpec:
    model: str
    port: int
    settings_file: Optional[str] = None
    overrides: tuple[tuple[str, str], ...] = ()
    binary: Optional[str] = None
    # Unset sizes take CONTEXT and BATCH; with the strategy, one pass per
    # prompt: -b and -ub as large as the context.
    context: Optional[int] = None
    batch: Optional[int] = None
    ubatch: Optional[int] = None
    threads: Threads = Threads()
    template: Optional[str] = None
    resources: Optional[EngineResources] = None
    # Derive the launch strategy from the machine characteristics fact. Without
    # it the launch resolves from its settings and options alone.
    strategy: bool = False
    # The directory that keeps each model's prompt cache across restarts, in a
    # subdirectory named after the model; None runs without a prompt cache.
    prompt_cache: Optional[str] = None
    # The IPv4 address the server listens on; see listen_host.
    host: str = LOOPBACK
    # The model's extra GGUF shards, the server's --shard/--ple values as given (comma separated
    # paths, any number of them): a per-layer embedding table kept apart from the model, say. The
    # engine's facts name the files the model loads from.
    shards: tuple[str, ...] = ()
    # The prompt cache's size limit per model in MiB; None takes PROMPT_CACHE_MIB.
    prompt_cache_mib: Optional[int] = None
    # The engine's event for each layer and each prompt graph, which the benchmarks read
    # (docs/architecture/native-observations.md). Without them the events file holds the
    # settings alone, and a long run does not grow it.
    layer_events: bool = True
    # The directory of the prefix caches that came with each model (the WiCi AI desktop app
    # downloads them), in a subdirectory named as the prompt cache's; the engine only reads
    # them. None passes none.
    prefix_cache: Optional[str] = None


def listen_host(text: str) -> str:
    """TEXT as the IPv4 address the server listens on.

    127.0.0.1 is this machine only and 0.0.0.0 every network interface. The
    launch confirms its listener by address, so a host name is not accepted.
    """
    try:
        return str(ipaddress.IPv4Address(text))
    except ValueError:
        raise ValueError("host must be an IPv4 address, such as 127.0.0.1 or 0.0.0.0") from None


def connect_host(host: str) -> str:
    """The address at which this machine reaches a server listening on HOST."""
    return LOOPBACK if host == "0.0.0.0" else host


# A context left unset: 0, which the engine resolves to the largest multiple of 8192 up to the
# model's trained context whose KV cache fits memory when full (it holds memory only for the
# cells in use).
AUTO_CONTEXT = 0
# One pass a prompt with the context unset: a batch the engine bounds by the context it chose.
ONE_PASS_BATCH = 1 << 20
BATCH = 512
# The disk prompt cache's size limit per model, unless the launch names one. A cache holds the
# conversations someone may go on with; the least recently used are dropped past the limit.
PROMPT_CACHE_MIB = 4096
# DeepSeek-V4.1's template drops earlier turns' thinking by default, so a
# history rendered again differs from the cached prompt; keep it.
DSV_TEMPLATE_KWARGS = '{"drop_thinking": false}'


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
    """Resolve the counts; a strict CPU set sizes the pools and bounds both counts.

    A decode CPU set sizes and bounds the decode count alone; the batch count
    is then always given, so the server's batch pool never inherits the decode
    pool's CPUs.
    """
    for count in (threads.decode, threads.batch):
        if count is not None and (type(count) is not int or not 1 <= count <= 256):
            raise ValueError("thread counts must be integers in 1..256")
    cpus, decode_cpus = threads.cpus, threads.decode_cpus
    # A server CPU mask holds GGML_MAX_N_THREADS bits.
    for ids in (cpus, decode_cpus):
        if len(set(ids)) != len(ids) or any(type(x) is not int or not 0 <= x < 512 for x in ids):
            raise ValueError("strict CPUs must be unique CPU IDs in 0..511")
    if cpus and decode_cpus:
        raise ValueError("a decode CPU set excludes a CPU set for both thread pools")
    if not cpus and not decode_cpus:
        return Threads(threads.decode, threads.batch or threads.decode)
    if platform == "darwin":
        raise ValueError("macOS cannot pin threads to CPUs")
    if decode_cpus:
        decode = threads.decode or len(decode_cpus)
        if decode > len(decode_cpus):
            raise ValueError("the decode thread count exceeds the decode CPUs")
        return Threads(decode, threads.batch or decode, (), tuple(sorted(decode_cpus)))
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


def cpu_defaults(raw: object, allowed: Sequence[int], given: Threads) -> Threads:
    """Default CPU work to the available fast cluster; preserve explicit counts."""
    if not isinstance(raw, Mapping) or type(raw.get("affinity_supported")) is not bool:
        raise ValueError("invalid CPU topology")
    clusters = raw.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("invalid CPU clusters")
    fast: set[int] = set()
    for cluster in clusters:
        if not isinstance(cluster, Mapping) or not isinstance(cluster.get("cpus"), list):
            raise ValueError("invalid CPU cluster")
        ids = cluster["cpus"]
        if any(type(cpu) is not int or cpu < 0 for cpu in ids):
            raise ValueError("invalid CPU IDs")
        if cluster.get("kind") == "performance":
            fast.update(ids)
    selected = sorted(set(allowed) & fast) or sorted(set(allowed))
    if not selected:
        raise ValueError("no available CPUs for inference")
    count = min(256, len(selected))
    decode, batch = given.decode or count, given.batch or count
    if raw["affinity_supported"] and max(selected) < 512 and max(decode, batch) <= len(selected):
        return Threads(decode, batch, tuple(selected))
    return Threads(decode, batch)


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
    if s.inputs.profile and s.inputs.profile not in PROFILES:
        raise ValueError("unknown model profile; choose " + ", ".join(PROFILES))
    if s.inputs.mode != "server":
        raise ValueError("owned typed launch currently supports server mode")
    sizes = [size for size in (spec.context, spec.batch, spec.ubatch) if size is not None]
    if not 1 <= spec.port <= 65535 or any(size < 1 for size in sizes):
        raise ValueError("invalid port, context or batch size")
    listen_host(spec.host)
    threads = threadpools(spec.threads, platform)
    protected = {
        "-m",
        "--model",
        "--shard",
        "--ple",
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
        "--chat-template-kwargs",
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
        "--prefix-cache",
        "--ssd-no-layer-events",
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
    resources = spec.resources
    if resources is not None and resources.platform not in (None, platform):
        raise ValueError("engine resource platform differs from the selected platform")
    binary_path = server_binary(
        platform, spec.binary, resources, s.inputs.repo, lambda path: bool(s.value("is_file", path))
    )
    if not binary_path:
        raise ValueError("provide --binary or an engine resource descriptor")
    binary = s.text("resolved_path", binary_path)
    if not s.value("is_file", binary) or not s.value("is_executable", binary):
        raise ValueError("missing server executable: " + binary)
    # The engine's facts of the model, read once for the profile and the strategy. They name the
    # files the engine loads the model from: its shards, as its split numbering and any --shard
    # values give them.
    facts = model_facts_from(s.value("model_facts", binary, model, *spec.shards))
    paths = list(facts.paths)
    for path in paths:
        if not s.value("is_file", path):
            raise ValueError("missing GGUF file: " + path)
    experts = facts.experts is not None
    profile = s.inputs.profile or ARCHITECTURES.get(facts.architecture or "")
    if not profile and experts:
        raise ValueError(
            f"no model profile for the GGUF architecture {facts.architecture}; choose one with "
            "--profile " + ", ".join(PROFILES)
        )
    # The engine's GPU backends: the descriptor's, else its build's; None when neither says.
    gpu_backends = resources.gpu_backends if resources is not None else None
    if gpu_backends is None:
        built = s.value("engine_gpu_backends", binary)
        gpu_backends = None if built is None else tuple(cast(list[str], built))
    bundled = PROFILES[profile] if profile else None
    template_path = spec.template
    if not template_path and profile and bundled:
        template_path = resources.template(profile) if resources else None
        if not template_path and s.inputs.repo:
            template_path = os_path.join(s.inputs.repo, "engine", "models", "templates", bundled)
        if not template_path:
            raise ValueError("provide --template or an engine resource descriptor")
    template = s.text("resolved_path", template_path) if template_path else None
    if template is not None and not s.value("is_file", template):
        raise ValueError("missing chat template: " + template)
    if configured is None:
        text = s.text("read_text", spec.settings_file) if spec.settings_file else None
        value = parse(
            text,
            spec.settings_file or "",
            overrides=spec.overrides,
            platform=platform,
            warn=lambda message: s.emit(message + "\n", "stderr"),
        )
    else:
        value = configured
    if value.platform != platform:
        raise ValueError("typed snapshot platform differs from execution policy")
    if profile == "dsv":
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
    planned: Optional[Planned] = None
    if spec.strategy:
        # The strategy fills what the user left unset; the checks below see the result.
        planned = plan(
            s,
            platform,
            memory,
            paths,
            lambda: facts,
            value,
            spec.threads,
            gpu_backends,
        )
        value, threads = planned.settings, threadpools(planned.threads, platform)
        if any(argument in protected for argument in planned.server_args[::2]):
            raise ValueError("the launch strategy names a protected server option")
        # This newly supported CPU model has no measured thread/compute priors on
        # many existing profiles. Avoid the unpriced all-core fallback on hybrid
        # machines; priced strategies and explicit user thread choices still win.
        costs = dict(cast(tuple[tuple[str, float], ...], value.at("place.costs").value))
        if (
            profile == "qwen4exp"
            and gpu_backends == ()
            and spec.threads == Threads()
            and not {"cpu_fixed_us", "cpu_route_us"} <= costs.keys()
            and value.at("cpu.decode_participants").value == 0
        ):
            runtime = runtime_facts_from(s.value("runtime_facts"))
            io_cpus = set(cast(tuple[int, ...], value.at("cpu.io_cpus").value))
            threads = threadpools(
                cpu_defaults(
                    s.value("cpu_topology"),
                    [cpu for cpu in runtime.cpus if cpu not in io_cpus],
                    spec.threads,
                ),
                platform,
            )
            chosen = dict(cast(Mapping[str, object], planned.receipt.get("chosen", {})))
            actual = {
                "decode": threads.decode,
                "batch": threads.batch,
                "cpus": list(threads.cpus),
                "decode_cpus": list(threads.decode_cpus),
            }
            strategy_receipt = dict(
                planned.receipt,
                model_defaults={
                    "reason": "qwen4exp CPU compute priors unavailable; use the live fast cluster",
                    "previous_threads": chosen.get("threads", {}),
                    "threads": actual,
                },
            )
            chosen["threads"] = actual
            strategy_receipt["chosen"] = chosen
            planned = replace(planned, threads=threads, receipt=strategy_receipt)
    if not experts and value.at("expert.source").value:
        raise ValueError("expert.source names routed experts; the model has none")
    if experts and not value.at("expert.source").value:
        if value.at("expert.source").origin != "default":
            raise ValueError("typed launch requires a nonempty explicit expert.source")
        # The engine indexes the routed experts from the model's own GGUF headers.
        selected = Setting(
            "expert.source",
            tuple(paths),
            "file",
            "the model's GGUF files",
            " source=model; input=its GGUF shards",
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
    # The launch reads the VRAM tier's GPUs through their driver: nvidia-smi for CUDA, amdkfd for
    # ROCm. With the engine's backends unknown, the engine itself refuses a budget it has no tier
    # for.
    tier = set(TIER_BACKENDS) & set(gpu_backends or ())
    if value.at("vram.budget").value != 0 and tier:
        if not s.value("gpu_available"):
            raise ValueError(
                "typed GPU execution requires the GPUs' driver: "
                "nvidia-smi for CUDA, amdkfd for ROCm"
            )
    # A setting of the expert tier needs an engine built with a backend that carries it.
    for item in value.entries:
        backend = schema()["properties"][item.key].get("backend")
        if backend and item.origin != "default" and gpu_backends is not None:
            if not set(backend.split(",")) & set(gpu_backends):
                raise ValueError(
                    f"{item.key} ({item.location}) configures the expert tier of "
                    + " or ".join(backend.split(","))
                    + ", which this engine was built without"
                )
    # The sink's I/O thread keeps its own CPUs; decode runs on the first decode threads.
    if set(cast(tuple[int, ...], value.at("cpu.io_cpus").value)) & set(
        threads.cpus + threads.decode_cpus
    ):
        raise ValueError("cpu.io_cpus overlaps the strict compute CPUs")
    participants = cast(int, value.at("cpu.decode_participants").value)
    if threads.decode is not None and participants > threads.decode:
        raise ValueError("cpu.decode_participants exceeds the decode thread count")
    settings = os_path.join(directory, "settings.ssd")
    telemetry = os_path.join(directory, "native-events.jsonl")
    s.ask("write_exclusive", settings, materialized_settings(value))
    if planned is not None:
        receipt = json.dumps(planned.receipt, ensure_ascii=False, indent=2) + "\n"
        s.ask("write_exclusive", os_path.join(directory, "strategy.json"), receipt)
    context = spec.context or AUTO_CONTEXT
    one_pass = planned is not None and planned.one_pass
    batch = spec.batch or ((context or ONE_PASS_BATCH) if one_pass else BATCH)
    ubatch = spec.ubatch or (batch if one_pass else BATCH)
    if planned is not None and one_pass:
        given = [name for name, size in (("-b", spec.batch), ("-ub", spec.ubatch)) if size]
        cast(list[str], planned.receipt["kept"]).extend("server " + name for name in given)
    command = [binary, "-m", model]
    for shard in spec.shards:
        command += ["--shard", shard]
    command += ["--ssd-settings", settings, "--ssd-telemetry", telemetry]
    if not spec.layer_events:
        command.append("--ssd-no-layer-events")
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
        str(context),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
    ]
    if threads.decode is not None:
        command += ["-t", str(threads.decode)]
    if threads.batch is not None:
        command += ["-tb", str(threads.batch)]
    if threads.cpus:
        mask = hex(sum(1 << cpu for cpu in threads.cpus))
        command += ["-C", mask, "-Cb", mask, "--cpu-strict", "1", "--cpu-strict-batch", "1"]
    elif threads.decode_cpus:
        # The batch pool has its own count (-tb above), so it keeps no CPU mask.
        command += ["-C", hex(sum(1 << cpu for cpu in threads.decode_cpus)), "--cpu-strict", "1"]
    if template is not None:
        command += ["--chat-template-file", template]
    if profile == "dsv":
        command += ["--chat-template-kwargs", DSV_TEMPLATE_KWARGS]
    sampling = SAMPLING.get(profile or "", ())
    named = {x.split("=", 1)[0] for x in s.inputs.extra}
    for option, number in zip(sampling[::2], sampling[1::2]):
        if option not in named:
            command += [option, number]
    if spec.prompt_cache is None:
        command += ["--cache-ram", "0"]
    else:
        # One directory per model, named as its file is without the shard suffix.
        match = re.fullmatch(r"(.+)-\d{5}-of-\d{5}\.gguf", os_path.basename(model))
        name = match.group(1) if match else os_path.basename(model)
        if name.endswith(".gguf"):
            name = name[: -len(".gguf")]
        if spec.prompt_cache_mib is not None and spec.prompt_cache_mib < 1:
            raise ValueError("--prompt-cache-mib must be positive")
        command += [
            "--cache-disk",
            os_path.join(spec.prompt_cache, name),
            "--cache-disk-mib",
            str(spec.prompt_cache_mib or PROMPT_CACHE_MIB),
        ]
        if spec.prefix_cache is not None:
            command += ["--prefix-cache", os_path.join(spec.prefix_cache, name)]
    command += [
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "-np",
        "1",
    ]
    if experts:
        # The SSD executor reads the routed experts; their mapped pages stay unread.
        command += ["--defer-experts"]
    # Flash Attention stays the server's auto choice on every platform: on for each layer whose
    # device runs it, which is every layer wherever the backends implement it.
    if platform == "linux":
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
    if planned is not None:
        command += list(planned.server_args)
    command += list(s.inputs.extra)
    s.argv = command
    output: dict[str, object] = {
        "settings": settings,
        "telemetry": telemetry,
        "effective": value.document(),
        "profile": profile,
        "launch": {
            "decode_threads": threads.decode,
            "batch_threads": threads.batch,
            "cpus": list(threads.cpus),
            "decode_cpus": list(threads.decode_cpus),
            "context": context,
            "batch": batch,
            "ubatch": ubatch,
        },
    }
    if planned is not None:
        output["strategy"] = planned.receipt
    s.emit(json.dumps(output, ensure_ascii=False) + "\n")


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
