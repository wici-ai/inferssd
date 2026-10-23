"""Pure Apple Silicon GGUF launch policy; filesystem and host facts are explicit."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping

from ssdash.launch.types import AwaitFact, Fact, LaunchInput, Reply, Resolution, State

PROFILES = {
    "dsv": (384, "deepseek-ai-DeepSeek-V4.1.jinja"),
    "muse": (256, "muse-spark.jinja"),
}
MAX_CACHE_MIB = ((1 << 64) - 1) >> 20
MODEL_OPTIONS = frozenset(
    (
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-dr",
        "--docker-repo",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
    )
)
REMOTE_MODEL_ENV = (
    "LLAMA_ARG_MODEL_URL",
    "LLAMA_ARG_DOCKER_REPO",
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
)


def _resolve(s: State) -> None:
    env = s.env
    profile, mode = s.inputs.profile, s.inputs.mode
    if profile not in PROFILES or mode not in ("chat", "server"):
        raise ValueError("Choose profile dsv or muse and mode chat or server")
    if s.text("platform_system") != "Darwin" or s.text("platform_machine") != "arm64":
        raise ValueError("This launcher requires an Apple Silicon Mac")
    model = env.get("SSD_LLAMA_MODEL")
    if not model:
        raise ValueError("Set SSD_LLAMA_MODEL to the first GGUF shard")
    if env.get("SSD_LLAMA_MAC_EXPERT_PACK"):
        raise ValueError(
            "Unset SSD_LLAMA_MAC_EXPERT_PACK for GGUF mode, or use the legacy pack launcher"
        )
    extra = s.inputs.extra[1:] if s.inputs.extra[:1] == ("--",) else s.inputs.extra
    if any(word.split("=", 1)[0] in MODEL_OPTIONS for word in extra) or any(
        env.get(name) for name in REMOTE_MODEL_ENV
    ):
        raise ValueError("Set the model with SSD_LLAMA_MODEL so the expert index matches it")
    first = s.text("resolved_path", model)
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", posixpath.basename(first))
    if match:
        prefix, number, count = match.groups()
        if number != "00001" or int(count) < 1:
            raise ValueError("SSD_LLAMA_MODEL must name the first GGUF shard")
        paths = [
            posixpath.join(posixpath.dirname(first), f"{prefix}-{index:05d}-of-{count}.gguf")
            for index in range(1, int(count) + 1)
        ]
    else:
        paths = [first]
    for path in paths:
        if not s.value("is_file", path):
            raise ValueError(f"Missing GGUF shard: {path}")
    build = s.text("resolved_path", env.get("SSD_LLAMA_BUILD_DIR", s.inputs.repo + "/build-mac"))
    binary = build + "/bin/" + ("llama-cli" if mode == "chat" else "llama-server")
    if not s.value("is_file", binary) or not s.value("is_executable", binary):
        raise ValueError(f"Missing executable: {binary}; run scripts/build-macos-metal.sh first")
    template = s.inputs.repo + "/models/templates/" + PROFILES[profile][1]
    if not s.value("is_file", template):
        raise ValueError(f"Missing model chat template: {template}")

    def integer(name: str, default: int, minimum: int, maximum: int) -> str:
        text = env.get(name, str(default))
        if not re.fullmatch(r"[0-9]+", text) or not minimum <= int(text) <= maximum:
            raise ValueError(f"{name} must be {minimum}..{maximum}")
        return str(int(text))

    settings = {
        "SSD_LLAMA_MAC_EXPERT_CACHE_MIB": integer(
            "SSD_LLAMA_MAC_EXPERT_CACHE_MIB", 4096, 0, MAX_CACHE_MIB
        ),
        "SSD_LLAMA_MAC_EXPERT_READ_AHEAD": integer("SSD_LLAMA_MAC_EXPERT_READ_AHEAD", 2, 0, 4),
        "SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT": integer(
            "SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT", 4096, 1, 16384
        ),
        "SSD_LLAMA_MAC_EXPERT_PREFILL": integer("SSD_LLAMA_MAC_EXPERT_PREFILL", 1, 0, 1),
    }
    if settings["SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT"] not in ("1", "4096", "16384"):
        raise ValueError("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT must be 1, 4096 or 16384")
    policy = env.get("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY", "lfu")
    if policy not in ("lfu", "lru"):
        raise ValueError("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY must be lfu or lru")
    ctx = integer("SSD_LLAMA_CTX", 8192, 1, 1048576)
    batch = integer("SSD_LLAMA_BATCH", 512, 1, 65536)
    ubatch = integer("SSD_LLAMA_UBATCH", 512, 1, 65536)
    threads = integer("SSD_LLAMA_THREADS", 8, 1, 1024)
    prefill_threads = integer("SSD_LLAMA_PREFILL_THREADS", 12, 1, 1024)
    port = integer("SSD_LLAMA_PORT", 8080, 1, 65535)
    cache_dir = posixpath.join(
        env.get("XDG_CACHE_HOME", env.get("HOME", "") + "/Library/Caches"), "ssdash/mac-alias"
    )
    s.ask("mkdir", cache_dir)
    memory = int(s.text("physical_memory"))
    free = int(s.text("disk_free", cache_dir))
    s.emit(
        f"Physical memory: {memory / 1024**3:.1f} GiB; expert cache limit: "
        f"{settings['SSD_LLAMA_MAC_EXPERT_CACHE_MIB']} MiB. Model buffers "
        f"and the OS need additional RAM.\n"
    )
    s.emit(
        f"Reading {len(paths)} GGUF file(s) directly; no expert-pack copy. "
        f"Index storage has {free / 1024**3:.1f} GiB free.\n"
    )
    alias = s.text("mac_alias", profile, cache_dir, *paths)
    s.assigned = list(settings.items()) + [
        ("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", alias),
        ("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY", policy),
    ]
    for name, value in {
        "LLAMA_MMAP_PREFETCH": "0",
        "LLAMA_DSV41_TOPK_REUSE": "0",
        "LLAMA_MAC_COMPACT_METAL": "1",
        "LLAMA_DROP_MMAP_TENSOR_AFTER_UPLOAD": "1",
        "LLAMA_MAC_ADAPTIVE_PREFILL": "0",
    }.items():
        if name not in env:
            s.assigned.append((name, value))
    command = [
        binary,
        "-m",
        first,
        "--mmap",
        "--no-repack",
        "--fit",
        "off",
        "--no-warmup",
        "--no-host",
        "-ngl",
        "99",
        "--cpu-moe",
        "--no-op-offload",
        "-c",
        ctx,
        "-b",
        batch,
        "-ub",
        ubatch,
        "-t",
        threads,
        "-tb",
        prefill_threads,
        "--chat-template-file",
        template,
        "--reasoning",
        "off",
        "--cache-ram",
        "0",
    ]
    if profile == "dsv":
        command.extend(("-ot", r"blk\.[0-9]+\.engram_embd\.weight=CPU"))
    if mode == "server":
        command.extend(("--host", "127.0.0.1", "--port", port, "-np", "1"))
    else:
        command.extend(("-n", "128"))
    command.extend(extra)
    s.argv = command
    s.emit(f"Starting {profile} {mode} with a fresh GGUF alias index: {alias}\n")


def resolve_darwin(inputs: LaunchInput, facts: Mapping[Fact, Reply]) -> Resolution:
    state = State(inputs, facts)
    try:
        _resolve(state)
    except AwaitFact as missing:
        return state.finish(None, missing.fact)
    except ValueError as error:
        state.emit(str(error) + "\n", "stderr")
        return state.finish(2)
    return state.finish(0)
