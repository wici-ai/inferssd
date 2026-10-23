"""Fixed inputs and receipts for native final-benchmark wrappers.

Model hashes come from an earlier complete verification. Only their recorded
file identities are rechecked here; large weight files are never rehashed by a
benchmark arm. This is not protection against a privileged concurrent writer.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import final_chat
from native_core import CORE, IDENTITY as CORE_IDENTITY

group_exists = CORE["process"].group_exists
stop_owned = CORE["process"].stop_owned

SHA = re.compile(r"[0-9a-f]{64}")
FILE_FIELDS = ("dev", "ino", "bytes", "mtime_ns", "ctime_ns")
MANAGED_ARGS = {"-m", "--model", "-c", "--ctx-size", "-np", "--parallel", "--host", "--port",
                "--warmup", "--no-warmup", "--fit", "--cache-ram", "--cache-disk", "--cache-disk-mib",
                "--chat-template-file", "--chat-template", "--reasoning", "--api-key", "--api-key-file",
                "--model-url", "-mu", "--hf-repo", "-hf", "--hf-file", "-hff", "--docker-repo",
                "--ssd-settings", "--ssd-telemetry", "--log-disable", "--log-file", "--verbosity", "--log-verbosity", "-lv"}
LEGACY_SOURCES = {"GGML_MOE_EXPERT_PACK", "GGML_MOE_EXPERT_PACK_LIST", "GGML_MOE_EXPERT_PACK_OVERLAY",
                  "GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA", "GGML_MOE_EXPERT_PACK_V2", "GGML_MOE_PAGECACHE_PACK",
                  "GGML_MOE_Q4_COLIBRI_PACK", "GGML_MOE_GLM52_COLIBRI_PACK", "GGML_MOE_Q4_COLIBRI_OVERLAY_PACK",
                  "GGML_MOE_Q4_COLIBRI_OVERLAY_MANIFEST", "GGML_MOE_LOSSLESS_PACK", "GGML_MOE_GLM52_LOSSLESS_PACK",
                  "GGML_MOE_DS4_LOSSLESS_PACK", "GGML_MOE_STREAM_DS4_TRIPLET_PACK", "SSD_LLAMA_MAC_EXPERT_PACK"}


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_identity(path, hash_file=False):
    path = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        if hash_file:
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    project = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if project(before) != project(after) or project(after) != project(path.stat()):
        raise ValueError("file changed during identity check")
    result = dict(zip(FILE_FIELDS, project(after)), path=str(path))
    if hash_file:
        result["sha256"] = digest.hexdigest()
    return result


def verify_file(spec, *, model=False):
    actual = file_identity(spec["path"], hash_file=not model)
    fields = FILE_FIELDS if model else ("sha256",)
    if any(actual[key] != spec[key] for key in fields):
        raise ValueError(f"recorded file identity differs: {spec['path']}")
    if model:
        actual["sha256"] = spec["sha256"]
    return actual


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def file_spec(spec, *, model=False):
    if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
            or not Path(spec["path"]).is_absolute()
            or not isinstance(spec.get("sha256"), str) or not SHA.fullmatch(spec["sha256"])):
        raise ValueError("absolute file path and lowercase SHA-256 required")
    if model:
        for name in FILE_FIELDS:
            positive(spec.get(name), name, 0 if name in ("dev", "mtime_ns", "ctime_ns") else 1)


def aliases(profile):
    return profile["aliases"] if "aliases" in profile else [profile["alias"]]


def validate_profile(profile):
    if profile.get("schema_version") != 1 or profile.get("platform") not in ("linux", "macos"):
        raise ValueError("native profile schema 1 and linux/macos platform required")
    if not re.fullmatch(r"[0-9a-f]{40}", profile.get("revision", "")):
        raise ValueError("full final source revision required")
    file_spec(profile["binary"])
    images = profile["images"]
    if not isinstance(images, list) or not images:
        raise ValueError("all expected project libraries are required")
    for spec in images:
        file_spec(spec)
    if len({x["path"] for x in images}) != len(images):
        raise ValueError("duplicate expected image")
    shards = profile["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("complete verified model shard list required")
    for shard in shards:
        file_spec(shard, model=True)
    if len({x["path"] for x in shards}) != len(shards):
        raise ValueError("duplicate model shard")
    names = [Path(x["path"]).name for x in shards]
    split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", names[0])
    if split:
        prefix, first, total = split.groups()
        expected = [f"{prefix}-{i:05d}-of-{int(total):05d}.gguf" for i in range(1, int(total) + 1)]
        if first != "00001" or names != expected or len({str(Path(x['path']).parent) for x in shards}) != 1:
            raise ValueError("all shards, in order from the first shard, are required")
    elif len(shards) != 1 or not names[0].endswith(".gguf"):
        raise ValueError("unrecognized model shard set")
    file_spec(profile["template"])
    if "aliases" in profile and "alias" in profile:
        raise ValueError("choose alias or aliases, not both")
    sources = aliases(profile)
    if not isinstance(sources, list) or not sources or (profile["platform"] == "macos" and len(sources) != 1):
        raise ValueError("Linux needs a nonempty alias list; Mac needs exactly one alias")
    for source in sources:
        file_spec(source)
    args = profile.get("args")
    if not isinstance(args, list) or any(not isinstance(x, str) or not x for x in args):
        raise ValueError("args must be an explicit string list")
    if any(word.split("=", 1)[0] in MANAGED_ARGS or word.startswith("--ssd-") for word in args):
        raise ValueError("wrapper owns model, context, template, port, slot, logging and warmup arguments")
    env = profile.get("environment")
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("explicit environment string mapping required")
    if any(k.startswith(("H2D_", "LLAMA_ARG_")) for k in env):
        raise ValueError("H2D caps and implicit CLI environment are not permitted")
    if any(env.get(key) for key in LEGACY_SOURCES):
        raise ValueError("formal runs use direct GGUF, not legacy expert packs")
    if "typed_settings" in profile:
        file_spec(profile["typed_settings"])
        if any(k.startswith(("GGML_KIMI_", "GGML_MOE_", "LLAMA_K3_", "LLAMA_DSV41_", "DS4_", "SSD_LLAMA_MAC_"))
               or re.match(r"TASK[0-9]+_", k) for k in env):
            raise ValueError("typed profiles reject retired environment controls")
        typed_snapshot(profile)
    elif env.get("GGML_MOE_EXPERT_GGUF_ALIAS_TSV") != ":".join(x["path"] for x in sources):
        raise ValueError("source environment must use the exact reviewed GGUF alias")
    if any(k.startswith("DYLD_") or k in ("LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG") for k in env):
        raise ValueError("wrapper owns runtime library search paths")
    if env.get("LD_PRELOAD"):
        preloads = env["LD_PRELOAD"].replace(":", " ").split()
        if any(path not in {x["path"] for x in images} or "h2d" in Path(path).name.lower() for path in preloads):
            raise ValueError("preloads must be pinned project images and must not cap H2D")
    resources = profile["resources"]
    for key in ("memory_budget_bytes", "host_reserve_bytes", "minimum_cpus"):
        positive(resources.get(key), key)
    if profile["platform"] == "linux":
        positive(resources.get("minimum_memlock_bytes"), "minimum_memlock_bytes", 0)
    if profile["platform"] == "macos":
        for key in ("max_compressor_growth_bytes", "max_swap_growth_bytes"):
            positive(resources.get(key), key, 0)
    settings = profile["effective_settings"]
    if settings.get("source_mode") != "direct-gguf" or settings.get("h2d_limit_bytes_per_second", 0) is not None:
        raise ValueError("direct GGUF and no H2D cap required")
    for key in ("expert_cache_budget_bytes", "vram_budget_bytes", "unified_memory_budget_bytes", "gpu_reserve_bytes"):
        positive(settings.get(key), key, 0)
    positive(settings.get("context_tokens"), "context_tokens", 257)
    for key in ("load_timeout_s", "cleanup_timeout_s"):
        final_chat.bounded_timeout(profile[key])
    if profile["cleanup_timeout_s"] > 30:
        raise ValueError("cleanup deadline must be <= 30 seconds")
    return profile


def typed_snapshot(profile):
    try:
        from inferssd.configuration.settings import parse
    except ImportError as error:
        raise ValueError("typed profiles require the installed settings framework") from error
    snapshot = parse(Path(profile["typed_settings"]["path"]).read_text(encoding="utf-8"),
                     profile["typed_settings"]["path"],
                     platform="darwin" if profile["platform"] == "macos" else "linux")
    if snapshot.at("expert.source").value != tuple(x["path"] for x in aliases(profile)):
        raise ValueError("typed expert.source differs from the exact reviewed aliases")
    return snapshot


def verify_inputs(profile):
    build = [verify_file(x) for x in [profile["binary"], *profile["images"]]]
    shards = [verify_file(x, model=True) for x in profile["shards"]]
    assets = [verify_file(x) for x in [profile["template"], *aliases(profile),
              *([profile["typed_settings"]] if "typed_settings" in profile else [])]]
    sources = {x["path"]: x["bytes"] for x in shards}
    count = 0
    for alias in aliases(profile):
        with Path(alias["path"]).open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                path = str(Path(row["source_path"]).resolve(strict=True))
                offset, size = int(row["offset"]), int(row["nbytes"])
                if path not in sources or offset < 0 or size <= 0 or offset + size > sources[path]:
                    raise ValueError("alias range is outside the verified GGUF shards")
                count += 1
    if not count:
        raise ValueError("empty expert alias")
    return {"build": build, "shards": shards, "assets": assets, "alias_entries": count}


def inherited_lock(environment):
    fd = int(environment["SSDASH_LOCK_FD"])
    if fd < 3 or not stat.S_ISREG(os.fstat(fd).st_mode):
        raise ValueError("an inherited regular-file device-lock descriptor is required")
    # The existing open-file-description is reused. Never open a second lock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def clean_environment(profile):
    inherited = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP", "TZ", "XDG_CACHE_HOME"}
    env = {k: v for k, v in os.environ.items() if k in inherited}
    env.update(profile["environment"])
    key = "DYLD_LIBRARY_PATH" if profile["platform"] == "macos" else "LD_LIBRARY_PATH"
    env[key] = str(Path(profile["binary"]["path"]).resolve().parent)
    return env


def effective_settings(profile):
    # Bind every argv/env/threshold to cross-tier identity, including values
    # whose runtime telemetry is dynamic (e.g. admitted cache capacity).
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    policy = {"args": profile["args"], "environment": profile["environment"],
              "template_sha256": profile["template"]["sha256"],
              "alias_sha256": [x["sha256"] for x in aliases(profile)]}
    if "typed_settings" in profile:
        policy["typed_values"] = typed_snapshot(profile).document()["values"]
    return dict(profile["effective_settings"], native_profile_sha256=digest,
                native_launch_policy=policy, native_resource_policy=profile["resources"],
                native_ownership_core=CORE_IDENTITY)


def server_argv(profile, port, telemetry=None):
    typed = []
    if "typed_settings" in profile:
        if telemetry is None:
            raise ValueError("typed profile requires its fresh context telemetry path")
        typed = ["--ssd-settings", profile["typed_settings"]["path"], "--ssd-telemetry", str(telemetry)]
    return [profile["binary"]["path"], *profile["args"], *typed, "-m", profile["shards"][0]["path"],
            "--chat-template-file", profile["template"]["path"], "--reasoning", "off",
            "-c", str(profile["effective_settings"]["context_tokens"]), "-np", "1", "--fit", "off",
            "--no-warmup", "--cache-ram", "0", "--host", "127.0.0.1", "--port", str(port)]


def command(argv, timeout=5):
    value = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if value.returncode:
        raise RuntimeError(f"command failed ({value.returncode}): {argv[0]}: {value.stderr.strip()}")
    return value.stdout
