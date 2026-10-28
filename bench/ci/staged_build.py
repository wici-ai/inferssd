"""Build and identify one source revision using only the trusted stage broker."""

import hashlib
import json
import os
import stat
import time
from pathlib import Path

from ci.contract import revision as full_revision
from ci.contract import save
from ci.isolated_device import protected
from ci.reference import relative


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts or "\0" in value:
        raise ValueError("explicit absolute reviewed stage path required")
    return value


def frozen_aliases(frozen, *, evidence_path=None):
    """Read the root freezer's complete internal hardlink groups, not caller aliases."""
    _, raw = sealed_file(frozen["manifest_path"], limit=32 << 20, contents=True)
    manifest = json.loads(raw)
    records = manifest["files"]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != frozen["manifest_sha256"]:
        raise ValueError("root freezer manifest differs from its receipt")
    if evidence_path is not None:
        save(evidence_path, manifest)
    root = Path(absolute(frozen["host_path"]))
    rows = {}
    for row in records:
        name = row["path"]
        if not relative(name) or name in rows:
            raise ValueError("invalid or duplicate root freezer path")
        rows[name] = row
    groups = {}
    for name, row in rows.items():
        target = row.get("alias_of", name)
        if (not relative(target) or target not in rows or "alias_of" in rows[target]
                or row["sha256"] != rows[target]["sha256"] or row["size"] != rows[target]["size"]):
            raise ValueError("invalid root freezer alias provenance")
        groups.setdefault(target, []).append(root / name)
    aliases = {}
    for target, members in groups.items():
        if len(members) < 2:
            continue
        proof = dict(paths=members, sha256=rows[target]["sha256"], size=rows[target]["size"])
        for member in members:
            aliases[member] = proof
    return aliases


def retain_mac_freeze(frozen, evidence_path):
    """Keep the root Mac freezer's exact bytes for hosted source/image checks."""
    _, raw = sealed_file(frozen["manifest_path"], limit=32 << 20, contents=True)
    if hashlib.sha256(raw).hexdigest() != frozen["manifest_sha256"] or frozen.get("readonly") is not True:
        raise ValueError("Mac root freezer manifest differs from its receipt")
    records = json.loads(raw)
    if (not isinstance(records, dict) or type(frozen.get("file_count")) is not int
            or not frozen["file_count"] or len(records) != frozen["file_count"]):
        raise ValueError("Mac root freezer inventory count differs")
    for name, row in records.items():
        if (not relative(name) or not isinstance(row, dict)
                or type(row.get("size")) is not int or row["size"] < 0
                or row.get("mode") not in (0o444, 0o555)
                or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError("Mac root freezer member is invalid")
    with Path(evidence_path).open("xb") as target:
        target.write(raw)


def alias_identity(path, info, aliases):
    proof = (aliases or {}).get(path)
    if info.st_nlink == 1 and proof is None:
        return None
    if proof is None or info.st_nlink != len(proof["paths"]) or info.st_size != proof["size"]:
        raise ValueError("multiply linked artifact lacks a complete root freezer alias proof")
    for member in proof["paths"]:
        observed = protected(member).lstat()
        if (not stat.S_ISREG(observed.st_mode) or observed.st_mode & 0o222
                or (observed.st_dev, observed.st_ino, observed.st_nlink) !=
                (info.st_dev, info.st_ino, info.st_nlink)):
            raise ValueError("root freezer alias does not share the immutable file identity")
    return proof


def sealed_file(path, *, limit=1 << 30, contents=False, aliases=None):
    """Read a root-sealed regular file, never execute a path obtained from it."""
    path = protected(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, chunks, size = hashlib.sha256(), [], 0
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222:
            raise ValueError("build identity requires an immutable regular file")
        proof = alias_identity(path, before, aliases)
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            if size > limit:
                raise ValueError("sealed build artifact exceeds its read bound")
            digest.update(block)
            if contents:
                chunks.append(block)
        after = os.fstat(stream.fileno())
        fields = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_nlink)
        if fields(before) != fields(after) or fields(after) != fields(path.stat()):
            raise ValueError("sealed artifact changed during identity read")
        alias_identity(path, after, aliases)
    result = {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}
    if proof is not None:
        if result["sha256"] != proof["sha256"]:
            raise ValueError("sealed aliases differ from the root freezer content hash")
        result["aliases"] = [str(member) for member in proof["paths"]]
    return (result, b"".join(chunks)) if contents else result


def compiler_cache(raw, expected):
    allowed = {"CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER"}
    if (not isinstance(expected, dict) or not {"CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"} <= set(expected)
            or not set(expected) <= allowed):
        raise ValueError("explicit reviewed C/C++ and optional CUDA compiler paths required")
    observed = {}
    for line in raw.decode("utf-8").splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.partition(":")[0]
        if name in allowed:
            if name in observed:
                raise ValueError("duplicate compiler identity in CMakeCache")
            observed[name] = value
    if observed != {key: absolute(value) for key, value in expected.items()}:
        raise ValueError("sealed CMakeCache compiler paths differ from reviewed toolchain")
    return observed


def validate_platform_flags(platform, flags):
    """Reject known incompatible profiles before acquiring a device lease.

    CMake remains authoritative for other options. Read the last definition,
    matching CMake, including typed -DKEY:BOOL=VALUE forms.
    """
    values = {}
    for flag in flags:
        if flag.startswith("-D") and "=" in flag:
            key, value = flag[2:].split("=", 1)
            values[key.split(":", 1)[0]] = value.upper()
    false = {"0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", ""}
    def enabled(key, default):
        value = values.get(key, default)
        return value not in false and not value.endswith("-NOTFOUND")
    if enabled("LLAMA_SSD_STORE", "ON" if platform == "linux" else "OFF"):
        if (platform != "linux" or not enabled("BUILD_SHARED_LIBS", "ON")
                or enabled("LLAMA_USE_SYSTEM_GGML", "OFF")):
            raise ValueError("LLAMA_SSD_STORE requires Linux, BUILD_SHARED_LIBS=ON and bundled ggml")


def run_build(session, side, settings, paths, output, revision):
    """Paths: platform/source/build/cwd/frozen, plus job_root for Darwin.

    Settings require cmake, cmake_flags, jobs, targets, timeout_s, toolchain,
    binary_relative and image_relatives. Output is a fresh evidence directory.
    Every command, including the final --version, runs as a sandboxed stage.
    """
    full_revision(revision)
    if side not in ("base", "candidate") or paths.get("platform") not in ("linux", "darwin"):
        raise ValueError("explicit build side/platform required")
    for key in ("source", "build", "cwd", "frozen"):
        absolute(paths[key])
    cmake = absolute(settings["cmake"])
    for key in ("cmake_flags", "targets", "image_relatives"):
        if not isinstance(settings[key], list) or any(not isinstance(v, str) or not v or "\0" in v for v in settings[key]):
            raise ValueError("explicit reviewed list required: " + key)
    validate_platform_flags(paths["platform"], settings["cmake_flags"])
    if (not settings["targets"] or not settings["image_relatives"]
            or not relative(settings["binary_relative"])
            or any(not relative(path) for path in settings["image_relatives"])):
        raise ValueError("build targets and safe relative output files required")
    timeout, jobs = settings["timeout_s"], settings["jobs"]
    if type(timeout) is not int or not 1 <= timeout <= 43200 or type(jobs) is not int or not 1 <= jobs <= 512:
        raise ValueError("finite stage timeout and positive build jobs required")
    cap = settings.get("output_max_bytes", 16 << 20)
    if type(cap) is not int or not 1 <= cap <= 64 << 20:
        raise ValueError("explicit bounded stage output size required")
    environment = dict(settings.get("environment", {}))
    freeze_fields = {}
    if paths["platform"] == "darwin":
        job = Path(absolute(paths["job_root"]))
        if Path(paths["build"]) != job / "work" / side or Path(paths["frozen"]) != job / "frozen" / side:
            raise ValueError("Mac build and freeze paths must be the exact owned side")
        freeze_fields = {"source": paths["build"], "name": side}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "purpose": "ci", "status": "failed",
               "side": side, "revision": revision, "stages": []}

    def command(name, kind, argv, cwd):
        stage = session.start(side + "-" + name, kind, argv, environment, cwd, timeout)
        result = session.wait(stage, deadline=time.monotonic() + timeout)
        logs, total = {}, 0
        for stream in ("stdout", "stderr"):
            path, offset, collected = output / (name + "-" + stream + ".log"), 0, []
            with path.open("xb") as target:
                while True:
                    data, offset, eof = session.output(stage, stream, offset)
                    total += len(data)
                    if total > cap:
                        raise ValueError("trusted stage output exceeds the retained log bound")
                    target.write(data)
                    collected.append(data)
                    if eof:
                        break
                    if not data:
                        raise ValueError("completed stage output did not reach EOF")
            logs[stream] = b"".join(collected)
        row = {"stage_id": side + "-" + name, "argv": argv, "cwd": cwd, "result": result,
               "logs": {key: {"path": name + "-" + key + ".log", "sha256": hashlib.sha256(value).hexdigest()}
                        for key, value in logs.items()}}
        receipt["stages"].append(row)
        native = result.get("receipt")
        if (result.get("running") is not False or result.get("returncode") != 0
                or result.get("cleanup_confirmed") is not True
                or result.get("state") in ("failed", "quarantined")
                or isinstance(native, dict) and native.get("passed") is not True):
            raise RuntimeError("sandboxed " + name + " failed or cleanup was not confirmed")
        return stage, logs["stdout"] + b"\n" + logs["stderr"]

    try:
        _, configured = command("configure", "build", [cmake, "-S", paths["source"], "-B", paths["build"], *settings["cmake_flags"]], paths["cwd"])
        if b"Manually-specified variables were not used by the project" in configured:
            raise ValueError("reviewed CMake flags include unused or retired controls")
        built, _ = command("build", "build", [cmake, "--build", paths["build"], "--parallel", str(jobs), "--target", *settings["targets"]], paths["cwd"])
        frozen = session.freeze(built, **freeze_fields)
        receipt["freeze"] = frozen
        if frozen.get("path") != paths["frozen"]:
            raise ValueError("broker freeze path differs from reviewed runtime layout")
        host = Path(absolute(frozen.get("host_path", frozen["path"])))
        aliases = frozen_aliases(frozen, evidence_path=output / "freeze-manifest.json") if paths["platform"] == "linux" and "manifest_path" in frozen else None
        if paths["platform"] == "darwin":
            if frozen["manifest_path"] != str(Path(paths["job_root"]) / "frozen" / (side + ".manifest.json")):
                raise ValueError("Mac root freezer manifest is outside the exact owned side")
            retain_mac_freeze(frozen, output / "freeze-manifest.json")
        cache, raw = sealed_file(host / "CMakeCache.txt", limit=8 << 20, contents=True, aliases=aliases)
        receipt["toolchain"] = compiler_cache(raw, settings["toolchain"])
        receipt.update(cmake_cache=cache, cmake_cache_sha256=cache["sha256"])
        receipt["binary"] = dict(sealed_file(host / settings["binary_relative"], aliases=aliases), runtime_path=str(Path(paths["frozen"]) / settings["binary_relative"]))
        receipt["images"] = [dict(sealed_file(host / name, aliases=aliases), runtime_path=str(Path(paths["frozen"]) / name)) for name in settings["image_relatives"]]
        _, version = command("version", "test", [receipt["binary"]["runtime_path"], "--version"], paths["frozen"])
        actual = version.decode("utf-8", errors="replace").strip()
        if revision[:7] not in actual:
            raise ValueError("sealed server version does not identify the reviewed source")
        receipt.update(actual_server_version=actual, status="passed")
        return receipt
    except BaseException as error:
        receipt["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        save(output / "build-receipt.json", receipt)
