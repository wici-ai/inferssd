"""Verify file identities for project images observed in a macOS process.

This is a point-in-time, file-backed image receipt, not an in-memory text hash
or a guarantee that the process cannot load another image later. Call outside
timed requests, after health and again before shutdown. Keep the build files
unchanged for the whole run. The system dyld shared cache is outside this scope.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time


CORE_FAMILIES = frozenset({"libllama", "libggml", "libggml-base"})
BACKEND_FAMILIES = frozenset({"libggml-cpu", "libggml-metal"})
_FAMILY = re.compile(r"^(lib(?:ggml|llama|mtmd|kimi)(?:-[A-Za-z_][A-Za-z0-9_-]*)?)(?:\.\d+)*\.dylib$")


class ImageIdentityError(RuntimeError):
    pass


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_lsof(data, pid):
    records = []
    current = None
    seen_pid = False
    for raw in data.split(b"\0"):
        # lsof adds a newline after each NUL-terminated process/file set.
        field = raw.lstrip(b"\n")
        if not field:
            continue
        tag, value = chr(field[0]), os.fsdecode(field[1:])
        if tag == "p":
            if seen_pid or value != str(pid):
                raise ImageIdentityError("lsof returned an unexpected process record")
            seen_pid = True
        elif tag == "f":
            if not seen_pid:
                raise ImageIdentityError("lsof file appeared before process record")
            if current is not None:
                records.append(current)
            current = {"descriptor": value}
        elif tag in {"D", "i", "n"}:
            if current is None or tag in current:
                raise ImageIdentityError("lsof returned malformed or duplicate file fields")
            current[tag] = value
        else:
            raise ImageIdentityError("lsof returned an unexpected field: " + tag)
    if current is not None:
        records.append(current)
    if not seen_pid or not records:
        raise ImageIdentityError("lsof returned no process images")
    for record in records:
        if record.get("descriptor") != "txt" or "n" not in record:
            raise ImageIdentityError("lsof returned an incomplete text-image record")
    return records


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _device(value):
    # Darwin dev_t is 32 bits; Python and lsof can print different signedness.
    return value & 0xFFFFFFFF


def _hash_observed(record, expected_path):
    try:
        observed_dev = int(record["D"], 16)
        observed_ino = int(record["i"], 10)
    except (KeyError, ValueError) as exc:
        raise ImageIdentityError("lsof image lacks a valid device/inode") from exc
    named_path = Path(record["n"])
    if not named_path.is_absolute():
        raise ImageIdentityError("lsof image path is not absolute")
    path = named_path.resolve(strict=True)
    if path != expected_path:
        raise ImageIdentityError("image path changed before hashing")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ImageIdentityError("image path is not a regular file")
    if _device(before.st_dev) != _device(observed_dev) or before.st_ino != observed_ino:
        raise ImageIdentityError("loaded image device/inode differs from current file")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if _identity(before) != _identity(opened):
            raise ImageIdentityError("image was replaced before open")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = path.stat()
    if (named_path.resolve(strict=True) != path or
            _identity(before) != _identity(after_fd) or
            _identity(before) != _identity(after_path)):
        raise ImageIdentityError("image identity or contents changed while hashing")
    return {"path": str(path), "observed_path": str(named_path),
            "device": _device(before.st_dev), "inode": before.st_ino,
            "size": before.st_size, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns, "sha256": digest.hexdigest()}


def snapshot_images(pid, binary, out_prefix, *, require_backends=False):
    """Save lsof raw evidence and verify the selected executable/project DSOs.

    Core images are mandatory at health. Set require_backends=True after model
    requests to additionally require CPU and Metal. All observed project images,
    including optional libmtmd/libkimi/other ggml backends, must resolve directly
    inside binary.parent. Errors leave a failed JSON receipt and raise.
    """
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = str(prefix)
    receipt = {"pid": pid, "scope": "observed project file-backed text images",
               "require_backends": require_backends, "success": False,
               "started_monotonic": time.monotonic(), "argv": [],
               "returncode": None, "timeout": False}
    stdout = stderr = b""
    try:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ImageIdentityError("PID must be a positive integer")
        binary = Path(binary).resolve(strict=True)
        receipt["binary"] = str(binary)
        receipt["bundle"] = str(binary.parent)
        receipt["argv"] = ["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-d", "txt", "-F0Dfin"]
        try:
            result = subprocess.run(receipt["argv"], capture_output=True, check=False, timeout=15)
            stdout, stderr = result.stdout, result.stderr
            receipt["returncode"] = result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            receipt["timeout"] = True
            raise ImageIdentityError("lsof image snapshot timed out") from exc
        if receipt["returncode"] != 0:
            raise ImageIdentityError("lsof image snapshot failed")
        records = _parse_lsof(stdout, pid)
        images = []
        found_main = False
        families = set()
        for record in records:
            named_path = Path(record["n"])
            name = named_path.name
            project = name.startswith(("libggml", "libllama", "libmtmd", "libkimi")) and ".dylib" in name
            # Resolve only candidate project images and the main executable.
            # Unrelated system text mappings need not exist outside dyld's cache.
            if not project and named_path.name != binary.name:
                continue
            path = named_path.resolve(strict=True)
            if path == binary:
                if found_main:
                    raise ImageIdentityError("duplicate main executable image")
                found_main = True
                family = "executable"
            else:
                match = _FAMILY.fullmatch(name)
                if match is None:
                    raise ImageIdentityError("unrecognized project library name: " + name)
                family = match.group(1)
                if path.parent != binary.parent:
                    raise ImageIdentityError("project image loaded outside selected build bundle: " + str(path))
                if family in families:
                    raise ImageIdentityError("multiple loaded images for project family: " + family)
                families.add(family)
            image = _hash_observed(record, path)
            image["family"] = family
            images.append(image)
        if not found_main:
            raise ImageIdentityError("selected main executable is not an observed text image")
        required = CORE_FAMILIES | (BACKEND_FAMILIES if require_backends else frozenset())
        missing = sorted(required - families)
        if missing:
            raise ImageIdentityError("required project images are not loaded: " + ", ".join(missing))
        receipt["images"] = sorted(images, key=lambda item: item["path"])
        receipt["success"] = True
        return receipt
    except Exception as exc:
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, ImageIdentityError):
            raise
        raise ImageIdentityError(receipt["error"]) from exc
    finally:
        receipt["ended_monotonic"] = time.monotonic()
        Path(stem + ".lsof.stdout").write_bytes(stdout)
        Path(stem + ".lsof.stderr").write_bytes(stderr)
        _save(Path(stem + ".images.json"), receipt)


def verify_image_continuity(before, after):
    """Require unchanged prior images; allow newly loaded same-bundle images."""
    if not before.get("success") or not after.get("success"):
        raise ImageIdentityError("cannot compare unsuccessful image snapshots")
    if any(before[key] != after[key] for key in ("pid", "binary", "bundle")):
        raise ImageIdentityError("image snapshots refer to different process/build selections")
    old = {item["path"]: item for item in before["images"]}
    new = {item["path"]: item for item in after["images"]}
    fields = ("family", "device", "inode", "size", "mtime_ns", "ctime_ns", "sha256")
    for path, item in old.items():
        if path not in new or any(item[key] != new[path][key] for key in fields):
            raise ImageIdentityError("previously observed image changed or disappeared: " + path)
    return {"unchanged": sorted(old), "added": sorted(set(new) - set(old))}
