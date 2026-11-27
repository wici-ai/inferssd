#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish prefix caches and name them in the prefix cache list (docs/prefix-cache.md).

Usage: APP_VERSIONING_TOKEN=... scripts/publish-prefix-caches.py ENTRIES.json [--dry-run]

ENTRIES.json is a list of the prefix caches to publish, each

    {"file": "<a .prompt file>", "engines": ["<engine commit>", ...], "os": "windows",
     "model": "<the model's first GGUF file name>", "app": "claude-code", "app_version": "2.1.328"}

Each file goes to cdn.wici.ai/apps/inferssd/ as prefix-cache-<sha256>.prompt through WiCi AI's release
service, as the engine's packages do; a file already there is skipped. Only then the list,
prefix-cache.json, names them. Entries add engines when the system, model, app, app version and
SHA-256 all match. Other entries are added, and each app's two latest versions for one model and
system stay. --dry-run reads the list and prints what would be uploaded and the list that would be
written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path
from typing import NoReturn

SERVICE = os.environ.get("INFERSSD_PUBLISH_SERVICE", "https://appapi.wici.ai/versioning/inferssd")
LIST = "prefix-cache.json"
SCHEMA = "inferssd-prefix-cache 1"
# The CDN in front of the service refuses Python's default User-Agent.
AGENT = "inferssd-publish"
# a cached prompt file: "LPCF", version 1, the fingerprint, the tokens (tools/server/inferssd/prompt-cache.cpp)
MAGIC, VERSION = 0x4643504C, 1
FINGERPRINT = "prefix-cache 1|"
BASE_PROVENANCE = "|adapter-cache=1"
KEEP_APP_VERSIONS = 2
LEGACY_REGISTRY = "engine/tools/server/inferssd/prompt-cache-legacy.inc"


def call(method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None,
         token: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": AGENT, **(headers or {})})
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def fail(message: str) -> NoReturn:
    sys.exit("publish-prefix-caches: " + message)


def prompt_metadata(path: Path, data: bytes) -> tuple[int, bool]:
    """Read the token count and adapter provenance from the supplied file."""
    if len(data) < 16:
        fail(f"{path} is not a cached prompt file")
    magic, version, n_fp = struct.unpack_from("<III", data)
    if magic != MAGIC or version != VERSION or n_fp > 4096 or len(data) < 16 + n_fp:
        fail(f"{path} is not a cached prompt file")
    fingerprint = data[12:12 + n_fp].decode("utf-8", "replace")
    if not fingerprint.startswith(FINGERPRINT):
        fail(f"{path} is a prompt cache of one server, not a prefix cache (fingerprint {fingerprint[:40]!r})")
    if "|lora=" in fingerprint:
        fail(f"{path} was computed with LoRA adapters and is not a portable base-model prefix cache")
    (n_tokens,) = struct.unpack_from("<I", data, 12 + n_fp)
    return n_tokens, fingerprint.endswith(BASE_PROVENANCE)


@cache
def legacy_prefixes(engine: str) -> frozenset[tuple[int, str]]:
    """Read the exact import inventory compiled into the named engine."""
    repository = Path(__file__).resolve().parents[1]
    try:
        inventory = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{engine}:{LEGACY_REGISTRY}"],
            encoding="utf-8", stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        fail(f"cannot verify the legacy import inventory for engine {engine}")
    entries = set()
    for line in inventory.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        match = re.fullmatch(r'\{([1-9]\d*), "([0-9a-f]{64})"\},', line)
        if not match:
            fail(f"invalid legacy import inventory for engine {engine}")
        entries.add((int(match[1]), match[2]))
    return frozenset(entries)


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def catalog_key(entry: dict) -> tuple[str, ...]:
    return entry["os"], entry["model"], entry["app"], entry["app_version"], entry["sha256"]


def merged(current: list[dict], new: list[dict]) -> list[dict]:
    files = [dict(entry) for entry in current]
    for entry in new:
        key = catalog_key(entry)
        same = next((f for f in files if catalog_key(f) == key), None)
        if same is None:
            files.append(entry)
        else:
            same["engines"] = sorted(set(same["engines"]) | set(entry["engines"]))
    kept = []
    for entry in files:
        versions = sorted({f["app_version"] for f in files
                           if (f["os"], f["model"], f["app"]) == (entry["os"], entry["model"], entry["app"])},
                          key=version_key, reverse=True)
        if entry["app_version"] in versions[:KEEP_APP_VERSIONS]:
            kept.append(entry)
    return kept


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = len(args) != len(sys.argv) - 1
    if len(args) != 1:
        fail("usage: publish-prefix-caches.py ENTRIES.json [--dry-run]")
    token = os.environ.get("APP_VERSIONING_TOKEN")
    if not token and not dry_run:
        fail("APP_VERSIONING_TOKEN is not set")
    base = Path(args[0]).parent
    planned = []
    for item in json.loads(Path(args[0]).read_text(encoding="utf-8")):
        path = base / item["file"]
        if item.get("os") not in ("macos", "windows", "linux"):
            fail(f"{path}: os is macos, windows or linux")
        if not item.get("engines") or not all(re.fullmatch(r"[0-9a-f]{40}", e) for e in item["engines"]):
            fail(f"{path}: engines are 40-hex engine commits")
        if not str(item.get("model", "")).endswith(".gguf") or "/" in item["model"] or "\\" in item["model"]:
            fail(f"{path}: model is the model's first GGUF file name")
        data = path.read_bytes()
        tokens, has_base_provenance = prompt_metadata(path, data)
        digest = hashlib.sha256(data).hexdigest()
        planned.append((path, data, has_base_provenance,
                        {"engines": sorted(item["engines"]), "os": item["os"], "model": item["model"],
                         "app": item["app"], "app_version": item["app_version"],
                         "tokens": tokens, "size": len(data), "sha256": digest}))

    status, body = call("GET", f"{SERVICE}/config/{LIST}")
    if status == 200:
        current = json.loads(body)
        if current.get("schema") != SCHEMA:
            fail(f"{LIST} has schema {current.get('schema')!r}")
        current_files = current["files"]
    elif status == 404:
        current_files = []
    else:
        fail(f"reading {LIST} answered {status}")

    published = {}
    for entry in current_files:
        published.setdefault(catalog_key(entry), entry)
    for path, _, has_base_provenance, entry in planned:
        if has_base_provenance:
            continue
        existing = published.get(catalog_key(entry))
        if existing is None or existing["size"] != entry["size"]:
            fail(f"{path} lacks base-model provenance; regenerate the prefix with an adapter-aware engine")
        for engine in sorted(set(entry["engines"]) - set(existing["engines"])):
            if (entry["size"], entry["sha256"]) not in legacy_prefixes(engine):
                fail(f"{path} is not in the legacy import inventory for engine {engine}")

    new = []
    for _, data, has_base_provenance, entry in planned:
        name = f"prefix-cache-{entry['sha256']}.prompt"
        if not has_base_provenance:
            # A verified legacy file only changes the engine list of the existing entry.
            existing = published[catalog_key(entry)]
            if not dry_run:
                status, served = call("GET", existing["url"])
                if status != 200 or len(served) != entry["size"] or hashlib.sha256(served).hexdigest() != entry["sha256"]:
                    fail(f"{existing['url']} does not serve {name} (status {status})")
            print(f"{'would add engines' if dry_run else 'adding engines'} to already published {name}")
            new.append({**existing, "engines": entry["engines"]})
            continue
        if dry_run:
            print(f"would upload {name} ({len(data)} bytes, {entry['tokens']} tokens)")
            new.append({**entry, "url": f"https://cdn.wici.ai/apps/inferssd/{name}"})
            continue
        status, body = call("POST", f"{SERVICE}/upload",
                            json.dumps({"name": name, "length": len(data), "sha256": entry["sha256"]}).encode(),
                            {"Content-Type": "application/json"}, token)
        if status != 200:
            fail(f"the upload request for {name} answered {status}: {body[:300]!r}")
        upload = json.loads(body)
        if upload["status"] == "ready":
            if upload.get("replacing"):
                fail(f"{name} is already published with other bytes")
            status, body = call("PUT", upload["upload_url"], data, upload["headers"])
            if status != 200:
                fail(f"uploading {name} answered {status}: {body[:300]!r}")
            print(f"uploaded {name} ({len(data)} bytes)")
        elif upload["status"] == "exists":
            print(f"{name} is already published")
        else:
            fail(f"unexpected upload answer {upload!r}")
        # The CDN must serve exactly these bytes before the list names them.
        status, served = call("GET", upload["url"])
        if status != 200 or hashlib.sha256(served).hexdigest() != entry["sha256"]:
            fail(f"{upload['url']} does not serve {name} (status {status})")
        new.append({**entry, "url": upload["url"]})

    listing = json.dumps({"schema": SCHEMA, "files": merged(current_files, new)}, indent=2).encode() + b"\n"
    if dry_run:
        print(f"would write {SERVICE}/config/{LIST}:")
        print(listing.decode())
        return
    status, body = call("PUT", f"{SERVICE}/config/{LIST}", listing, {"Content-Type": "application/json"}, token)
    if status != 200:
        fail(f"writing {LIST} answered {status}: {body[:300]!r}")
    print(f"{json.loads(body)['url']} updates {len(new)} prefix cache(s)")


if __name__ == "__main__":
    main()
