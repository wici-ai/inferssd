#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish prefix caches and name them in the prefix cache list (docs/prefix-cache.md).

Usage: APP_VERSIONING_TOKEN=... scripts/publish-prefix-caches.py ENTRIES.json [--dry-run]

ENTRIES.json is a list of the prefix caches to publish, each

    {"file": "<a .prompt file>", "engines": ["<engine commit>", ...], "os": "windows",
     "model": "<the model's first GGUF file name>", "app": "claude-code", "app_version": "2.1.328"}

Each file goes to cdn.wici.ai/apps/inferssd/ as prefix-cache-<sha256>.prompt through WiCi AI's release
service, as the engine's packages do; a file already there is skipped. Only then the list,
prefix-cache.json, names them: an entry with the same SHA-256 as a listed one adds its engines to it,
another is added, and of each app's prefix caches for one model and system the two latest app versions
stay. --dry-run reads the list and prints what would be uploaded and the list that would be written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

SERVICE = os.environ.get("INFERSSD_PUBLISH_SERVICE", "https://appapi.wici.ai/versioning/inferssd")
LIST = "prefix-cache.json"
SCHEMA = "inferssd-prefix-cache 1"
# The CDN in front of the service refuses Python's default User-Agent.
AGENT = "inferssd-publish"
# a cached prompt file: "LPCF", version 1, the fingerprint, the tokens (tools/server/inferssd/prompt-cache.cpp)
MAGIC, VERSION = 0x4643504C, 1
FINGERPRINT = "prefix-cache 1|"
KEEP_APP_VERSIONS = 2


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


def fail(message: str) -> None:
    sys.exit("publish-prefix-caches: " + message)


def prompt_tokens(path: Path) -> int:
    """The number of tokens of a prefix cache file; fails for a file that is not one."""
    with path.open("rb") as f:
        magic, version, n_fp = struct.unpack("<III", f.read(12))
        if magic != MAGIC or version != VERSION or n_fp > 4096:
            fail(f"{path} is not a cached prompt file")
        fingerprint = f.read(n_fp).decode("utf-8", "replace")
        if not fingerprint.startswith(FINGERPRINT):
            fail(f"{path} is a prompt cache of one server, not a prefix cache (fingerprint {fingerprint[:40]!r})")
        (n_tokens,) = struct.unpack("<I", f.read(4))
        return n_tokens


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def merged(current: list[dict], new: list[dict]) -> list[dict]:
    files = [dict(entry) for entry in current]
    for entry in new:
        same = next((f for f in files if f["sha256"] == entry["sha256"]), None)
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
        digest = hashlib.sha256(data).hexdigest()
        planned.append((data, {"engines": sorted(item["engines"]), "os": item["os"], "model": item["model"],
                               "app": item["app"], "app_version": item["app_version"],
                               "tokens": prompt_tokens(path), "size": len(data), "sha256": digest}))

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

    new = []
    for data, entry in planned:
        name = f"prefix-cache-{entry['sha256']}.prompt"
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
    print(f"{json.loads(body)['url']} names {len(new)} new prefix cache(s)")


if __name__ == "__main__":
    main()
