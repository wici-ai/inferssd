# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


@pytest.fixture
def publisher():
    script = Path(__file__).resolve().parents[2] / "scripts" / "publish-prefix-caches.py"
    spec = importlib.util.spec_from_file_location("prefix_cache_publisher", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(tmp_path, monkeypatch, suffix):
    fingerprint = ("prefix-cache 1|this model|old-build" + suffix).encode()
    prompt = (
        struct.pack("<III", 0x4643504C, 1, len(fingerprint)) + fingerprint + struct.pack("<I", 96)
    )
    (tmp_path / "prefix.prompt").write_bytes(prompt)
    entries = tmp_path / "entries.json"
    entries.write_text(
        json.dumps(
            [
                {
                    "file": "prefix.prompt",
                    "engines": ["a" * 40],
                    "os": "windows",
                    "model": "model.gguf",
                    "app": "claude-code",
                    "app_version": "2.1.328",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["publish-prefix-caches.py", str(entries), "--dry-run"])
    return prompt


def test_rejects_adapted_file_before_contacting_service(publisher, tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, "|adapter-cache=1|lora=0123456789abcdef")
    monkeypatch.setattr(
        publisher, "call", lambda *_args: pytest.fail("unexpected release service request")
    )
    with pytest.raises(SystemExit, match="computed with LoRA adapters"):
        publisher.main()


def published_entry(prompt):
    digest = hashlib.sha256(prompt).hexdigest()
    return {
        "engines": ["b" * 40],
        "os": "windows",
        "model": "model.gguf",
        "app": "claude-code",
        "app_version": "2.1.328",
        "tokens": 96,
        "size": len(prompt),
        "sha256": digest,
        "url": f"https://cdn.wici.ai/apps/inferssd/prefix-cache-{digest}.prompt",
    }


@pytest.mark.parametrize("mismatch", ["sha256", "size"])
def test_rejects_unproven_file_missing_exact_published_match(
    publisher, tmp_path, monkeypatch, mismatch
):
    entry = published_entry(prepare(tmp_path, monkeypatch, ""))
    entry[mismatch] = {"sha256": "0" * 64, "size": entry["size"] + 1}[mismatch]
    listing_url = f"{publisher.SERVICE}/config/{publisher.LIST}"
    calls = []

    def call(method, url, *_args):
        calls.append((method, url))
        return 200, json.dumps({"schema": publisher.SCHEMA, "files": [entry]}).encode()

    monkeypatch.setattr(publisher, "call", call)
    with pytest.raises(SystemExit, match="lacks base-model provenance"):
        publisher.main()
    assert calls == [("GET", listing_url)]


def test_base_marker_allows_new_publication(publisher, tmp_path, monkeypatch, capsys):
    prompt = prepare(tmp_path, monkeypatch, "|adapter-cache=1")
    monkeypatch.setattr(publisher, "call", lambda *_args: (404, b""))
    publisher.main()
    name = f"prefix-cache-{hashlib.sha256(prompt).hexdigest()}.prompt"
    assert f"would upload {name} ({len(prompt)} bytes, 96 tokens)" in capsys.readouterr().out


@pytest.mark.parametrize("dry_run", [True, False])
def test_adds_engine_to_verified_legacy_entry_without_uploading(
    publisher, tmp_path, monkeypatch, capsys, dry_run
):
    prompt = prepare(tmp_path, monkeypatch, "")
    entry = {**published_entry(prompt), "app_version": "2.1.327"}
    expected = {"schema": publisher.SCHEMA, "files": [{**entry, "engines": ["a" * 40, "b" * 40]}]}
    listing_url = f"{publisher.SERVICE}/config/{publisher.LIST}"
    if not dry_run:
        monkeypatch.setattr(sys, "argv", sys.argv[:-1])
        monkeypatch.setenv("APP_VERSIONING_TOKEN", "token")

    calls = []

    def call(method, url, body=None, *_args):
        calls.append((method, url))
        if (method, url) == ("GET", listing_url):
            return 200, json.dumps({"schema": publisher.SCHEMA, "files": [entry]}).encode()
        if (method, url) == ("GET", entry["url"]):
            return 200, prompt
        if (method, url) == ("PUT", listing_url):
            assert json.loads(body) == expected
            return 200, json.dumps({"url": listing_url}).encode()
        pytest.fail(f"unexpected release service request: {method} {url}")

    monkeypatch.setattr(publisher, "call", call)
    publisher.main()
    if dry_run:
        output = capsys.readouterr().out
        assert "would upload" not in output
        assert json.loads(output.split(f"would write {listing_url}:\n", 1)[1]) == expected
        assert calls == [("GET", listing_url)]
    else:
        assert calls == [("GET", listing_url), ("GET", entry["url"]), ("PUT", listing_url)]


@pytest.mark.parametrize("change", [b"x", b"xy"])
def test_does_not_update_legacy_entry_when_cdn_bytes_differ(
    publisher, tmp_path, monkeypatch, change
):
    prompt = prepare(tmp_path, monkeypatch, "")
    entry = published_entry(prompt)
    listing_url = f"{publisher.SERVICE}/config/{publisher.LIST}"
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    monkeypatch.setenv("APP_VERSIONING_TOKEN", "token")

    def call(method, url, *_args):
        if (method, url) == ("GET", listing_url):
            return 200, json.dumps({"schema": publisher.SCHEMA, "files": [entry]}).encode()
        if (method, url) == ("GET", entry["url"]):
            return 200, prompt[:-1] + change
        pytest.fail(f"unexpected release service request: {method} {url}")

    monkeypatch.setattr(publisher, "call", call)
    with pytest.raises(SystemExit, match="does not serve prefix-cache-"):
        publisher.main()
