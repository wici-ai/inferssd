# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import struct
import subprocess
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


def set_legacy_inventory(publisher, monkeypatch, entry):
    def inventory(engine):
        assert engine == "a" * 40
        return frozenset({(entry["size"], entry["sha256"])})

    monkeypatch.setattr(publisher, "legacy_prefixes", inventory)


@pytest.mark.parametrize("mismatch", ["sha256", "size", "os", "app_version"])
def test_rejects_unproven_file_missing_exact_published_match(
    publisher, tmp_path, monkeypatch, mismatch
):
    entry = published_entry(prepare(tmp_path, monkeypatch, ""))
    entry[mismatch] = {
        "sha256": "0" * 64,
        "size": entry["size"] + 1,
        "os": "macos",
        "app_version": "2.1.327",
    }[mismatch]
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
@pytest.mark.parametrize(
    "other_identity",
    [None, {"os": "macos"}, {"app_version": "2.1.327"}, {"os": "macos", "app_version": "2.1.327"}],
)
def test_adds_engine_to_verified_legacy_entry_without_uploading(
    publisher, tmp_path, monkeypatch, capsys, dry_run, other_identity
):
    prompt = prepare(tmp_path, monkeypatch, "")
    entry = published_entry(prompt)
    set_legacy_inventory(publisher, monkeypatch, entry)
    other_rows = (
        [{**entry, **other_identity, "url": entry["url"] + "?alias=1"}]
        if other_identity is not None
        else []
    )
    expected = {
        "schema": publisher.SCHEMA,
        "files": [*other_rows, {**entry, "engines": ["a" * 40, "b" * 40]}],
    }
    listing_url = f"{publisher.SERVICE}/config/{publisher.LIST}"
    if not dry_run:
        monkeypatch.setattr(sys, "argv", sys.argv[:-1])
        monkeypatch.setenv("APP_VERSIONING_TOKEN", "token")

    calls = []

    def call(method, url, body=None, *_args):
        calls.append((method, url))
        if (method, url) == ("GET", listing_url):
            return 200, json.dumps(
                {"schema": publisher.SCHEMA, "files": [*other_rows, entry]}
            ).encode()
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
    set_legacy_inventory(publisher, monkeypatch, entry)
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


def test_legacy_import_inventory_is_read_at_the_advertised_engine(
    publisher, tmp_path, monkeypatch, capsys
):
    entry = published_entry(prepare(tmp_path, monkeypatch, ""))
    repository = tmp_path / "engine-source"
    registry = repository / publisher.LEGACY_REGISTRY
    registry.parent.mkdir(parents=True)
    monkeypatch.setattr(publisher, "__file__", str(repository / "scripts/publish-prefix-caches.py"))

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgSign=false",
                *arguments,
            ],
            cwd=repository,
            text=True,
        ).strip()

    git("init", "-q")
    other_digest = "0" * 64
    registry.write_text(f'    {{{entry["size"]}, "{other_digest}"}},\n', encoding="utf-8")
    git("add", publisher.LEGACY_REGISTRY)
    git("commit", "-qm", "Name the original legacy asset")
    earlier = git("rev-parse", "HEAD")
    with registry.open("a", encoding="utf-8") as output:
        output.write(f'    {{{entry["size"]}, "{entry["sha256"]}"}},\n')
    git("add", publisher.LEGACY_REGISTRY)
    git("commit", "-qm", "Add the published legacy asset")
    later = git("rev-parse", "HEAD")

    listing_url = f"{publisher.SERVICE}/config/{publisher.LIST}"

    def call(method, url, *_args):
        assert (method, url) == ("GET", listing_url)
        return 200, json.dumps({"schema": publisher.SCHEMA, "files": [entry]}).encode()

    monkeypatch.setattr(publisher, "call", call)
    entries_path = tmp_path / "entries.json"
    entries = json.loads(entries_path.read_text(encoding="utf-8"))
    entries[0]["engines"] = [earlier]
    entries_path.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(
        SystemExit, match=f"not in the legacy import inventory for engine {earlier}"
    ):
        publisher.main()

    entries[0]["engines"] = [later]
    entries_path.write_text(json.dumps(entries), encoding="utf-8")
    publisher.main()
    output = capsys.readouterr().out
    listing = json.loads(output.split(f"would write {listing_url}:\n", 1)[1])
    assert listing["files"] == [{**entry, "engines": sorted([later, "b" * 40])}]
