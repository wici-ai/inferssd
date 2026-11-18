"""The common package publisher keeps each platform's latest list separate."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "suffix,platform,extension,listing",
    [
        ("macos-arm64", "darwin-arm64", ".tar.gz", "latest-macos.json"),
        ("windows-x64-vulkan", "windows-x64", ".zip", "latest-windows-vulkan.json"),
        ("windows-x64-cuda", "windows-x64", ".zip", "latest-windows-cuda.json"),
    ],
)
def test_publish_verified_package_to_its_platform_list(
    tmp_path, monkeypatch, suffix, platform, extension, listing
):
    script = Path(__file__).resolve().parents[2] / "scripts/publish-package.py"
    spec = importlib.util.spec_from_file_location("publisher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = "inferssd-0.1.5-" + suffix
    archive = tmp_path / (name + extension)
    archive.write_bytes(b"verified package")
    package = {
        "schema": "inferssd-package 1",
        "platform": platform,
        "name": name,
        "file": archive.name,
        "size": archive.stat().st_size,
        "version": "0.1.5",
        "tag": "v0.1.5",
        "build": 401,
        "label": "261109.401.aaaaaaa",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (tmp_path / (name + ".json")).write_text(json.dumps(package))
    url, requests = "https://cdn.example/" + archive.name, []

    def call(method, address, body=None, headers=None, token=None):
        requests.append((method, address, body))
        if method == "GET" and address.endswith("/config/" + listing):
            return 404, b""
        if method == "POST":
            assert json.loads(body)["sha256"] == package["sha256"]
            return 200, json.dumps({"status": "exists", "url": url}).encode()
        if method == "GET" and address == url:
            return 200, archive.read_bytes()
        if method == "PUT" and address.endswith("/config/" + listing):
            assert body == json.dumps({**package, "url": url}, indent=2).encode() + b"\n"
            return 200, b'{"url":"https://config.example/list"}'
        raise AssertionError((method, address))

    monkeypatch.setattr(module, "call", call)
    monkeypatch.setenv("APP_VERSIONING_TOKEN", "test-only-token")
    monkeypatch.setattr(module.sys, "argv", [str(script), str(tmp_path)])
    module.main()
    assert requests[0][1].endswith("/config/" + listing)
    assert requests[-1][1].endswith("/config/" + listing)


@pytest.mark.parametrize(
    "change",
    [
        {"tag": None},
        {"tag": "v0.1.6"},
        {"name": "inferssd-261109.401.aaaaaaa-macos-arm64"},
    ],
)
def test_only_a_release_package_named_by_its_version_is_published(tmp_path, monkeypatch, change):
    script = Path(__file__).resolve().parents[2] / "scripts/publish-package.py"
    spec = importlib.util.spec_from_file_location("publisher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = "inferssd-0.1.5-macos-arm64"
    archive = tmp_path / (name + ".tar.gz")
    archive.write_bytes(b"verified package")
    package = {
        "schema": "inferssd-package 1",
        "platform": "darwin-arm64",
        "name": name,
        "file": archive.name,
        "size": archive.stat().st_size,
        "version": "0.1.5",
        "tag": "v0.1.5",
        "build": 401,
        "label": "261109.401.aaaaaaa",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        **change,
    }
    (tmp_path / (name + ".json")).write_text(json.dumps(package))

    def call(*arguments, **keywords):
        raise AssertionError("nothing is uploaded")

    monkeypatch.setattr(module, "call", call)
    monkeypatch.setenv("APP_VERSIONING_TOKEN", "test-only-token")
    monkeypatch.setattr(module.sys, "argv", [str(script), str(tmp_path)])
    with pytest.raises(SystemExit, match="release|version"):
        module.main()
