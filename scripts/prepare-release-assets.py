#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare one numbered release from the checked macOS and Windows packages of one commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

# Each package by the suffix of its name: its platform, archive extension and package list.
PACKAGES = {
    "macos-arm64": ("darwin-arm64", ".tar.gz", "latest-macos.json"),
    "windows-x64-vulkan": ("windows-x64", ".zip", "latest-windows-vulkan.json"),
    "windows-x64-cuda": ("windows-x64", ".zip", "latest-windows-cuda.json"),
}


def checked_package(
    directory: Path, suffix: str, version: str, revision: str
) -> dict[str, Any]:
    platform, extension, _ = PACKAGES[suffix]
    listings = sorted(directory.glob(f"inferssd-*-{suffix}.json"))
    if len(listings) != 1:
        raise ValueError(
            f"expected one checked {suffix} package, found {len(listings)}"
        )
    package = json.loads(listings[0].read_text(encoding="utf-8"))
    if (
        package.get("schema") != "inferssd-package 1"
        or package.get("platform") != platform
    ):
        raise ValueError(f"wrong package schema/platform: {listings[0]}")
    if package.get("version") != version or package.get("revision") != revision:
        raise ValueError(
            f"{platform} package does not match the release version and tag commit"
        )
    name = package.get("name", "")
    if not re.fullmatch(rf"inferssd-[A-Za-z0-9_.-]+-{suffix}", name):
        raise ValueError("invalid package name")
    if package.get("file") != name + extension:
        raise ValueError("archive filename does not match package name")
    if (
        type(package.get("build")) is not int
        or package["build"] <= 0
        or not package.get("label")
    ):
        raise ValueError("package has no build identity")
    archive = directory / package["file"]
    with archive.open("rb") as stream:
        digest = (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )
    if (
        type(package.get("size")) is not int
        or archive.stat().st_size != package["size"]
        or digest != package.get("sha256")
    ):
        raise ValueError(
            f"{platform} archive size/SHA-256 does not match its checked manifest"
        )
    return package


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"installer pin marker is missing or ambiguous: {old}")
    return text.replace(old, new)


def release_base(tag: str, repository: str) -> str:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("a numbered version tag is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GitHub repository")
    return f"https://github.com/{repository}/releases/download/{tag}"


def pinned_powershell(root: Path, tag: str, repository: str) -> str:
    """install.ps1 with its default ref and package lists pinned to the release."""
    base = release_base(tag, repository)
    powershell = replace_once(
        (root / "scripts/install.ps1").read_text(encoding="utf-8"),
        "$DefaultRef = 'main'",
        "$DefaultRef = '" + tag + "'",
    )
    for variable, listing in (
        ("VulkanPackages", "latest-windows-vulkan.json"),
        ("CudaPackages", "latest-windows-cuda.json"),
    ):
        powershell = replace_once(
            powershell,
            f"${variable} = 'https://appapi.wici.ai/versioning/inferssd/config/{listing}'",
            f"${variable} = '{base}/{listing}'",
        )
    return powershell


def signed_powershell(signed: Path, pinned: str) -> bytes:
    """The signed copy of the pinned install.ps1: exactly its bytes, then the signature block."""
    data = signed.read_bytes()
    body = pinned.encode("utf-8")
    if not data.startswith(body) or not data[len(body):].startswith(
        b"\r\n# SIG # Begin signature block\r\n"
    ):
        raise ValueError("the signed install.ps1 is not the release's pinned installer")
    return data


def prepare(
    root: Path,
    directories: dict[str, Path],
    output: Path,
    tag: str,
    revision: str,
    repository: str,
    signed_installer: Path | None = None,
) -> list[str]:
    base = release_base(tag, repository)
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("a numbered version tag and its exact commit SHA are required")
    version = re.search(
        r'^version = "([^"]+)"$',
        (root / "framework/pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if version is None or version[1] != tag[1:]:
        raise ValueError("tag does not match the framework version")
    if set(directories) != set(PACKAGES):
        raise ValueError(
            "a numbered release holds every package: " + ", ".join(PACKAGES)
        )
    packages = {
        suffix: checked_package(directories[suffix], suffix, tag[1:], revision)
        for suffix in PACKAGES
    }
    if (
        len(
            {
                tuple(p.get(key) for key in ("build", "label", "date"))
                for p in packages.values()
            }
        )
        != 1
    ):
        raise ValueError("the packages do not share one build identity")
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output is not empty; existing assets were preserved")
    shell = replace_once(
        (root / "scripts/install.sh").read_text(encoding="utf-8"),
        "DEFAULT_REF=main",
        "DEFAULT_REF=" + tag,
    )
    shell = replace_once(
        shell,
        "PACKAGES=https://appapi.wici.ai/versioning/inferssd/config/latest-macos.json",
        "PACKAGES=" + base + "/latest-macos.json",
    )
    powershell = pinned_powershell(root, tag, repository)
    installer = (
        signed_powershell(signed_installer, powershell)
        if signed_installer
        else powershell.encode("utf-8")
    )
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for suffix, package in packages.items():
        shutil.copy2(directories[suffix] / package["file"], output / package["file"])
        latest = PACKAGES[suffix][2]
        (output / latest).write_text(
            json.dumps({**package, "url": base + "/" + package["file"]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        names.extend((package["file"], latest))
    (output / "install.sh").write_text(shell, encoding="utf-8")
    (output / "install.sh").chmod(0o755)
    (output / "install.ps1").write_bytes(installer)
    return names + ["install.sh", "install.ps1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pinned-install-ps1",
        type=Path,
        help="only write the release's pinned install.ps1 here, for signing (with --tag and --repository)",
    )
    for suffix in PACKAGES:
        parser.add_argument("--" + suffix, type=Path, help=f"the checked {suffix} package")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--repository", required=True)
    parser.add_argument(
        "--signed-install-ps1", type=Path, help="the signed copy of the pinned install.ps1"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.pinned_install_ps1:
        args.pinned_install_ps1.parent.mkdir(parents=True, exist_ok=True)
        args.pinned_install_ps1.write_bytes(
            pinned_powershell(root, args.tag, args.repository).encode("utf-8")
        )
        return
    missing = [
        "--" + name.replace("_", "-")
        for name in [suffix.replace("-", "_") for suffix in PACKAGES] + ["output", "revision"]
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    files = prepare(
        root,
        {suffix: getattr(args, suffix.replace("-", "_")) for suffix in PACKAGES},
        args.output,
        args.tag,
        args.revision,
        args.repository,
        args.signed_install_ps1,
    )
    print(json.dumps({"tag": args.tag, "revision": args.revision, "assets": files}))


if __name__ == "__main__":
    main()
