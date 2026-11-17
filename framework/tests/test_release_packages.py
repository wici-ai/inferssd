import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_release_assets", ROOT / "scripts/prepare-release-assets.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
REVISION = "1" * 40


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "source"
    (root / "framework").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "framework/pyproject.toml").write_text('version = "0.1.3"\n')
    for name in ("install.sh", "install.ps1"):
        shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
    directories = {}
    for suffix, (platform, extension, _) in release.PACKAGES.items():
        directory = tmp_path / suffix
        directory.mkdir()
        name = f"inferssd-261109.250.1111111-{suffix}"
        archive = directory / (name + extension)
        archive.write_bytes((suffix + " checked package").encode())
        package = {
            "schema": "inferssd-package 1",
            "name": name,
            "platform": platform,
            "file": archive.name,
            "size": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "version": "0.1.3",
            "revision": REVISION,
            "build": 250,
            "date": "2026-11-09",
            "label": "261109.250.1111111",
        }
        (directory / (name + ".json")).write_text(json.dumps(package))
        directories[suffix] = directory
    return root, directories, tmp_path / "assets"


def prepare(inputs, tag="v0.1.3", revision=REVISION, signed_installer=None):
    root, directories, output = inputs
    return release.prepare(
        root, directories, output, tag, revision, "wici-ai/inferssd", signed_installer
    )


SIGNATURE = b"\r\n# SIG # Begin signature block\r\n# MIIx\r\n# SIG # End signature block\r\n"


def test_one_release_contains_every_package_and_both_pinned_installers(inputs):
    root, _, output = inputs
    originals = {
        name: (root / "scripts" / name).read_bytes() for name in ("install.sh", "install.ps1")
    }
    files = prepare(inputs)
    assert len(files) == 8 and set(files) == {p.name for p in output.iterdir()}
    assert {
        "install.sh",
        "install.ps1",
        "latest-macos.json",
        "latest-windows-vulkan.json",
        "latest-windows-cuda.json",
    } <= set(files)
    for platform, _, listing in release.PACKAGES.values():
        package = json.loads((output / listing).read_text())
        assert package["platform"] == platform and package["revision"] == REVISION
        assert (
            package["url"]
            == "https://github.com/wici-ai/inferssd/releases/download/v0.1.3/" + package["file"]
        )
        assert (output / package["file"]).exists()
    shell = (output / "install.sh").read_text()
    powershell = (output / "install.ps1").read_text()
    assert "DEFAULT_REF=v0.1.3" in shell
    assert (
        "PACKAGES=https://github.com/wici-ai/inferssd/releases/download/v0.1.3/latest-macos.json"
        in shell
    )
    assert "$DefaultRef = 'v0.1.3'" in powershell
    base = "https://github.com/wici-ai/inferssd/releases/download/v0.1.3/"
    assert f"$VulkanPackages = '{base}latest-windows-vulkan.json'" in powershell
    assert f"$CudaPackages = '{base}latest-windows-cuda.json'" in powershell
    for name, original in originals.items():
        assert (root / "scripts" / name).read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "0.1.4"),
        ("revision", "2" * 40),
        ("build", 251),
        ("label", "another-build"),
        ("sha256", "0" * 64),
        ("size", 1),
    ],
)
def test_other_version_commit_build_or_changed_bytes_cannot_enter_numbered_release(
    inputs, field, value
):
    _, directories, output = inputs
    listing = next(directories["windows-x64-cuda"].glob("*.json"))
    package = json.loads(listing.read_text())
    package[field] = value
    listing.write_text(json.dumps(package))
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not output.exists()


def test_missing_platform_and_existing_assets_are_not_silently_replaced(inputs):
    _, directories, output = inputs
    manifest = next(directories["windows-x64-vulkan"].glob("*.json"))
    saved = manifest.read_bytes()
    manifest.unlink()
    with pytest.raises(ValueError, match="expected one checked"):
        prepare(inputs)
    manifest.write_bytes(saved)
    output.mkdir()
    existing = output / "install.ps1"
    existing.write_text("published bytes")
    with pytest.raises(ValueError, match="existing assets were preserved"):
        prepare(inputs)
    assert existing.read_text() == "published bytes"


def test_the_signed_pinned_installer_is_published_byte_for_byte(inputs, tmp_path):
    root, _, output = inputs
    pinned = release.pinned_powershell(root, "v0.1.3", "wici-ai/inferssd")
    signed = tmp_path / "signed-install.ps1"
    signed.write_bytes(pinned.encode() + SIGNATURE)
    prepare(inputs, signed_installer=signed)
    assert (output / "install.ps1").read_bytes() == signed.read_bytes()
    assert "$DefaultRef = 'v0.1.3'" in pinned


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body.replace(b"v0.1.3", b"v0.1.2", 1) + SIGNATURE,
        lambda body: body + b"Write-Output 'added'\n" + SIGNATURE,
        lambda body: body,
    ],
)
def test_a_signed_installer_must_be_the_pinned_installer(inputs, tmp_path, change):
    root, _, output = inputs
    pinned = release.pinned_powershell(root, "v0.1.3", "wici-ai/inferssd").encode()
    signed = tmp_path / "signed-install.ps1"
    signed.write_bytes(change(pinned))
    with pytest.raises(ValueError, match="pinned installer"):
        prepare(inputs, signed_installer=signed)
    assert not output.exists()


def test_the_signing_job_writes_the_same_pinned_installer(tmp_path):
    written = tmp_path / "out" / "install.ps1"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare-release-assets.py"),
            "--pinned-install-ps1",
            str(written),
            "--tag",
            "v0.1.3",
            "--repository",
            "wici-ai/inferssd",
        ],
        check=True,
    )
    expected = release.pinned_powershell(ROOT, "v0.1.3", "wici-ai/inferssd")
    assert written.read_bytes() == expected.encode()


def test_published_windows_packages_are_signed_before_their_check():
    yaml = pytest.importorskip("yaml")
    jobs = yaml.load(
        (ROOT / ".github/workflows/package-windows.yml").read_text(), Loader=yaml.BaseLoader
    )["jobs"]
    sign = jobs["sign"]
    assert sign["needs"] == "build" and sign["environment"] == "windows-signing"
    assert "inputs.source_ref" in sign["if"] and "inputs.publish" in sign["if"]
    uploads = [s for s in sign["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    assert {s["with"]["name"] for s in uploads if s["with"].get("overwrite") == "true"} == {
        "package-windows-x64-vulkan",
        "package-windows-x64-cuda",
    }
    assert set(jobs["check"]["needs"]) == {"build", "sign"}
    assert set(jobs["publish"]["needs"]) == {"sign", "check"}
    assert "needs.sign.result == 'success'" in jobs["publish"]["if"]


def test_tag_is_the_project_version_and_exact_tag_commit(inputs):
    with pytest.raises(ValueError, match="framework version"):
        prepare(inputs, tag="v0.1.4")
    with pytest.raises(ValueError, match="tag commit"):
        prepare(inputs, revision="3" * 40)


def test_tag_workflow_waits_for_both_reusable_checked_packages():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    jobs = workflow["jobs"]
    for name in ("macos", "windows"):
        assert jobs[name]["uses"] == f"./.github/workflows/package-{name}.yml"
        assert jobs[name]["with"]["source_ref"] == "${{ needs.validate.outputs.revision }}"
    assert set(jobs["release"]["needs"]) == {"validate", "macos", "windows"}
    preparation = next(
        s["run"] for s in jobs["release"]["steps"] if "Prepare both" in s.get("name", "")
    )
    assert "--signed-install-ps1 dist/install-windows/install.ps1" in preparation
    group = jobs["release"]["concurrency"]["group"]
    for name in ("macos", "windows"):
        package_workflow = yaml.load(
            (ROOT / f".github/workflows/package-{name}.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        assert package_workflow["jobs"]["publish"]["concurrency"]["group"] == group
    publication = next(
        s["run"] for s in jobs["release"]["steps"] if "Publish the common" in s.get("name", "")
    )
    assert "scripts/publish-package.py dist/macos" in publication
    assert "scripts/publish-package.py dist/windows-vulkan\n" in publication
    assert "scripts/publish-package.py dist/windows-cuda" in publication
    assert "--draft --verify-tag" in publication and "--draft=false" in publication
    assert "its assets will not be replaced" in publication
