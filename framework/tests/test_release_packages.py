import hashlib
import importlib.util
import json
import re
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
    (root / "scripts").mkdir(parents=True)
    for name in ("install.sh", "install.ps1"):
        shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
    directories = {}
    for suffix, (platform, extension, _) in release.PACKAGES.items():
        directory = tmp_path / suffix
        directory.mkdir()
        name = f"inferssd-261109.250-{suffix}"
        archive = directory / (name + extension)
        archive.write_bytes((suffix + " checked package").encode())
        package = {
            "schema": "inferssd-package 1",
            "name": name,
            "platform": platform,
            "file": archive.name,
            "size": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "version": "261109.250",
            "tag": "261109.250",
            "revision": REVISION,
            "build": 250,
            "date": "2026-11-09",
            "label": "261109.250.1111111",
        }
        (directory / (name + ".json")).write_text(json.dumps(package))
        directories[suffix] = directory
    return root, directories, tmp_path / "assets"


def prepare(inputs, tag="261109.250", revision=REVISION, signed_installer=None):
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
            == "https://github.com/wici-ai/inferssd/releases/download/261109.250/" + package["file"]
        )
        assert (output / package["file"]).exists()
    shell = (output / "install.sh").read_text()
    powershell = (output / "install.ps1").read_text()
    assert "DEFAULT_REF=261109.250" in shell
    assert (
        "PACKAGES=https://github.com/wici-ai/inferssd/releases/download/261109.250/latest-macos.json"
        in shell
    )
    assert "$DefaultRef = '261109.250'" in powershell
    base = "https://github.com/wici-ai/inferssd/releases/download/261109.250/"
    assert f"$VulkanPackages = '{base}latest-windows-vulkan.json'" in powershell
    assert f"$CudaPackages = '{base}latest-windows-cuda.json'" in powershell
    for name, original in originals.items():
        assert (root / "scripts" / name).read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "261109.251"),
        ("tag", None),
        ("tag", "261109.251"),
        ("tag", "v261109.250"),
        ("name", "inferssd-261109.250.1111111-windows-x64-cuda"),
        ("revision", "2" * 40),
        ("build", 251),
        ("label", "another-build"),
        ("sha256", "0" * 64),
        ("size", 1),
    ],
)
def test_other_version_commit_build_or_changed_bytes_cannot_enter_a_release(inputs, field, value):
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
    pinned = release.pinned_powershell(root, "261109.250", "wici-ai/inferssd")
    signed = tmp_path / "signed-install.ps1"
    signed.write_bytes(pinned.encode() + SIGNATURE)
    prepare(inputs, signed_installer=signed)
    assert (output / "install.ps1").read_bytes() == signed.read_bytes()
    assert "$DefaultRef = '261109.250'" in pinned


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body.replace(b"261109.250", b"261109.249", 1) + SIGNATURE,
        lambda body: body + b"Write-Output 'added'\n" + SIGNATURE,
        lambda body: body,
    ],
)
def test_a_signed_installer_must_be_the_pinned_installer(inputs, tmp_path, change):
    root, _, output = inputs
    pinned = release.pinned_powershell(root, "261109.250", "wici-ai/inferssd").encode()
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
            "261109.250",
            "--repository",
            "wici-ai/inferssd",
        ],
        check=True,
    )
    expected = release.pinned_powershell(ROOT, "261109.250", "wici-ai/inferssd")
    assert written.read_bytes() == expected.encode()


def test_released_windows_packages_are_signed_before_their_check():
    yaml = pytest.importorskip("yaml")
    jobs = yaml.load(
        (ROOT / ".github/workflows/package-windows.yml").read_text(), Loader=yaml.BaseLoader
    )["jobs"]
    sign = jobs["sign"]
    assert sign["needs"] == "build" and sign["environment"] == "windows-signing"
    assert sign["if"] == "inputs.release_tag"
    uploads = [s for s in sign["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    # Each backend's signing replaces that backend's built package with the signed one.
    assert {s["with"]["name"] for s in uploads if s["with"].get("overwrite") == "true"} == {
        "package-windows-x64-${{ matrix.backend }}",
    }
    assert set(jobs["check"]["needs"]) == {"build", "sign"}


def test_the_tag_and_commit_must_be_the_packages(inputs):
    with pytest.raises(ValueError, match="release version and tag commit"):
        prepare(inputs, tag="261109.251")
    # A tag of the old form names no release now.
    with pytest.raises(ValueError, match="<yymmdd>.<build>"):
        prepare(inputs, tag="v0.1.3")
    with pytest.raises(ValueError, match="tag commit"):
        prepare(inputs, revision="3" * 40)


def test_release_notes_put_highlights_and_installation_before_the_changes(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 261118.294 - 2026-11-18\n\nSigned packages.\n\n"
        "## 0.1.4 - 2026-11-10\n\nOlder.\n"
    )
    changes = "## What's Changed\n* ci: one release (#1)\n"
    notes = release.release_notes(tmp_path, "261118.294", "wici-ai/inferssd", changes)
    base = "https://github.com/wici-ai/inferssd/releases/download/261118.294"
    assert notes.startswith("Signed packages.\n\n### Install or upgrade\n")
    assert f"curl -fsSL {base}/install.sh | sh" in notes
    assert f"irm {base}/install.ps1 | iex" in notes
    assert notes.endswith(changes) and "Older." not in notes
    # A release without a section of its own has the installation and the changes alone.
    plain = release.release_notes(tmp_path, "261119.301", "wici-ai/inferssd", changes)
    assert plain.startswith("### Install or upgrade\n") and plain.endswith(changes)


def test_release_notes_name_pull_requests_and_credit_every_contributor(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n")
    pull = "https://github.com/wici-ai/inferssd/pull"
    changes = (
        "## What's Changed\n"
        f"* engine: one thing by @Chikati in {pull}/164\n"
        f"* model: another by @Bi666666 in {pull}/292\n"
        f"* docs: a third by @Chikati in {pull}/310\n"
        f"* ci: bump by @dependabot[bot] in {pull}/311\n"
        "* an entry GitHub may write otherwise\n"
        "\n## New Contributors\n"
        f"* @Bi666666 made their first contribution in {pull}/292\n"
        "\n**Full Changelog**: https://github.com/wici-ai/inferssd/compare/v0.1.4...v0.1.5\n"
    )
    notes = release.release_notes(tmp_path, "261119.301", "wici-ai/inferssd", changes)
    changed = notes[notes.index("## What's Changed") :]
    assert changed == (
        "## What's Changed\n"
        "* engine: one thing (#164)\n"
        "* model: another (#292)\n"
        "* docs: a third (#310)\n"
        "* ci: bump (#311)\n"
        "* an entry GitHub may write otherwise\n"
        "\n## Contributors\n\n"
        "@Chikati, @Bi666666\n"
        "\n**Full Changelog**: https://github.com/wici-ai/inferssd/compare/v0.1.4...v0.1.5\n"
    )


def test_every_published_package_is_a_release_of_main():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.load(
        (ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    # The version is main's head's own: a run's one input says whether a GitHub release is made.
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"github_release"}
    assert inputs["github_release"]["type"] == "boolean"
    assert inputs["github_release"]["default"] == "true"
    jobs = workflow["jobs"]
    release_with = {
        "source_ref": "${{ needs.validate.outputs.revision }}",
        "release_tag": "${{ needs.validate.outputs.tag }}",
    }
    assert jobs["macos"]["uses"] == "./.github/workflows/package-macos.yml"
    assert jobs["macos"]["with"] == release_with
    # Each Windows package in a call of its own, so one does not wait for the other.
    for backend in ("vulkan", "cuda"):
        call = jobs[f"windows-{backend}"]
        assert call["uses"] == "./.github/workflows/package-windows.yml"
        assert call["with"] == {**release_with, "backends": f'["{backend}"]'}
    # Every secret the Windows package workflow reads is declared, required and passed by name: a
    # reusable workflow's job reads an environment secret only when its caller passes it, and an
    # unpassed one is an empty string, which once left the signing job without its credential.
    windows_text = (ROOT / ".github/workflows/package-windows.yml").read_text()
    windows = yaml.load(windows_text, Loader=yaml.BaseLoader)
    declared = windows["on"]["workflow_call"]["secrets"]
    assert set(re.findall(r"secrets\.([A-Z_]+)", windows_text)) == set(declared)
    signing = {"ES_USERNAME", "ES_PASSWORD", "ES_CREDENTIAL_ID", "ES_TOTP_SECRET"}
    assert set(declared) == {"APP_VERSIONING_TOKEN"} | signing
    assert all(entry["required"] == "true" for entry in declared.values())
    for backend in ("vulkan", "cuda"):
        assert jobs[f"windows-{backend}"]["secrets"] == {
            name: f"${{{{ secrets.{name} }}}}" for name in declared
        }
    choice = next(s["run"] for s in jobs["validate"]["steps"] if s.get("id") == "version")
    assert "refs/heads/main" in choice
    assert "--date=format-local:%y%m%d HEAD).$(git rev-list --count HEAD)" in choice
    # A version that is already a tag stops the run before anything is built or published.
    assert 'git rev-parse -q --verify "refs/tags/$tag"' in choice
    # Each package reaches the CDN and its package list once its own package checks pass, not
    # when the slowest does.
    published = {
        "publish-macos": ("macos", "package-macos", "dist/macos"),
        "publish-windows-vulkan": (
            "windows-vulkan",
            "package-windows-x64-vulkan",
            "dist/windows-vulkan",
        ),
        "publish-windows-cuda": ("windows-cuda", "package-windows-x64-cuda", "dist/windows-cuda"),
    }
    for job, (package, artifact, dist) in published.items():
        assert set(jobs[job]["needs"]) == {"validate", package}
        steps = jobs[job]["steps"]
        assert any(s.get("with", {}).get("name") == artifact for s in steps)
        run = next(s["run"] for s in steps if "Publish the" in s.get("name", ""))
        assert run.rstrip().endswith(f"scripts/publish-package.py {dist}")
    # The GitHub release comes last, once every package is published, and only when asked for.
    release = jobs["release"]
    assert release["if"] == "inputs.github_release"
    assert set(release["needs"]) == {"validate", *(p for p, _, _ in published.values()), *published}
    preparation = next(s["run"] for s in release["steps"] if "Prepare both" in s.get("name", ""))
    assert "--signed-install-ps1 dist/install-windows/install.ps1" in preparation
    publication = next(
        s["run"] for s in release["steps"] if "Publish the release" in s.get("name", "")
    )
    assert "--draft --target" in publication and "--draft=false --latest" in publication
    assert "its release and assets will not be replaced" in publication
    assert "publish-package.py" not in yaml.dump(release)
    # The Windows package workflow packages the backends it is given, each signed on its own, one
    # signing at a time; the Vulkan package's signing also signs the pinned installer, once a run.
    backends = windows["on"]["workflow_call"]["inputs"]["backends"]
    assert backends["required"] == "false" and backends["default"] == '["vulkan", "cuda"]'
    every = """${{ fromJSON(inputs.backends || '["vulkan", "cuda"]') }}"""
    for job in ("build", "sign", "check"):
        assert windows["jobs"][job]["strategy"]["matrix"]["backend"] == every
    assert "inputs.backends" in windows["concurrency"]["group"]
    # The CUDA build takes the organization's larger runner; signing and checking stay on the
    # standard one.
    build_on = windows["jobs"]["build"]["runs-on"]
    assert build_on == (
        "${{ matrix.backend == 'cuda' && 'release-packaging-runner-windows' || 'windows-2025' }}"
    )
    assert (
        windows["jobs"]["sign"]["runs-on"] == windows["jobs"]["check"]["runs-on"] == "windows-2025"
    )
    sign = windows["jobs"]["sign"]
    assert sign["concurrency"] == {"group": "windows-signing", "cancel-in-progress": "false"}
    installer = [
        s
        for s in sign["steps"]
        if "install" in (s.get("name", "") + str(s.get("with", {}))).lower()
    ]
    assert installer and all(s["if"] == "matrix.backend == 'vulkan'" for s in installer)
    # The package workflows build and check; only Release publishes. Packages are built only for a
    # release and by hand: no pull request or merge builds one.
    for name in ("macos", "windows"):
        package_workflow = yaml.load(
            (ROOT / f".github/workflows/package-{name}.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        assert set(package_workflow["on"]) == {"workflow_call", "workflow_dispatch"}
        assert "publish" not in package_workflow["jobs"]
        assert "inputs" not in (package_workflow["on"]["workflow_dispatch"] or {})
