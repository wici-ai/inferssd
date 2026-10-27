"""Fixed Mac public metadata view; no sudo, device, broker or model execution."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import isolated_device as isolation, selection_contract as contract
from ci.contract import digest
from test_ci_mac_operator import JOB, MANIFEST, mac_config
from test_ci_selection import POLICY


PUBLIC = Path("/Library/InferSSD-CI/inputs/operator-" + JOB + "/manifest.json")


@pytest.mark.parametrize("logical", [
    "/tmp/manifest.json", "/Library/InferSSD-CI/private/job-manifests/../" + JOB + ".json",
    "/Library/InferSSD-CI/private/job-manifests/" + JOB.upper() + ".json",
    "/Library/InferSSD-CI/private/job-manifests/" + JOB[:-1] + ".json",
    str(PUBLIC), MANIFEST + "/extra", None,
])
def test_mac_view_rejects_noncanonical_private_identity(logical):
    with pytest.raises(ValueError, match="exact private job"):
        isolation.manifest_read_path(dict(paths=dict(platform="darwin"), purpose="selected-integration", manifest_path=logical))


def test_mac_view_ignores_arbitrary_caller_fallback():
    config = dict(paths=dict(platform="darwin"), purpose="selected-integration", manifest_path=MANIFEST,
                  runner_manifest_path="/tmp/attacker.json")
    assert isolation.manifest_read_path(config) == PUBLIC


@pytest.fixture
def bound_files(tmp_path, monkeypatch):
    config, manifest, plan = mac_config()
    config, manifest = contract.bind(config, manifest, plan, POLICY)
    mirror = tmp_path / "public-manifest.json"
    mirror.write_text(json.dumps(manifest))
    mirror.chmod(0o444)
    config["manifest_sha256"] = digest(mirror)
    file = tmp_path / "config.json"
    file.write_text(json.dumps(config))
    observed = []
    def protected(path, **_kwargs):
        path = Path(path)
        observed.append(str(path))
        # Model the actual private directory denial. No root-directory permission
        # claim is made by this portable test; the post-deploy runner probe is separate.
        if path == Path(MANIFEST):
            raise PermissionError("private broker manifest is root-only")
        if path == PUBLIC:
            return mirror
        assert path == file
        return file
    monkeypatch.setattr(isolation, "protected", protected)
    return config, manifest, mirror, file, observed


def test_binding_reads_exact_copy_without_accessing_private_directory(bound_files):
    config, _, _, file, observed = bound_files
    assert contract.protected_binding(file, digest(file), config["selection_plan_sha256"], JOB) == config
    assert observed == [str(file), str(PUBLIC)]
    assert config["manifest_path"] == MANIFEST


@pytest.mark.parametrize("fault", ["changed-bytes", "writable", "wrong-job"])
def test_bound_read_view_rejects_invalid_copy(bound_files, fault):
    config, manifest, mirror, file, _ = bound_files
    if fault == "changed-bytes":
        mirror.chmod(0o644)
        mirror.write_bytes(mirror.read_bytes() + b" ")
        mirror.chmod(0o444)
    elif fault == "writable":
        mirror.chmod(0o644)
    else:
        manifest["job_id"] = "b" * 32
        mirror.chmod(0o644)
        mirror.write_text(json.dumps(manifest))
        mirror.chmod(0o444)
        config["manifest_sha256"] = digest(mirror)
        file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash differs|readonly|binding differs"):
        contract.protected_binding(file, digest(file), config["selection_plan_sha256"], JOB)


def test_missing_public_copy_does_not_fall_back_to_private(tmp_path, monkeypatch):
    config = dict(paths=dict(platform="darwin"), purpose="selected-integration", manifest_path=MANIFEST, manifest_sha256="a" * 64)
    calls = []
    def absent(path):
        calls.append(Path(path))
        raise FileNotFoundError(path)
    monkeypatch.setattr(isolation, "protected", absent)
    with pytest.raises(FileNotFoundError):
        isolation.protected_manifest(config)
    assert calls == [PUBLIC]


def test_existing_protection_rejects_symlink_and_runner_owned_copy(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="root-owned, non-symlink"):
        isolation.protected(link)
    original = Path.lstat
    class RunnerOwned:
        st_uid = 62001
        st_mode = 0o100444
    monkeypatch.setattr(Path, "lstat", lambda path: RunnerOwned() if path == target else original(path))
    with pytest.raises(ValueError, match="root-owned, non-symlink"):
        isolation.protected(target)


@pytest.mark.parametrize("platform,purpose", [("linux", "selected-integration"), ("darwin", "manual-readiness"), ("darwin", None)])
def test_linux_and_legacy_mac_read_path_owner_write_semantics_are_unchanged(tmp_path, monkeypatch, platform, purpose):
    manifest = tmp_path / "linux-manifest.json"
    manifest.write_text("{}")
    manifest.chmod(0o644)
    logical = Path(MANIFEST) if platform == "darwin" else manifest
    config = dict(paths=dict(platform=platform), manifest_path=str(logical), manifest_sha256=digest(manifest))
    if purpose is not None:
        config["purpose"] = purpose
    seen = []
    def protected(path):
        seen.append(path)
        return manifest
    monkeypatch.setattr(isolation, "protected", protected)
    assert isolation.manifest_read_path(config) == logical
    assert isolation.protected_manifest(config) == manifest
    assert seen == [logical]
