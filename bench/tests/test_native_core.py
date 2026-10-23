"""The standalone bundle is complete and sealed before any ownership module loads."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_native
import native_core
from ssdash.platform import guardian, lease, process


def test_installed_identity_matches_installed_core():
    assert native_core.CORE["lease"] is lease
    assert native_core.CORE["process"] is process
    assert native_core.CORE["guardian"] is guardian
    assert native_core.IDENTITY["mode"] == "installed"


def test_explicit_bundle_is_self_contained(tmp_path):
    destination = tmp_path / "wrapper"
    receipt = bundle_native.bundle(destination, Path(lease.__file__).parent)
    assert "ownership/lease.py" in receipt and "ownership/process.py" in receipt
    assert "native-core.json" in receipt and "run_abba.py" in receipt
    loaded, identity = native_core.load_core(destination / "native-core.json")
    assert identity["mode"] == "bundle"
    assert set(loaded) == {"lease", "process", "guardian"}
    assert all(str(destination) in row["path"] for row in identity["modules"].values())
    spec = importlib.util.spec_from_file_location("standalone_native_core", destination / "native_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IDENTITY == identity


@pytest.mark.parametrize("change", ["hash", "version", "path", "missing", "member"])
def test_bad_bundle_never_falls_back_to_installed_core(tmp_path, change):
    destination = tmp_path / "wrapper"
    bundle_native.bundle(destination, Path(lease.__file__).parent)
    path = destination / "native-core.json"
    manifest = json.loads(path.read_text())
    if change == "hash":
        (destination / "ownership/process.py").write_text("raise AssertionError('must not import')")
    elif change == "version":
        manifest["api_version"] = 2
    elif change == "path":
        manifest["modules"]["lease"]["path"] = "../elsewhere.py"
    elif change == "missing":
        (destination / "ownership/process.py").unlink()
    else:
        del manifest["modules"]["process"]
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        native_core.load_core(path)
