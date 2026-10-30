"""One fake-broker orchestration check; no native build or device execution."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import staged_build


@pytest.mark.parametrize("platform,bad_toolchain", [("linux", False), ("darwin", False), ("linux", True)])
def test_build_freeze_cache_validation_then_sandboxed_version(tmp_path, monkeypatch, platform, bad_toolchain):
    # The fake broker cannot create root-owned artifacts. All other sealing/read
    # checks run against real immutable temporary files; this is no native proof.
    monkeypatch.setattr(staged_build, "protected", lambda path: Path(path))
    frozen = tmp_path / "sealed"
    (frozen / "bin").mkdir(parents=True)
    (frozen / "lib").mkdir()
    for path, content in (
        (frozen / "CMakeCache.txt", b"CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc\nCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"),
        (frozen / "bin/inferssd", b"fake server; never executed"),
        (frozen / "lib/libinferssd.so", b"fake library; never loaded"),
    ):
        path.write_bytes(content)
        path.chmod(0o444)
    settings = {
        "cmake": "/usr/bin/cmake", "cmake_flags": ["-G", "Ninja"], "jobs": 4,
        "targets": ["inferssd"], "timeout_s": 60,
        "binary_relative": "bin/inferssd", "image_relatives": ["lib/libinferssd.so"],
        "toolchain": {"CMAKE_C_COMPILER": "/usr/bin/cc", "CMAKE_CXX_COMPILER": "/wrong/compiler" if bad_toolchain else "/usr/bin/c++"},
    }
    paths = {"platform": platform, "source": "/source/base", "build": "/scratch/base",
             "cwd": "/scratch", "frozen": "/build/base"}
    if platform == "darwin":
        paths.update(job_root=str(tmp_path / "job"), build=str(tmp_path / "job/work/base"),
                     cwd=str(tmp_path / "job/work"), frozen=str(frozen))
        # The Mac platform owns a fixed job/frozen/<side> layout.
        new = tmp_path / "job/frozen/base"
        new.parent.mkdir(parents=True)
        frozen.rename(new)
        frozen = new
        paths["frozen"] = str(new)
    events = []

    class Session:
        def start(self, name, kind, argv, environment, cwd, timeout):
            events.append(("start", name, kind, argv, cwd))
            assert environment == {} and timeout == 60
            return name

        def wait(self, stage, **kwargs):
            assert "deadline" in kwargs
            return {"running": False, "returncode": 0, "cleanup_confirmed": True,
                    "receipt": {"passed": True}}

        def output(self, stage, stream, offset):
            assert offset == 0
            body = b"version aaaaaaa\n" if stage.endswith("version") and stream == "stdout" else b"ok\n"
            return body, len(body), True

        def freeze(self, stage, **fields):
            events.append(("freeze", stage, fields))
            assert stage == "base-build"
            assert fields == ({"source": paths["build"], "name": "base"} if platform == "darwin" else {})
            if platform == "darwin":
                records = {str(p.relative_to(frozen)):dict(size=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(), mode=0o444)
                    for p in frozen.rglob("*") if p.is_file()}
                manifest = frozen.parent / "base.manifest.json"
                manifest.write_text(json.dumps(records)); manifest.chmod(0o444)
                return dict(path=paths["frozen"], manifest_path=str(manifest),
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), file_count=len(records), readonly=True)
            return {"path": paths["frozen"], "host_path": str(frozen), "manifest_sha256": "b" * 64}

    if bad_toolchain:
        with pytest.raises(ValueError, match="compiler paths"):
            staged_build.run_build(Session(), "base", settings, paths, tmp_path / "result", "a" * 40)
        assert not any(event[0] == "start" and event[1] == "base-version" for event in events)
    else:
        receipt = staged_build.run_build(Session(), "base", settings, paths, tmp_path / "result", "a" * 40)
        assert receipt["status"] == "passed" and len(receipt["stages"]) == 3
        assert receipt["binary"]["runtime_path"] == paths["frozen"] + "/bin/inferssd"
        assert events[-1] == ("start", "base-version", "test", [paths["frozen"] + "/bin/inferssd", "--version"], paths["frozen"])
    stored = json.loads((tmp_path / "result/build-receipt.json").read_text())
    assert stored["status"] == ("failed" if bad_toolchain else "passed")
    assert events[0][3] == ["/usr/bin/cmake", "-S", "/source/base", "-B", paths["build"], "-G", "Ninja"]
    assert events[1][3] == ["/usr/bin/cmake", "--build", paths["build"], "--parallel", "4", "--target", "inferssd"]


@pytest.mark.parametrize("mutation", [None, "outside-link", "replaced-member"])
def test_only_complete_root_freezer_alias_groups_are_readable(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(staged_build, "protected", lambda path: Path(path))
    root = tmp_path / "frozen"
    root.mkdir()
    content = b"one shared library inode; not executed"
    names = ["libengine.so.1.2", "libengine.so.1", "libengine.so"]
    original = root / names[0]
    original.write_bytes(content)
    original.chmod(0o444)
    for name in names[1:]:
        os.link(original, root / name)
    sha256 = hashlib.sha256(content).hexdigest()
    records = [dict(path=name, size=len(content), sha256=sha256,
                    **({"alias_of": names[0]} if index else {}))
               for index, name in enumerate(names)]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": records}))
    manifest.chmod(0o444)
    receipt = dict(host_path=str(root), manifest_path=str(manifest),
                   manifest_sha256=hashlib.sha256(encoded).hexdigest())
    aliases = staged_build.frozen_aliases(receipt)
    with pytest.raises(ValueError, match="complete root freezer alias proof"):
        staged_build.sealed_file(original)
    if mutation == "outside-link":
        os.link(original, tmp_path / "unproven-outside-alias")
    elif mutation == "replaced-member":
        (root / names[-1]).unlink()
        (root / names[-1]).write_bytes(content)
        (root / names[-1]).chmod(0o444)
    if mutation:
        with pytest.raises(ValueError, match="alias"):
            staged_build.sealed_file(original, aliases=aliases)
    else:
        result = staged_build.sealed_file(root / names[1], aliases=aliases)
        assert result["sha256"] == sha256
        assert set(result["aliases"]) == {str(root / name) for name in names}


@pytest.mark.parametrize("platform,flags", [
    ("darwin", ["-DLLAMA_USE_SYSTEM_GGML=ON"]),
    ("darwin", ["-DLLAMA_SSD_STORE:BOOL=TRUE", "-DLLAMA_USE_SYSTEM_GGML:BOOL=TRUE"]),
    ("linux", ["-DLLAMA_USE_SYSTEM_GGML=ON"]),
    ("linux", ["-DLLAMA_SSD_STORE=OFF", "-DLLAMA_SSD_STORE=ON", "-DLLAMA_USE_SYSTEM_GGML=ON"]),
])
def test_incompatible_store_profile_rejected_before_build(platform, flags):
    with pytest.raises(ValueError, match="LLAMA_SSD_STORE requires the bundled ggml"):
        staged_build.validate_platform_flags(platform, flags)


@pytest.mark.parametrize("platform,flags", [
    ("darwin", ["-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON"]),
    ("darwin", ["-DLLAMA_SSD_STORE=ON", "-DBUILD_SHARED_LIBS=ON"]),
    ("linux", ["-DLLAMA_SSD_STORE=ON", "-DBUILD_SHARED_LIBS=ON"]),
    ("linux", ["-DBUILD_SHARED_LIBS=OFF"]),
    ("linux", ["-DLLAMA_SSD_STORE=OFF", "-DLLAMA_USE_SYSTEM_GGML=ON"]),
])
def test_supported_platform_profiles(platform, flags):
    staged_build.validate_platform_flags(platform, flags)
