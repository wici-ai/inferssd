"""Only recovery refusal checks; no build, device or second completion."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import resume_base
from ci.contract import digest, save
from ci.reference import LEGACY_REVISION


def test_configuration_only_allows_new_manifest_and_identical_settings_relocation():
    old = dict(manifest_path="/old", manifest_sha256="a" * 64,
               base={"revision": LEGACY_REVISION}, candidate={"revision": "b" * 40,
               "settings": {"host_path": "/old.ssd", "runtime_path": "/models/settings.ssd", "sha256": "c" * 64}},
               host={"memory": 64}, comparison_binding={"vram.budget": 0})
    new = copy.deepcopy(old)
    new.update(manifest_path="/new", manifest_sha256="d" * 64, resumed_base={})
    new["candidate"]["settings"]["host_path"] = "/new.ssd"
    resume_base.same_configuration(old, new)
    for field, value in (("host", {"memory": 63}), ("base", {"revision": "e" * 40}),
                         ("comparison_binding", {"vram.budget": 8 << 30})):
        changed = copy.deepcopy(new)
        changed[field] = value
        with pytest.raises(ValueError, match="original two arms"):
            resume_base.same_configuration(old, changed)
    new["candidate"]["settings"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="original two arms"):
        resume_base.same_configuration(old, new)


def test_sealed_manifest_rejects_changed_or_unlisted_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(resume_base, "sealed_file", lambda path, **kw:
        ({"sha256": digest(Path(path)), "bytes": Path(path).stat().st_size}, b"")
        if kw.get("contents") else {"sha256": digest(Path(path)), "bytes": Path(path).stat().st_size})
    root = tmp_path / "artifact"
    root.mkdir()
    item = root / "raw.sse"
    item.write_text("original")
    manifest = tmp_path / "manifest.json"
    save(manifest, dict(schema_version=1, job_id="a" * 32, run_id="123", run_attempt=1,
                        files={"raw.sse": digest(item)}))
    spec = dict(artifact_root=str(root), manifest_path=str(manifest), manifest_sha256=digest(manifest))
    resume_base.sealed_evidence(spec)
    item.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        resume_base.sealed_evidence(spec)
    item.write_text("original")
    (root / "unbound.json").write_text("{}")
    with pytest.raises(ValueError, match="complete retained artifact"):
        resume_base.sealed_evidence(spec)


def completed_fixture(tmp_path):
    evidence = tmp_path / "evidence"
    output = evidence / "base/benchmark"
    output.mkdir(parents=True)
    old = {"manifest_sha256": "a" * 64, "base": {"benchmark": {"request_timeout_s": 2700}}}
    save(evidence / "checkpoint-config.json", old)
    job_id, lease = "b" * 32, {"inode": 42, "device": 1}
    opened = dict(job_id=job_id, manifest_sha256=old["manifest_sha256"], lease=lease, opened=True)
    identity = dict(worker_pid=12, worker_uid=996, birth_identity="boot:123",
                    scope_identity="/scope", handle="handle", stage_id="base-benchmark-512")
    stopped = dict(identity, receipt={"lease": lease}, cleanup_confirmed=True)
    save(evidence / "job-opened.json", opened)
    save(evidence / "job-finished.json", dict(opened, cleanup_confirmed=True, errors=[],
             stages={"base-benchmark-512": stopped["receipt"]}))
    save(evidence / "checkpoint.json", dict(status="failed", cleanup_confirmed=True, arms={"base": {}},
        error="RuntimeError: benchmark arm failed; retain its raw evidence",
        config_sha256=digest(evidence / "checkpoint-config.json")))
    result = {"status": "passed", "output_tokens": 256}
    save(output / "result.json", result)
    save(output / "benchmark.json", dict(status="failed", side="base", revision=LEGACY_REVISION,
        completion_requests=1, samples_per_tier=1, output_tokens=256, tier="512", cleanup_confirmed=True,
        cleanup_error="RuntimeError: native benchmark did not stop normally and cleanly", result=result,
        result_artifact={"path": "result.json", "sha256": digest(output / "result.json")}))
    for name, value in (("started", identity), ("stopped", stopped),
                        ("runtime-before", dict(identity, ready=True)), ("runtime-after", dict(identity, ready=True))):
        save(output / (name + ".json"), value)
    return evidence, old, {"job_id": job_id}


def test_completed_base_preserves_failed_json_and_requires_single_complete_sample(tmp_path, monkeypatch):
    evidence, old, seal = completed_fixture(tmp_path)
    output = evidence / "base/benchmark"
    monkeypatch.setattr(resume_base, "verify_stopped", lambda row: "trusted-controller-stop")
    monkeypatch.setattr(resume_base.run_final, "read_result", lambda path, *args: json.loads(path.read_text()))
    original = (output / "benchmark.json").read_bytes()
    assert resume_base.completed_base(evidence, old, seal)[0]["output_tokens"] == 256
    assert (output / "benchmark.json").read_bytes() == original
    for changes in ({"completion_requests": 2}, {"error": "native failure"},
                    {"log_errors": {"stderr": "incomplete"}}, {"cleanup_confirmed": False},
                    {"result": {"status": "different"}}):
        changed = json.loads(original)
        changed.update(changes)
        (output / "benchmark.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            resume_base.completed_base(evidence, old, seal)
    (output / "benchmark.json").write_bytes(original)
    after = json.loads((output / "runtime-after.json").read_text())
    after["birth_identity"] = "reused-process"
    (output / "runtime-after.json").write_text(json.dumps(after))
    with pytest.raises(ValueError, match="runtime identity"):
        resume_base.completed_base(evidence, old, seal)


def test_recovery_keeps_failed_raw_bytes_but_new_derived_directory_is_writable(tmp_path, monkeypatch):
    evidence, old, seal = completed_fixture(tmp_path)
    old.update(candidate={"settings": {"host_path": "/settings.ssd"}}, manifest_path="/manifest.json")
    (evidence / "checkpoint-config.json").write_text(json.dumps(old))
    source = {"sha256": "e" * 64}
    save(evidence / "base/alias-identity.json", source)
    save(evidence / "model-identity-before.json", {"same": "model"})
    (evidence / "base").chmod(0o555)
    (evidence / "base/benchmark").chmod(0o555)
    raw = evidence / "base/benchmark/benchmark.json"
    raw.chmod(0o444)
    original = raw.read_bytes()
    monkeypatch.setattr(resume_base, "sealed_evidence", lambda spec: (
        evidence, dict(seal, run_id="123", run_attempt=1)))
    monkeypatch.setattr(resume_base, "completed_base", lambda *args: (
        {"output_tokens": 256}, {"lease": {"inode": 42}},
        {"status": "failed", "error": "original failure", "config_sha256": "f" * 64}))
    monkeypatch.setattr(resume_base, "retained_build", lambda *args: {"binary": {"sha256": "b" * 64}})
    monkeypatch.setattr(resume_base, "sealed_file", lambda *args, **kwargs: ({"sha256": old["manifest_sha256"]}, b""))
    original_load = resume_base.load
    monkeypatch.setattr(resume_base, "load", lambda path: {} if str(path) == "/manifest.json" else original_load(path))
    current = dict(old, resumed_base={"artifact_root": "/sealed"})
    binding = {"sides": {"base": {"launch_policy": {"alias_sha256": [None]}}}}

    def receipt(config, side, spec, build, output, binding, model, *, evidence):
        assert evidence == raw.parent
        save(output / "comparison-receipt.json", {"verified": True})
        return {"verified": True}

    output = tmp_path / "new-base"
    resume_base.recover(current, "c" * 32, output, binding, {"same": "model"},
        receipt=receipt, alias_asset=lambda *args: source, models=lambda *args: {"same": "model"})
    assert raw.read_bytes() == original == (output / "benchmark/benchmark.json").read_bytes()
    assert (output / "benchmark/comparison-receipt.json").exists()
    assert load_recovery(output)["original_status"] == "failed"


def load_recovery(output):
    return json.loads((output / "recovery.json").read_text())
