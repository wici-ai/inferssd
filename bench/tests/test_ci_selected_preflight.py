"""Selected pre-timing protocol only; no device, model or timing execution."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import selected_integration as execution, selected_preflight as preflight
from ci.contract import digest, load, save
from ci.selection_contract import linux_stages
from test_ci_selection import impact, plan, template, POLICY
from test_ci_staged_benchmark import CONFIG


def setup(tmp_path, monkeypatch, *, change=None, fault=None, paths=None):
    value = plan(paths or ["engine/src/ssd/model-adapter.cpp"])
    config = template(value)
    needs = impact.device_requirements(value, config["device"])
    manifest_path = tmp_path / "manifest.json"
    config.update(selection_plan=value, selection_plan_sha256=impact.fingerprint(value),
                  manifest_sha256="a" * 64, paths=config["sides"]["candidate"]["paths"],
                  manifest_path=str(manifest_path), policy=POLICY)
    for side, data in config["sides"].items():
        for model, native in data["models"].items():
            launch = dict(CONFIG, model=model, revision=value[side],
                          argv=["/build/" + side + "/bin/llama-server"])
            native["smoke"] = copy.deepcopy(launch)
            native["benchmarks"] = {tier: copy.deepcopy(launch)
                for tier in needs["performance"][model]["tiers"]}
    if change:
        change(config)
    manifest = dict(stages=linux_stages(config, needs, ("base", "candidate")))
    save(manifest_path, manifest)
    config_path = tmp_path / "config.json"
    save(config_path, config)
    events, sessions = [], []

    class Session:
        closed = False
        job_id = "f" * 32
        def __init__(self, *args, **kwargs): sessions.append(self)
        def open(self):
            events.append(("open",))
            return dict(manifest_sha256="a" * 64)
        def finish(self):
            events.append(("finish",))
            return dict(cleanup_confirmed=True)
        def close(self): self.closed = True

    def smoke(session, spec, output, *, stage_id):
        events.append(("smoke", stage_id))
        output.mkdir(parents=True, exist_ok=False)
        result = dict(status="passed", cleanup_confirmed=True, completion_requests=1,
                      model=spec["model"], revision=spec["revision"], output_tokens=8)
        if fault in ("failed", "unclean") and stage_id == "base-muse-preflight-512":
            result["status" if fault == "failed" else "cleanup_confirmed"] = "failed" if fault == "failed" else False
            result["error"] = "retained original synthetic preflight failure"
        save(output / "smoke.json", result)
        save(output / "response.json", dict(tokens_predicted=8, truncated=False, stop_type="limit"))
        save(output / "stopped.json", dict(running=False, returncode=0, cleanup_confirmed=result["cleanup_confirmed"]))
        return result

    def receipt(config, manifest, side, model, spec, build, output, *, destination=None):
        events.append(("receipt", model, side, "timed" if "result.json" in {p.name for p in output.iterdir()} else "short"))
        typed = {"placement.vram_experts": 32}
        if model == "muse" and side == "candidate" and fault == "typed-mismatch":
            typed["placement.vram_experts"] = 0
        if fault == "timed-drift" and (output / "result.json").exists():
            typed["placement.vram_experts"] = 0
        result = dict(model_identity=dict(model=model),
            server=dict(pid=1 if side == "base" else 2, start_identity=side),
            build=dict(loaded_libraries=[dict(path="/build/" + side + "/libggml-cuda.so", sha256=side)]),
            host=dict(hardware="synthetic", os="synthetic", memory_budget_bytes=1,
                os_page_cache_treatment=spec["os_page_cache_treatment"],
                effective_settings=dict(native_launch_policy=dict(typed_values=typed),
                                        actual_resource_policy=dict(memory_max=1))))
        if model == "muse" and side == "candidate":
            if fault == "resource-mismatch": result["host"]["memory_budget_bytes"] = 2
            if fault == "image-mismatch": result["build"]["loaded_libraries"][0]["path"] = "/unexpected/libggml-cpu.so"
            if fault == "receipt-failure": raise ValueError("retained synthetic native snapshot refusal")
        save(destination if destination is not None else output / "comparison-receipt.json", result)
        return result

    def benchmark(session, side, spec, output):
        events.append(("timed", spec["model"], side))
        assert load(args.output / "performance-preflight.json")["status"] == "passed"
        output.mkdir(parents=True, exist_ok=False)
        measured = dict(metrics=dict(decode_first128_tps=1, decode_second128_tps=1,
            decode_tps=1, ttft_s=1, server_prefill_tps=1),
            input_tokens=1, timings=dict(prompt_n=1, prompt_ms=1000), prompt_ids=[1], prompt_sha256="a" * 64,
            output_tokens=256, tokens=[11] * 256)
        save(output / "result.json", measured)
        return dict(status="passed", cleanup_confirmed=True, completion_requests=1)

    monkeypatch.setattr(execution, "protected_binding", lambda *args: config)
    monkeypatch.setattr(execution, "source_manifest", lambda *args: "d" * 40)
    real_digest = execution.digest
    monkeypatch.setattr(execution, "digest", lambda p: "e" * 64 if Path(p).name == "control-source.json" else real_digest(p))
    monkeypatch.setattr(execution, "StageSession", Session)
    monkeypatch.setattr(execution, "run_build", lambda session, side, *args:
        dict(binary=dict(runtime_path="/build/" + side + "/bin/llama-server"), toolchain={}))
    monkeypatch.setattr(execution, "selected_numerical", lambda session, config, build, cases, output:
        {case: dict(status="passed") for case in cases})
    monkeypatch.setattr(execution, "smoke", smoke)
    monkeypatch.setattr(preflight, "smoke", smoke)
    monkeypatch.setattr(preflight, "receipt", receipt)
    monkeypatch.setattr(execution, "receipt", receipt)
    monkeypatch.setattr(execution, "benchmark", benchmark)
    monkeypatch.setattr(execution.run_final, "read_result", lambda path, *args: load(path))
    args = SimpleNamespace(output=tmp_path / "out", config=config_path, config_sha256=digest(config_path),
        selection_plan_sha256=impact.fingerprint(value), job_id="f" * 32, socket="/unused")
    return args, config, manifest, needs, events, sessions


def test_every_cell_is_checked_before_first_timing_and_identical_functional_is_reused(tmp_path, monkeypatch):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch)
    assert execution.run(args) == 0
    timed = next(i for i, event in enumerate(events) if event[0] == "timed")
    assert len([e for e in events[:timed] if e[0] == "receipt"]) == 4
    assert [e[1] for e in events if e[0] == "smoke"] == ["candidate-smoke-deepseek", "candidate-smoke-muse",
        "base-deepseek-preflight-512", "base-muse-preflight-512"]
    assert len([e for e in events if e[0] == "timed"]) == 4
    assert len(sessions) == 1 and sessions[0].closed and events[-1] == ("finish",)
    assert set(preflight.retained(args.output, config, needs)) == {
        (model, "512", side) for model in needs["models"] for side in ("base", "candidate")}


@pytest.mark.parametrize("fault", ["failed", "unclean", "typed-mismatch", "resource-mismatch", "image-mismatch", "receipt-failure"])
def test_late_cell_failure_preserves_raw_and_prevents_every_timed_request(tmp_path, monkeypatch, fault):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch, fault=fault)
    assert execution.run(args) == 1
    assert not any(e[0] == "timed" for e in events)
    assert load(args.output / "performance-preflight.json")["status"] == "failed"
    assert load(args.output / "preflight/muse/512/base/smoke.json")["completion_requests"] == 1
    assert len([e for e in events if e == ("smoke", "base-muse-preflight-512")]) == 1
    assert sessions[0].closed and events[-1] == ("finish",)


def test_different_candidate_launch_needs_its_own_single_short_request(tmp_path, monkeypatch):
    def change(config):
        config["sides"]["candidate"]["models"]["muse"]["benchmarks"]["512"]["cwd"] = "/other-scratch"
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch, change=change)
    assert execution.run(args) == 0
    assert events.count(("smoke", "candidate-muse-preflight-512")) == 1
    preflight.retained(args.output, config, needs)


def test_every_selected_tier_is_preflighted_before_any_timing(tmp_path, monkeypatch):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch,
        paths=["engine/src/llama-kv-cache-dsv4.cpp"])
    assert execution.run(args) == 0
    timed = next(i for i, event in enumerate(events) if event[0] == "timed")
    assert len([e for e in events[:timed] if e[0] == "receipt"]) == 4
    assert ("smoke", "base-deepseek-preflight-4096") in events[:timed]
    assert len([e for e in events if e[0] == "timed"]) == 4
    assert set(preflight.retained(args.output, config, needs)) == {
        ("deepseek", tier, side) for tier in ("512", "4096") for side in ("base", "candidate")}


@pytest.mark.parametrize("missing", ["environment", "port", "timeout_s", "os_page_cache_treatment"])
def test_missing_declaration_refuses_before_session(tmp_path, monkeypatch, missing):
    def change(config):
        del config["sides"]["base"]["models"]["muse"]["benchmarks"]["512"][missing]
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch, change=change)
    assert execution.run(args) == 1 and not sessions and not events
    assert "explicit performance launch field" in load(args.output / "integration.json")["error"]


@pytest.mark.parametrize("stage", ["base-muse-preflight-512", "candidate-smoke-muse"])
def test_missing_exact_stage_authorization_refuses_before_session(tmp_path, monkeypatch, stage):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch)
    del manifest["stages"][stage]
    Path(config["manifest_path"]).unlink()
    save(config["manifest_path"], manifest)
    assert execution.run(args) == 1 and not sessions and not events
    assert "authorize" in load(args.output / "integration.json")["error"]


def test_timed_runtime_drift_stops_after_one_arm_without_retry(tmp_path, monkeypatch):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch, fault="timed-drift")
    assert execution.run(args) == 1
    assert [e for e in events if e[0] == "timed"] == [("timed", "deepseek", "base")]
    assert "differs from its successful short preflight" in load(args.output / "integration.json")["error"]


@pytest.mark.parametrize("damage", ["missing", "late-cell", "receipt", "cleanup", "reuse"])
def test_retained_preflight_cannot_be_missing_incomplete_changed_or_unclean(tmp_path, monkeypatch, damage):
    args, config, manifest, needs, events, sessions = setup(tmp_path, monkeypatch)
    assert execution.run(args) == 0
    path = args.output / "performance-preflight.json"
    gate = load(path)
    if damage == "missing": path.unlink()
    elif damage == "late-cell":
        gate["cells"].pop()
        path.unlink()
        save(path, gate)
    elif damage == "receipt":
        path = args.output / "preflight/muse/512/base/comparison-receipt.json"
        path.unlink()
        save(path, {})
    elif damage == "cleanup":
        path = args.output / "native-muse/stopped.json"
        path.unlink()
        save(path, dict(running=False, returncode=0, cleanup_confirmed=False))
    else:
        config["sides"]["candidate"]["models"]["muse"]["smoke"]["environment"] = {"CHANGED": "1"}
    with pytest.raises((ValueError, FileNotFoundError)):
        preflight.retained(args.output, config, needs)
