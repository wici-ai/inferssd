"""Bounded operator performance contracts; synthetic transport, no devices."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import selection_contract as contract, selected_integration as execution
from ci import selected_preflight, staged_benchmark
from ci.contract import load, save
from ci.impact import fingerprint
from ci.selected_receipt import retained_typed_settings
from test_ci_selection import plan, template, POLICY
from test_ci_selected_preflight import setup
from test_ci_staged_benchmark import CONFIG, Session


def performance(config):
    config.update(validation_mode=contract.PERFORMANCE_MODE,
                  execution_devices=["cloud-5090"],
                  performance_protocol=contract.performance_protocol(), noise_bands={})
    for side in config["sides"].values():
        for data in side["models"].values():
            data["benchmarks"] = {tier: spec for tier, spec in data["benchmarks"].items() if tier == "512"}
            for spec in data["benchmarks"].values():
                if "argv" in spec:
                    spec["preflight_argv"] = list(spec["argv"])


def bound_config(paths):
    value = plan(paths)
    config = template(value)
    performance(config)
    sides = ("base", "candidate") if any(cell["performance"]["required"] for cell in value["cells"]) else ("candidate",)
    manifest = dict(job_id="a" * 32, source_revisions={side: value[side] for side in sides})
    return contract.bind(config, manifest, value, POLICY)


def test_operator_preserves_impact_but_stages_only_the_explicit_512_checkpoint():
    config, manifest = bound_config(["engine/src/llama-kv-cache-dsv4.cpp"])
    original = copy.deepcopy(config["selection_plan"])
    needs = contract.bound(config, manifest, "a" * 32)
    assert all(cell["performance"]["tiers"] == ["512", "4096"] for cell in original["cells"] if cell["required"])
    assert needs["performance"] == {"deepseek": dict(required=True, tiers=["512"], output_tokens=256)}
    assert config["selection_plan"] == original
    assert manifest["source_revisions"] == {side: original[side] for side in ("base", "candidate")}
    assert all("4096" not in name for name in manifest["stages"])
    timed = {name for name in manifest["stages"] if "-benchmark-" in name}
    assert timed == {side + "-deepseek-r" + pair + "-benchmark-512" for side in ("base", "candidate") for pair in "12"}


@pytest.mark.parametrize("damage", ["missing-protocol", "wrong-prompt", "output-count", "bool-count", "extra-tier", "missing-base",
                                    "missing-bands", "invalid-bands"])
def test_new_mode_never_silently_defaults_or_accepts_one_sided_performance(damage):
    config, manifest = bound_config(["engine/src/models/muse.cpp"])
    if damage == "missing-protocol": config.pop("performance_protocol")
    elif damage == "wrong-prompt": config["performance_protocol"]["prompt_sha256"] = "0" * 64
    elif damage == "output-count": config["performance_protocol"]["output_tokens"] = 8
    elif damage == "bool-count": config["performance_protocol"]["samples_per_side"] = True
    elif damage == "extra-tier": config["sides"]["base"]["models"]["muse"]["benchmarks"]["4096"] = {}
    elif damage == "missing-bands": config.pop("noise_bands")
    elif damage == "invalid-bands": config["noise_bands"] = {"cloud-5090": {"muse": {"512": {}}}}
    else: manifest["source_revisions"].pop("base")
    with pytest.raises(ValueError):
        contract.bind(config, manifest, config["selection_plan"], POLICY)


def test_control_only_scope_keeps_performance_not_required_and_candidate_only():
    config, manifest = bound_config(["bench/ci/compare.py"])
    assert contract.runtime_requirements(config)["performance"] == {}
    assert set(manifest["source_revisions"]) == {"candidate"}
    assert not any("benchmark" in name for name in manifest["stages"])


def test_new_mode_never_runs_4k_and_runs_each_planned_arm_once_in_abba_order(tmp_path, monkeypatch):
    args, config, manifest, _, events, sessions = setup(tmp_path, monkeypatch,
        paths=["engine/src/llama-kv-cache-dsv4.cpp"], change=performance)
    assert execution.run(args) == 0
    assert [e for e in events if e[0] == "timed"] == [("timed", "deepseek", side)
                                                      for side in ("base", "candidate", "candidate", "base")]
    assert args.stages == ["base-deepseek-r1-benchmark-512", "candidate-deepseek-r1-benchmark-512",
                           "candidate-deepseek-r2-benchmark-512", "base-deepseek-r2-benchmark-512"]
    assert all((args.output / "deepseek/512" / pair / "comparison.json").is_file() for pair in ("r1", "r2"))
    cell = load(args.output / "deepseek/512/comparison.json")
    assert cell["status"] == "passed" and cell["order"] == ["r1/base", "r1/candidate", "r2/candidate", "r2/base"]
    assert not any("4096" in str(event) for event in events)
    gate = load(args.output / "performance-preflight.json")
    assert [(cell["model"], cell["tier"]) for cell in gate["cells"]] == [("deepseek", "512")]
    assert load(args.output / "integration.json")["performance_protocol"] == contract.performance_protocol()
    assert len(sessions) == 1 and sessions[0].closed


def test_needs_review_retains_every_arm_and_completes_the_other_cell(tmp_path, monkeypatch):
    args, config, manifest, _, events, sessions = setup(tmp_path, monkeypatch, change=performance)

    def diverged(path, *args):
        value = load(path)
        if "deepseek/512/r2/candidate" in str(path):
            value["tokens"][64] = 99
        return value

    monkeypatch.setattr(execution.run_final, "read_result", diverged)
    assert execution.run(args) == 1
    report = load(args.output / "integration.json")
    assert report["status"] == "needs-review" and report["cleanup_confirmed"]
    assert len(report["cells"]) == 2
    deepseek, muse = (cell["performance"][0] for cell in report["cells"])
    assert deepseek["status"] == "needs-review" and deepseek["review_reasons"] == ["generated-tokens-differ"]
    assert deepseek["output_workload"]["differences"] == [dict(
        arm="r2/candidate", first_difference_one_based=65, reference_token=11, token=99)]
    assert muse["status"] == "passed"
    assert [e for e in events if e[0] == "timed"] == [("timed", model, side)
        for model in ("deepseek", "muse") for side in ("base", "candidate", "candidate", "base")]
    for model in ("deepseek", "muse"):
        assert (args.output / model / "512/comparison.json").is_file()
        assert all((args.output / model / "512" / pair / side / "result.json").is_file()
                   for pair in ("r1", "r2") for side in ("base", "candidate"))
    assert sessions[0].closed


@pytest.mark.parametrize("fault", ["missing-stage", "corrupt-prompt"])
def test_all_timed_authorization_and_fixed_prompt_are_checked_before_lease(tmp_path, monkeypatch, fault):
    args, config, manifest, _, events, sessions = setup(tmp_path, monkeypatch, change=performance)
    if fault == "missing-stage":
        del manifest["stages"]["candidate-muse-r2-benchmark-512"]
        Path(config["manifest_path"]).unlink()
        save(config["manifest_path"], manifest)
    else:
        monkeypatch.setattr(selected_preflight.final_chat, "prompt_path", lambda tier: (_ for _ in ()).throw(ValueError("fixed prompt changed")))
    assert execution.run(args) == 1 and not sessions and not events
    assert "error" in load(args.output / "integration.json")


def test_benchmark_checks_prompt_before_native_start(tmp_path, monkeypatch):
    monkeypatch.setattr(staged_benchmark.final_chat, "prompt_path", lambda tier: (_ for _ in ()).throw(ValueError("fixed prompt changed")))
    session = Session()
    with pytest.raises(ValueError, match="fixed prompt changed"):
        staged_benchmark.benchmark(session, "base", CONFIG, tmp_path / "arm")
    assert session.started == [] and session.completions == 0


@pytest.mark.parametrize("damage", ["missing", "same-file", "other-argument", "other-side", "traversal", "duplicate"])
def test_explicit_preflight_launch_refuses_collisions_or_unrelated_differences(damage):
    spec = dict(argv=["/build/base/bin/llama-server", "--ssd-telemetry", "/scratch/base/timed.jsonl"],
                preflight_argv=["/build/base/bin/llama-server", "--ssd-telemetry", "/scratch/base/short.jsonl"])
    if damage == "missing": spec.pop("preflight_argv")
    elif damage == "same-file": spec["preflight_argv"] = list(spec["argv"])
    elif damage == "other-argument": spec["preflight_argv"].append("--no-mmap")
    elif damage == "other-side": spec["preflight_argv"][-1] = "/scratch/candidate/short.jsonl"
    elif damage == "traversal": spec["preflight_argv"][-1] = "/scratch/base/../short.jsonl"
    else: spec["preflight_argv"] += ["--ssd-telemetry", "/scratch/base/other.jsonl"]
    with pytest.raises(ValueError): selected_preflight.launch(spec, "base", required=True)


@pytest.mark.parametrize("damage", [None, "no-telemetry", "preflight-collision", "functional-collision",
                                    "outside-scratch", "pair-zero", "unpaired-preflight"])
def test_each_timed_arm_derives_its_own_private_event_file(damage):
    binary = "/build/base/bin/inferssd-server"
    spec = dict(argv=[binary, "--ssd-telemetry", "/scratch/base/deepseek-benchmark-512.jsonl"],
                preflight_argv=[binary, "--ssd-telemetry", "/scratch/base/deepseek-preflight.jsonl"])
    functional, pair = None, 1
    if damage == "no-telemetry": spec = dict(argv=[binary, "--fixture"], preflight_argv=[binary, "--fixture"])
    elif damage == "preflight-collision": spec["preflight_argv"][-1] = "/scratch/base/deepseek-benchmark-512-r1.jsonl"
    elif damage == "functional-collision": functional = dict(argv=[binary, "--ssd-telemetry", "/scratch/base/deepseek-benchmark-512-r1.jsonl"])
    elif damage == "outside-scratch": spec["argv"][-1] = "/scratch/base/../deepseek.jsonl"
    elif damage == "pair-zero": pair = 0
    elif damage == "unpaired-preflight": spec["preflight_argv"].append("--other")
    if damage in (None, "no-telemetry"):
        arms = [selected_preflight.arm_launch(spec, "base", k, required=True)["argv"] for k in (1, 2)]
        if damage is None:
            assert [argv[-1] for argv in arms] == ["/scratch/base/deepseek-benchmark-512-r1.jsonl",
                                                   "/scratch/base/deepseek-benchmark-512-r2.jsonl"]
            assert all(argv[:-1] == spec["argv"][:-1] for argv in arms)
        else:
            assert arms == [spec["argv"]] * 2
    else:
        with pytest.raises(ValueError):
            selected_preflight.arm_launch(spec, "base", pair, required=True, functional=functional)


def test_real_exclusive_event_files_survive_short_and_timed_servers_without_deletion(tmp_path, monkeypatch):
    def distinct(config):
        performance(config)
        for side, data in config["sides"].items():
            for model, native in data["models"].items():
                flag = ["--ssd-telemetry", "/scratch/" + side + "/" + model + "-preflight.jsonl"]
                native["smoke"]["argv"] += flag
                bench = native["benchmarks"]["512"]
                bench["preflight_argv"] = list(native["smoke"]["argv"])
                bench["argv"] += ["--ssd-telemetry", "/scratch/" + side + "/" + model + "-benchmark512.jsonl"]

    args, config, manifest, _, events, sessions = setup(tmp_path, monkeypatch, change=distinct)
    old_smoke, old_timed = execution.smoke, execution.benchmark
    paths = []

    def open_events(spec):
        argv = spec["argv"]
        name = argv[argv.index("--ssd-telemetry") + 1]
        path = tmp_path / "worker" / Path(name).relative_to("/scratch")
        path.parent.mkdir(parents=True, exist_ok=True)
        # This is the native writer's actual O_CREAT|O_EXCL lifecycle contract.
        with path.open("xb") as stream: stream.write(b"retained\n")
        paths.append(path)

    def smoke(session, spec, output, **kwargs):
        open_events(spec)
        return old_smoke(session, spec, output, **kwargs)

    def timed(session, side, spec, output):
        open_events(spec)
        return old_timed(session, side, spec, output)

    monkeypatch.setattr(execution, "smoke", smoke)
    monkeypatch.setattr(selected_preflight, "smoke", smoke)
    monkeypatch.setattr(execution, "benchmark", timed)
    assert execution.run(args) == 0
    assert len(paths) == len(set(paths)) == 12
    assert all(path.read_bytes() == b"retained\n" for path in paths)
    assert len([event for event in events if event[0] == "smoke"]) == 4
    assert len([event for event in events if event[0] == "timed"]) == 8
    assert {path.name for path in paths if "benchmark" in path.name} == {
        model + "-benchmark512-r" + pair + ".jsonl" for model in ("deepseek", "muse") for pair in "12"}
    with pytest.raises(FileExistsError):
        open_events(selected_preflight.arm_launch(config["sides"]["base"]["models"]["muse"]["benchmarks"]["512"],
                                                  "base", 2, required=True))
    assert all(path.is_file() for path in paths)
    assert sessions[0].closed


@pytest.mark.parametrize("side_name", ["base", "candidate"])
@pytest.mark.parametrize("damage", [None, "other-side", "other-job", "non-source", "outside-assets", "inventory"])
def test_retained_alias_mapping_is_side_and_current_job_exact(tmp_path, side_name, damage):
    job = "a" * 32
    host = "/var/lib/inferssd-ci-broker/jobs/" + job
    path = tmp_path / "freeze.json"
    rows = [dict(path="assets/model.tsv", size=17, sha256="b" * 64)]
    save(path, dict(files=rows))
    side = dict(paths=dict(frozen="/build/" + side_name), models=dict(muse=dict(model_identity=dict(shards=[]))))
    build = dict(freeze=dict(side=side_name, path="/build/" + side_name,
        host_path=host + "/frozen/" + side_name, manifest_sha256=fingerprint(rows), files=1))
    native = {"expert.source": ["/build/" + side_name + "/assets/model.tsv"], "placement.vram_experts": 32}
    expected = {"expert.source": ["<sealed-build>/assets/model.tsv"], "placement.vram_experts": 32}
    if damage == "other-side": native["expert.source"] = ["/build/" + ("candidate" if side_name == "base" else "base") + "/assets/model.tsv"]
    elif damage == "other-job": build["freeze"]["host_path"] = host.replace(job, "c" * 32) + "/frozen/" + side_name
    elif damage == "non-source": native["placement.vram_experts"] = 0
    elif damage == "outside-assets": native["expert.source"] = ["/tmp/assets/model.tsv"]
    elif damage == "inventory": build["freeze"]["manifest_sha256"] = "0" * 64
    if damage in ("other-job", "inventory"):
        with pytest.raises(ValueError): retained_typed_settings(native, expected, side, "muse", build, path, host + "/manifest.json")
    else:
        assert retained_typed_settings(native, expected, side, "muse", build, path, host + "/manifest.json") is (damage is None)
