"""CI contracts use synthetic raw records; these tests are not real device results."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from test_final import write_artifacts
import test_native_wrapper

profile = test_native_wrapper.profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_common as native
import run_final
from ci import compare, github_gate
from ci.contract import (
    DEVICES,
    MODELS,
    authorize,
    classify,
    digest,
    load,
    policy_read,
    runner_config,
)


@pytest.fixture
def policy():
    path = Path(__file__).resolve().parents[1] / "ci/policy.json"
    value = policy_read(load(path))
    value["_sha256"] = digest(path)
    return value


def pr():
    repo = {"full_name": "wici-ai/inferssd"}
    return {
        "number": 12,
        "state": "open",
        "draft": False,
        "author_association": "COLLABORATOR",
        "head": {"repo": repo, "sha": "b" * 40},
        "base": {"repo": repo, "ref": "main", "sha": "a" * 40},
        "merge_commit_sha": "c" * 40,
    }


@pytest.mark.parametrize(
    "path,tiers,code",
    [
        ("docs/example.md", ["512"], False),
        ("engine/tests/test-quantize.cpp", ["512"], True),
        ("engine/ggml/src/ggml-cpu/ops.cpp", ["512", "4096"], True),
        ("framework/src/inferssd/platform/worker.py", ["512", "4096"], True),
        ("bench/ci/driver.py", ["512", "4096"], True),
        (".github/workflows/ci.yml", ["512"], True),
    ],
)
def test_every_code_pr_and_resource_changes_are_classified(policy, path, tiers, code):
    result = classify([path], policy)
    assert result["tiers"] == tiers and result["code"] is code


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["head"].update(repo={"full_name": "fork/inferssd"}),
        lambda p: p.update(author_association="CONTRIBUTOR"),
        lambda p: p.update(draft=True),
        lambda p: p.update(state="closed"),
        lambda p: p.update(merge_commit_sha=None),
        lambda p: p["base"].update(ref="other"),
    ],
)
def test_unsafe_or_stale_pr_cannot_reach_a_runner(policy, change):
    value = copy.deepcopy(pr())
    change(value)
    with pytest.raises(ValueError):
        authorize(value, "wici-ai/inferssd", policy)
    with pytest.raises(ValueError, match="stale"):
        authorize(pr(), "wici-ai/inferssd", policy, observed_sha="d" * 40)


def test_activation_off_never_reads_event_or_calls_api(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REPOSITORY", "wici-ai/inferssd")
    monkeypatch.setenv("INFERSSD_CI_ENABLED", "false")
    monkeypatch.setattr(
        github_gate, "GitHub", lambda _: pytest.fail("no API while unprovisioned")
    )
    github_gate.preflight(
        SimpleNamespace(policy=Path(__file__).parents[1] / "ci/policy.json")
    )
    assert (
        "enabled=false" in output.read_text() and "not activated" in output.read_text()
    )


def test_typed_native_profile_binds_settings_and_actual_source(profile, tmp_path):
    typed = tmp_path / "settings.ssd"
    typed.write_text(
        "inferssd-settings 1\nexpert.source="
        + json.dumps([profile["alias"]["path"]])
        + "\n"
    )
    profile["typed_settings"] = {"path": str(typed), "sha256": digest(typed)}
    profile["environment"] = {}
    native.validate_profile(profile)
    native.verify_inputs(profile)
    argv = native.server_argv(profile, 8888, tmp_path / "native-events.jsonl")
    assert argv[argv.index("--ssd-settings") + 1] == str(typed)
    assert "typed_values" in native.effective_settings(profile)["native_launch_policy"]
    profile["environment"] = {"GGML_KIMI_CPU_SINK_THREADS": "8"}
    with pytest.raises(ValueError, match="retired"):
        native.validate_profile(profile)
    profile["environment"] = {}
    typed.write_text("inferssd-settings 1\nexpert.source=[]\n")
    with pytest.raises(ValueError, match="source"):
        native.validate_profile(profile)


def test_formal_defaults_are_not_replaced_by_ci_subset(tmp_path):
    with pytest.raises(ValueError, match="formal"):
        run_final.run(
            {}, tmp_path / "out", tmp_path / "lock", tmp_path / "STOP", order=("512",)
        )
    assert not list(tmp_path.iterdir())


def fixture_config(device, tmp_path):
    return {
        "schema_version": 1,
        "activation_ready": True,
        "device": device,
        "platform": "darwin" if device == "local-mbp" else "linux",
        "lock_path": str(tmp_path / "lock"),
        "stop_path": str(tmp_path / "STOP"),
        "workspace_root": str(tmp_path / "work"),
        "min_free_bytes": 1,
        "max_workspace_bytes": 100000000,
        "stage_policy": {
            "platform": "darwin" if device == "local-mbp" else "linux",
            "lock_path": str(tmp_path / "lock"),
            "stop_path": str(tmp_path / "STOP"),
            "memory_budget_bytes": 1024,
            "host_reserve_bytes": 0,
            "minimum_memlock_bytes": 0,
            "max_compressor_growth_bytes": 0,
            "max_swap_growth_bytes": 0,
            "runtime_seconds": 5,
            "cleanup_seconds": 1,
        },
        "build": {
            "argv": ["fixture-only"],
            "timeout_s": 5,
            "template_relatives": {m: "template.jinja" for m in (*MODELS, "tiny")},
        },
        "validation": {
            "argv": ["fixture-only"],
            "timeout_s": 5,
            "baseline": {
                "tests": {
                    "num": {"expect": "passed", "kind": "numerical"},
                    "life": {"expect": "passed", "kind": "lifecycle"},
                }
            },
        },
        "profiles": {
            m: {"path": "/synthetic/" + m, "sha256": "f" * 64}
            for m in (*MODELS, "tiny")
        },
        "request_timeout_s": 1,
        "arm_timeout_s": 2,
    }



def measurement_manifest(parent, config, rows):
    # Mirror the one-run manifest/ledger emitted by run_final; no device executes.
    run_id = str(parent)
    (parent / "manifest.json").write_text(json.dumps({
        "purpose": "ci", "config": config, "order": ["512"],
        "samples_per_tier": 1, "run_id": run_id,
    }))
    (parent / "summary.json").write_text(json.dumps({
        "purpose": "ci", "complete": True, "samples_per_tier": 1,
        "order": ["512"], "run_id": run_id, "interrupted_signals": [], "rows": rows,
    }))
    (parent / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

def synthetic_device(tmp_path, device, plan, policy):
    root = tmp_path / device
    root.mkdir()
    config = fixture_config(device, tmp_path)
    (root / "runner-config.json").write_text(json.dumps(config))
    plan["device_config_sha256"][device] = digest(root / "runner-config.json")
    (root / "ci-manifest.json").write_text(
        json.dumps(
            {
                "purpose": "ci",
                "device": device,
                "plan": plan,
                "policy_sha256": policy["_sha256"],
            }
        )
    )
    for side in ("base", "candidate"):
        (root / ("build-" + side + ".json")).write_text(
            json.dumps(
                {"status": "passed", "revision": plan[side], "toolchain": "synthetic"}
            )
        )
        xml = '<testsuite><testcase name="num"/><testcase name="life"/></testsuite>'
        (root / ("validation-" + side + ".xml")).write_text(xml)
        report = compare.GATE.compare(
            {"num": "passed", "life": "passed"}, config["validation"]["baseline"]
        )
        report.update(revision=plan[side], cleanup_confirmed=True, purpose="ci")
        (root / ("validation-" + side + ".json")).write_text(json.dumps(report))
        for stage in ("build", "validation"):
            folder = root / ("stage-" + stage + "-" + side)
            folder.mkdir()
            (folder / "ownership.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "server": {
                            "owned_processes_gone": True,
                            "resource_scope_gone": True,
                        },
                    }
                )
            )
    for model in ("tiny", *MODELS):
        for side in ("base", "candidate"):
            parent = root / model / side
            parent.mkdir(parents=True)
            config_run = {
                "revision": plan[side],
                "device": device,
                "model": model,
                "request_timeout_s": 1,
            }
            write_artifacts(parent / "512", "512", config_run)
            receipt = load(parent / "512/wrapper.json")
            receipt["server"]["start_identity"] = side
            receipt["host"]["effective_settings"]["native_launch_policy"] = {
                "typed_values": {"cpu.threads": 2},
                "args": [],
            }
            (parent / "512/wrapper.json").write_text(json.dumps(receipt))
            result = load(parent / "512/result.json")
            result["rendered_prompt_sha256"] = "e" * 64
            (parent / "512/result.json").write_text(json.dumps(result))
            measurement_manifest(parent, config_run, [{
                "tier": "512", "status": "passed", "result": result,
                "wrapper_receipt_sha256": digest(parent / "512/wrapper.json"),
                "owned_group_cleanup_confirmed": True, "wrapper_cleanup_confirmed": True,
            }])
    a = load(root / "tiny/base/512/result.json")
    b = load(root / "tiny/candidate/512/result.json")
    (root / "tiny-numerical.json").write_text(
        json.dumps(compare.compare_outputs.compare(a, b))
    )
    return root


@pytest.fixture
def synthetic(tmp_path, policy):
    plan = dict(
        authorize(pr(), "wici-ai/inferssd", policy),
        tiers=["512"],
        device_config_sha256={},
        policy_sha256=policy["_sha256"],
    )
    roots = [synthetic_device(tmp_path, d, plan, policy) for d in DEVICES]
    # All manifests contain the complete plan, not an order-dependent prefix.
    for root in roots:
        manifest = load(root / "ci-manifest.json")
        manifest["plan"] = plan
        (root / "ci-manifest.json").write_text(json.dumps(manifest))
    return roots, plan, policy


def test_three_device_aggregation_revalidates_raw_data_without_execution(synthetic):
    roots, plan, policy = synthetic
    result = compare.aggregate(roots, plan, policy)
    assert result["passed"] and result["formal_samples"] == 0
    assert sum(len(x["rows"]) for x in result["devices"]) == 6
    with pytest.raises(ValueError, match="every real device"):
        compare.aggregate(roots[:2], plan, policy)


@pytest.mark.parametrize(
    "tamper", ["raw", "summary", "config", "validation", "tiny", "cleanup", "duplicate"]
)
def test_anomaly_never_becomes_an_automatic_accept(synthetic, tamper):
    roots, plan, policy = synthetic
    root = roots[0]
    if tamper == "raw":
        path = root / "muse/candidate/512/result.events.sse"
        path.write_bytes(path.read_bytes()[:-40])
    elif tamper == "summary":
        path = root / "deepseek/candidate/summary.json"
        value = load(path)
        value["rows"][0]["result"]["metrics"]["decode_tps"] *= 2
        path.write_text(json.dumps(value))
    elif tamper == "config":
        with (root / "runner-config.json").open("a") as stream:
            stream.write(" ")
    elif tamper == "validation":
        (root / "validation-base.xml").write_text(
            '<testsuite><testcase name="num"/></testsuite>'
        )
    elif tamper == "tiny":
        path = root / "tiny-numerical.json"
        value = load(path)
        value["exact"] = False
        path.write_text(json.dumps(value))
    elif tamper == "cleanup":
        path = root / "stage-validation-base/ownership.json"
        value = load(path)
        value["server"]["owned_processes_gone"] = False
        path.write_text(json.dumps(value))
    else:
        path = root / "muse/candidate/summary.json"
        value = load(path)
        value["rows"].append(value["rows"][0])
        path.write_text(json.dumps(value))
    try:
        result = compare.aggregate(roots, plan, policy)
    except ValueError:
        return
    assert not result["passed"]


def test_observed_slowdown_blocks_both_windows_and_separate_ttft(synthetic):
    roots, plan, policy = synthetic
    a, ar = compare.cell(roots[0], "muse", "base", "512", plan["base"], DEVICES[0])
    b, br = compare.cell(
        roots[0], "muse", "candidate", "512", plan["candidate"], DEVICES[0]
    )
    b = copy.deepcopy(b)
    b["metrics"]["decode_first128_tps"] *= 0.94
    b["metrics"]["ttft_s"] *= 1.11
    result = compare.compare_pair(a, b, ar, br, policy)
    assert result["regressions"] == ["decode_first128_tps", "ttft_s"]
    assert result["status"] == "inconclusive" and not result["passed"]
    br["host"]["memory_budget_bytes"] += 1
    with pytest.raises(ValueError, match="resource"):
        compare.compare_pair(a, b, ar, br, policy)


def test_unprovisioned_example_cannot_be_used_as_a_real_device():
    with pytest.raises(ValueError, match="activated"):
        runner_config(
            load(Path(__file__).parents[1] / "ci/runner-config.example.json"),
            DEVICES[0],
        )


def test_unknown_code_extension_cannot_bypass_device_gates(policy):
    assert classify(["new_backend/kernel.newlanguage"], policy)["code"]
    assert classify(["Makefile"], policy)["code"]


def test_preflight_pins_all_three_slots_without_accepting_an_incomplete_file_list(
    tmp_path, monkeypatch
):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"inputs": {"pull_request": "12"}}))
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_REPOSITORY": "wici-ai/inferssd",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_OUTPUT": str(output),
        "INFERSSD_CI_ENABLED": "true",
        "INFERSSD_CI_DEVICE_CONFIG_HASHES": json.dumps({d: "a" * 64 for d in DEVICES}),
    }.items():
        monkeypatch.setenv(key, value)
    current = pr()
    current["changed_files"] = 1
    calls = []

    class API:
        def __init__(self, *_):
            pass

        def call(self, path, data=None, **kw):
            calls.append((path, data))
            if path == "pulls/12":
                return current
            if path.startswith("pulls/12/files"):
                return [{"filename": "engine/src/llama-context.cpp"}]
            if path == "check-runs":
                return {"id": 33}
            raise AssertionError(path)

    monkeypatch.setattr(github_gate, "GitHub", API)
    monkeypatch.setattr(
        github_gate.subprocess, "check_output", lambda *a, **kw: "d" * 40
    )
    args = SimpleNamespace(
        policy=Path(__file__).parents[1] / "ci/policy.json", plan=tmp_path / "plan"
    )
    github_gate.preflight(args)
    plan = load(args.plan)
    assert plan["tiers"] == ["512", "4096"] and plan["control_revision"] == "d" * 40
    assert plan["check_id"] == 33 and "enabled=true" in output.read_text()
    assert (
        len(
            json.loads(
                next(
                    x.split("=", 1)[1]
                    for x in output.read_text().splitlines()
                    if x.startswith("matrix=")
                )
            )["include"]
        )
        == 3
    )
    current["changed_files"] = 2
    args.plan = tmp_path / "never"
    with pytest.raises(ValueError, match="incomplete"):
        github_gate.preflight(args)
    assert not args.plan.exists()


def test_finalizer_missing_device_publishes_failure_without_executing_artifacts(
    tmp_path, monkeypatch, policy
):
    plan = dict(
        authorize(pr(), "wici-ai/inferssd", policy),
        tiers=["512"],
        check_id=4,
        policy_sha256=policy["_sha256"],
    )
    path = tmp_path / "plan"
    path.write_text(json.dumps(plan))
    calls = []

    class API:
        def __init__(self, *_):
            pass

        def call(self, target, data=None, **kw):
            calls.append((target, data))
            if target == "pulls/12":
                return pr()
            return {}

    monkeypatch.setenv("GITHUB_REPOSITORY", "wici-ai/inferssd")
    monkeypatch.setattr(github_gate, "GitHub", API)
    assert (
        github_gate.finalize(
            SimpleNamespace(
                policy=Path(__file__).parents[1] / "ci/policy.json",
                plan=path,
                artifacts=tmp_path / "absent",
                output=tmp_path / "comparison",
            )
        )
        == 1
    )
    assert calls[-1][1]["conclusion"] == "failure"
    assert not load(tmp_path / "comparison")["passed"]


@pytest.mark.parametrize("fail_measurement", [False, True])
def test_driver_builds_and_validates_both_sources_before_single_attempt_cells(
    tmp_path, monkeypatch, profile, policy, fail_measurement
):
    from ci import driver

    config = fixture_config(DEVICES[0], tmp_path)
    config["build"]["argv"] = ["build-fixture", "{receipt}", "{revision}", "{build}"]
    config["validation"]["argv"] = ["validate-fixture", "{results}", "{revision}"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    Path(config["lock_path"]).touch()
    typed = tmp_path / "typed.ssd"
    typed.write_text(
        "inferssd-settings 1\nexpert.source="
        + json.dumps([profile["alias"]["path"]])
        + "\n"
    )
    profile = copy.deepcopy(profile)
    profile.update(
        environment={}, typed_settings={"path": str(typed), "sha256": digest(typed)}
    )
    plan = dict(
        authorize(pr(), "wici-ai/inferssd", policy),
        tiers=["512"],
        device_config_sha256={DEVICES[0]: digest(config_path)},
        policy_sha256=policy["_sha256"],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    repository = tmp_path / "repository"
    repository.mkdir()
    log = []

    def git(_repository, *args):
        if args[0] == "rev-parse":
            return plan["candidate"]
        if args[0] == "show":
            return plan["base"] + " " + plan["head"]
        if args[0] == "diff":
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(driver, "git", git)

    def worktree(argv, **kw):
        assert argv[3:6] == ["worktree", "add", "--detach"]
        Path(argv[6]).mkdir()

    monkeypatch.setattr(driver.subprocess, "run", worktree)

    def stage(argv, directory, cfg, lock_fd, timeout, interrupted):
        directory.mkdir()
        ownership = {
            "status": "passed",
            "server": {"owned_processes_gone": True, "resource_scope_gone": True},
        }
        (directory / "ownership.json").write_text(json.dumps(ownership))
        log.append(argv[0])
        if argv[0] == "build-fixture":
            build = Path(argv[3])
            build.mkdir()
            current = copy.deepcopy(profile)
            current["revision"] = argv[2]
            source = build.with_name(build.name.replace("-build", "-source"))
            template = source / "template.jinja"
            template.write_bytes(Path(current["template"]["path"]).read_bytes())
            current["template"] = {"path": str(template), "sha256": digest(template)}
            for name in ("binary", "images"):
                specs = [current[name]] if name == "binary" else current[name]
                for index, spec in enumerate(specs):
                    target = build / (name + str(index))
                    target.write_bytes(Path(spec["path"]).read_bytes())
                    spec.update(path=str(target), sha256=digest(target))
            Path(argv[1]).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "revision": argv[2],
                        "profiles": {m: current for m in (*MODELS, "tiny")},
                        "toolchain": "synthetic-only",
                    }
                )
            )
        else:
            Path(argv[1]).write_text(
                '<testsuite><testcase name="num"/><testcase name="life"/></testsuite>'
            )
        return ownership

    monkeypatch.setattr(driver, "command_stage", stage)

    def measurement(
        cfg,
        target,
        lock_path,
        stop_path,
        *,
        order,
        purpose,
        inherited_lock_fd,
        environment,
    ):
        log.append("measure:" + cfg["model"])
        assert purpose == "ci" and order == ("512",) and inherited_lock_fd >= 3
        assert "GITHUB_TOKEN" not in environment and "PYTHONPATH" not in environment
        if fail_measurement:
            return 1
        target.mkdir(parents=True)
        write_artifacts(target / "512", "512", cfg)
        receipt = load(target / "512/wrapper.json")
        receipt["server"]["start_identity"] = cfg["revision"]
        receipt["host"]["effective_settings"]["native_launch_policy"] = {
            "typed_values": {"cpu.threads": 2}
        }
        (target / "512/wrapper.json").write_text(json.dumps(receipt))
        result = load(target / "512/result.json")
        result["rendered_prompt_sha256"] = "e" * 64
        (target / "512/result.json").write_text(json.dumps(result))
        measurement_manifest(target, cfg, [{
            "tier": "512", "status": "passed", "result": result,
            "wrapper_receipt_sha256": digest(target / "512/wrapper.json"),
            "owned_group_cleanup_confirmed": True, "wrapper_cleanup_confirmed": True,
        }])
        return 0

    monkeypatch.setattr(driver.run_final, "run", measurement)
    args = SimpleNamespace(
        policy=Path(__file__).parents[1] / "ci/policy.json",
        plan=plan_path,
        config=config_path,
        device=DEVICES[0],
        repository=repository,
        output=tmp_path / "output",
    )
    if fail_measurement:
        with pytest.raises(RuntimeError, match="no automatic retry"):
            driver.run(args)
        assert log == [
            "build-fixture",
            "build-fixture",
            "validate-fixture",
            "validate-fixture",
            "measure:tiny",
        ]
    else:
        assert driver.run(args) == 0
        assert log == [
            "build-fixture",
            "build-fixture",
            "validate-fixture",
            "validate-fixture",
            "measure:tiny",
            "measure:tiny",
            "measure:deepseek",
            "measure:deepseek",
            "measure:muse",
            "measure:muse",
        ]
    assert load(args.output / "ci-attempt.json")["passed"] is not fail_measurement


@pytest.mark.parametrize("unused_flag", [False, True])
def test_build_adapter_binds_actual_outputs_and_refuses_unused_controls(
    tmp_path, monkeypatch, profile, unused_flag
):
    from ci import build as builder

    source, output = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    (source / "template.jinja").write_bytes(
        Path(profile["template"]["path"]).read_bytes()
    )
    typed = tmp_path / "typed.ssd"
    typed.write_text(
        "inferssd-settings 1\nexpert.source="
        + json.dumps([profile["alias"]["path"]])
        + "\n"
    )
    profile.update(
        environment={}, typed_settings={"path": str(typed), "sha256": digest(typed)}
    )
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text(json.dumps(profile))
    cmake, compiler = tmp_path / "cmake", tmp_path / "compiler"
    cmake.write_bytes(b"synthetic metadata only")
    compiler.write_bytes(b"synthetic metadata only")
    config = {
        "build": {
            "cmake": str(cmake),
            "cmake_flags": ["-DREVIEWED=ON"],
            "targets": ["llama-server", "fixtures"],
            "jobs": 2,
            "binary_relative": "server",
            "image_relatives": ["lib.so"],
            "template_relatives": {m: "template.jinja" for m in (*MODELS, "tiny")},
        },
        "profiles": {
            m: {"path": str(blueprint), "sha256": digest(blueprint)}
            for m in (*MODELS, "tiny")
        },
    }
    config_path = tmp_path / "config"
    config_path.write_text(json.dumps(config))
    calls = []

    def cmake_call(argv, **kw):
        calls.append(argv)
        if "-S" in argv:
            return SimpleNamespace(
                stdout="Manually-specified variables were not used by the project"
                if unused_flag
                else "configured"
            )
        output.mkdir()
        (output / "server").write_bytes(b"synthetic binary metadata")
        (output / "lib.so").write_bytes(b"synthetic library metadata")
        (output / "CMakeCache.txt").write_text(
            "CMAKE_CXX_COMPILER:FILEPATH=" + str(compiler) + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", cmake_call)
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    args = SimpleNamespace(
        config=config_path,
        source=source,
        build=output,
        revision="a" * 40,
        receipt=tmp_path / "receipt",
    )
    if unused_flag:
        with pytest.raises(ValueError, match="unused/retired"):
            builder.build(args)
        assert len(calls) == 1 and not args.receipt.exists()
    else:
        builder.build(args)
        receipt = load(args.receipt)
        assert len(calls) == 2 and receipt["actual_server_version"] == "a" * 40
        assert receipt["profiles"]["muse"]["template"]["path"] == str(
            source / "template.jinja"
        )
        assert receipt["profiles"]["tiny"]["binary"]["sha256"] == digest(
            output / "server"
        )
        assert receipt["toolchain"]["CMAKE_CXX_COMPILER:FILEPATH"]["sha256"] == digest(
            compiler
        )
