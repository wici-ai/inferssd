"""Mac selected CI contracts over synthetic root receipts; no devices or models."""

import copy
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import impact, selection_contract as contract, selected_aggregate as aggregate
from ci import selected_controller as controller, selected_preflight as preflight
from ci import selected_receipt as receipt
from ci.contract import digest, save
from test_ci_selection import POLICY, plan, template

JOB = "a" * 32
JOB_ROOT = "/Library/InferSSD-CI/jobs/" + JOB
MANIFEST = "/Library/InferSSD-CI/private/job-manifests/" + JOB + ".json"
RESOURCES = dict(max_processes=128, memory_budget_bytes=40 << 30, host_reserve_bytes=3 << 30)


def mac_config():
    value = plan(["bench/ci/macos-sandbox/executor.py"])
    config = template(value, "local-mbp")
    config.update(manifest_path=MANIFEST, validation_mode=contract.PERFORMANCE_MODE,
        execution_devices=["local-mbp"], performance_protocol=contract.performance_protocol(),
        host=dict(resources=RESOURCES))
    config["numerical"]["backend"] = "MTL0"
    for side, data in config["sides"].items():
        data["paths"] = dict(platform="darwin", job_root=JOB_ROOT,
            source="/Library/InferSSD-CI/inputs/operator-" + JOB + "/" + side,
            build=JOB_ROOT + "/work/" + side, cwd=JOB_ROOT + "/work/" + side,
            frozen=JOB_ROOT + "/frozen/" + side)
    manifest = dict(job_id=JOB, source_revisions={s:value[s] for s in ("base", "candidate")},
                    selection_lock_identity=[16777231, 81219804])
    return config, manifest, value


def test_mac_executor_paths_select_only_mac_required_512_without_changing_linux_rules():
    for path in ("bench/ci/macos-sandbox/executor.py", "bench/ci/macos-sandbox/recover.py", "bench/ci/operator_mac_stage.py"):
        p = plan([path]); selected = impact.selected(p)
        assert len(selected) == 2 and {row["device"] for row in selected} == {"local-mbp"}
        assert all(row["performance"]["tiers"] == ["512"] for row in selected)
        assert not any(case.startswith("cuda-") for row in selected for case in row["numerical"])
    shared = plan(["bench/ci/selected_controller.py"])
    assert len(impact.selected(shared)) == 6 and not any(row["performance"]["required"] for row in shared["cells"])
    cuda = plan(["engine/ggml/src/ggml-cuda/ssd/mxfp4-compute-adapter.cpp"])
    assert {row["device"] for row in impact.selected(cuda)} == {"cloud-5090", "dgx-spark"}


def test_mac_only_mode_binds_original_lease_resource_and_both_source_sides():
    config, manifest, value = mac_config()
    config, manifest = contract.bind(config, manifest, value, POLICY)
    assert contract.bound(config, manifest, JOB)["performance"] == {m:dict(required=True, tiers=["512"], output_tokens=256) for m in ("deepseek", "muse")}
    assert manifest["selection_resource_policy"] == RESOURCES
    assert manifest["selection_lock_identity"] == [16777231, 81219804]
    assert config["selection_plan"] == value


@pytest.mark.parametrize("fault", ["lease", "resources", "other-job", "wrong-side", "linux-backend"])
def test_mac_scope_cannot_use_another_job_platform_or_unbound_resource(fault):
    config, manifest, value = mac_config()
    if fault == "lease": manifest.pop("selection_lock_identity")
    elif fault == "resources": config["host"]["resources"] = dict(memory_max=40 << 30)
    elif fault == "other-job": config["manifest_path"] = MANIFEST.replace(JOB, "b" * 32)
    elif fault == "wrong-side": config["sides"]["base"]["paths"]["frozen"] = JOB_ROOT + "/frozen/candidate"
    else: config["numerical"]["backend"] = "CUDA0"
    with pytest.raises(ValueError): contract.bind(config, manifest, value, POLICY)


def test_mac_launch_uses_current_job_distinct_private_telemetry():
    spec = dict(argv=[JOB_ROOT + "/frozen/base/bin/llama-server", "--ssd-telemetry", JOB_ROOT + "/work/base/timed.jsonl"],
                preflight_argv=[JOB_ROOT + "/frozen/base/bin/llama-server", "--ssd-telemetry", JOB_ROOT + "/work/base/short.jsonl"])
    assert preflight.launch(spec, "base", required=True, job_root=JOB_ROOT)["argv"] == spec["preflight_argv"]
    for path in (JOB_ROOT + "/work/base/timed.jsonl", JOB_ROOT.replace(JOB, "b" * 32) + "/work/base/short.jsonl", JOB_ROOT + "/work/candidate/short.jsonl", JOB_ROOT + "/work/base/../short.jsonl"):
        altered = copy.deepcopy(spec); altered["preflight_argv"][-1] = path
        with pytest.raises(ValueError): preflight.launch(altered, "base", required=True, job_root=JOB_ROOT)


@pytest.fixture
def mac_raw(tmp_path):
    side = "candidate"; frozen = JOB_ROOT + "/frozen/" + side; binary = frozen + "/bin/llama-server"
    folder = tmp_path / "native-deepseek"; folder.mkdir(); build_dir = tmp_path / "build-candidate"; build_dir.mkdir()
    launch = dict(argv=[binary, "--fixture"], environment=dict(LANG="C"), cwd=JOB_ROOT + "/work/candidate")
    data = dict(smoke=launch, model_identity=dict(shards=[dict(runtime_path="/Library/InferSSD-CI/models/frozen/model.gguf")]),
                settings=dict(runtime_path="/Library/InferSSD-CI/inputs/operator-" + JOB + "/settings.ssd"), template=dict(runtime_path="/Library/InferSSD-CI/public/template.jinja", sha256="c" * 64))
    cfg = dict(revision="c" * 40, paths=dict(platform="darwin"), manifest_path=MANIFEST, manifest_sha256="f" * 64,
        host=dict(hardware=dict(cpu="M5 Max"), os=dict(sysname="Darwin", release="fixture", machine="arm64"), resources=RESOURCES),
        sides=dict(candidate=dict(revision="c" * 40, paths=dict(frozen=frozen, job_root=JOB_ROOT), models=dict(deepseek=data))))
    manifest = dict(selection_resource_policy=RESOURCES, selection_lock_identity=[16777231, 81219804])
    save(tmp_path / "root-manifest.json", manifest)
    inventory = {"assets/source.tsv":dict(size=32, mode=0o444, sha256="a" * 64)}
    save(build_dir / "freeze-manifest.json", inventory)
    build = dict(binary=dict(runtime_path=binary, sha256="b" * 64), images=[dict(runtime_path=frozen + "/lib/libggml.dylib", sha256="d" * 64)],
        freeze=dict(path=frozen, manifest_path=JOB_ROOT + "/frozen/candidate.manifest.json", readonly=True,
                    manifest_sha256=digest(build_dir / "freeze-manifest.json"), file_count=1))
    save(build_dir / "build-receipt.json", build)
    started = dict(worker_pid=123, worker_uid=62002, birth_identity=dict(pid=123, uid=62002, start_sec=1700000000, start_usec=1), scope_identity=JOB_ROOT)
    common = dict(**started, job_id=JOB, job_root=JOB_ROOT, manifest_sha256=cfg["manifest_sha256"], lock_dev_inode=manifest["selection_lock_identity"])
    observed = dict(common, running=True, ready=True, argv=launch["argv"], resource_policy=RESOURCES,
        members=[dict(pid=123, rss_bytes=1024)], rss_bytes=1024, listener_pids=[123],
        lsof="p123\nu62002\nftxt\nn" + binary + "\nfmem\nn" + frozen + "/lib/libggml.dylib\nfmem\nn/usr/lib/libSystem.B.dylib\nftxt\nn/Library/InferSSD-CI/models/frozen/model.gguf\n")
    stopped = dict(common, running=False, state="failed", reason="requested stop", returncode=0, cleanup_confirmed=True,
                   cleanup=dict(confirmed=True, remaining=[], worker_user_domain_retired=dict(returncode=0)))
    native = {"expert.source":[frozen + "/assets/source.tsv"], "cpu.threads":4}
    effective = dict(ambient_environment=launch["environment"], native_resource_policy=RESOURCES,
        native_resource_enforcement="root-sampled-worker-rss-processes-host-reserve",
        native_launch_policy=dict(template_sha256="c" * 64, typed_values=dict(native, **{"expert.source":["<sealed-build>/assets/source.tsv"]})),
        system_images=[dict(path="/usr/lib/libSystem.B.dylib", sha256="e" * 64)])
    runtime = dict(model_identity=data["model_identity"], build=dict(revision="c" * 40,
        loaded_libraries=[dict(path=binary, sha256="b" * 64), dict(path=frozen + "/lib/libggml.dylib", sha256="d" * 64)]),
        server=dict(pid=123, start_identity=started["birth_identity"]), host=dict(hardware=cfg["host"]["hardware"], os=cfg["host"]["os"], effective_settings=effective))
    save(folder / "started.json", started); save(folder / "stopped.json", stopped)
    save(folder / "runtime-before.json", observed); save(folder / "runtime-after.json", observed)
    (folder / "stderr.log").write_text('[ssd.settings] ' + json.dumps(dict(values=native)) + '\n')
    save(folder / "runtime-receipt.json", runtime)
    return tmp_path, cfg, manifest, folder, launch


def test_mac_runtime_retained_rechecks_real_shaped_probe_lsof_sampled_cap_and_stop(mac_raw):
    root, config, _, folder, launch = mac_raw
    value = aggregate.runtime_retained(root, config, "candidate", "deepseek", folder, launch, folder / "runtime-receipt.json")
    assert value["host"]["effective_settings"]["native_resource_enforcement"].startswith("root-sampled")


@pytest.mark.parametrize("fault", ["pid", "uid", "birth", "scope", "argv", "policy", "rss", "rss-sum", "members", "listener", "lsof", "lsof-owner", "lock", "manifest", "loaded-sha", "enforcement", "stop-residual", "stop-domain", "stop-lock", "source-job", "source-inventory"])
def test_mac_raw_identity_resource_image_or_cleanup_failure_is_not_green(mac_raw, fault):
    root, config, manifest, folder, launch = mac_raw
    probe = json.loads((folder / "runtime-after.json").read_text())
    stopped = json.loads((folder / "stopped.json").read_text())
    value = json.loads((folder / "runtime-receipt.json").read_text())
    if fault == "pid": probe["worker_pid"] = 124
    elif fault == "uid": probe["worker_uid"] = 501
    elif fault == "birth": probe["birth_identity"]["start_usec"] += 1
    elif fault == "scope": probe["scope_identity"] = JOB_ROOT.replace(JOB, "b" * 32)
    elif fault == "argv": probe["argv"].append("--other")
    elif fault == "policy": probe["resource_policy"]["host_reserve_bytes"] += 1
    elif fault == "rss": probe["members"][0]["rss_bytes"] = probe["rss_bytes"] = RESOURCES["memory_budget_bytes"] + 1
    elif fault == "rss-sum": probe["rss_bytes"] += 1
    elif fault == "members": probe["members"].append(probe["members"][0])
    elif fault == "listener": probe["listener_pids"] = [124]
    elif fault == "lsof": probe["lsof"] += "fmem\nn/tmp/unreviewed.dylib\n"
    elif fault == "lsof-owner": probe["lsof"] = probe["lsof"].replace("u62002", "u501")
    elif fault == "lock": probe["lock_dev_inode"][1] += 1
    elif fault == "manifest": probe["manifest_sha256"] = "0" * 64
    elif fault == "loaded-sha": value["build"]["loaded_libraries"][0]["sha256"] = "0" * 64
    elif fault == "enforcement": value["host"]["effective_settings"].pop("native_resource_enforcement")
    elif fault == "stop-residual": stopped["cleanup"]["remaining"] = [dict(pid=124)]
    elif fault == "stop-domain": stopped["cleanup"]["worker_user_domain_retired"]["returncode"] = 1
    elif fault == "stop-lock": stopped["lock_dev_inode"][1] += 1
    elif fault == "source-job": config["manifest_path"] = MANIFEST.replace(JOB, "b" * 32)
    else: (root / "build-candidate/freeze-manifest.json").write_text('{}\n')
    (folder / "runtime-after.json").write_text(json.dumps(probe)); (folder / "stopped.json").write_text(json.dumps(stopped))
    (folder / "runtime-receipt.json").write_text(json.dumps(value))
    with pytest.raises(ValueError): aggregate.runtime_retained(root, config, "candidate", "deepseek", folder, launch, folder / "runtime-receipt.json")


def test_mac_only_scoped_result_never_makes_linux_or_full_coverage_green(monkeypatch):
    value = plan(["engine/src/llama-model.cpp"])
    scope = dict(validation_mode=contract.PERFORMANCE_MODE, execution_devices=["local-mbp"])
    jobs = dict(**{"local-mbp":dict(job_id=JOB)})
    rows = [dict(device="local-mbp", model=m, status="inconclusive", functional="passed", numerical="passed", performance=[dict(tier="512", status="inconclusive")]) for m in ("deepseek", "muse")]
    monkeypatch.setattr(aggregate, "device_report", lambda *a: rows)
    report = aggregate.operator_aggregate(value, POLICY, Path("/unused"), jobs, jobs, scope)
    assert report["execution_scoped_status"] == report["mac_scoped_status"] == "inconclusive"
    assert report["linux_scoped_status"] == "not-in-this-run" and report["passed"] is False
    assert not report["full_impact_coverage_complete"]
    assert all(row["status"] == "deferred-not-in-this-run" for row in report["cells"] if row["device"] != "local-mbp")
    text = io.StringIO(); controller.operator_summary(text, report)
    assert "Mac scope: **inconclusive**" in text.getvalue() and "does not disable swap" in text.getvalue()


def test_mac_source_alias_is_exact_for_each_current_job_side(mac_raw):
    root, config, _, _, _ = mac_raw
    frozen = JOB_ROOT + "/frozen/base"
    side = copy.deepcopy(config["sides"]["candidate"])
    side["paths"]["frozen"] = frozen
    build = json.loads((root / "build-candidate/build-receipt.json").read_text().replace("/frozen/candidate", "/frozen/base"))
    expected = {"expert.source":["<sealed-build>/assets/source.tsv"], "cpu.threads":4}
    actual = dict(expected, **{"expert.source":[frozen + "/assets/source.tsv"]})
    manifest = root / "build-candidate/freeze-manifest.json"
    assert receipt.retained_typed_settings(actual, expected, side, "deepseek", build, manifest, MANIFEST)
    actual["expert.source"] = [JOB_ROOT + "/frozen/candidate/assets/source.tsv"]
    assert not receipt.retained_typed_settings(actual, expected, side, "deepseek", build, manifest, MANIFEST)


@pytest.mark.parametrize("fault", ["hash", "count", "writable", "escape", "readonly"])
def test_mac_freeze_keeps_exact_root_inventory_and_rejects_untrusted_metadata(tmp_path, monkeypatch, fault):
    from ci import staged_build
    monkeypatch.setattr(staged_build, "protected", lambda path: Path(path))
    records = {"assets/source.tsv":dict(size=32, mode=0o444, sha256="a" * 64)}
    if fault == "writable": records["assets/source.tsv"]["mode"] = 0o644
    if fault == "escape": records["../outside"] = records.pop("assets/source.tsv")
    source = tmp_path / "root.manifest.json"; save(source, records); source.chmod(0o444)
    frozen = dict(manifest_path=str(source), manifest_sha256=digest(source), readonly=True, file_count=1)
    if fault == "hash": frozen["manifest_sha256"] = "b" * 64
    elif fault == "count": frozen["file_count"] = 2
    elif fault == "readonly": frozen["readonly"] = False
    with pytest.raises(ValueError): staged_build.retain_mac_freeze(frozen, tmp_path / "retained.json")
    assert not (tmp_path / "retained.json").exists()


def test_mac_and_linux_smoke_do_not_hide_missing_stop_log_eof(tmp_path):
    from ci.staged_smoke import collect_log
    class Session:
        def output(self, *args): return b"", 0, False
    with pytest.raises(ValueError, match="EOF"): collect_log(Session(), object(), tmp_path)


@pytest.fixture
def mac_performance_raw(mac_raw):
    from test_final import write_artifacts
    from ci.compare import compare_pair
    import run_final
    root, config, _, original, _ = mac_raw
    source_build = json.loads((root / "build-candidate/build-receipt.json").read_text())
    source_receipt = json.loads((original / "runtime-receipt.json").read_text())
    plan = dict(base="a" * 40, candidate="c" * 40)
    preflight = {}
    for side, pid in (("base", 124), ("candidate", 123)):
        def side_copy(value):
            return json.loads(json.dumps(value).replace("/frozen/candidate", "/frozen/" + side).replace("/work/candidate", "/work/" + side))
        config["sides"][side] = side_copy(copy.deepcopy(config["sides"]["candidate"]))
        config["sides"][side]["revision"] = plan[side]
        native = config["sides"][side]["models"]["deepseek"]
        spec = dict(native["smoke"], request_timeout_s=300)
        native["benchmarks"] = {"512":spec}
        build_dir = root / ("build-" + side); build_dir.mkdir(exist_ok=True)
        build = side_copy(source_build)
        (build_dir / "build-receipt.json").write_text(json.dumps(build))
        if side == "base":
            (build_dir / "freeze-manifest.json").write_bytes((root / "build-candidate/freeze-manifest.json").read_bytes())
        folder = root / "deepseek/512" / side
        folder.parent.mkdir(parents=True, exist_ok=True)
        write_artifacts(folder, "512", dict(device="local-mbp", model="deepseek", revision=plan[side], request_timeout_s=300))
        for name in ("started.json", "stopped.json", "runtime-before.json", "runtime-after.json"):
            value = side_copy(json.loads((original / name).read_text()))
            value["worker_pid"] = pid; value["birth_identity"]["pid"] = pid
            if "members" in value:
                value["members"][0]["pid"] = pid; value["listener_pids"] = [pid]
                value["lsof"] = value["lsof"].replace("p123\n", "p" + str(pid) + "\n")
            (folder / name).write_text(json.dumps(value))
        (folder / "stderr.log").write_text((original / "stderr.log").read_text().replace("/frozen/candidate", "/frozen/" + side))
        value = side_copy(source_receipt); value["server"]["pid"] = pid; value["server"]["start_identity"]["pid"] = pid
        value["build"]["revision"] = plan[side]
        value["host"].update(memory_budget_bytes=RESOURCES["memory_budget_bytes"], os_page_cache_treatment="not-forced; uncontrolled")
        save(folder / "comparison-receipt.json", value); preflight["deepseek", "512", side] = value
        save(folder / "benchmark.json", dict(status="passed", cleanup_confirmed=True, completion_requests=1,
            revision=plan[side], tier="512", model="deepseek", result=json.loads((folder / "result.json").read_text())))
    results = {side:run_final.read_result(root / "deepseek/512" / side / "result.json", "512", 300) for side in plan}
    pair = compare_pair(results["base"], results["candidate"], preflight["deepseek", "512", "base"], preflight["deepseek", "512", "candidate"], POLICY)
    save(root / "deepseek/512/comparison.json", pair)
    return root, config, plan, preflight


@pytest.mark.parametrize("damage", [None, "missing-sse", "different-job-source"])
def test_mac_timed_pair_replays_real_protocol_sse_and_both_side_runtime(mac_performance_raw, damage):
    root, config, plan, preflight = mac_performance_raw
    if damage == "missing-sse": (root / "deepseek/512/base/result.events.sse").unlink()
    elif damage == "different-job-source": config["manifest_path"] = MANIFEST.replace(JOB, "b" * 32)
    if damage:
        with pytest.raises((ValueError, OSError)):
            aggregate.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
    else:
        pair = aggregate.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
        assert pair["status"] == "passed" and pair["measurement_identity"]["output_tokens"] == 256
        assert pair["metrics"]["decode_first128_tps"]["base"] == pair["metrics"]["decode_second128_tps"]["candidate"] == 1
        assert pair["evidence"]["base"]["result.events.sse"]["sha256"] == digest(root / "deepseek/512/base/result.events.sse")


def test_selected_mac_failure_does_not_relabel_successful_linux_scope(monkeypatch):
    value = plan(["engine/src/llama-model.cpp"])
    scope = dict(validation_mode=contract.PERFORMANCE_MODE, execution_devices=list(contract.OPERATOR_DEVICES))
    jobs = {d:dict(job_id=JOB) for d in contract.OPERATOR_DEVICES}
    def reports(root, plan, policy, job, child, scope):
        device = root.parent.name
        if device == "local-mbp": raise ValueError("Mac stopped UID cleanup missing")
        return [dict(device=device, model=m, status="passed", functional="passed", numerical="passed", performance=[dict(tier="512", status="passed")]) for m in ("deepseek", "muse")]
    monkeypatch.setattr(aggregate, "device_report", reports)
    result = aggregate.operator_aggregate(value, POLICY, Path("/unused"), jobs, jobs, scope)
    assert result["status"] == result["execution_scoped_status"] == result["mac_scoped_status"] == "failed"
    assert result["linux_scoped_status"] == "passed" and result["passed"] is False
    assert not result["full_impact_coverage_complete"] and result["errors"]
