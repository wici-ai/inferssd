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
from ci import selected_receipt as receipt, mac_host_gate, mac_host_state
from ci.contract import digest, save
from test_ci_selection import POLICY, plan, template
from test_ci_mac_host_gate import snapshot as host_snapshot

JOB = "a" * 32
JOB_ROOT = "/Library/InferSSD-CI/jobs/" + JOB
MANIFEST = "/Library/InferSSD-CI/private/job-manifests/" + JOB + ".json"
RESOURCES = dict(max_processes=128, memory_budget_bytes=40 << 30, host_reserve_bytes=3 << 30)


def mac_config():
    value = plan(["bench/ci/macos-sandbox/executor.py"])
    config = template(value, "local-mbp")
    config.update(manifest_path=MANIFEST, validation_mode=contract.PERFORMANCE_MODE,
        execution_devices=["local-mbp"], performance_protocol=contract.performance_protocol(),
        host=dict(hardware=dict(memory_total_bytes=48 << 30), resources=RESOURCES))
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
        host=dict(hardware=dict(cpu="M5 Max", memory_total_bytes=48 << 30), os=dict(sysname="Darwin", release="fixture", machine="arm64"), resources=RESOURCES),
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
        members=[dict(pid=123, rss_bytes=45 << 30)], rss_bytes=45 << 30, listener_pids=[123],
        memory_metric="physical-footprint", physical_footprint_bytes=20 << 30, memory_exited_pids=[],
        memory_members=[dict(pid=123, rss_bytes=45 << 30, birth_identity=copy.deepcopy(started["birth_identity"]),
            physical_footprint_bytes=20 << 30, resident_bytes=45 << 30, start_abstime=100, exit_abstime=0)],
        lsof="p123\nu62002\nftxt\nn" + binary + "\nfmem\nn" + frozen + "/lib/libggml.dylib\nfmem\nn/usr/lib/libSystem.B.dylib\nftxt\nn/Library/InferSSD-CI/models/frozen/model.gguf\n")
    stopped = dict(common, running=False, state="failed", reason="requested stop", returncode=0, cleanup_confirmed=True,
                   cleanup=dict(confirmed=True, remaining=[], worker_user_domain_retired=dict(returncode=0)))
    native = {"expert.source":[frozen + "/assets/source.tsv"], "host.budget_mib":4096}
    effective = dict(ambient_environment=launch["environment"], native_resource_policy=RESOURCES,
        native_resource_enforcement="root-sampled-worker-physical-footprint-processes-host-reserve",
        native_launch_policy=dict(template_sha256="c" * 64, typed_values=dict(native, **{"expert.source":["<sealed-build>/assets/source.tsv"]})),
        system_images=[dict(path="/usr/lib/libSystem.B.dylib", sha256="e" * 64)])
    runtime = dict(model_identity=data["model_identity"], build=dict(revision="c" * 40,
        loaded_libraries=[dict(path=binary, sha256="b" * 64), dict(path=frozen + "/lib/libggml.dylib", sha256="d" * 64)]),
        server=dict(pid=123, start_identity=started["birth_identity"]), host=dict(hardware=cfg["host"]["hardware"], os=cfg["host"]["os"], effective_settings=effective))
    save(folder / "started.json", started); save(folder / "stopped.json", stopped)
    save(folder / "runtime-before.json", observed); save(folder / "runtime-after.json", observed)
    (folder / "stderr.log").write_text('[ssd.settings] ' + json.dumps(dict(values=native)) + '\n')
    save(folder / "runtime-receipt.json", runtime)
    host_raw(folder, cfg)
    return tmp_path, cfg, manifest, folder, launch


def host_raw(folder, config):
    """Retain parseable synthetic OS output, not a mocked green gate result."""
    expected = mac_host_gate.policy(config)
    before, after = host_snapshot(1), host_snapshot(3.1)
    save(folder / mac_host_gate.QUIET, dict(schema_version=1, purpose="mac-host-quiet",
        policy=expected, before=before, after=after,
        assessment=mac_host_state.compare_snapshots(before, after, expected["limits"])))
    for name, second in zip(mac_host_gate.SNAPSHOTS, (4, 6, 8)):
        save(folder / name, host_snapshot(second))
    return mac_host_gate.retained_arm(folder, expected)


def test_mac_runtime_retained_rechecks_real_shaped_probe_lsof_sampled_cap_and_stop(mac_raw):
    root, config, _, folder, launch = mac_raw
    probe = json.loads((folder / "runtime-after.json").read_text())
    assert probe["rss_bytes"] > RESOURCES["memory_budget_bytes"]
    assert receipt.mac_memory_sample(probe, RESOURCES) == 20 << 30
    value = aggregate.runtime_retained(root, config, "candidate", "deepseek", folder, launch, folder / "runtime-receipt.json")
    assert value["host"]["effective_settings"]["native_resource_enforcement"] == receipt.MAC_RESOURCE_ENFORCEMENT


@pytest.mark.parametrize("fault", [None, "duplicate-exit", "foreign-exit", "live-exit", "server-exit", "unaccounted"])
def test_mac_memory_partition_preserves_exited_inventory_without_charging_it(mac_raw, fault):
    _, _, _, folder, _ = mac_raw
    probe = json.loads((folder / "runtime-after.json").read_text())
    # ps can retain a zombie's old RSS; the root rusage sampler excludes it.
    probe["members"].append(dict(pid=124, rss_bytes=2 << 30))
    probe["rss_bytes"] += 2 << 30
    probe["memory_exited_pids"] = [124]
    if fault == "duplicate-exit": probe["memory_exited_pids"].append(124)
    elif fault == "foreign-exit": probe["memory_exited_pids"] = [125]
    elif fault == "live-exit": probe["memory_exited_pids"].append(123)
    elif fault == "server-exit":
        probe["memory_members"] = []
        probe["memory_exited_pids"].append(123)
        probe["physical_footprint_bytes"] = 0
    elif fault == "unaccounted": probe["memory_exited_pids"] = []
    if fault:
        with pytest.raises(ValueError): receipt.mac_memory_sample(probe, RESOURCES)
    else:
        assert receipt.mac_memory_sample(probe, RESOURCES) == 20 << 30
        assert len(probe["members"]) == 2 and probe["rss_bytes"] == 47 << 30


def test_mac_memory_sums_multiple_live_workers_at_exact_budget(mac_raw):
    _, _, _, folder, _ = mac_raw
    probe = json.loads((folder / "runtime-after.json").read_text())
    child = dict(pid=124, rss_bytes=1 << 30)
    probe["members"].append(child)
    probe["rss_bytes"] += child["rss_bytes"]
    probe["memory_members"].append(dict(child,
        birth_identity=dict(pid=124, uid=62002, start_sec=1700000001, start_usec=2),
        physical_footprint_bytes=20 << 30, resident_bytes=1 << 30, start_abstime=101, exit_abstime=0))
    probe["physical_footprint_bytes"] = 40 << 30
    assert receipt.mac_memory_sample(probe, RESOURCES) == RESOURCES["memory_budget_bytes"]


def test_mac_live_receipt_reads_alias_from_native_freeze_path(mac_raw, monkeypatch):
    from hashlib import sha256
    from inferssd.configuration.settings import parse

    root, config, manifest, folder, launch = mac_raw
    build = json.loads((root / "build-candidate/build-receipt.json").read_text())
    assert "host_path" not in build["freeze"]
    frozen = build["freeze"]["path"]
    data = config["sides"]["candidate"]["models"]["deepseek"]
    shard = data["model_identity"]["shards"][0]
    shard["stat"] = dict(bytes=128)
    alias_path = frozen + "/assets/source.tsv"
    settings_raw = ('inferssd-settings 1\nexpert.source = ["' + alias_path + '"]\n').encode()
    payloads = {data["settings"]["runtime_path"]: settings_raw,
                data["template"]["runtime_path"]: b"template fixture",
                alias_path: ("source_path\texpert_idx\toffset\tnbytes\n" +
                             shard["runtime_path"] + "\t0\t0\t32\n").encode()}
    for key in ("settings", "template"):
        item = data[key]
        item.update(host_path=item["runtime_path"], sha256=sha256(payloads[item["runtime_path"]]).hexdigest())
    launch["argv"] = [build["binary"]["runtime_path"], "-m", shard["runtime_path"],
                      "--ssd-settings", data["settings"]["runtime_path"],
                      "--chat-template-file", data["template"]["runtime_path"]]
    launch["os_page_cache_treatment"] = "not-forced; uncontrolled"
    for name in ("runtime-before.json", "runtime-after.json"):
        observed = json.loads((folder / name).read_text())
        observed["argv"] = launch["argv"]
        (folder / name).write_text(json.dumps(observed))
    native = parse(settings_raw.decode(), data["settings"]["runtime_path"], platform="darwin").document()
    (folder / "stderr.log").write_text("[ssd.settings] " + json.dumps(native) + "\n")
    uname = receipt.os.uname()
    config["host"]["os"] = dict(sysname=uname.sysname, release=uname.release, machine=uname.machine)
    reads = []

    def sealed_file(path, *, contents=False):
        path = str(path)
        reads.append(path)
        raw = payloads[path]
        identity = dict(path=path, bytes=len(raw), sha256=sha256(raw).hexdigest())
        return (identity, raw) if contents else identity

    monkeypatch.setattr(receipt, "sealed_file", sealed_file)
    monkeypatch.setattr(receipt, "checked_model", lambda *a: data["model_identity"])
    result = receipt.receipt(config, manifest, "candidate", "deepseek", launch, build, folder)
    assert reads.count(alias_path) == 1
    assert result["host"]["effective_settings"]["native_resource_enforcement"] == receipt.MAC_RESOURCE_ENFORCEMENT
    assert result["host"]["effective_settings"]["native_launch_policy"]["typed_values"]["expert.source"] == [
        "<sealed-build>/assets/source.tsv"]
    assert aggregate.runtime_retained(root, config, "candidate", "deepseek", folder, launch,
        folder / "comparison-receipt.json") == result


@pytest.mark.parametrize("fault", ["pid", "uid", "birth", "scope", "argv", "policy", "footprint", "rss-sum", "members", "listener", "lsof", "lsof-owner", "lock", "manifest", "loaded-sha", "enforcement", "legacy-enforcement", "stop-residual", "stop-domain", "stop-lock", "source-job", "source-inventory",
    "memory-metric", "legacy-metric", "unknown-memory", "accounting-error", "footprint-sum", "memory-member-missing", "memory-member-duplicate", "memory-member-foreign", "memory-member-uid", "memory-member-birth", "memory-member-rss", "memory-member-start", "memory-member-exited"])
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
    elif fault == "footprint": probe["memory_members"][0]["physical_footprint_bytes"] = probe["physical_footprint_bytes"] = RESOURCES["memory_budget_bytes"] + 1
    elif fault == "rss-sum": probe["rss_bytes"] += 1
    elif fault == "members": probe["members"].append(probe["members"][0])
    elif fault == "listener": probe["listener_pids"] = [124]
    elif fault == "lsof": probe["lsof"] += "fmem\nn/tmp/unreviewed.dylib\n"
    elif fault == "lsof-owner": probe["lsof"] = probe["lsof"].replace("u62002", "u501")
    elif fault == "lock": probe["lock_dev_inode"][1] += 1
    elif fault == "manifest": probe["manifest_sha256"] = "0" * 64
    elif fault == "loaded-sha": value["build"]["loaded_libraries"][0]["sha256"] = "0" * 64
    elif fault == "enforcement": value["host"]["effective_settings"].pop("native_resource_enforcement")
    elif fault == "legacy-enforcement": value["host"]["effective_settings"]["native_resource_enforcement"] = "root-sampled-worker-rss-processes-host-reserve"
    elif fault == "stop-residual": stopped["cleanup"]["remaining"] = [dict(pid=124)]
    elif fault == "stop-domain": stopped["cleanup"]["worker_user_domain_retired"]["returncode"] = 1
    elif fault == "stop-lock": stopped["lock_dev_inode"][1] += 1
    elif fault == "source-job": config["manifest_path"] = MANIFEST.replace(JOB, "b" * 32)
    elif fault == "source-inventory": (root / "build-candidate/freeze-manifest.json").write_text('{}\n')
    elif fault == "memory-metric": probe.pop("memory_metric")
    elif fault == "legacy-metric": probe["memory_metric"] = "rss"
    elif fault == "unknown-memory": probe["physical_footprint_bytes"] = None
    elif fault == "accounting-error": probe["memory_accounting_error"] = "PermissionError: proc_pid_rusage refused"
    elif fault == "footprint-sum": probe["physical_footprint_bytes"] += 1
    elif fault == "memory-member-missing": probe["memory_members"] = []
    elif fault == "memory-member-duplicate": probe["memory_members"].append(probe["memory_members"][0])
    elif fault == "memory-member-foreign": probe["memory_members"][0]["pid"] = 124
    elif fault == "memory-member-uid": probe["memory_members"][0]["birth_identity"]["uid"] = 501
    elif fault == "memory-member-birth": probe["memory_members"][0]["birth_identity"]["start_usec"] += 1
    elif fault == "memory-member-rss": probe["memory_members"][0]["rss_bytes"] += 1
    elif fault == "memory-member-start": probe["memory_members"][0]["start_abstime"] = 0
    elif fault == "memory-member-exited": probe["memory_members"][0]["exit_abstime"] = 1
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
    expected = {"expert.source":["<sealed-build>/assets/source.tsv"], "host.budget_mib":4096}
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
    preflight, host_states = {}, {}
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
                value["memory_members"][0]["pid"] = pid
                value["memory_members"][0]["birth_identity"]["pid"] = pid
                value["lsof"] = value["lsof"].replace("p123\n", "p" + str(pid) + "\n")
            (folder / name).write_text(json.dumps(value))
        (folder / "stderr.log").write_text((original / "stderr.log").read_text().replace("/frozen/candidate", "/frozen/" + side))
        value = side_copy(source_receipt); value["server"]["pid"] = pid; value["server"]["start_identity"]["pid"] = pid
        value["build"]["revision"] = plan[side]
        value["host"].update(memory_budget_bytes=RESOURCES["memory_budget_bytes"], os_page_cache_treatment="not-forced; uncontrolled")
        save(folder / "comparison-receipt.json", value); preflight["deepseek", "512", side] = value
        save(folder / "benchmark.json", dict(status="passed", cleanup_confirmed=True, completion_requests=1,
            revision=plan[side], tier="512", model="deepseek", result=json.loads((folder / "result.json").read_text())))
        host_states[side] = host_raw(folder, config)
    results = {side:run_final.read_result(root / "deepseek/512" / side / "result.json", "512", 300) for side in plan}
    pair = compare_pair(results["base"], results["candidate"], preflight["deepseek", "512", "base"], preflight["deepseek", "512", "candidate"], POLICY)
    pair = mac_host_gate.pair(pair, host_states["base"], host_states["candidate"])
    save(root / "deepseek/512/comparison.json", pair)
    return root, config, plan, preflight


@pytest.mark.parametrize("damage", [None, "missing-sse", "different-job-source", "missing-host", "changed-host-raw"])
def test_mac_timed_pair_replays_real_protocol_sse_and_both_side_runtime(mac_performance_raw, damage):
    root, config, plan, preflight = mac_performance_raw
    if damage == "missing-sse": (root / "deepseek/512/base/result.events.sse").unlink()
    elif damage == "different-job-source": config["manifest_path"] = MANIFEST.replace(JOB, "b" * 32)
    elif damage == "missing-host": (root / "deepseek/512/base" / mac_host_gate.SNAPSHOTS[1]).unlink()
    elif damage == "changed-host-raw":
        path = root / "deepseek/512/candidate" / mac_host_gate.SNAPSHOTS[1]
        value = json.loads(path.read_text()); value["raw"]["swap"]["stdout"] = "invalid observation"
        path.write_text(json.dumps(value))
    if damage:
        with pytest.raises((ValueError, OSError)):
            aggregate.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
    else:
        pair = aggregate.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
        assert pair["status"] == "passed" and pair["measurement_identity"]["output_tokens"] == 256
        assert pair["metrics"]["decode_first128_tps"]["base"] == pair["metrics"]["decode_second128_tps"]["candidate"] == 1
        assert pair["evidence"]["base"]["result.events.sse"]["sha256"] == digest(root / "deepseek/512/base/result.events.sse")
        assert pair["environment"]["status"] == "passed"
        assert pair["evidence"]["base"][mac_host_gate.QUIET]["sha256"] == digest(root / "deepseek/512/base" / mac_host_gate.QUIET)


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
