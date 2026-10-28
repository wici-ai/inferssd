"""Operator/report-only contracts. Synthetic API/raw only; no physical claims."""

import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import impact, selected_controller as controller, selected_aggregate as aggregation
from ci import selection_contract as contract
from ci import compare
from ci.contract import digest, load, save
from ci.stage_session import BrokerError
from test_private_controller import event_context
from test_ci_selection import template

POLICY = load(Path(__file__).parents[1] / "ci/policy.json")


@pytest.fixture
def operator(tmp_path, monkeypatch):
    event, env = event_context(tmp_path)
    event["sender"]["id"] = 17
    env.update(GITHUB_TOKEN="synthetic-private-only", GITHUB_TRIGGERING_ACTOR="maintainer",
               INFERSSD_CI_ENABLED="true", GITHUB_STEP_SUMMARY=str(tmp_path / "summary"))
    plan = impact.plan(["bench/ci/selected_controller.py"], POLICY,
                       base="a" * 40, head="b" * 40, candidate="c" * 40)
    devices = ["cloud-5090"]
    jobs = {d: dict(device=d, job_id="f" * 32, source_revision=plan["candidate"],
                  config_sha256="1" * 64, selection_plan_sha256=impact.fingerprint(plan),
                  control_revision="e" * 40, control_manifest_sha256="2" * 64) for d in devices}
    envelope = dict(schema_version=1, purpose=controller.OPERATOR_PURPOSE,
        repository=POLICY["repository"], **{k: plan[k] for k in ("base", "head", "candidate")},
        selection=plan, selection_plan_sha256=impact.fingerprint(plan),
        execution_devices=devices, validation_mode=contract.FUNCTIONAL_MODE,
        prepared_by=dict(login="maintainer", id=17), controller_repository=POLICY["controller_repository"],
        controller_workflow_sha=env["GITHUB_WORKFLOW_SHA"], policy_sha256=impact.fingerprint(POLICY), jobs=jobs)
    event["inputs"] = dict(prepared_envelope=json.dumps(envelope))
    args = SimpleNamespace(policy=Path(__file__).parents[1] / "ci/policy.json", control=tmp_path / "control",
                           plan=tmp_path / "admission.json", artifacts=tmp_path / "artifacts", output=tmp_path / "report.json")
    args.control.mkdir()
    save(args.control / "control-source.json", dict(synthetic=True))
    for job in jobs.values(): job["control_manifest_sha256"] = digest(args.control / "control-source.json")
    calls = []
    permissions = {"maintainer": "admin"}
    child = dict(id=73, workflow_id=91, path=".github/workflows/integration-smoke.yml", event="workflow_dispatch",
                 head_branch="main", head_sha=env["GITHUB_SHA"], run_attempt=1,
                 display_title="integration-" + "f" * 32, status="completed", conclusion="success")
    class Private:
        def __init__(self, repository, token, **kwargs):
            assert repository == POLICY["controller_repository"]
            assert token == env["GITHUB_TOKEN"]
        def call(self, path, data=None, method=None):
            assert data is None and method is None, "report-only code must not publish/dispatch in this test"
            calls.append(path)
            if path == "": return dict(private=True, default_branch="main")
            if path == "branches/main": return dict(protected=True, commit=dict(sha=env["GITHUB_SHA"]))
            if path.startswith("collaborators/"):
                return dict(permission=permissions.get(path.split("/")[1], "read"))
            assert path == "actions/runs/73"
            return child
    monkeypatch.setattr(controller, "API", Private)
    monkeypatch.setattr(controller, "source_manifest", lambda _: "e" * 40)
    def event_save():
        event["inputs"]["prepared_envelope"] = json.dumps(envelope)
        Path(env["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    event_save()
    return SimpleNamespace(args=args, env=env, event=event, envelope=envelope, plan=plan,
                           calls=calls, permissions=permissions, child=child, write_event=event_save)


def test_operator_admission_uses_only_private_token_and_sealed_source(operator):
    a = operator
    assert controller.operator_preflight(a.args, a.env) == 0
    admitted = load(a.args.plan)
    assert admitted["operator"] == dict(login="maintainer", id=17, triggering_actor="maintainer")
    assert admitted["selection"] == a.plan and admitted["execution_devices"] == ["cloud-5090"]
    assert not any("pulls" in p or "check-runs" in p or "statuses" in p for p in a.calls)


@pytest.mark.parametrize("fault", ["actor", "actor-id", "controller", "policy", "scope", "empty", "duplicate", "missing-job", "source", "plan", "unknown-field", "job-control", "job-manifest"])
def test_operator_seal_or_scope_changes_fail_before_admission(operator, fault):
    a = operator; e = a.envelope
    if fault == "actor": e["prepared_by"]["login"] = "other"
    elif fault == "actor-id": e["prepared_by"]["id"] = True
    elif fault == "controller": e["controller_workflow_sha"] = "0" * 40
    elif fault == "policy": e["policy_sha256"] = "0" * 64
    elif fault == "scope": e["execution_devices"] = ["local-mbp"]
    elif fault == "empty": e["execution_devices"] = []
    elif fault == "duplicate": e["execution_devices"] *= 2
    elif fault == "missing-job": e["jobs"] = {}
    elif fault == "source": e["candidate"] = "0" * 40
    elif fault == "plan": e["selection"]["cells"][-1]["required"] = False
    elif fault == "job-control": e["jobs"]["cloud-5090"]["control_revision"] = "0" * 40
    elif fault == "job-manifest": e["jobs"]["cloud-5090"]["control_manifest_sha256"] = "0" * 64
    else: e["runtime_token"] = "forbidden"
    a.write_event()
    with pytest.raises((ValueError, KeyError)):
        controller.operator_preflight(a.args, a.env)
    assert not a.args.plan.exists()


@pytest.mark.parametrize("fault", ["rerun", "trigger-permission", "initial-permission", "disabled", "event-id"])
def test_actual_operator_and_rerun_permissions_are_checked(operator, fault):
    a = operator
    if fault == "rerun": a.env["GITHUB_RUN_ATTEMPT"] = "2"
    elif fault == "trigger-permission": a.env["GITHUB_TRIGGERING_ACTOR"] = "reader"
    elif fault == "initial-permission": a.permissions["maintainer"] = "read"
    elif fault == "disabled": a.env["INFERSSD_CI_ENABLED"] = "false"
    else: a.event["sender"]["id"] = 18; a.write_event()
    with pytest.raises(ValueError): controller.operator_preflight(a.args, a.env)


def test_dispatch_json_never_accepts_duplicate_or_nonfinite_data(operator):
    for raw in ('{"selection":1,"selection":2}', '{"x":NaN}', 'x' * (60 * 1024 + 1)):
        with pytest.raises((ValueError, BrokerError)): controller.read_operator_envelope(dict(inputs=dict(prepared_envelope=raw)))


def ready(a):
    controller.operator_preflight(a.args, a.env)
    a.args.artifacts.mkdir()
    save(a.args.artifacts / "dispatch-state.json", dict(selection_plan_sha256=impact.fingerprint(a.plan), errors={},
          children={"cloud-5090": dict(a.child)}))


def test_report_only_returns_scoped_pass_without_source_api_or_publication(operator, monkeypatch):
    a = operator; ready(a)
    def raw_aggregate(plan, policy, artifacts, jobs, children, scope):
        assert scope == dict(validation_mode=contract.FUNCTIONAL_MODE, execution_devices=["cloud-5090"])
        return dict(passed=True, full_impact_coverage_complete=False, cells=[], **scope)
    monkeypatch.setattr(controller, "operator_aggregate", raw_aggregate)
    assert controller.operator_report(a.args, a.env) == 0
    report = load(a.args.output)
    assert report["candidate"] == a.plan["candidate"] and report["passed"] is True
    assert report["full_impact_coverage_complete"] is False
    summary = Path(a.env["GITHUB_STEP_SUMMARY"]).read_text()
    assert "PR publication is not enabled" in summary and "0/16" in summary
    assert not any("check-runs" in p or "statuses" in p or "pulls" in p for p in a.calls)


@pytest.mark.parametrize("fault", ["missing-admission", "missing-child", "child-failed", "child-rerun", "envelope-changed", "unclean"])
def test_report_failure_is_terminal_not_skipped_or_promoted(operator, monkeypatch, fault):
    a = operator; ready(a)
    monkeypatch.setattr(controller, "operator_aggregate", lambda *a: (_ for _ in ()).throw(ValueError("unclean raw")))
    if fault == "missing-admission": a.args.plan.unlink()
    elif fault == "missing-child":
        state = load(a.args.artifacts / "dispatch-state.json"); state["children"] = {}
        (a.args.artifacts / "dispatch-state.json").write_text(json.dumps(state))
    elif fault == "child-failed": a.child["conclusion"] = "failure"
    elif fault == "child-rerun": a.child["run_attempt"] = 2
    elif fault == "envelope-changed":
        admitted = load(a.args.plan); admitted["jobs"]["cloud-5090"]["job_id"] = "0" * 32
        a.args.plan.write_text(json.dumps(admitted))
    assert controller.operator_report(a.args, a.env) == 1
    assert load(a.args.output)["passed"] is False
    assert "failed" in Path(a.env["GITHUB_STEP_SUMMARY"]).read_text()


def test_functional_scope_preserves_canonical_required_and_defers_perf():
    p = impact.plan(["engine/src/llama-kv-cache-dsv4.cpp"], POLICY,
                    base="a" * 40, head="b" * 40, candidate="c" * 40)
    before = copy.deepcopy(p)
    cfg = template(p)
    cfg.update(validation_mode=contract.FUNCTIONAL_MODE, execution_devices=["cloud-5090"])
    for side in cfg["sides"].values():
        for model in side["models"].values(): model["benchmarks"] = {}
    manifest = dict(job_id="f" * 32, source_revisions={"candidate": p["candidate"]})
    cfg, manifest = contract.bind(cfg, manifest, p, POLICY)
    assert p == before and all(c["performance"]["tiers"] == ["512", "4096"] for c in impact.selected(p))
    assert contract.bound(cfg, manifest, "f" * 32)["performance"] == {}
    assert not any("benchmark" in name or "preflight" in name for name in manifest["stages"])
    manifest["selection"]["execution_devices"] = ["dgx-spark"]
    with pytest.raises(ValueError): contract.bound(cfg, manifest, "f" * 32)


def test_operator_aggregate_reports_omitted_devices_without_editing_required(operator, monkeypatch):
    a = operator
    scope = contract.execution_scope(a.envelope)
    def result(root, plan, policy, job, child, received):
        assert received == scope
        return [dict(device=job["device"], model=m, status="passed", functional="passed",
                     numerical="passed", performance="not-required") for m in ("deepseek", "muse")]
    monkeypatch.setattr(aggregation, "device_report", result)
    report = aggregation.operator_aggregate(a.plan, POLICY, Path("/unused"), a.envelope["jobs"],
        {"cloud-5090": a.child}, scope)
    assert report["passed"] is True and report["full_impact_coverage_complete"] is False
    assert {r["status"] for r in report["cells"] if r["device"] == "local-mbp"} == {"deferred-offline"}
    assert {r["status"] for r in report["cells"] if r["device"] == "dgx-spark"} == {"deferred-not-in-this-run"}
    assert all(c["required"] for c in a.plan["cells"])
    with pytest.raises(ValueError): aggregation.operator_aggregate(a.plan, POLICY, Path("/unused"), {}, {}, scope)
    monkeypatch.setattr(aggregation, "device_report", lambda *a: (_ for _ in ()).throw(ValueError("missing raw cleanup")))
    report = aggregation.operator_aggregate(a.plan, POLICY, Path("/unused"), a.envelope["jobs"], {"cloud-5090": a.child}, scope)
    assert report["passed"] is False and report["errors"]


@pytest.mark.parametrize("clean", [True, False])
def test_scoped_executor_builds_only_candidate_and_never_calls_performance(tmp_path, monkeypatch, clean):
    from ci import selected_integration as execution
    p = impact.plan(["bench/ci/selected_controller.py"], POLICY,
                    base="a" * 40, head="b" * 40, candidate="c" * 40)
    cfg = template(p)
    cfg.update(validation_mode=contract.FUNCTIONAL_MODE, execution_devices=["cloud-5090"])
    manifest = dict(job_id="f" * 32, source_revisions={"candidate": p["candidate"]})
    cfg, manifest = contract.bind(cfg, manifest, p, POLICY)
    path, manifest_path = tmp_path / "config.json", tmp_path / "manifest.json"
    save(manifest_path, manifest)
    cfg["manifest_path"] = str(manifest_path)
    cfg["manifest_sha256"] = digest(manifest_path)
    for data in cfg["sides"]["candidate"]["models"].values():
        data["smoke"].update(argv=["/build/candidate/bin/llama-server"])
    save(path, cfg)
    calls = []
    class Session:
        closed = False
        def __init__(self, *args, **kwargs): pass
        def open(self): calls.append("open"); return dict(manifest_sha256=cfg["manifest_sha256"])
        def finish(self): calls.append("finish"); return dict(cleanup_confirmed=clean)
        def close(self): self.closed = True
    monkeypatch.setattr(execution, "StageSession", Session)
    monkeypatch.setattr(execution, "protected_binding", lambda *a: cfg)
    monkeypatch.setattr(execution, "source_manifest", lambda *a: cfg["control_revision"])
    real_digest = execution.digest
    monkeypatch.setattr(execution, "digest", lambda p: cfg["control_manifest_sha256"]
        if Path(p).name == "control-source.json" else real_digest(p))
    def build(session, side, *args):
        calls.append("build-" + side)
        return dict(binary=dict(runtime_path="/build/candidate/bin/llama-server"))
    monkeypatch.setattr(execution, "run_build", build)
    monkeypatch.setattr(execution, "selected_numerical", lambda session, config, build, cases, output:
        {case: dict(status="passed", synthetic=True) for case in cases})
    def smoke(session, native, folder, **kwargs):
        calls.append("smoke-" + native["model"]); folder.mkdir()
        return dict(status="passed", completion_requests=1, cleanup_confirmed=True)
    monkeypatch.setattr(execution, "smoke", smoke)
    def receipt(*args, destination):
        calls.append("runtime-receipt"); save(destination, dict(synthetic=True))
    monkeypatch.setattr(execution, "receipt", receipt)
    monkeypatch.setattr(execution, "benchmark", lambda *a, **k: pytest.fail("functional scope must not time a request"))
    args = SimpleNamespace(output=tmp_path / "out", config=path, config_sha256=digest(path),
        selection_plan_sha256=impact.fingerprint(p), job_id="f" * 32, socket="/unused")
    assert execution.run(args) == (0 if clean else 1)
    report = load(args.output / "integration.json")
    assert report["validation_mode"] == contract.FUNCTIONAL_MODE
    assert report["execution_devices"] == ["cloud-5090"] and report["cleanup_confirmed"] is clean
    assert "build-base" not in calls and calls.count("runtime-receipt") == 2 and calls[-1] == "finish"
    assert all(cell["performance"] == [] and cell["runtime_receipt_sha256"] for cell in report["cells"])
    assert load(args.output / "performance-preflight.json")["cells"] == []


@pytest.fixture
def functional_raw(tmp_path):
    root = tmp_path; folder = root / "native-deepseek"; folder.mkdir()
    (root / "build-candidate").mkdir()
    binary = dict(runtime_path="/build/candidate/bin/llama-server", sha256="a" * 64)
    save(root / "build-candidate/build-receipt.json", dict(binary=binary))
    started = dict(worker_pid=123, worker_uid=995, birth_identity="boot:5", scope_identity="/system.slice/fresh.service")
    resources = dict(cpus_allowed_list="0-3", cgroup_limits=dict(memory_max="100", memory_swap_max="0"), memlock_bytes=1)
    launch = dict(argv=[binary["runtime_path"], "--fixture"], environment={}, cwd="/scratch/candidate")
    data = dict(smoke=launch, model_identity=dict(variant="fixture", shards=[]), template=dict(sha256="b" * 64))
    cfg = dict(revision="c" * 40, sides=dict(candidate=dict(models=dict(deepseek=data))),
               host=dict(hardware="fixture", os=dict(sysname="Linux"), resources=resources))
    raw = dict(**started, running=True, ready=True, cmdline=launch["argv"], executable=binary["runtime_path"],
        executable_sha256=binary["sha256"], cpus_allowed_list="0-3", cgroup_limits=resources["cgroup_limits"],
        proc_limits="Max locked memory 1 1 bytes\n", environment=dict(PWD=launch["cwd"], LANG="C"),
        proc_maps="100-200 r-xp 0000 08:01 9 /build/candidate/bin/llama-server\n")
    save(folder / "runtime-before.json", raw); save(folder / "runtime-after.json", raw)
    save(folder / "started.json", started)
    save(folder / "stopped.json", dict(running=False, returncode=0, cleanup_confirmed=True))
    (folder / "stderr.log").write_text('[ssd.settings] {"values":{"fixture":1}}\n')
    receipt = dict(model_identity=data["model_identity"], build=dict(revision=cfg["revision"],
        loaded_libraries=[dict(path=binary["runtime_path"], sha256=binary["sha256"])]),
        server=dict(pid=123, start_identity="boot:5"), host=dict(hardware="fixture", os=cfg["host"]["os"],
        effective_settings=dict(native_resource_policy=resources, ambient_environment=dict(LANG="C"),
            native_launch_policy=dict(typed_values=dict(fixture=1), template_sha256="b" * 64))))
    save(folder / "runtime-receipt.json", receipt)
    return root, folder, cfg, dict(runtime_receipt_sha256=digest(folder / "runtime-receipt.json"))


def test_functional_aggregate_rechecks_retained_probes_not_just_receipt_hash(functional_raw):
    root, _, config, cell = functional_raw
    aggregation.functional_retained(root, config, "deepseek", cell)


@pytest.mark.parametrize("fault", ["argv", "resource", "pid", "memlock", "maps", "ambient", "settings", "model", "template", "cleanup"])
def test_functional_retained_identity_or_resources_cannot_be_rehashed_into_acceptance(functional_raw, fault):
    root, folder, cfg, cell = functional_raw
    if fault in ("model", "template"):
        receipt = load(folder / "runtime-receipt.json")
        if fault == "model": receipt["model_identity"]["variant"] = "different"
        else: receipt["host"]["effective_settings"]["native_launch_policy"]["template_sha256"] = "0" * 64
        (folder / "runtime-receipt.json").write_text(json.dumps(receipt))
        cell["runtime_receipt_sha256"] = digest(folder / "runtime-receipt.json")
    elif fault == "settings": (folder / "stderr.log").write_text('[ssd.settings] {"values":{"fixture":2}}\n')
    elif fault == "cleanup": (folder / "stopped.json").write_text(json.dumps(dict(running=False, returncode=0, cleanup_confirmed=False)))
    else:
        raw = load(folder / "runtime-after.json")
        if fault == "argv": raw["cmdline"] = ["/other"]
        elif fault == "resource": raw["cgroup_limits"]["memory_max"] = "200"
        elif fault == "pid": raw["worker_pid"] = 124
        elif fault == "memlock": raw["proc_limits"] = "Max locked memory 2 2 bytes\n"
        elif fault == "maps": raw["proc_maps"] = raw["proc_maps"].replace("08:01 9", "08:01 10")
        else: raw["environment"]["LANG"] = "different"
        (folder / "runtime-after.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError): aggregation.functional_retained(root, cfg, "deepseek", cell)


@pytest.fixture
def functional_alias_raw(functional_raw):
    root, folder, cfg, cell = functional_raw
    records = [dict(path="assets/deepseek-alias.tsv", size=12, sha256="d" * 64)]
    save(root / "build-candidate/freeze-manifest.json", dict(files=records))
    build = load(root / "build-candidate/build-receipt.json")
    build["freeze"] = dict(side="candidate", path="/build/candidate",
        host_path="/var/lib/inferssd-ci-broker/jobs/" + "f" * 32 + "/frozen/candidate",
        files=len(records), manifest_sha256=impact.fingerprint(records))
    (root / "build-candidate/build-receipt.json").write_text(json.dumps(build))
    cfg["sides"]["candidate"]["paths"] = dict(frozen="/build/candidate")
    cfg["manifest_path"] = "/var/lib/inferssd-ci-broker/jobs/" + "f" * 32 + "/manifest.json"
    receipt = load(folder / "runtime-receipt.json")
    receipt["host"]["effective_settings"]["native_launch_policy"]["typed_values"]["expert.source"] = ["<sealed-build>/assets/deepseek-alias.tsv"]
    (folder / "runtime-receipt.json").write_text(json.dumps(receipt))
    cell["runtime_receipt_sha256"] = digest(folder / "runtime-receipt.json")
    (folder / "stderr.log").write_text('[ssd.settings] {"values":{"fixture":1,"expert.source":["/build/candidate/assets/deepseek-alias.tsv"]}}\n')
    return root, folder, cfg, cell


@pytest.mark.parametrize("native_source", ["/build/candidate/assets/deepseek-alias.tsv", "<sealed-build>/assets/deepseek-alias.tsv"])
def test_functional_retained_accepts_only_the_same_sealed_alias(functional_alias_raw, native_source):
    root, folder, cfg, cell = functional_alias_raw
    (folder / "stderr.log").write_text('[ssd.settings] ' + json.dumps(dict(values={"fixture": 1, "expert.source": [native_source]})) + '\n')
    aggregation.functional_retained(root, cfg, "deepseek", cell)


def test_functional_retained_keeps_exact_source_semantics_without_linux_remapping(functional_raw):
    root, folder, cfg, cell = functional_raw
    source = "/Library/InferSSD-CI/jobs/owned/frozen/candidate/assets/deepseek-alias.tsv"
    values = {"fixture": 1, "expert.source": [source]}
    receipt = load(folder / "runtime-receipt.json")
    receipt["host"]["effective_settings"]["native_launch_policy"]["typed_values"] = values
    (folder / "runtime-receipt.json").write_text(json.dumps(receipt))
    cell["runtime_receipt_sha256"] = digest(folder / "runtime-receipt.json")
    (folder / "stderr.log").write_text('[ssd.settings] ' + json.dumps(dict(values=values)) + '\n')
    aggregation.functional_retained(root, cfg, "deepseek", cell)


@pytest.mark.parametrize("fault", ["prefix", "other-build", "parent", "dot", "double-slash", "asset", "missing-asset", "inventory-hash", "host-root", "other-job", "manifest-root", "runtime-root", "other-setting"])
def test_functional_retained_source_spelling_cannot_hide_another_input(functional_alias_raw, fault):
    root, folder, cfg, cell = functional_alias_raw
    source = "/build/candidate/assets/deepseek-alias.tsv"
    native = {"fixture": 1, "expert.source": [source]}
    replacements = {"prefix": "/tmp/candidate/assets/deepseek-alias.tsv",
        "other-build": "/build/candidate-extra/assets/deepseek-alias.tsv",
        "parent": "/build/candidate/assets/../assets/deepseek-alias.tsv",
        "dot": "/build/candidate/assets/./deepseek-alias.tsv",
        "double-slash": "/build/candidate/assets//deepseek-alias.tsv",
        "asset": "/build/candidate/assets/muse-alias.tsv"}
    if fault in replacements:
        native["expert.source"] = [replacements[fault]]
    elif fault == "other-setting": native["fixture"] = 2
    elif fault == "runtime-root": cfg["sides"]["candidate"]["paths"]["frozen"] = "/tmp/candidate"
    elif fault == "manifest-root": cfg["manifest_path"] = "/tmp/manifest.json"
    else:
        build = load(root / "build-candidate/build-receipt.json")
        if fault == "inventory-hash": build["freeze"]["manifest_sha256"] = "0" * 64
        elif fault == "host-root": build["freeze"]["host_path"] = "/tmp/frozen/candidate"
        elif fault == "other-job": build["freeze"]["host_path"] = "/var/lib/inferssd-ci-broker/jobs/" + "e" * 32 + "/frozen/candidate"
        else:
            (root / "build-candidate/freeze-manifest.json").write_text(json.dumps(dict(files=[])))
            build["freeze"].update(files=0, manifest_sha256=impact.fingerprint([]))
        (root / "build-candidate/build-receipt.json").write_text(json.dumps(build))
    (folder / "stderr.log").write_text('[ssd.settings] ' + json.dumps(dict(values=native)) + '\n')
    with pytest.raises(ValueError): aggregation.functional_retained(root, cfg, "deepseek", cell)


def test_operator_execute_reuses_exact_child_and_cancellation_protocol(operator, monkeypatch):
    a = operator; controller.operator_preflight(a.args, a.env)
    calls = []
    def start(api, job, sha, parent):
        calls.append(("start", job["device"], sha, parent))
        return dict(id=73, workflow_id=91, controller_sha=sha, job_id=job["job_id"],
                    inputs=dict(selection_parent=parent))
    monkeypatch.setattr(controller.selection_dispatch, "start", start)
    monkeypatch.setattr(controller.selection_dispatch, "wait", lambda *a, **k: dict(operator.child))
    monkeypatch.setattr(controller.selection_dispatch, "collect", lambda *a, **k: calls.append(("collect",)))
    monkeypatch.setattr(controller.selection_dispatch, "cancel", lambda api, child:
        calls.append(("cancel", child["id"])) or dict(cancel_requested=True, cleanup_confirmed=False))
    assert controller.execute(a.args, a.env) == 0
    assert calls[0] == ("start", "cloud-5090", a.env["GITHUB_SHA"], "42-1")
    assert calls[-1] == ("cancel", 73)
    state = load(a.args.artifacts / "dispatch-state.json")
    assert set(state["children"]) == {"cloud-5090"} and state["errors"] == {}
    assert load(a.args.artifacts / "cancellation-requests.json")["cloud-5090-dispatch.json"]["cleanup_confirmed"] is False


def test_operator_execute_refuses_modified_admission_before_any_child(operator):
    a = operator; controller.operator_preflight(a.args, a.env)
    admitted = load(a.args.plan); admitted["jobs"]["cloud-5090"]["job_id"] = "0" * 32
    a.args.plan.write_text(json.dumps(admitted))
    with pytest.raises(ValueError, match="envelope changed"): controller.execute(a.args, a.env)
    assert not a.args.artifacts.exists()


def test_performance_envelope_is_admitted_without_changing_canonical_impact(operator):
    a = operator
    a.envelope["validation_mode"] = contract.PERFORMANCE_MODE
    original = copy.deepcopy(a.envelope["selection"])
    a.write_event()
    assert controller.operator_preflight(a.args, a.env) == 0
    admitted = load(a.args.plan)
    assert admitted["validation_mode"] == contract.PERFORMANCE_MODE
    assert admitted["selection"] == original


def test_failed_terminal_child_retains_exact_raw_once_but_does_not_green_execution(operator, monkeypatch):
    a = operator
    controller.operator_preflight(a.args, a.env)
    a.child["conclusion"] = "failure"
    calls = []
    monkeypatch.setattr(controller.selection_dispatch, "start", lambda api, job, sha, parent:
        dict(id=73, workflow_id=91, controller_sha=sha, job_id=job["job_id"], inputs=dict(selection_parent=parent)))
    monkeypatch.setattr(controller.selection_dispatch, "wait", lambda *a, **k:
        (_ for _ in ()).throw(ValueError("selected child did not succeed")))
    monkeypatch.setattr(controller.selection_dispatch, "collect", lambda api, run, job, output, **k:
        calls.append((run["id"], run["conclusion"], job["job_id"])))
    monkeypatch.setattr(controller, "cancel_children", lambda *a: 0)
    assert controller.execute(a.args, a.env) == 1
    assert calls == [(73, "failure", "f" * 32)]
    state = load(a.args.artifacts / "dispatch-state.json")
    assert state["errors"] and state["children"]["cloud-5090"]["conclusion"] == "failure"


@pytest.mark.parametrize("retained", [True, False])
def test_inconclusive_child_metrics_survive_report_without_success(operator, monkeypatch, retained):
    a = operator
    ready(a)
    a.child["conclusion"] = "failure"
    state = load(a.args.artifacts / "dispatch-state.json")
    state["errors"] = {"cloud-5090": "child exited nonzero"}
    if retained:
        state["retained_failed_children"] = {"cloud-5090": 73}
    a.args.artifacts.joinpath("dispatch-state.json").write_text(json.dumps(state))
    monkeypatch.setattr(controller, "operator_aggregate", lambda *args:
        dict(passed=False, status="inconclusive", linux_scoped_status="inconclusive",
             full_impact_coverage_complete=False, cells=[dict(device="cloud-5090", model="muse",
                 status="inconclusive", functional="passed", numerical="passed", performance=[])], errors=[]))
    assert controller.operator_report(a.args, a.env) == 1
    result = load(a.args.output)
    assert result["status"] == ("inconclusive" if retained else "failed") and result["passed"] is False
    assert result["child_workflow_errors"] == state["errors"]
    assert "inconclusive" in Path(a.env["GITHUB_STEP_SUMMARY"]).read_text()


@pytest.mark.parametrize("fault", ["missing", "inconclusive", "unclean"])
def test_performance_scope_never_hides_missing_or_inconclusive_cell(operator, monkeypatch, fault):
    a = operator
    plan = impact.plan(["engine/ggml/src/ggml-cpu/ssd/cpu-sink.cpp"], POLICY,
                       base=a.plan["base"], head=a.plan["head"], candidate=a.plan["candidate"])
    scope = dict(validation_mode=contract.PERFORMANCE_MODE, execution_devices=["cloud-5090"])
    original = copy.deepcopy(plan)
    models = {cell["model"] for cell in plan["cells"] if cell["device"] == "cloud-5090" and cell["required"]}
    def measured(*args):
        if fault == "unclean":
            raise ValueError("timed scope still active")
        rows = [dict(device="cloud-5090", model=model, status="inconclusive" if fault == "inconclusive" else "passed",
                     functional="passed", numerical="passed", performance=[dict(tier="512", status="inconclusive")],
                     performance_deferred_tiers=["4096"]) for model in models]
        return rows[:-1] if fault == "missing" else rows
    monkeypatch.setattr(aggregation, "device_report", measured)
    result = aggregation.operator_aggregate(plan, POLICY, Path("/unused"), a.envelope["jobs"], {"cloud-5090": a.child}, scope)
    assert result["passed"] is False
    assert result["status"] == ("inconclusive" if fault == "inconclusive" else "failed")
    assert result["linux_scoped_status"] == result["status"] and not result["full_impact_coverage_complete"]
    assert plan == original
    assert all(row["status"] == "deferred-offline" for row in result["cells"] if row["device"] == "local-mbp")


def test_successful_linux512_is_separate_from_required_4k_and_mac(operator, monkeypatch):
    a = operator
    plan = impact.plan(["engine/ggml/src/ggml-cpu/ssd/cpu-sink.cpp"], POLICY,
                       base=a.plan["base"], head=a.plan["head"], candidate=a.plan["candidate"])
    scope = dict(validation_mode=contract.PERFORMANCE_MODE, execution_devices=["cloud-5090"])
    monkeypatch.setattr(aggregation, "device_report", lambda *args: [dict(device="cloud-5090", model=cell["model"],
        status="passed", functional="passed", numerical="passed", performance=[dict(tier="512", status="passed")],
        performance_deferred_tiers=["4096"]) for cell in plan["cells"] if cell["device"] == "cloud-5090" and cell["required"]])
    result = aggregation.operator_aggregate(plan, POLICY, Path("/unused"), a.envelope["jobs"], {"cloud-5090": a.child}, scope)
    assert result["passed"] and result["linux_scoped_status"] == "passed"
    assert result["full_impact_coverage_complete"] is False


def test_summary_shows_both_decode_windows_signed_percent_and_ttft_direction():
    metrics = {name: dict(base=10, candidate=11, change_percent=10) for name in compare.METRICS}
    metrics["ttft_s"] = dict(base=2, candidate=1.8, change_percent=-10)
    pair = dict(tier="512", status="inconclusive", metrics=metrics,
                measurement_identity=dict(prompt_sha256="a" * 64, prompt_ids_sha256="b" * 64,
                                          conditions_sha256="c" * 64))
    report = dict(passed=False, status="inconclusive", full_impact_coverage_complete=False,
        base="d" * 40, head="e" * 40, candidate="f" * 40, cells=[dict(device="cloud-5090", model="muse",
            status="inconclusive", functional="passed", numerical="passed", performance=[pair],
            performance_deferred_tiers=["4096"])])
    stream = io.StringIO()
    controller.operator_summary(stream, report)
    text = stream.getvalue()
    assert "0–128 base tok/s" in text and "128–256 base tok/s" in text
    assert "+10.000%" in text and "-10.000%" in text and "lower TTFT is better" in text
    assert "Request TTFT seconds" in text and "Native prefill tok/s" in text and "Whole decode tok/s" in text
    assert "Declared device scope: **inconclusive**" in text and "deferred tiers: 4096" in text
    assert "template IDs SHA" in text and "conditions/settings SHA" in text and "0/16" in text


@pytest.mark.parametrize("devices", [["local-mbp"], ["cloud-5090", "local-mbp"], ["cloud-5090"]])
def test_summary_describes_mac_footprint_protection_and_diagnostic_rss(devices):
    report = dict(passed=True, full_impact_coverage_complete=False,
                  execution_devices=devices, cells=[])
    stream = io.StringIO()
    controller.operator_summary(stream, report)
    text = stream.getvalue()
    if "local-mbp" in devices:
        assert "Mac memory protection monitors worker physical footprint/process count and host reserve" in text
        assert "RSS is diagnostic only" in text
        assert "does not disable swap or provide a Linux memory.max hard cap" in text
    else:
        assert "Mac memory protection" not in text
        assert "RSS" not in text
    assert "worker RSS/process count" not in text


@pytest.fixture
def performance_raw(functional_raw):
    """Real validators over synthetic SSE/probe bytes; no device or model runs."""
    from test_final import write_artifacts
    root, original, config, _ = functional_raw
    plan = dict(base="a" * 40, candidate="c" * 40)
    preflight = {}
    for side, pid in (("base", 321), ("candidate", 123)):
        config["sides"][side] = copy.deepcopy(config["sides"]["candidate"])
        config["sides"][side]["revision"] = plan[side]
        model = config["sides"][side]["models"]["deepseek"]
        launch = copy.deepcopy(model["smoke"])
        launch["argv"][0] = f"/build/{side}/bin/llama-server"
        launch["preflight_argv"] = [*launch["argv"], "--ssd-telemetry", f"/scratch/{side}/preflight.jsonl"]
        launch["argv"] += ["--ssd-telemetry", f"/scratch/{side}/benchmark512.jsonl"]
        launch["request_timeout_s"] = 300
        model["benchmarks"] = {"512": launch}
        build = root / ("build-" + side)
        build.mkdir(exist_ok=True)
        (build / "build-receipt.json").write_text(json.dumps(dict(binary=dict(
            runtime_path=launch["argv"][0], sha256="a" * 64))))
        parent = root / "deepseek/512"
        parent.mkdir(parents=True, exist_ok=True)
        folder = parent / side
        write_artifacts(folder, "512", dict(device="cloud-5090", model="deepseek", revision=plan[side], request_timeout_s=300))
        for name in ("started.json", "runtime-before.json", "runtime-after.json", "stopped.json"):
            data = json.loads((original / name).read_text().replace("/build/candidate", "/build/" + side))
            if "worker_pid" in data:
                data.update(worker_pid=pid, birth_identity="birth-" + side, scope_identity="scope-" + side)
            if "cmdline" in data:
                data["cmdline"] = launch["argv"]
            (folder / name).write_text(json.dumps(data))
        (folder / "stderr.log").write_bytes((original / "stderr.log").read_bytes())
        receipt = json.loads((original / "runtime-receipt.json").read_text().replace("/build/candidate", "/build/" + side))
        receipt["build"]["revision"] = plan[side]
        receipt["server"] = dict(pid=pid, start_identity="birth-" + side)
        receipt["host"].update(memory_budget_bytes=100, os_page_cache_treatment="uncontrolled")
        save(folder / "comparison-receipt.json", receipt)
        preflight["deepseek", "512", side] = copy.deepcopy(receipt)
        save(folder / "benchmark.json", dict(status="passed", cleanup_confirmed=True, completion_requests=1,
             revision=plan[side], tier="512", model="deepseek", result=load(folder / "result.json")))
    def comparison():
        results = {side: aggregation.run_final.read_result(root / "deepseek/512" / side / "result.json", "512", 300)
                   for side in ("base", "candidate")}
        receipts = {side: load(root / "deepseek/512" / side / "comparison-receipt.json") for side in results}
        pair = compare.compare_pair(results["base"], results["candidate"], receipts["base"], receipts["candidate"], POLICY)
        (root / "deepseek/512/comparison.json").write_text(json.dumps(pair))
        return pair
    comparison()
    return root, config, plan, preflight, comparison


def test_performance_retained_replays_sse_and_both_side_native_probes(performance_raw):
    root, config, plan, preflight, _ = performance_raw
    pair = aggregation.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
    assert pair["passed"] and pair["status"] == "passed"
    assert pair["measurement_identity"]["output_tokens"] == 256
    assert pair["sources"] == plan
    assert pair["evidence"]["base"]["result.events.sse"]["sha256"] == digest(root / "deepseek/512/base/result.events.sse")


def test_performance_inconclusive_keeps_full_verified_pair_values(performance_raw):
    root, config, plan, preflight, rebuild = performance_raw
    folder = root / "deepseek/512/candidate"
    records = [json.loads(line) for line in (folder / "result.events.jsonl").read_text().splitlines()]
    records[-1]["event"]["timings"]["prompt_ms"] *= 1.01
    (folder / "result.events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (folder / "result.events.sse").write_text("".join("data: " + json.dumps(row["event"]) + "\n\n" for row in records))
    result = load(folder / "result.json")
    rebuilt = aggregation.run_final.stream_chat.summarize(records, result["start_ns"], result["end_ns"], result["prompt_ids"], 256)
    aggregation.run_final.final_chat.validate_timings(rebuilt)
    result.update(rebuilt)
    (folder / "result.json").write_text(json.dumps(result))
    arm = load(folder / "benchmark.json"); arm["result"] = result
    (folder / "benchmark.json").write_text(json.dumps(arm))
    rebuild()
    pair = aggregation.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)
    assert pair["status"] == "inconclusive" and not pair["passed"]
    assert pair["observed_slowdowns"] == ["server_prefill_tps"]
    assert pair["metrics"]["server_prefill_tps"]["change_percent"] < 0 and pair["evidence"]["candidate"]


@pytest.mark.parametrize("fault", ["base-resource", "candidate-argv", "receipt-resource", "missing-sse", "unclean", "repeated", "source"])
def test_performance_identity_raw_or_cleanup_failure_never_passes(performance_raw, fault):
    root, config, plan, preflight, _ = performance_raw
    folder = root / "deepseek/512/candidate"
    if fault == "missing-sse":
        (folder / "result.events.sse").unlink()
    elif fault in ("base-resource", "candidate-argv"):
        target = root / "deepseek/512/base/runtime-after.json" if fault == "base-resource" else folder / "runtime-after.json"
        raw = load(target)
        if fault == "base-resource": raw["cgroup_limits"]["memory_max"] = "200"
        else: raw["cmdline"] = ["/other"]
        target.write_text(json.dumps(raw))
    elif fault == "receipt-resource":
        path = folder / "comparison-receipt.json"; raw = load(path)
        raw["host"]["effective_settings"]["native_resource_policy"]["memlock_bytes"] = 2
        path.write_text(json.dumps(raw))
    elif fault == "unclean":
        path = folder / "stopped.json"; raw = load(path); raw["cleanup_confirmed"] = False
        path.write_text(json.dumps(raw))
    else:
        path = folder / "benchmark.json"; raw = load(path)
        raw["completion_requests"] = 2 if fault == "repeated" else 1
        if fault == "source": raw["revision"] = plan["base"]
        path.write_text(json.dumps(raw))
    with pytest.raises((OSError, ValueError)):
        aggregation.performance_retained(root, config, "deepseek", "512", plan, POLICY, preflight)


def cancellation_dispatch(operator):
    a = operator
    a.args.artifacts.mkdir()
    child = dict(id=73, workflow_id=91, controller_sha=a.env["GITHUB_SHA"], job_id="f" * 32,
                 inputs=dict(selection_parent="42-1"))
    save(a.args.artifacts / "cloud-5090-dispatch.json", child)
    return child


def test_cancel_reentry_preserves_first_receipt_and_only_rechecks_exact_closed_child(operator):
    a = operator; cancellation_dispatch(a)
    assert controller.cancel_children(a.args, a.env) == 0
    path = a.args.artifacts / "cancellation-requests.json"; original = path.read_bytes()
    assert controller.cancel_children(a.args, a.env) == 0
    assert path.read_bytes() == original
    recheck = load(next(a.args.artifacts.glob("cancellation-recheck-*.json")))
    result = recheck["results"]["cloud-5090-dispatch.json"]
    assert result["run_id"] == 73 and result["status"] == "completed"
    assert result["prior_outcome_unknown"] is False and result["cleanup_confirmed"] is False
    assert a.calls.count("actions/runs/73") == 2


@pytest.mark.parametrize("state", ["in_progress", "completed"])
def test_cancel_reentry_does_not_repeat_an_ambiguous_request(operator, monkeypatch, state):
    a = operator; cancellation_dispatch(a); posts = []
    def ambiguous(api, child):
        posts.append(child["id"])
        raise RuntimeError("POST response lost")
    monkeypatch.setattr(controller.selection_dispatch, "cancel", ambiguous)
    assert controller.cancel_children(a.args, a.env) == 1
    path = a.args.artifacts / "cancellation-requests.json"; original = path.read_bytes()
    a.child["status"] = state
    assert controller.cancel_children(a.args, a.env) == 1
    assert posts == [73] and path.read_bytes() == original
    recheck = load(next(a.args.artifacts.glob("cancellation-recheck-*.json")))
    assert recheck["results"]["cloud-5090-dispatch.json"]["prior_outcome_unknown"] is True


def test_cancel_claim_without_result_is_uncertain_and_never_reposted(operator, monkeypatch):
    a = operator; child = cancellation_dispatch(a)
    save(a.args.artifacts / "cancellation-cloud-5090-dispatch-attempt.json",
         dict(schema_version=1, parent="42-1", child_sha256=impact.fingerprint(child), run_id=73, cleanup_confirmed=False))
    monkeypatch.setattr(controller.selection_dispatch, "cancel", lambda *a: pytest.fail("must not repeat an uncertain request"))
    assert controller.cancel_children(a.args, a.env) == 1
    assert load(a.args.artifacts / "cancellation-requests.json")["cloud-5090-dispatch.json"]["prior_outcome_unknown"] is True


def test_cancel_reentry_rejects_changed_child_identity(operator):
    a = operator; cancellation_dispatch(a)
    assert controller.cancel_children(a.args, a.env) == 0
    original = (a.args.artifacts / "cancellation-requests.json").read_bytes()
    a.child["head_sha"] = "0" * 40
    assert controller.cancel_children(a.args, a.env) == 1
    assert (a.args.artifacts / "cancellation-requests.json").read_bytes() == original
    recheck = load(next(a.args.artifacts.glob("cancellation-recheck-*.json")))
    assert "identity differs" in recheck["results"]["cloud-5090-dispatch.json"]["error"]


def test_cancel_reentry_does_not_trust_a_prior_receipt_for_another_run(operator):
    a = operator; cancellation_dispatch(a)
    save(a.args.artifacts / "cancellation-requests.json", {
        "cloud-5090-dispatch.json": dict(run_id=74, cancellation_requested=False, cleanup_confirmed=False)})
    assert controller.cancel_children(a.args, a.env) == 1
    recheck = load(next(a.args.artifacts.glob("cancellation-recheck-*.json")))
    assert recheck["results"]["cloud-5090-dispatch.json"]["prior_outcome_unknown"] is True


class MetadataAPI:
    def __init__(self, runs):
        self.runs, self.calls = list(runs), []

    def call(self, path, data=None, method=None):
        self.calls.append((path, data, method))
        if method == "POST":
            return None
        assert path == "actions/runs/73" and data is None
        value = self.runs.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def metadata_observer(a, runs):
    a.args.artifacts.mkdir(exist_ok=True)
    child = dict(id=73, workflow_id=91, controller_sha=a.env["GITHUB_SHA"], job_id="f" * 32)
    api, pauses, now = MetadataAPI(runs), [], [0]
    def sleep(seconds):
        pauses.append(seconds)
        now[0] += seconds
    observed = controller.ChildMetadataAPI(api, child, a.args.artifacts, "test",
                                          sleep=sleep, clock=lambda: now[0])
    return observed, api, pauses


@pytest.mark.parametrize("title", [None, "", "InferSSD reviewed integration smoke"])
@pytest.mark.parametrize("status", ["queued", "requested"])
def test_child_metadata_pending_title_requires_later_full_identity_and_retains_every_get(operator, title, status):
    a = operator
    pending = dict(a.child, display_title=title, status=status, conclusion=None)
    expected = dict(a.child, name="a workflow label is not the run title", actor={"login": "github-actions[bot]"})
    observed, api, pauses = metadata_observer(a, [pending, expected])
    assert observed.call("actions/runs/73") == expected
    rows = [load(path) for path in sorted(a.args.artifacts.glob("child-metadata-*.json"))]
    assert [row["run"] for row in rows] == [pending, expected]
    assert rows[0]["differences"] == {"display_title": dict(expected=a.child["display_title"], actual=title)}
    assert rows[0]["pending_title"] is True and rows[1]["differences"] == {}
    assert rows[1]["pending_title"] is False and pauses == [2]
    assert api.calls == [("actions/runs/73", None, None)] * 2


@pytest.mark.parametrize("field,value", [("id", 74), ("workflow_id", 92), ("workflow_id", True),
    ("path", ".github/workflows/other.yml"), ("event", "push"), ("head_branch", "other"),
    ("head_sha", "0" * 40), ("run_attempt", 2), ("run_attempt", True),
    ("display_title", "integration-" + "e" * 32), ("display_title", "unknown title"),
    ("display_title", " ")])
def test_child_metadata_definite_identity_mismatch_fails_without_wait_or_post(operator, field, value):
    a = operator
    run = dict(a.child, status="queued", conclusion=None, display_title="")
    run[field] = value
    observed, api, pauses = metadata_observer(a, [run])
    with pytest.raises(ValueError, match="identity differs"):
        controller.selection_dispatch.cancel(observed, observed.child)
    row = load(next(a.args.artifacts.glob("child-metadata-*.json")))
    assert row["run"] == run and field in row["differences"] and row["pending_title"] is False
    assert len(api.calls) == 1 and pauses == []


@pytest.mark.parametrize("status,conclusion", [("in_progress", None), ("completed", "failure"),
    ("completed", "success"), ("queued", "failure"), ("waiting", None)])
def test_child_metadata_generic_title_is_not_pending_after_execution_or_conclusion(operator, status, conclusion):
    a = operator
    observed, api, pauses = metadata_observer(a, [dict(a.child, status=status, conclusion=conclusion,
                                                      display_title="InferSSD reviewed integration smoke")])
    with pytest.raises(ValueError, match="identity differs"):
        observed.call("actions/runs/73")
    assert len(api.calls) == 1 and pauses == []


def test_child_metadata_observation_is_finite_and_never_accepts_placeholder(operator):
    a = operator
    pending = dict(a.child, status="queued", conclusion=None, display_title=None)
    observed, api, pauses = metadata_observer(a, [pending] * 6)
    with pytest.raises(TimeoutError, match="finite observation"):
        controller.selection_dispatch.cancel(observed, observed.child)
    assert len(api.calls) == len(list(a.args.artifacts.glob("child-metadata-*.json"))) == 6
    assert pauses == [2] * 5


def test_child_metadata_missing_immutable_field_and_transport_error_do_not_retry(operator):
    a = operator
    run = dict(a.child, status="queued", conclusion=None, display_title="")
    del run["head_sha"]
    observed, api, pauses = metadata_observer(a, [run])
    with pytest.raises(ValueError, match="head_sha"):
        observed.call("actions/runs/73")
    assert len(api.calls) == 1 and pauses == []
    observed, api, pauses = metadata_observer(a, [OSError("synthetic transport failure")])
    with pytest.raises(OSError, match="transport"):
        observed.call("actions/runs/73")
    assert len(api.calls) == 1 and pauses == []


def test_child_metadata_cancel_posts_once_only_after_exact_observed_identity(operator, monkeypatch):
    a = operator
    child = cancellation_dispatch(a)
    pending = dict(a.child, status="queued", conclusion=None, display_title="")
    exact = dict(a.child, status="in_progress", conclusion=None)
    api = MetadataAPI([pending, exact, dict(a.child)])
    monkeypatch.setattr(controller, "API", lambda *args, **kwargs: api)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    assert controller.cancel_children(a.args, a.env) == 1
    original = (a.args.artifacts / "cancellation-requests.json").read_bytes()
    assert load(a.args.artifacts / "cancellation-cloud-5090-dispatch-attempt.json")["child_sha256"] == impact.fingerprint(child)
    assert controller.cancel_children(a.args, a.env) == 0
    assert (a.args.artifacts / "cancellation-requests.json").read_bytes() == original
    assert api.calls == [("actions/runs/73", None, None)] * 2 + [
        ("actions/runs/73/cancel", {}, "POST"), ("actions/runs/73", None, None)]
    assert len(list(a.args.artifacts.glob("child-metadata-*.json"))) == 3


def test_child_metadata_snapshot_failure_cannot_authorize_cancel(operator, monkeypatch):
    observed, api, _ = metadata_observer(operator, [operator.child])
    monkeypatch.setattr(controller, "save", lambda *args: (_ for _ in ()).throw(OSError("snapshot storage failed")))
    with pytest.raises(OSError, match="snapshot storage"):
        controller.selection_dispatch.cancel(observed, observed.child)
    assert api.calls == [("actions/runs/73", None, None)]


def test_child_metadata_wait_uses_returned_id_without_redispatch(operator):
    a = operator
    pending = dict(a.child, status="queued", conclusion=None, display_title="")
    observed, api, pauses = metadata_observer(a, [pending, a.child])
    assert controller.selection_dispatch.wait(observed, observed.child) == a.child
    assert api.calls == [("actions/runs/73", None, None)] * 2 and pauses == [2]


def test_child_metadata_does_not_retry_an_ambiguous_post(operator):
    observed, api, pauses = metadata_observer(operator, [operator.child])
    with pytest.raises(ValueError, match="fully observed exact identity"):
        observed.call("actions/runs/73/cancel", {}, method="POST")
    assert api.calls == []
    observed.call("actions/runs/73")
    def failed_post(path, data=None, method=None):
        api.calls.append((path, data, method))
        raise OSError("ambiguous write")
    api.call = failed_post
    with pytest.raises(OSError, match="ambiguous write"):
        observed.call("actions/runs/73/cancel", {}, method="POST")
    assert api.calls == [("actions/runs/73", None, None), ("actions/runs/73/cancel", {}, "POST")] and pauses == []


@pytest.mark.parametrize("wrong_job", [False, True])
def test_real_canonical_get_retry_keeps_child_identity_and_every_successful_snapshot(
    operator, monkeypatch, capsys, wrong_job
):
    a = operator
    a.args.artifacts.mkdir(exist_ok=True)
    now, pauses, calls = [100.0], [], []

    def sleep(seconds):
        pauses.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(controller.time, "sleep", sleep)
    pending = dict(a.child, status="queued", conclusion=None,
                   display_title="InferSSD reviewed integration smoke")
    exact = dict(a.child, status="queued", conclusion=None)
    if wrong_job:
        exact["display_title"] = "integration-" + "e" * 32
    replies = [pending, 500, exact]
    canonical = controller.private_controller

    def open_request(request, *, timeout):
        assert request.get_method() == "GET" and request.data is None
        calls.append(timeout)
        value = replies.pop(0)
        if value == 500:
            raise urllib.error.HTTPError(request.full_url, 500, "never-log", {}, None)
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(canonical.urllib.request, "urlopen", open_request)
    child = dict(id=73, workflow_id=91, controller_sha=a.env["GITHUB_SHA"], job_id="f" * 32)
    api = canonical.API(POLICY["controller_repository"], "never-log-secret")
    observed = controller.ChildMetadataAPI(api, child, a.args.artifacts, "test")
    if wrong_job:
        with pytest.raises(ValueError, match="identity differs"):
            observed.call("actions/runs/73")
        assert observed.identity_confirmed is False
    else:
        assert observed.call("actions/runs/73") == exact
        assert observed.identity_confirmed is True
    assert calls == [10, 8, 7.75] and pauses == [2, .25]
    snapshots = [load(path) for path in sorted(a.args.artifacts.glob("child-metadata-*.json"))]
    assert [row["run"] for row in snapshots] == [pending, exact]
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [row["decision"] for row in rows] == ["return", "retry", "return"]
    assert rows[1]["http_status"] == 500
    assert "never-log" not in json.dumps(rows)


def test_real_child_get_cannot_extend_the_ten_second_title_budget(operator, monkeypatch):
    a = operator
    a.args.artifacts.mkdir(exist_ok=True)
    now, calls = [100.0], []
    monkeypatch.setattr(controller.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    canonical = controller.private_controller
    pending = dict(a.child, status="queued", conclusion=None, display_title="")

    def open_request(request, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            return io.BytesIO(json.dumps(pending).encode())
        now[0] += timeout
        raise TimeoutError("synthetic read timeout")

    monkeypatch.setattr(canonical.urllib.request, "urlopen", open_request)
    child = dict(id=73, workflow_id=91, controller_sha=a.env["GITHUB_SHA"], job_id="f" * 32)
    observed = controller.ChildMetadataAPI(canonical.API(POLICY["controller_repository"], "synthetic"),
                                          child, a.args.artifacts, "test")
    with pytest.raises(TimeoutError):
        observed.call("actions/runs/73")
    assert calls == [10, 8] and now[0] == 110
    assert observed.identity_confirmed is False
    assert len(list(a.args.artifacts.glob("child-metadata-*.json"))) == 1
