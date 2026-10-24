"""Operator/report-only contracts. Synthetic API/raw only; no physical claims."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import impact, selected_controller as controller, selected_aggregate as aggregation
from ci import selection_contract as contract
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
