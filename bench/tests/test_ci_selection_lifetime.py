"""Protocol-only cancellation and refusal; no device or performance evidence."""

import copy
from pathlib import Path
import signal
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import impact, selected_controller, selection_dispatch, selection_lifetime
from ci.contract import load, save
from ci.private_controller import context
from ci.stage_session import StageSession
from ci.staged_benchmark import verify_stopped

POLICY = load(Path(__file__).parents[1] / "ci/policy.json")


class ParentAPI:
    def __init__(self):
        self.run = dict(workflow_id=3, path=selection_lifetime.WORKFLOW,
                        event="workflow_dispatch", head_branch="main", head_sha="a" * 40,
                        run_attempt=2, status="in_progress", conclusion=None)
        self.jobs = [dict(name="control", status="in_progress", conclusion=None)]

    def call(self, path):
        if path == "actions/workflows/device-performance.yml":
            return dict(id=3, path=selection_lifetime.WORKFLOW, state="active")
        if path == "actions/runs/1":
            return self.run
        assert path == "actions/runs/1/attempts/2/jobs?per_page=100"
        return dict(total_count=len(self.jobs), jobs=self.jobs)


@pytest.mark.parametrize("field,value", [("head_sha", "b" * 40), ("run_attempt", 3),
    ("status", "completed"), ("conclusion", "cancelled"), ("path", ".github/workflows/other.yml")])
def test_parent_identity_or_lifetime_refuses_before_lease(field, value):
    api = ParentAPI()
    api.run[field] = value
    watch = selection_lifetime.ParentWatch(api, "1-2", "a" * 40)
    with pytest.raises(ValueError):
        watch.start()
    assert watch.thread is None and watch.session is None


def test_publisher_running_does_not_keep_cancelled_control_alive():
    api = ParentAPI()
    api.jobs = [dict(name="control", status="completed", conclusion="cancelled"),
                dict(name="publish", status="in_progress", conclusion=None)]
    with pytest.raises(ValueError):
        selection_lifetime.active_parent(api, "1-2", "a" * 40)


@pytest.mark.parametrize("lost", [False, True])
def test_parent_lost_or_cancelled_interrupts_borrowed_socket(lost):
    api, aborted = ParentAPI(), threading.Event()
    watch = selection_lifetime.ParentWatch(api, "1-2", "a" * 40, interval=.001)
    watch.attach(SimpleNamespace(abort=aborted.set))
    if lost:
        api.call = lambda _: (_ for _ in ()).throw(OSError("parent unavailable"))
    else:
        api.run["conclusion"] = "cancelled"
    # Exercise the exact monitor branch without launching a process or broker.
    thread = threading.Thread(target=watch._run)
    thread.start()
    assert aborted.wait(1)
    thread.join(1)
    with pytest.raises(RuntimeError):
        watch.stop()


def test_abort_shutdown_interrupts_inflight_rpc_and_exposes_eof_to_guardian():
    client, guardian = socket.socketpair()
    session = object.__new__(StageSession)
    session.connection, session.closed = client, False
    received = []
    def read():
        try:
            received.append(client.recv(1))
        except OSError:
            received.append("interrupted")
    reader = threading.Thread(target=read)
    reader.start()
    session.abort()
    reader.join(1)
    guardian.settimeout(1)
    assert not reader.is_alive() and received in ([b""], ["interrupted"]) and guardian.recv(1) == b""
    assert session.closed  # Disconnect has produced no positive cleanup receipt.
    guardian.close()


def test_parent_interrupt_cancels_every_recorded_exact_child(tmp_path, monkeypatch):
    from test_private_controller import event_context
    event, env = event_context(tmp_path)
    identity = context(POLICY, env, event)
    plan = impact.plan(["engine/ggml/src/ggml-cuda/ssd/mxfp4.cpp"], POLICY,
                       base="a" * 40, head="b" * 40, candidate="c" * 40)
    jobs = {d: dict(device=d, job_id=str(i + 1) * 32, source_revision="c" * 40,
                   config_sha256="d" * 64, selection_plan_sha256=impact.fingerprint(plan),
                   control_revision="e" * 40, control_manifest_sha256="f" * 64)
            for i, d in enumerate(POLICY["required_devices"][:2])}
    args = SimpleNamespace(policy=tmp_path / "policy", control=tmp_path / "control",
                           plan=tmp_path / "plan.json", artifacts=tmp_path / "artifacts")
    save(args.plan, dict(selection=plan, jobs=jobs, controller=identity,
        policy_sha256=impact.fingerprint(POLICY), control_revision="e" * 40))
    monkeypatch.setattr(selected_controller, "environment", lambda *a: (POLICY, event, identity))
    monkeypatch.setattr(selected_controller, "bind_context", lambda *a: None)
    monkeypatch.setattr(selected_controller, "source_manifest", lambda *a: "e" * 40)
    monkeypatch.setattr(selected_controller, "API", lambda *a, **k: object())
    monkeypatch.setattr(selection_dispatch, "start", lambda api, job, sha, parent:
        dict(id=int(job["job_id"][0]), workflow_id=3, controller_sha=sha,
             job_id=job["job_id"], inputs=dict(selection_parent=parent)))
    cancelled = []
    monkeypatch.setattr(selection_dispatch, "cancel", lambda api, child:
        cancelled.append(child["id"]) or dict(cleanup_confirmed=False))
    def terminate(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(selection_dispatch, "wait", terminate)
    previous = signal.getsignal(signal.SIGTERM)
    env["GITHUB_TOKEN"] = "synthetic-private-only"
    with pytest.raises(KeyboardInterrupt):
        selected_controller.execute(args, env)
    assert cancelled == [1, 2]
    assert signal.getsignal(signal.SIGTERM) == previous
    assert load(args.artifacts / "dispatch-state.json")["children"] == {}
    assert all(x["cleanup_confirmed"] is False for x in
               load(args.artifacts / "cancellation-requests.json").values())


def test_cancel_rechecks_exact_child_identity_before_post():
    class API:
        def call(self, path, data=None, method=None):
            assert method is None
            return dict(workflow_id=3, path=selection_dispatch.WORKFLOW,
                event="workflow_dispatch", head_branch="main", head_sha="b" * 40,
                run_attempt=1, display_title="integration-" + "f" * 32)
    with pytest.raises(ValueError):
        selection_dispatch.cancel(API(), dict(id=4, workflow_id=3,
            controller_sha="a" * 40, job_id="f" * 32))


def test_manual_operator_artifact_cannot_satisfy_parent(tmp_path, monkeypatch):
    from ci import selected_aggregate
    root = tmp_path / ("f" * 32)
    root.mkdir()
    save(root / "integration.json", dict(status="passed", authorization_path="operator-root-staged",
                                         selection_parent=""))
    save(tmp_path / "dispatch.json", dict(selection_parent=""))
    monkeypatch.setattr(selected_aggregate, "sealed", lambda *a: None)
    with pytest.raises(ValueError, match="manual operator"):
        selected_aggregate.device_report(root, {}, POLICY, {}, {})


def test_publication_reloads_current_pr_after_raw_aggregation(tmp_path, monkeypatch):
    from test_private_controller import event_context, pr
    event, env = event_context(tmp_path)
    env.update(DEVICE_CHECK_TOKEN="synthetic-app-only", DEVICE_CHECK_APP_ID="88",
               GITHUB_TOKEN="synthetic-private-read")
    identity = context(POLICY, env, event)
    plan = impact.plan(["README.md"], POLICY, base="a" * 40,
                       head="b" * 40, candidate="c" * 40)
    admitted = dict(pr=12, base="a" * 40, head="b" * 40, candidate="c" * 40,
        authorization={}, controller=identity, selection=plan, jobs={}, normal_ci={"id": 1},
        control_revision="d" * 40, policy_sha256=impact.fingerprint(POLICY))
    args = SimpleNamespace(policy=Path(__file__).parents[1] / "ci/policy.json",
        control=tmp_path / "control", plan=tmp_path / "plan.json",
        artifacts=tmp_path / "artifacts", output=tmp_path / "aggregate.json")
    save(args.plan, admitted)
    args.artifacts.mkdir()
    save(args.artifacts / "dispatch-state.json", dict(selection_plan_sha256=impact.fingerprint(plan),
                                                     errors={}, children={}))
    state = dict(aggregated=False, reads=0)
    class API:
        def __init__(self, *args, **kwargs): pass
        def call(self, path, data=None, method=None):
            if path == "pulls/12":
                state["reads"] += 1
                return dict(pr(), fresh=state["aggregated"])
            assert path == "check-runs"
            return dict(app=dict(id=88), head_sha=data["head_sha"], external_id=data["external_id"])
    def aggregate(*args):
        state["aggregated"] = True
        return dict(passed=True)
    def admission(api, value, policy, head):
        assert value["fresh"], "publication reused the pre-aggregation PR object"
        return admitted
    monkeypatch.setattr(selected_controller, "API", API)
    monkeypatch.setattr(selected_controller, "source_manifest", lambda *a: "d" * 40)
    monkeypatch.setattr(selected_controller, "aggregate", aggregate)
    monkeypatch.setattr(selected_controller, "admission", admission)
    monkeypatch.setattr(selected_controller, "current_ci", lambda *a: {"id": 1})
    monkeypatch.setattr(selected_controller, "changed_paths", lambda *a: ["README.md"])
    assert selected_controller.finalize(args, env) == 0
    assert state["reads"] == 2


def mac_stop():
    job = "a" * 32
    return dict(running=False, state="failed", reason="requested stop", returncode=0,
        job_id=job, worker_uid=62002, job_root="/Library/InferSSD-CI/jobs/" + job,
        scope_identity="/Library/InferSSD-CI/jobs/" + job, cleanup_confirmed=True,
        cleanup=dict(confirmed=True, remaining=[], worker_user_domain_retired=dict(returncode=0)))


def test_known_mac_requested_stop_is_not_a_resource_failure():
    assert verify_stopped(mac_stop()) == "mac-requested-stop"


@pytest.mark.parametrize("case", ["STOP", "stage deadline", "process or memory reserve budget",
                                  "code", "remaining", "domain", "scope", "quarantine"])
def test_mac_stop_never_accepts_other_failed_or_unclean_status(case):
    value = copy.deepcopy(mac_stop())
    if case == "code": value["returncode"] = 1
    elif case == "remaining": value["cleanup"]["remaining"] = [42]
    elif case == "domain": value["cleanup"]["worker_user_domain_retired"]["returncode"] = 1
    elif case == "scope": value["scope_identity"] = "different"
    elif case == "quarantine": value["state"] = "quarantined"
    else: value["reason"] = case
    with pytest.raises(ValueError):
        verify_stopped(value)
