"""Security admission negatives use synthetic API/broker records, no device execution."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import load, policy_read

from ci import isolated_device as isolated
from ci import private_controller as controller

POLICY = Path(__file__).parents[1] / "ci/policy.json"


def policy():
    return policy_read(load(POLICY))


def pr():
    repo = {"full_name": "wici-ai/inferssd"}
    return {
        "number": 12,
        "state": "open",
        "draft": False,
        "mergeable": True,
        "user": {"login": "author"},
        "head": {"repo": repo, "ref": "device-ci/approved/topic", "sha": "b" * 40},
        "base": {"repo": repo, "ref": "main", "sha": "a" * 40},
        "merge_commit_sha": "c" * 40,
    }


class API:
    def __init__(self):
        self.branch = {"protected": True, "commit": {"sha": "b" * 40}}
        self.reviews = [
            {
                "id": 17,
                "state": "APPROVED",
                "body": "[device-ci-approved:" + "b" * 40 + "]",
                "commit_id": "b" * 40,
                "user": {"login": "reviewer"},
            }
        ]
        self.parents = ["a" * 40, "b" * 40]
        self.run = {
            "id": 91,
            "run_attempt": 2,
            "workflow_id": 5,
            "path": ".github/workflows/ci.yml",
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "head_sha": "b" * 40,
            "pull_requests": [
                {"number": 12, "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40}}
            ],
        }
        self.calls = []
        self.permission = "write"

    def call(self, path, data=None, method=None):
        self.calls.append((path, data))
        if path.startswith("branches/"):
            return self.branch
        if path.startswith("commits/"):
            return {"parents": [{"sha": s} for s in self.parents]}
        if "/reviews?" in path:
            return self.reviews
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        if path == "actions/workflows/ci.yml":
            return {"id": 5}
        if path.startswith("actions/workflows/5/runs?"):
            return {"workflow_runs": [self.run]}
        if path == "actions/runs/91":
            return self.run
        raise AssertionError(path)


def admitted(api=None):
    return controller.admission(api or API(), pr(), policy(), "b" * 40)


def test_exact_reviewed_staging_admission_records_authorization():
    plan = admitted()
    assert plan["base"] == "a" * 40 and plan["candidate"] == "c" * 40
    assert plan["authorization"]["reviews"] == [
        {"id": 17, "reviewer": "reviewer", "head": "b" * 40}
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["head"]["repo"].update(full_name="external/fork"),
        lambda p: p["head"].update(ref="arbitrary-internal-branch"),
        lambda p: p["head"].update(sha="d" * 40),
        lambda p: p.update(mergeable=False),
        lambda p: p.update(draft=True),
        lambda p: p.update(state="closed"),
        lambda p: p["base"].update(ref="other"),
    ],
)
def test_external_or_unreviewed_candidate_is_never_admitted(mutation):
    value = copy.deepcopy(pr())
    mutation(value)
    api = API()
    with pytest.raises(ValueError):
        controller.admission(api, value, policy(), "b" * 40)
    assert api.calls == []


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.branch.update(protected=False),
        lambda a: a.branch["commit"].update(sha="d" * 40),
        lambda a: a.parents.reverse(),
        lambda a: a.reviews[0].update(commit_id="d" * 40),
        lambda a: a.reviews[0]["user"].update(login="author"),
        lambda a: a.reviews.append(
            {
                "id": 18,
                "state": "CHANGES_REQUESTED",
                "commit_id": "b" * 40,
                "user": {"login": "reviewer"},
            }
        ),
        lambda a: setattr(a, "permission", "read"),
    ],
)
def test_protection_parent_review_and_permission_fail_closed(change):
    api = API()
    change(api)
    with pytest.raises(ValueError):
        controller.admission(api, pr(), policy(), "b" * 40)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(conclusion="failure"),
        lambda r: r.update(status="in_progress"),
        lambda r: r.update(workflow_id=6),
        lambda r: r.update(path=".github/workflows/fake.yml"),
        lambda r: r["pull_requests"][0]["base"].update(sha="d" * 40),
        lambda r: r["pull_requests"][0]["head"].update(sha="d" * 40),
        lambda r: r.update(head_sha="d" * 40),
    ],
)
def test_latest_hosted_ci_must_bind_current_base_workflow_and_success(change):
    api = API()
    change(api.run)
    with pytest.raises(ValueError):
        controller.current_ci(api, admitted(), policy())


def fork_run(api):
    # GitHub's run for a fork PR names the fork and links no PR.
    api.run.update(pull_requests=[], head_repository={"full_name": "contributor/inferssd"})
    return dict(admitted(), head_repository="contributor/inferssd")


def test_fork_hosted_ci_is_bound_by_the_fork_and_exact_head():
    api = API()
    plan = fork_run(api)
    assert controller.current_ci(api, plan, policy())["head"] == "b" * 40


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(head_repository={"full_name": "someone-else/inferssd"}),
        lambda r: r.update(head_repository=None),
        lambda r: r.update(head_sha="c" * 40),
        lambda r: r.update(pull_requests=[{"number": 12, "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40}}]),
        lambda r: r.update(conclusion="failure"),
        lambda r: r.update(event="pull_request_target"),
    ],
)
def test_fork_hosted_ci_refuses_another_fork_head_or_result(change):
    api = API()
    plan = fork_run(api)
    change(api.run)
    with pytest.raises(ValueError):
        controller.current_ci(api, plan, policy())


def test_same_repository_pr_still_needs_the_pr_link():
    api = API()
    api.run.update(pull_requests=[], head_repository={"full_name": "wici-ai/inferssd"})
    with pytest.raises(ValueError):
        controller.current_ci(api, dict(admitted(), head_repository="wici-ai/inferssd"), policy())


def event_context(tmp_path):
    p = policy()
    event = {
        "repository": {"private": True, "full_name": p["controller_repository"]},
        "sender": {"login": "maintainer"},
        "inputs": {"pull_request": "12", "head_sha": "b" * 40},
    }
    env = {
        "GITHUB_REPOSITORY": p["controller_repository"],
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": p["controller_ref"],
        "GITHUB_WORKFLOW_REF": p["controller_repository"]
        + "/"
        + p["controller_workflow"]
        + "@"
        + p["controller_ref"],
        "GITHUB_WORKFLOW_SHA": "d" * 40,
        "GITHUB_SHA": "d" * 40,
        "GITHUB_ACTOR": "maintainer",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    path = tmp_path / "event"
    path.write_text(json.dumps(event))
    env["GITHUB_EVENT_PATH"] = str(path)
    return event, env


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_REPOSITORY", "wici-ai/inferssd"),
        ("GITHUB_REF", "refs/heads/unreviewed"),
        ("GITHUB_WORKFLOW_SHA", "e" * 40),
        ("GITHUB_EVENT_NAME", "workflow_run"),
    ],
)
def test_public_wrong_ref_or_chained_trigger_cannot_admit(tmp_path, key, value):
    event, env = event_context(tmp_path)
    env[key] = value
    with pytest.raises(ValueError):
        controller.context(policy(), env, event)


def test_current_run_attempt_cannot_replay_older_plan(tmp_path):
    event, env = event_context(tmp_path)
    plan = {"controller": controller.context(policy(), env, event)}
    env["GITHUB_RUN_ATTEMPT"] = "2"
    with pytest.raises(ValueError, match="another controller"):
        controller.bind_context(plan, policy(), env)


def test_check_identity_binds_entire_plan_and_controller_attempt(tmp_path):
    event, env = event_context(tmp_path)
    plan = dict(admitted(), controller=controller.context(policy(), env, event))
    first = controller.check_body(plan, {"passed": True}, 123)
    plan["base"] = "e" * 40
    second = controller.check_body(plan, {"passed": True}, 123)
    assert (
        first["head_sha"] == plan["head"]
        and first["external_id"] != second["external_id"]
    )
    assert first["_expected_app_id"] == 123 and "/42/attempts/1" in first["details_url"]


def description():
    return {
        "schema_version": 1,
        "nonce": "challenge",
        "broker_uid": 0,
        "controller_uid": 501,
        "execution_uid": 601,
        "network": "loopback-only",
        "candidate_access": "sandboxed-stages",
        "proof_sha256": "a" * 64,
        "gates": {gate: True for gate in isolated.REQUIRED_GATES},
    }


def test_ephemeral_or_whole_driver_lower_uid_is_not_an_isolation_claim():
    config = {
        "isolation": {
            "backend": "external-reviewed-broker-v1",
            "proof_sha256": "a" * 64,
        }
    }
    value = description()
    assert isolated.isolation(config, "challenge", value, 501) == value
    for field, wrong in [
        ("execution_uid", 501),
        ("execution_uid", 0),
        ("nonce", "replay"),
        ("candidate_access", "whole-driver-worker"),
        ("network", "unrestricted"),
    ]:
        changed = copy.deepcopy(value)
        changed[field] = wrong
        with pytest.raises(ValueError):
            isolated.isolation(config, "challenge", changed, 501)
    for gate in isolated.REQUIRED_GATES:
        changed = copy.deepcopy(value)
        changed["gates"][gate] = False
        with pytest.raises(ValueError):
            isolated.isolation(config, "challenge", changed, 501)


def test_public_workflow_cannot_schedule_physical_runner_and_operator_report_has_no_source_writer():
    root = Path(__file__).parents[2]
    public = (root / ".github/workflows/device-performance.yml").read_text()
    private = (root / "bench/ci/controller/device-performance.yml").read_text()
    assert "self-hosted" not in public and "workflow_run:" not in public
    assert "prepared_envelope:" in private and "environment: physical-devices" in private
    assert "SOURCE_READ_TOKEN" not in private and "DEVICE_CHECK_TOKEN" not in private and "CHECK_APP" not in private
    device = (root / "bench/ci/controller/integration-smoke.workflow.yml").read_text()
    assert (
        "source-data" not in device
        and "SOURCE_READ_TOKEN" not in device
        and "CHECK_APP" not in device
    )
    assert "driver.py" not in device and "staged_integration.py" in device
    assert "selected_controller.py execute" in private
    assert "selected_controller.py operator_preflight" in private and "selected_controller.py operator_report" in private
    assert "if: always() &&" in private[private.index("\n  report:"):]


def test_default_broker_template_refuses_execution():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "ci/controller/broker-unprovisioned.py"),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 78 and b"refused" in result.stderr


def test_finalizer_publishes_only_current_exact_plan_with_dedicated_app(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from ci.contract import digest

    event, env = event_context(tmp_path)
    control = tmp_path / "control"
    control.mkdir()
    source = control / "pinned.py"
    source.write_text("trusted bytes\n")
    (control / "control-source.json").write_text(
        json.dumps({"revision": "e" * 40, "files": {"pinned.py": digest(source)}})
    )
    api = API()
    plan = dict(
        admitted(),
        code=False,
        controller=controller.context(policy(), env, event),
        control_revision="e" * 40,
        policy_sha256=digest(POLICY),
        normal_ci=controller.current_ci(api, admitted(), policy()),
    )
    plan_path = tmp_path / "plan"
    plan_path.write_text(json.dumps(plan))
    env.update(
        DEVICE_CHECK_TOKEN="synthetic-not-a-credential", DEVICE_CHECK_APP_ID="123"
    )
    replies = []
    original = api.call

    def call(path, data=None, method=None):
        if path == "pulls/12":
            return pr()
        if path == "check-runs":
            replies.append(data)
            return dict(data, app={"id": 123})
        return original(path, data, method)

    api.call = call
    monkeypatch.setattr(controller, "API", lambda *_: api)
    args = SimpleNamespace(
        control=control,
        policy=POLICY,
        plan=plan_path,
        artifacts=tmp_path / "artifacts",
        output=tmp_path / "result",
    )
    assert controller.finalize(args, env) == 0
    assert (
        replies[0]["conclusion"] == "success" and replies[0]["head_sha"] == plan["head"]
    )
    # A changed current base yields a failure, rather than reusing the earlier success.
    api.parents = ["f" * 40, "b" * 40]
    args.output = tmp_path / "failed"
    assert controller.finalize(args, env) == 1
    assert replies[-1]["conclusion"] == "failure"
    # A replayed attempt is rejected before even reaching the publication API.
    env["GITHUB_RUN_ATTEMPT"] = "2"
    before = len(replies)
    with pytest.raises(ValueError, match="another controller"):
        controller.finalize(args, env)
    assert len(replies) == before


def test_export_origin_manifest_rejects_changed_control_bytes(tmp_path):
    from ci.contract import digest

    path = tmp_path / "trusted.py"
    path.write_text("control\n")
    (tmp_path / "control-source.json").write_text(
        json.dumps({"revision": "a" * 40, "files": {"trusted.py": digest(path)}})
    )
    assert controller.source_manifest(tmp_path) == "a" * 40
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="reviewed manifest"):
        controller.source_manifest(tmp_path)


def test_another_maintainer_requesting_changes_blocks_existing_approval():
    api = API()
    api.reviews.append(
        {
            "id": 18,
            "state": "CHANGES_REQUESTED",
            "commit_id": "b" * 40,
            "user": {"login": "another-maintainer"},
        }
    )
    with pytest.raises(ValueError, match="outstanding"):
        controller.admission(api, pr(), policy(), "b" * 40)


def test_ordinary_code_review_without_explicit_physical_sha_authorization_is_not_admitted():
    api = API()
    api.reviews[0]["body"] = "Code looks fine, ordinary approval only"
    with pytest.raises(ValueError, match="approve this exact"):
        controller.admission(api, pr(), policy(), "b" * 40)


def test_hosted_source_pack_is_data_only_and_binds_real_merge_parents(tmp_path):
    import subprocess

    from ci.contract import digest

    from ci import pack_sources

    source = tmp_path / "source"
    source.mkdir()

    def git(*argv):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-C",
                str(source),
                *argv,
            ],
            text=True,
        ).strip()

    git("init", "-b", "main")
    (source / "CMakeLists.txt").write_text(
        "This candidate must never be executed by preparation\n"
    )
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "topic")
    (source / "candidate.txt").write_text("untrusted data only\n")
    git("add", ".")
    git("commit", "-m", "head")
    head = git("rev-parse", "HEAD")
    git("checkout", "main")
    git("merge", "--no-ff", "-m", "candidate", "topic")
    candidate = git("rev-parse", "HEAD")
    admission = tmp_path / "admission"
    admission.write_text(
        json.dumps({"base": base, "head": head, "candidate": candidate})
    )
    bundle = tmp_path / "source.bundle"
    plan = tmp_path / "plan"
    pack_sources.pack(source, admission, bundle, plan)
    assert load(plan)["source_bundle_sha256"] == digest(bundle)
    refs = subprocess.check_output(
        ["git", "bundle", "list-heads", str(bundle)], text=True
    )
    assert all(
        "refs/inferssd-ci/" + name in refs for name in ("base", "head", "candidate")
    )
    broken = tmp_path / "broken"
    broken.write_text(json.dumps({"base": head, "head": base, "candidate": candidate}))
    with pytest.raises(ValueError, match="parents"):
        pack_sources.pack(source, broken, tmp_path / "no-bundle", tmp_path / "no-plan")
