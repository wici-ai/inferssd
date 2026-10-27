"""Review entry contracts; no gh, SSH, source execution or device operations."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import review_operator as entry

POLICY = json.loads(
    (Path(__file__).resolve().parents[1] / "ci/policy.json").read_text()
)
REPO = POLICY["repository"]
BASE, HEAD, CANDIDATE = "a" * 40, "b" * 40, "c" * 40


class API:
    def __init__(self):
        self.pr = dict(
            number=58,
            state="open",
            draft=False,
            user={"login": "maintainer"},
            head={"sha": HEAD, "repo": {"full_name": REPO}},
            base={"sha": BASE, "repo": {"full_name": REPO}},
            merge_commit_sha=CANDIDATE,
        )
        self.reviews = [
            dict(
                id=71,
                user={"login": "maintainer"},
                state="COMMENTED",
                commit_id=HEAD,
                body="/ci test",
                submitted_at=datetime.now(timezone.utc).isoformat(),
            )
        ]
        self.permission = "write"
        self.run = dict(
            id=87,
            workflow_id=45,
            path=entry.prepare.WORKFLOW,
            event="workflow_dispatch",
            head_branch="main",
            head_sha="d" * 40,
            run_attempt=1,
            actor={"id": 9},
            status="in_progress",
            html_url="https://example.invalid/run/87",
        )
        self.posts = []

    def call(self, path, data=None):
        if data is not None:
            self.posts.append((path, data))
            return None
        if "/collaborators/" in path:
            return {"permission": self.permission}
        if path == f"repos/{REPO}/pulls/58":
            return copy.deepcopy(self.pr)
        if path.startswith(f"repos/{REPO}/pulls/58/reviews?"):
            page = int(path.rsplit("page=", 1)[1])
            return copy.deepcopy(self.reviews[(page - 1) * 100 : page * 100])
        if path.startswith(f"repos/{REPO}/pulls?state=open"):
            return [copy.deepcopy(self.pr)]
        if path == f"repos/{REPO}/git/commits/{CANDIDATE}":
            return {"sha": CANDIDATE, "parents": [{"sha": BASE}, {"sha": HEAD}]}
        if "/actions/runs/" in path:
            return copy.deepcopy(self.run)
        raise AssertionError(path)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(entry, "current_ci", lambda *_: {"id": 12, "attempt": 1})
    monkeypatch.setattr(
        entry.prepare, "access", lambda *_: ({"login": "maintainer"}, "d" * 40)
    )
    return API()


@pytest.fixture
def args(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    inputs = tmp_path / "profiles.json"
    inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "devices": {
                    d: {"ssh": {"sudo_password_required": False}}
                    for d in entry.prepare.DEVICES
                }
            }
        )
    )
    return SimpleNamespace(
        source=tmp_path,
        inputs=inputs,
        policy=policy,
        state=tmp_path,
        gh="unused-gh",
        fetch_remote="ssdash",
        max_authorization_age_s=86400,
        run_timeout_s=14400,
    )


def record(api, args):
    value = entry.authorization(api, POLICY, 58, 71, HEAD)
    value.update(
        inputs_sha256=entry.digest(args.inputs), policy_sha256=entry.digest(args.policy)
    )
    path = args.state / (entry.key(REPO, 58, api.reviews[0]) + ".json")
    entry.save_record(path, value, "queued")
    return path, value


def test_exact_comment_self_pr_authorization(api):
    value = entry.authorization(api, POLICY, 58, 71, HEAD)
    assert value["reviewer"] == api.pr["user"]["login"]
    assert value["head"] == HEAD and value["candidate"] == CANDIDATE


@pytest.mark.parametrize(
    "body",
    [
        "/ci test extra",
        "> /ci test",
        "```\n/ci test\n```",
        "/ci test\n/ci cancel",
        "/ci linux",
        "please run tests",
    ],
)
def test_incidental_or_ambiguous_text_refuses(api, body):
    api.reviews[0]["body"] = body
    with pytest.raises(entry.Stale):
        entry.authorization(api, POLICY, 58, 71, HEAD)


@pytest.mark.parametrize(
    "fault",
    ["head", "commit", "closed", "draft", "wrong-base-repo", "permission", "dismissed"],
)
def test_source_review_permission_changes_refuse(api, fault):
    if fault == "head":
        api.pr["head"]["sha"] = "e" * 40
    elif fault == "commit":
        api.reviews[0]["commit_id"] = "e" * 40
    elif fault == "closed":
        api.pr["state"] = "closed"
    elif fault == "draft":
        api.pr["draft"] = True
    elif fault == "wrong-base-repo":
        api.pr["base"]["repo"]["full_name"] = "external/fork"
    elif fault == "permission":
        api.permission = "read"
    else:
        api.reviews[0]["state"] = "DISMISSED"
    with pytest.raises(entry.Stale):
        entry.authorization(api, POLICY, 58, 71, HEAD)


@pytest.mark.parametrize(
    "state,body", [("COMMENTED", "/ci cancel"), ("CHANGES_REQUESTED", "stop")]
)
def test_later_maintainer_can_revoke(api, state, body):
    api.reviews.append(dict(api.reviews[0], id=72, state=state, body=body))
    with pytest.raises(entry.Stale, match="cancelled"):
        entry.authorization(api, POLICY, 58, 71, HEAD)


def test_expired_authorization(api):
    with pytest.raises(entry.Stale, match="expired"):
        entry.authorization(api, POLICY, 58, 71, HEAD, now=9e9)


def test_hosted_ci_pending_is_persistently_queued(api, args, monkeypatch):
    path, value = record(api, args)

    def pending(*_):
        raise ValueError("normal hosted CI has not completed")

    monkeypatch.setattr(entry, "current_ci", pending)
    monkeypatch.setattr(
        entry.prepare, "prepare", lambda *_: pytest.fail("must not prepare")
    )
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "queued" and "hosted CI" in value["waiting"]


def test_exact_head_changes_during_staging_never_dispatch(api, args, monkeypatch):
    path, value = record(api, args)
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: None)

    def stage(*_, **__):
        api.pr["head"]["sha"] = "e" * 40

    monkeypatch.setattr(entry.prepare, "stage", stage)
    monkeypatch.setattr(
        entry.prepare, "dispatch", lambda *_: pytest.fail("stale dispatch")
    )
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "stale" and value["no_automatic_retry"] is True


def test_prepared_then_staged_then_fresh_dispatch(api, args, monkeypatch):
    path, value = record(api, args)
    calls = []
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(
        entry.prepare, "prepare", lambda *_, **__: calls.append("prepare")
    )
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: calls.append("stage"))

    def dispatch(*_, **__):
        calls.append("dispatch")
        return api.run

    monkeypatch.setattr(entry.prepare, "dispatch", dispatch)
    entry.advance(args, api, POLICY, path, value)
    assert calls == ["prepare", "stage", "dispatch"] and value["status"] == "running"


@pytest.mark.parametrize("devices", [
    ["cloud-5090", "dgx-spark"], ["local-mbp"],
    ["cloud-5090", "dgx-spark", "local-mbp"],
])
def test_review_prepares_only_explicit_current_profile_scope(api, args, monkeypatch, devices):
    args.inputs.write_text(json.dumps(dict(schema_version=1, devices={
        device: dict(ssh=dict(sudo_password_required=False)) for device in reversed(devices)
    })))
    path, value = record(api, args)
    prepared = []
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(entry.prepare, "prepare", lambda actual, **_: prepared.append(actual.execution_devices))
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: api.run)
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "running" and prepared == [devices]


@pytest.mark.parametrize("devices", [{}, {"unknown": {}}, ["local-mbp"]])
def test_review_refuses_invalid_profile_scope_before_prepare(api, args, monkeypatch, devices):
    args.inputs.write_text(json.dumps(dict(schema_version=1, devices=devices)))
    path, value = record(api, args)
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: pytest.fail("invalid scope fetch"))
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: pytest.fail("invalid scope prepare"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "interrupted" and "device profiles" in value["error"]


def test_review_refuses_changed_profile_scope_before_prepare(api, args, monkeypatch):
    path, value = record(api, args)
    args.inputs.write_text(json.dumps(dict(schema_version=1, devices={
        "local-mbp": dict(ssh=dict(sudo_password_required=False))
    })))
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: pytest.fail("changed profile fetch"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "stale" and "profile/policy changed" in value["error"]


def test_background_never_prompts_or_stores_sudo_password(api, args, monkeypatch):
    args.inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "devices": {
                    d: {"ssh": {"sudo_password_required": True}}
                    for d in entry.prepare.DEVICES
                }
            }
        )
    )
    path, value = record(api, args)
    monkeypatch.setattr(
        entry.prepare, "stage", lambda *_: pytest.fail("password fallback")
    )
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "queued" and "NOPASSWD" in value["waiting"]


def test_repeated_event_is_not_dispatched_again(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(path, value, "completed", conclusion="success")
    monkeypatch.setattr(entry, "advance", lambda *_: pytest.fail("duplicate event"))
    entry.tick(args, api)
    assert (
        len(list(args.state.glob("*.json"))) == 3
    )  # profile, policy, one durable event


def test_removed_command_does_not_queue_or_rewrite_history(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(path, value, "completed", conclusion="success")
    retained = path.read_bytes()
    api.reviews[0]["body"] = "/ci linux"
    api.reviews.append(dict(api.reviews[0], id=72))
    monkeypatch.setattr(entry, "advance", lambda *_: pytest.fail("old command replay"))
    entry.tick(args, api)
    assert path.read_bytes() == retained
    assert len(list(args.state.glob("*.json"))) == 3


def test_fresh_test_review_gets_new_record_and_preserves_old_result(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(path, value, "completed", conclusion="success")
    retained = path.read_bytes()
    api.reviews[0]["body"] = "/ci linux"
    fresh = dict(api.reviews[0], id=72, body="/ci test")
    api.reviews.append(fresh)
    advanced = []
    monkeypatch.setattr(entry, "advance", lambda *values: advanced.append(values[-1]))
    entry.tick(args, api)
    assert path.read_bytes() == retained
    assert [row["review_id"] for row in advanced] == [72]
    assert (args.state / (entry.key(REPO, 58, fresh) + ".json")).is_file()


def test_interrupted_ambiguous_dispatch_never_reposts(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(
        path, value, "dispatching", prepared=str(args.state / "missing-prepared")
    )
    monkeypatch.setattr(
        entry.prepare, "dispatch", lambda *_: pytest.fail("second dispatch")
    )
    entry.tick(args, api)
    assert entry.load(path)["status"] == "interrupted"


def test_cancel_only_recorded_parent_and_no_cleanup_claim(api, args):
    path, value = record(api, args)
    entry.save_record(
        path,
        value,
        "running",
        run=copy.deepcopy(api.run),
        dispatched_at=entry.time.time(),
    )
    api.pr["head"]["sha"] = "e" * 40
    entry.monitor(args, api, POLICY, path, value)
    entry.monitor(args, api, POLICY, path, value)
    assert api.posts == [
        (f"repos/{POLICY['controller_repository']}/actions/runs/87/cancel", {})
    ]
    assert value["cleanup_confirmed"] is False and value["status"] == "cancelling"


def test_changed_run_attempt_cannot_cancel_other_attempt(api, args):
    path, value = record(api, args)
    entry.save_record(
        path,
        value,
        "running",
        run=copy.deepcopy(api.run),
        dispatched_at=entry.time.time(),
    )
    api.run["run_attempt"] = 2
    with pytest.raises(ValueError, match="identity"):
        entry.monitor(args, api, POLICY, path, value)
    assert not api.posts


def test_transient_cancel_failure_retries_same_exact_parent_until_ack(api, args):
    path, value = record(api, args)
    entry.save_record(
        path,
        value,
        "running",
        run=copy.deepcopy(api.run),
        dispatched_at=entry.time.time(),
    )
    api.pr["head"]["sha"] = "e" * 40
    call = api.call

    def transient(endpoint, data=None):
        result = call(endpoint, data)
        if endpoint.endswith("/cancel") and len(api.posts) == 1:
            raise RuntimeError("transient cancellation transport failure")
        return result

    api.call = transient
    entry.monitor(args, api, POLICY, path, value)
    assert value["cancel_attempts"] == 1 and "cancellation_error" in value
    entry.monitor(args, api, POLICY, path, value)
    entry.monitor(args, api, POLICY, path, value)
    endpoint = f"repos/{POLICY['controller_repository']}/actions/runs/87/cancel"
    assert api.posts == [(endpoint, {}), (endpoint, {})]
    assert value["cancel_attempts"] == 2 and value["cancellation_requested"] is True
    assert len(value["cancel_errors"]) == 1 and "cancellation_error" not in value
    assert value["cleanup_confirmed"] is False


def test_completion_retains_workflow_result_without_inventing_cleanup(api, args):
    path, value = record(api, args)
    entry.save_record(
        path,
        value,
        "running",
        run=copy.deepcopy(api.run),
        dispatched_at=entry.time.time(),
    )
    api.run.update(status="completed", conclusion="failure")
    entry.monitor(args, api, POLICY, path, value)
    assert value["status"] == "completed" and value["conclusion"] == "failure"
    assert "cleanup_confirmed" not in value


def test_external_fork_author_maintainer_review_service_dispatch(
    api, args, monkeypatch
):
    api.pr["head"]["repo"]["full_name"] = "external/fork"
    api.pr["user"]["login"] = "external-author"
    path, value = record(api, args)
    monkeypatch.setattr(
        entry.prepare, "access", lambda *_: ({"login": "service-account"}, "d" * 40)
    )
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: api.run)
    entry.advance(args, api, POLICY, path, value)
    assert value["reviewer"] == "maintainer" and value["status"] == "running"


def test_external_author_without_write_cannot_authorize(api):
    api.pr["head"]["repo"]["full_name"] = "external/fork"
    api.pr["user"]["login"] = api.reviews[0]["user"]["login"] = "external-author"
    api.permission = "read"
    with pytest.raises(entry.Stale, match="permission"):
        entry.authorization(api, POLICY, 58, 71, HEAD)


def test_fork_fetch_uses_source_pull_refs_not_fork_remote(api, args, monkeypatch):
    calls = []

    def git(_, *argv):
        calls.append(argv)
        if argv[0] == "cat-file":
            raise ValueError("missing object")
        if argv[:2] == ("remote", "get-url"):
            return ("git@github.com:" + REPO + ".git\n").encode()
        return b""

    monkeypatch.setattr(entry.prepare, "git", git)
    value = entry.authorization(api, POLICY, 58, 71, HEAD)
    entry.fetch_objects(args, value, REPO)
    assert calls[-1][-3:] == ("refs/pull/58/head", "refs/pull/58/merge", BASE)
    assert "external/fork" not in str(calls)
