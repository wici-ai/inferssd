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
MAIN = "f" * 40


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
        self.main = MAIN
        self.artifacts, self.downloads = [], {}

    def call(self, path, data=None):
        if data is not None:
            self.posts.append((path, data))
            return None
        if path == f"repos/{REPO}/branches/main":
            return {"commit": {"sha": self.main}}
        if path.endswith("/artifacts?per_page=100"):
            return {"artifacts": copy.deepcopy(self.artifacts)}
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

    def raw(self, path):
        return self.downloads[path]


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(entry, "current_ci", lambda *_: {"id": 12, "attempt": 1})
    monkeypatch.setattr(
        entry.prepare, "access", lambda *_: ({"login": "maintainer"}, "d" * 40)
    )
    return API()


@pytest.fixture
def args(tmp_path, monkeypatch):
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
    def git(_, *argv):
        if argv[0] == "cat-file":
            return b""
        assert argv[:2] == ("diff", "--name-only")
        return b"bench/ci/review_operator.py\0"

    monkeypatch.setattr(entry.prepare, "git", git)
    # Main's head already has its baseline job, so PR-only tests queue nothing else.
    entry.save_record(tmp_path / (entry.baseline_key(REPO, MAIN) + ".json"),
                      dict(schema_version=1, kind="baseline", repository=REPO, commit=MAIN, created_at=0.0),
                      "ingested")
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


def prepared_envelope(actual, **_):
    identities = {name: getattr(actual, name) for name in ("base", "head", "candidate")}
    paths = []
    for target in (actual.head, actual.candidate):
        raw = entry.prepare.git(actual.source, "diff", "--name-only", "-z",
                                "--no-renames", "--no-ext-diff", actual.base, target)
        paths.extend(item.decode() for item in raw.split(b"\0") if item)
    impact = entry.plan(paths, POLICY, **identities)
    actual.output.mkdir(parents=True, exist_ok=True)
    value = dict(**identities, selection=impact,
                 selection_plan_sha256=entry.fingerprint(impact),
                 execution_devices=actual.execution_devices,
                 jobs={device: {} for device in actual.execution_devices})
    (actual.output / "envelope.json").write_text(json.dumps(value))
    return value


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
    monkeypatch.setattr(entry.prepare, "prepare", prepared_envelope)

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
    def prepare(actual, **kwargs):
        calls.append("prepare")
        return prepared_envelope(actual, **kwargs)

    monkeypatch.setattr(entry.prepare, "prepare", prepare)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: calls.append("stage"))

    def dispatch(*_, **__):
        calls.append("dispatch")
        return api.run

    monkeypatch.setattr(entry.prepare, "dispatch", dispatch)
    entry.advance(args, api, POLICY, path, value)
    assert calls == ["prepare", "stage", "dispatch"] and value["status"] == "running"


@pytest.mark.parametrize("exception,devices", [
    (None, ["cloud-5090", "dgx-spark", "local-mbp"]),
    ({"pull_request": 58, "head": HEAD}, ["local-mbp"]),
    ({"pull_request": 64, "head": HEAD}, ["cloud-5090", "dgx-spark"]),
    ({"pull_request": 58, "head": "e" * 40}, ["cloud-5090", "dgx-spark"]),
])
def test_review_scope_uses_exact_exception_or_canonical_default(
    api, args, monkeypatch, exception, devices
):
    profiles = dict(schema_version=1, devices={
        device: dict(ssh=dict(sudo_password_required=False)) for device in reversed(devices)
    })
    if exception is not None:
        profiles["temporary_mac_acceptance"] = exception
    args.inputs.write_text(json.dumps(profiles))
    path, value = record(api, args)
    prepared = []
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    def prepare(actual, **kwargs):
        prepared.append(actual.execution_devices)
        return prepared_envelope(actual, **kwargs)

    monkeypatch.setattr(entry.prepare, "prepare", prepare)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: api.run)
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "running" and prepared == [devices]


@pytest.mark.parametrize("exception,missing", [
    (None, ["cloud-5090", "dgx-spark"]),
    ({"pull_request": 64, "head": HEAD}, ["cloud-5090", "dgx-spark"]),
    ({"pull_request": 58, "head": "e" * 40}, ["cloud-5090", "dgx-spark"]),
])
def test_mac_only_profiles_never_silently_reduce_other_required_coverage(
    api, args, monkeypatch, exception, missing
):
    profiles = dict(schema_version=1, devices={
        "local-mbp": dict(ssh=dict(sudo_password_required=False))
    })
    if exception is not None:
        profiles["temporary_mac_acceptance"] = exception
    args.inputs.write_text(json.dumps(profiles))
    path, value = record(api, args)
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_: pytest.fail("missing profiles"))
    assert entry.advance(args, api, POLICY, path, value) is False
    assert value["status"] == "queued"
    assert value["waiting"] == "required device profiles unavailable: " + ", ".join(missing)


def test_default_scope_filters_devices_by_both_complete_source_diffs(api, args, monkeypatch):
    calls = []

    def git(_, *argv):
        if argv[0] == "cat-file":
            return b""
        calls.append(argv)
        return (b"README.md\0" if argv[-1] == HEAD else
                b"engine/ggml/src/ggml-cuda/ssd/kernel.cu\0")

    monkeypatch.setattr(entry.prepare, "git", git)
    identity = entry.authorization(api, POLICY, 58, 71, HEAD)
    profiles = entry.load(args.inputs)
    profiles["devices"].pop("local-mbp")
    execution, impact = entry.review_scope(args, POLICY, identity, profiles)
    assert execution == ["cloud-5090", "dgx-spark"]
    assert impact["changed_paths"] == ["README.md", "engine/ggml/src/ggml-cuda/ssd/kernel.cu"]
    assert [call[-2:] for call in calls] == [(BASE, HEAD), (BASE, CANDIDATE)]


@pytest.mark.parametrize("exception", [
    None, {}, {"pull_request": True, "head": HEAD},
    {"pull_request": 58, "head": "bad"},
    {"pull_request": 58, "head": HEAD, "devices": ["local-mbp"]},
])
def test_malformed_temporary_exception_refuses(api, args, monkeypatch, exception):
    profiles = entry.load(args.inputs)
    profiles["temporary_mac_acceptance"] = exception
    args.inputs.write_text(json.dumps(profiles))
    path, value = record(api, args)
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: pytest.fail("invalid exception fetch"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "interrupted"


@pytest.mark.parametrize("status", ["prepared", "staged"])
@pytest.mark.parametrize("tamper", ["scope", "plan", "plan-hash", "identity", "jobs"])
def test_old_prepared_scope_or_impact_cannot_stage_or_dispatch(
    api, args, monkeypatch, status, tamper
):
    path, value = record(api, args)
    output = path.parent / path.stem / "prepared"
    prepared_envelope(SimpleNamespace(source=args.source, output=output,
        execution_devices=list(entry.prepare.DEVICES), base=BASE, head=HEAD, candidate=CANDIDATE))
    envelope = entry.load(output / "envelope.json")
    if tamper == "scope":
        envelope["execution_devices"] = ["local-mbp"]
    elif tamper == "plan":
        envelope["selection"]["changed_paths"] = ["README.md"]
    elif tamper == "plan-hash":
        envelope["selection_plan_sha256"] = "0" * 64
    elif tamper == "identity":
        envelope["head"] = "e" * 40
    else:
        envelope["jobs"].pop("dgx-spark")
    (output / "envelope.json").write_text(json.dumps(envelope))
    entry.save_record(path, value, status, prepared=str(output))
    monkeypatch.setattr(entry.prepare, "stage", lambda *_ , **__: pytest.fail("old scope stage"))
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_ , **__: pytest.fail("old scope dispatch"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "stale" and "execution scope/impact" in value["error"]


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
        len(list(args.state.glob("*.json"))) == 4
    )  # profile, policy, main's baseline job, one durable event


def test_unrelated_profile_wait_does_not_starve_exact_mac_acceptance(api, args, monkeypatch):
    profiles = entry.load(args.inputs)
    profiles["devices"] = {"local-mbp": profiles["devices"]["local-mbp"]}
    profiles["temporary_mac_acceptance"] = dict(pull_request=64, head=HEAD)
    args.inputs.write_text(json.dumps(profiles))
    identity = entry.authorization(api, POLICY, 58, 71, HEAD)
    records = []
    for index, number in enumerate((58, 64)):
        value = dict(identity, pr=number, inputs_sha256=entry.digest(args.inputs),
                     policy_sha256=entry.digest(args.policy))
        path = args.state / (str(index) * 64 + ".json")
        entry.save_record(path, value, "queued")
        records.append((path, value))
    monkeypatch.setattr(entry, "authorization", lambda *values, **_: dict(identity, pr=values[2]))
    monkeypatch.setattr(entry, "pages", lambda *_: [])
    monkeypatch.setattr(entry.prepare, "prepare", prepared_envelope)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    dispatched = []

    def dispatch(actual, **_):
        dispatched.append(entry.load(actual.prepared / "envelope.json"))
        return api.run

    monkeypatch.setattr(entry.prepare, "dispatch", dispatch)
    entry.tick(args, api)
    assert entry.load(records[0][0])["status"] == "queued"
    assert "cloud-5090, dgx-spark" in entry.load(records[0][0])["waiting"]
    assert entry.load(records[1][0])["status"] == "running"
    assert len(dispatched) == 1 and dispatched[0]["execution_devices"] == ["local-mbp"]


def test_tick_stops_after_one_actual_preparation_attempt(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(path, value, "queued")
    other = dict(value, review_id=72)
    entry.save_record(args.state / ("0" * 64 + ".json"), other, "queued")
    monkeypatch.setattr(entry, "pages", lambda *_: [])
    calls = []

    def advance(*values):
        calls.append(values[-1]["review_id"])
        return True

    monkeypatch.setattr(entry, "advance", advance)
    entry.tick(args, api)
    assert calls == [72]


def test_removed_command_does_not_queue_or_rewrite_history(api, args, monkeypatch):
    path, value = record(api, args)
    entry.save_record(path, value, "completed", conclusion="success")
    retained = path.read_bytes()
    api.reviews[0]["body"] = "/ci linux"
    api.reviews.append(dict(api.reviews[0], id=72))
    monkeypatch.setattr(entry, "advance", lambda *_: pytest.fail("old command replay"))
    entry.tick(args, api)
    assert path.read_bytes() == retained
    assert len(list(args.state.glob("*.json"))) == 4


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
    monkeypatch.setattr(entry.prepare, "prepare", prepared_envelope)
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


def baseline_path(args, commit=MAIN):
    return args.state / (entry.baseline_key(REPO, commit) + ".json")


def test_new_main_head_queues_one_baseline_job_that_measures_it_on_every_unattended_device(api, args, monkeypatch):
    api.main = "e" * 40
    monkeypatch.setattr(entry, "pages", lambda *_: [])
    fetched, prepared, dispatched = [], [], []
    monkeypatch.setattr(entry, "fetch_objects", lambda *values: fetched.append(values[-1]))

    def prepare(actual, **kwargs):
        prepared.append(actual)
        value = entry.plan([], POLICY, baseline=True, base=actual.base, head=actual.head, candidate=actual.candidate)
        actual.output.mkdir(parents=True)
        (actual.output / "envelope.json").write_text(json.dumps(dict(
            base=actual.base, head=actual.head, candidate=actual.candidate, selection=value,
            selection_plan_sha256=entry.fingerprint(value), execution_devices=actual.execution_devices,
            jobs={device: {} for device in actual.execution_devices})))

    monkeypatch.setattr(entry.prepare, "prepare", prepare)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: dispatched.append(1) or api.run)
    observed = entry.tick(args, api)
    record = entry.load(baseline_path(args, "e" * 40))
    assert observed["main_head"] == "e" * 40 and record["status"] == "running" and record["kind"] == "baseline"
    actual = prepared[0]
    assert actual.baseline is True and (actual.base, actual.head, actual.candidate) == ("e" * 40,) * 3
    assert actual.execution_devices == list(entry.prepare.DEVICES) and fetched == [("e" * 40,)]
    # The same head never queues a second job, and a running parent blocks every other start.
    api.run["status"] = "in_progress"
    entry.tick(args, api)
    assert len(prepared) == 1 and len(dispatched) == 1
    assert len(list(args.state.glob("*.json"))) == 4


@pytest.mark.parametrize("change,devices", [
    ("sudo", ["cloud-5090", "local-mbp"]),
    ("mac-acceptance", ["cloud-5090", "dgx-spark"]),
])
def test_baseline_job_runs_only_on_devices_staged_unattended(api, args, monkeypatch, change, devices):
    profiles = entry.load(args.inputs)
    if change == "sudo":
        profiles["devices"]["dgx-spark"]["ssh"]["sudo_password_required"] = True
    else:
        profiles["temporary_mac_acceptance"] = dict(pull_request=64, head=HEAD)
    args.inputs.write_text(json.dumps(profiles))
    path = baseline_path(args, "e" * 40)
    record = dict(schema_version=1, kind="baseline", repository=REPO, commit="e" * 40, created_at=1.0)
    entry.save_record(path, record, "queued")
    seen = []
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(entry.prepare, "prepare", lambda actual, **_: seen.append(actual.execution_devices)
                        or (_ for _ in ()).throw(RuntimeError("stop after scope")))
    entry.advance_baseline(args, api, POLICY, path, record)
    assert seen == [devices] and record["status"] == "interrupted" and record["no_automatic_retry"] is True


@pytest.mark.parametrize("first", ["pr", "baseline"])
def test_pr_and_baseline_jobs_start_in_queue_order(api, args, monkeypatch, first):
    path, value = record(api, args)
    entry.save_record(path, value, "queued", created_at=2.0 if first == "baseline" else 1.0)
    entry.save_record(baseline_path(args, "e" * 40), dict(schema_version=1, kind="baseline", repository=REPO,
                      commit="e" * 40, created_at=1.0 if first == "baseline" else 2.0), "queued")
    api.main = "e" * 40
    monkeypatch.setattr(entry, "pages", lambda *_: [])
    started = []
    monkeypatch.setattr(entry, "advance", lambda *values: started.append("pr"))
    monkeypatch.setattr(entry, "advance_baseline", lambda *values: started.append("baseline"))
    entry.tick(args, api)
    assert started == [first]


def test_pr_without_base_history_is_prepared_and_decided_needs_review_not_delayed(api, args, monkeypatch):
    # No scheduling for a missing base: the PR runs in queue order and its cells say so.
    path, value = record(api, args)
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    calls = []
    monkeypatch.setattr(entry.prepare, "prepare", lambda actual, **kwargs: calls.append(actual) or
                        prepared_envelope(actual, **kwargs))
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: api.run)
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "running" and len(calls) == 1 and not getattr(calls[0], "baseline", False)


def test_failed_staging_of_a_baseline_job_is_never_retried(api, args, monkeypatch):
    path = baseline_path(args, "e" * 40)
    record = dict(schema_version=1, kind="baseline", repository=REPO, commit="e" * 40, created_at=1.0)
    entry.save_record(path, record, "queued")
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    monkeypatch.setattr(entry.prepare, "prepare", lambda actual, **kwargs: (actual.output.mkdir(parents=True),
        (actual.output / "envelope.json").write_text(json.dumps(dict(
            base="e" * 40, head="e" * 40, candidate="e" * 40,
            selection=entry.plan([], POLICY, baseline=True, base="e" * 40, head="e" * 40, candidate="e" * 40),
            selection_plan_sha256=entry.fingerprint(entry.plan([], POLICY, baseline=True, base="e" * 40,
                                                               head="e" * 40, candidate="e" * 40)),
            execution_devices=actual.execution_devices, jobs={d: {} for d in actual.execution_devices})))))

    def held(*_, **__):
        raise RuntimeError("root staging failed (exit 1); retain fresh job and stop without retry: lease held")

    monkeypatch.setattr(entry.prepare, "stage", held)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: pytest.fail("dispatch after failed staging"))
    entry.advance_baseline(args, api, POLICY, path, record)
    assert record["status"] == "interrupted" and record["no_automatic_retry"] is True
    monkeypatch.setattr(entry, "advance_baseline", lambda *_: pytest.fail("automatic retry"))
    monkeypatch.setattr(entry, "pages", lambda *_: [])
    api.main = "e" * 40
    entry.tick(args, api)


def test_baseline_parent_is_cancelled_only_at_its_deadline_and_needs_no_review(api, args, monkeypatch):
    path = baseline_path(args, "e" * 40)
    record = dict(schema_version=1, kind="baseline", repository=REPO, commit="e" * 40, created_at=1.0)
    entry.save_record(path, record, "running", run=copy.deepcopy(api.run), dispatched_at=entry.time.time())
    monkeypatch.setattr(entry, "authorization", lambda *_, **__: pytest.fail("a baseline job has no review"))
    entry.monitor(args, api, POLICY, path, record)
    assert record["status"] == "running" and not api.posts
    record["dispatched_at"] -= args.run_timeout_s + 1
    entry.monitor(args, api, POLICY, path, record)
    assert record["status"] == "cancelling" and "deadline" in record["cancellation_reason"]


def aggregate_zip(report):
    import io
    import zipfile
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("aggregate.json", json.dumps(report))
    return raw.getvalue()


@pytest.mark.parametrize("damage", [None, "other-envelope", "missing-artifact", "digest"])
def test_completed_baseline_job_adds_only_its_verified_cells_to_the_history(api, args, damage):
    from test_ci_performance_series import baseline_row, report_for
    commit = "a" * 40  # the synthetic baseline cells measure BASE
    path = baseline_path(args, commit)
    prepared = path.parent / path.stem / "prepared"
    prepared.mkdir(parents=True)
    envelope = dict(base=commit, head=commit, candidate=commit, execution_devices=["cloud-5090"],
                    jobs={"cloud-5090": dict(job_id="f" * 32)})
    (prepared / "envelope.json").write_text(json.dumps(envelope))
    record = dict(schema_version=1, kind="baseline", repository=REPO, commit=commit, created_at=1.0,
                  prepared=str(prepared))
    entry.save_record(path, record, "running", run=copy.deepcopy(api.run), dispatched_at=entry.time.time())
    report = report_for([baseline_row(), baseline_row("deepseek", token_change=(2, "candidate"))])
    report["envelope_sha256"] = "0" * 64 if damage == "other-envelope" else entry.fingerprint(envelope)
    raw = aggregate_zip(report)
    if damage != "missing-artifact":
        api.artifacts = [dict(id=5, name="selected-aggregate-87-1", expired=False,
                              digest="sha256:" + ("0" * 64 if damage == "digest" else entry.hashlib.sha256(raw).hexdigest()))]
    api.downloads[f"repos/{POLICY['controller_repository']}/actions/artifacts/5/zip"] = raw
    api.run.update(status="completed", conclusion="failure", updated_at="2026-10-29T08:00:00Z")
    entry.monitor(args, api, POLICY, path, record)
    entry.ingest_baseline(args, api, POLICY, path, record)
    root = entry.history.store_root(args.inputs)
    if damage is None:
        assert record["status"] == "ingested"
        assert record["stored"] == [dict(device="cloud-5090", model="muse", tier="512", result="stored")]
        assert record["skipped"][0]["cell"] == "cloud-5090/deepseek/512"
        stored = entry.history.load_all(root)
        assert len(stored) == 1 and stored[0]["run_id"] == 87 and stored[0]["measured_at"] == "2026-10-29T08:00:00Z"
    else:
        assert record["status"] == "ingest-failed" and record["no_automatic_retry"] is True
        assert entry.history.load_all(root) == []
