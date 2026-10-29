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
        # A revision compared with itself has no changed paths.
        return b"" if argv[-2] == argv[-1] else b"bench/ci/review_operator.py\0"

    monkeypatch.setattr(entry.prepare, "git", git)
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
    value.update(entry.trusted_inputs(args))
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
    impact = entry.plan(paths, POLICY, calibration=getattr(actual, "calibration_pairs", None) is not None,
                        **identities)
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


def test_calibrate_command_has_the_same_exact_head_authorization(api):
    api.reviews[0]["body"] = "/ci calibrate"
    value = entry.authorization(api, POLICY, 58, 71, HEAD)
    assert value["command"] == "calibrate" and value["head"] == HEAD
    assert entry.prepared_identity(value) == dict(base=HEAD, head=HEAD, candidate=HEAD)
    api.permission = "read"
    with pytest.raises(entry.Stale):
        entry.authorization(api, POLICY, 58, 71, HEAD)


def test_later_cancel_also_revokes_a_calibration(api):
    api.reviews[0]["body"] = "/ci calibrate"
    api.reviews.append(dict(api.reviews[0], id=72, body="/ci cancel"))
    with pytest.raises(entry.Stale, match="cancelled"):
        entry.authorization(api, POLICY, 58, 71, HEAD)


def test_calibrate_review_queues_bounded_pairs_and_prepares_head_against_itself(api, args, monkeypatch):
    api.reviews[0]["body"] = "/ci calibrate"
    args.calibration_pairs = 3
    monkeypatch.setattr(entry, "fetch_objects", lambda *_: None)
    prepared = []

    def prepare(actual, **kwargs):
        prepared.append(actual)
        return prepared_envelope(actual, **kwargs)

    monkeypatch.setattr(entry.prepare, "prepare", prepare)
    monkeypatch.setattr(entry.prepare, "stage", lambda *_, **__: None)
    monkeypatch.setattr(entry.prepare, "dispatch", lambda *_, **__: api.run)
    entry.tick(args, api)
    record = entry.load(args.state / (entry.key(REPO, 58, api.reviews[0]) + ".json"))
    assert record["command"] == "calibrate" and record["calibration_pairs"] == 3 and record["status"] == "running"
    actual = prepared[0]
    assert (actual.base, actual.head, actual.candidate, actual.calibration_pairs) == (HEAD, HEAD, HEAD, 3)
    envelope = entry.load(actual.output / "envelope.json")
    assert envelope["selection"]["purpose"] == "noise-calibration"
    assert envelope["execution_devices"] == list(entry.prepare.DEVICES)


@pytest.mark.parametrize("pairs", [None, 5, True])
def test_calibration_record_without_a_bounded_pair_count_never_prepares(api, args, monkeypatch, pairs):
    api.reviews[0]["body"] = "/ci calibrate"
    path, value = record(api, args)
    if pairs is not None:
        value["calibration_pairs"] = pairs
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: pytest.fail("unbounded calibration"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "stale" and "pair count" in value["error"]


@pytest.mark.parametrize("change", ["new-bands", "invalid-bands"])
def test_noise_bands_changed_after_queueing_never_prepare(api, args, monkeypatch, change):
    from test_ci_performance_series import banded
    path, value = record(api, args)
    assert value["noise_bands_sha256"] == entry.fingerprint({})
    bands = args.inputs.with_name(entry.prepare.BANDS_FILE)
    bands.write_text(json.dumps(banded({("cloud-5090", "muse", "512"): "c" * 64}))
                     if change == "new-bands" else '{"other": {}}')
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: pytest.fail("changed operator bands"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == ("stale" if change == "new-bands" else "interrupted")
    assert ("noise bands" if change == "new-bands" else "noise bands are keyed") in value["error"]


def test_edited_command_after_queueing_is_stale(api, args, monkeypatch):
    path, value = record(api, args)
    api.reviews[0]["body"] = "/ci calibrate"
    monkeypatch.setattr(entry.prepare, "prepare", lambda *_, **__: pytest.fail("command changed"))
    entry.advance(args, api, POLICY, path, value)
    assert value["status"] == "stale" and "command" in value["error"]


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
        len(list(args.state.glob("*.json"))) == 3
    )  # profile, policy, one durable event


def test_unrelated_profile_wait_does_not_starve_exact_mac_acceptance(api, args, monkeypatch):
    profiles = entry.load(args.inputs)
    profiles["devices"] = {"local-mbp": profiles["devices"]["local-mbp"]}
    profiles["temporary_mac_acceptance"] = dict(pull_request=64, head=HEAD)
    args.inputs.write_text(json.dumps(profiles))
    identity = entry.authorization(api, POLICY, 58, 71, HEAD)
    records = []
    for index, number in enumerate((58, 64)):
        value = dict(identity, pr=number, **entry.trusted_inputs(args))
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
