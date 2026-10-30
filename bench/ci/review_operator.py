#!/usr/bin/env python3
"""Poll exact-head `/ci test` PR reviews and main's head on the trusted preparation host.

Each new main head queues one baseline job: that commit profiles the machine
and is measured four times per model on every enabled device. When it
completes, its verified cells and the characteristics file it profiled join the
baseline history next to the device profiles. PR and baseline jobs run one at a
time in the order they were queued, so a PR queued after a merge finds its base
measured; a PR whose base has no history gets needs-review cells. A PR job
launches from the characteristics of each device's most recent baseline job,
and cannot be prepared for a device that has none yet.
"""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import history
from ci import operator_prepare as prepare
from ci import selection_dispatch
from ci.contract import MEASURED_AT, digest, load, policy_read, revision
from ci.impact import device_requirements, fingerprint, plan
from ci.private_controller import current_ci


class Stale(ValueError):
    """Authorization is no longer valid; never dispatch its sealed source."""


class Waiting(ValueError):
    """An authenticated prerequisite has not completed yet."""


class SourceAPI:
    def __init__(self, api, repository):
        self.api, self.repository = api, repository

    def call(self, path):
        return self.api.call("repos/" + self.repository + "/" + path)


def pages(api, path):
    result = []
    for page in range(1, 33):
        rows = api.call(
            path + ("&" if "?" in path else "?") + f"per_page=100&page={page}"
        )
        if not isinstance(rows, list):
            raise ValueError("incomplete GitHub list response")
        result.extend(rows)
        if len(rows) < 100:
            return result
    raise ValueError("GitHub list exceeded complete pagination bound")


def marker(review):
    if review.get("state") not in ("COMMENTED", "APPROVED"):
        return None
    # A whole comment avoids authorizing quoted examples or incidental prose.
    return {"/ci test": "run", "/ci cancel": "cancel"}.get(
        review.get("body", "").strip()
    )


def writable(api, repository, login):
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", login):
        return False
    value = api.call("repos/" + repository + "/collaborators/" + login + "/permission")
    return value.get("permission") in ("admin", "maintain", "write")


def key(repository, number, review):
    raw = f"{repository}#{number}#{review['id']}#{review['commit_id']}"
    return hashlib.sha256(raw.encode()).hexdigest()


def authorization(
    api, policy, number, review_id, head=None, now=None, max_age=86400, require_ci=True
):
    repo = policy["repository"]
    pr = api.call(f"repos/{repo}/pulls/{number}")
    if (
        type(number) is not int
        or number <= 0
        or pr.get("state") != "open"
        or pr.get("draft") is not False
        or pr.get("base", {}).get("repo", {}).get("full_name") != repo
    ):
        raise Stale(
            "PR is not an open nondraft request against the fixed source repository"
        )
    actual = revision(pr["head"]["sha"])
    if head is not None and actual != head:
        raise Stale("review no longer identifies the current head")
    rows = pages(api, f"repos/{repo}/pulls/{number}/reviews")
    found = [row for row in rows if row.get("id") == review_id]
    if len(found) != 1:
        raise Stale("review is missing or ambiguous")
    review = found[0]
    login = review.get("user", {}).get("login")
    if (
        marker(review) != "run"
        or review.get("commit_id") != actual
        or not writable(api, repo, login)
    ):
        raise Stale("review command, exact commit or maintainer permission was revoked")
    submitted = datetime.fromisoformat(review["submitted_at"].replace("Z", "+00:00"))
    if (
        submitted.tzinfo is None
        or (now if now is not None else time.time()) - submitted.timestamp() > max_age
    ):
        raise Stale("review authorization expired")
    for row in rows:
        if (
            row.get("commit_id") == actual
            and row.get("id", 0) > review_id
            and (marker(row) == "cancel" or row.get("state") == "CHANGES_REQUESTED")
            and writable(api, repo, row.get("user", {}).get("login"))
        ):
            raise Stale("a later maintainer review cancelled physical testing")
    base = revision(pr["base"]["sha"])
    candidate = pr.get("merge_commit_sha")
    if not candidate:
        raise Waiting("GitHub has not produced the exact test merge")
    candidate = revision(candidate)
    remote = api.call(f"repos/{repo}/git/commits/{candidate}")
    if remote.get("sha") != candidate or [
        p.get("sha") for p in remote.get("parents", [])
    ] != [base, actual]:
        raise Waiting("test merge does not identify current base and reviewed head")
    value = dict(
        repository=repo,
        pr=number,
        review_id=review_id,
        reviewer=login,
        review_commit_id=actual,
        base=base,
        head=actual,
        candidate=candidate,
    )
    if require_ci:
        try:
            value["hosted_ci"] = current_ci(SourceAPI(api, repo), value, policy)
        except ValueError as error:
            raise Waiting(str(error)) from error
    return value


def atomic(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def save_record(path, value, status, **fields):
    value.update(status=status, updated_at=time.time(), **fields)
    atomic(path, value)


def fetch_objects(args, identity, repository, refs=None):
    """Fetch missing source objects: a PR's refs and base, or the given exact refs."""
    try:
        for name in ("base", "head", "candidate"):
            prepare.git(args.source, "cat-file", "-e", identity[name] + "^{commit}")
        return
    except ValueError:
        pass
    remote = (
        prepare.git(args.source, "remote", "get-url", args.fetch_remote)
        .decode()
        .strip()
    )
    allowed = {
        f"https://github.com/{repository}.git",
        f"https://github.com/{repository}",
        f"git@github.com:{repository}.git",
        f"git@github.com:{repository}",
    }
    if remote not in allowed:
        raise ValueError(
            "configured fetch remote does not identify the fixed source repository"
        )
    prepare.git(
        args.source,
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        "--no-recurse-submodules",
        args.fetch_remote,
        *(refs or ("refs/pull/" + str(identity["pr"]) + "/head", "refs/pull/" + str(identity["pr"]) + "/merge",
                   identity["base"])),
    )


def same_authorization(old, new):
    if any(
        old[name] != new[name]
        for name in ("repository", "pr", "review_id", "base", "head", "candidate")
    ):
        raise Stale("prepared source/base changed; a new review is required")


def review_scope(args, policy, identity, profiles):
    declared = profiles.get("devices")
    if (
        profiles.get("schema_version") != 1
        or not isinstance(declared, dict)
        or not declared
        or not set(declared) <= set(prepare.DEVICES)
    ):
        raise ValueError("nonempty reviewed device profiles required")
    enabled = prepare.DEVICES
    if "temporary_mac_acceptance" in profiles:
        exception = profiles["temporary_mac_acceptance"]
        if (
            not isinstance(exception, dict)
            or set(exception) != {"pull_request", "head"}
            or type(exception["pull_request"]) is not int
            or exception["pull_request"] <= 0
        ):
            raise ValueError("exact temporary Mac acceptance PR/head required")
        revision(exception["head"])
        enabled = (
            ("local-mbp",)
            if exception["pull_request"] == identity["pr"]
            and exception["head"] == identity["head"]
            else prepare.LINUX_DEVICES
        )
    fetch_objects(args, identity, policy["repository"])
    paths = []
    # Include merge-resolution paths, exactly as the preparation plan does.
    for target in (identity["head"], identity["candidate"]):
        changed = prepare.git(
            args.source, "diff", "--name-only", "-z", "--no-renames",
            "--no-ext-diff", identity["base"], target,
        )
        paths.extend(item.decode("utf-8") for item in changed.split(b"\0") if item)
    impact = plan(paths, policy, **{
        name: identity[name] for name in ("base", "head", "candidate")
    })
    execution = [
        device for device in enabled if device_requirements(impact, device)["models"]
    ]
    if not execution:
        raise Waiting("canonical impact requires no currently enabled device; do not stage")
    missing = [device for device in execution if device not in declared]
    if missing:
        raise Waiting("required device profiles unavailable: " + ", ".join(missing))
    if any(
        declared[d]["ssh"].get("sudo_password_required") is not False
        for d in execution
    ):
        raise Waiting(
            "background staging requires existing root access or the fixed narrow NOPASSWD helper"
        )
    return execution, impact


def prepared_scope(prepared, execution, impact):
    envelope = load(prepared / "envelope.json")
    if (
        envelope.get("execution_devices") != execution
        or envelope.get("selection") != impact
        or envelope.get("selection_plan_sha256") != fingerprint(impact)
        or not isinstance(envelope.get("jobs"), dict)
        or set(envelope["jobs"]) != set(execution)
        or any(envelope.get(name) != impact[name]
               for name in ("base", "head", "candidate"))
    ):
        raise Stale("prepared execution scope/impact differs from reviewed profile policy")


def run_identity(run, saved):
    fields = (
        "id",
        "workflow_id",
        "path",
        "event",
        "head_branch",
        "head_sha",
        "run_attempt",
    )
    if (
        any(run.get(name) != saved.get(name) for name in fields)
        or run.get("actor", {}).get("id") != saved.get("actor", {}).get("id")
        or type(run.get("id")) is not int
        or run["id"] <= 0
        or run.get("run_attempt") != 1
    ):
        raise ValueError("recorded parent workflow identity changed")


def monitor(args, api, policy, path, record):
    saved = record["run"]
    endpoint = f"repos/{policy['controller_repository']}/actions/runs/{saved['id']}"
    run = api.call(endpoint)
    run_identity(run, saved)
    if run.get("status") == "completed":
        save_record(
            path,
            record,
            "completed",
            conclusion=run.get("conclusion"),
            html_url=run.get("html_url"),
            completed_at=run.get("updated_at"),
            cleanup_source="Actions selected report/artifacts; not inferred here",
        )
        return
    reason = (
        record.get("cancellation_reason") if record["status"] == "cancelling" else None
    )
    try:
        if record.get("kind") != "baseline":  # a baseline job has no review to re-check
            prepare.access(api, policy)
            fresh = authorization(
                api,
                policy,
                record["pr"],
                record["review_id"],
                record["head"],
                max_age=args.max_authorization_age_s,
            )
            same_authorization(record, fresh)
            if (
                digest(args.inputs) != record["inputs_sha256"]
                or digest(args.policy) != record["policy_sha256"]
            ):
                raise Stale("trusted profile/policy changed during preparation")
        if time.time() - record["dispatched_at"] > args.run_timeout_s:
            raise Stale("recorded parent exceeded its finite deadline")
    except (ValueError, RuntimeError) as error:
        reason = str(error)
    if reason and record.get("cancellation_requested") is not True:
        # Persist before POST. API acknowledgement is never a worker-cleanup proof.
        save_record(
            path,
            record,
            "cancelling",
            cancel_attempted=True,
            cancel_attempts=record.get("cancel_attempts", 0) + 1,
            cancellation_reason=reason,
            cleanup_confirmed=False,
        )
        try:
            api.call(endpoint + "/cancel", {})
            record.pop("cancellation_error", None)
            save_record(path, record, "cancelling", cancellation_requested=True)
        except Exception as error:
            message = type(error).__name__ + ": " + str(error)
            save_record(
                path,
                record,
                "cancelling",
                cancellation_error=message,
                cancel_errors=record.get("cancel_errors", [])
                + [
                    dict(
                        attempt=record["cancel_attempts"],
                        error=message,
                        timestamp=time.time(),
                    )
                ],
            )


def advance(args, api, policy, path, record):
    attempted = record["status"] != "queued"
    try:
        fresh = authorization(
            api,
            policy,
            record["pr"],
            record["review_id"],
            record["head"],
            max_age=args.max_authorization_age_s,
        )
        same_authorization(record, fresh)
        if (
            digest(args.inputs) != record["inputs_sha256"]
            or digest(args.policy) != record["policy_sha256"]
        ):
            raise Stale("trusted profile/policy changed; authorize a fresh attempt")
        prepared = path.parent / path.stem / "prepared"
        profiles = load(args.inputs)
        if digest(args.inputs) != record["inputs_sha256"]:
            raise Stale("trusted profiles changed while reading execution scope")
        execution, impact = review_scope(args, policy, fresh, profiles)
        if record["status"] == "queued":
            attempted = True
            prepared.parent.mkdir(mode=0o700, exist_ok=True)
            save_record(path, record, "preparing", prepared=str(prepared))
            prepare.prepare(
                SimpleNamespace(
                    source=args.source,
                    inputs=args.inputs,
                    output=prepared,
                    policy=args.policy,
                    execution_devices=execution,
                    gh=args.gh,
                    **{k: fresh[k] for k in ("base", "head", "candidate")},
                ),
                api=api,
            )
            save_record(path, record, "prepared")
        prepared_scope(prepared, execution, impact)
        if record["status"] == "prepared":
            save_record(path, record, "staging")
            prepare.stage(SimpleNamespace(prepared=prepared, gh=args.gh), password=None)
            save_record(path, record, "staged")
        # A source/review/CI change during export or root staging must refuse dispatch.
        fresh = authorization(
            api,
            policy,
            record["pr"],
            record["review_id"],
            record["head"],
            max_age=args.max_authorization_age_s,
        )
        same_authorization(record, fresh)
        if (
            digest(args.inputs) != record["inputs_sha256"]
            or digest(args.policy) != record["policy_sha256"]
        ):
            raise Stale("trusted profile/policy changed before dispatch")
        prepared_scope(prepared, execution, impact)
        save_record(path, record, "dispatching")
        run = prepare.dispatch(SimpleNamespace(prepared=prepared, gh=args.gh), api=api)
        save_record(path, record, "running", run=run, dispatched_at=time.time())
    except Waiting as error:
        if record["status"] in ("queued", "prepared", "staged"):
            save_record(path, record, record["status"], waiting=str(error))
        else:
            save_record(
                path, record, "interrupted", error=str(error), no_automatic_retry=True
            )
    except Exception as error:
        save_record(
            path,
            record,
            "stale" if isinstance(error, Stale) else "interrupted",
            error=type(error).__name__ + ": " + str(error),
            no_automatic_retry=True,
        )
    # A queued prerequisite wait has not begun preparation or acquired a device.
    # It must not starve another independently authorized PR in this poll cycle.
    return attempted or record["status"] != "queued"


def baseline_key(repository, commit):
    return hashlib.sha256(f"baseline#{repository}#{commit}".encode()).hexdigest()


def baseline_needed(args, policy, head):
    """Whether main `head` needs a baseline job of its own, and the measured commit it was compared with.

    The files of the most recently measured main commit are compared with head's,
    by content rather than ancestry. Only a path that the impact rules give a
    performance cell needs a new measurement. Without one, the PR jobs on head
    compare against each cell's most recent record.
    """
    records = history.load_all(history.store_root(args.inputs))
    if not records:
        return True, None
    measured = max(records, key=lambda record: (record["measured_at"], record["commit"]))["commit"]
    if measured == head:
        return False, measured
    try:
        fetch_objects(args, dict(base=head, head=head, candidate=head), policy["repository"], (head,))
        raw = prepare.git(args.source, "diff", "--no-renames", "--name-only", "-z", measured, head)
    except ValueError:
        return True, measured
    paths = [path for path in raw.decode().split("\0") if path]
    selection = plan(paths, policy, base=measured, head=head, candidate=head)
    return any(cell["performance"]["required"] for cell in selection["cells"]), measured


def advance_baseline(args, api, policy, path, record):
    """Prepare, stage and dispatch the baseline job of one main commit; never retried."""
    commit = record["commit"]
    try:
        profiles = load(args.inputs)
        declared = profiles.get("devices", {})
        exception = profiles.get("temporary_mac_acceptance")
        enabled = prepare.LINUX_DEVICES if exception is not None else prepare.DEVICES
        # Only devices staged unattended; a device needing a sudo password is left out.
        execution = [device for device in enabled if device in declared
                     and declared[device]["ssh"].get("sudo_password_required") is False]
        if not execution:
            raise ValueError("no enabled device can be staged unattended")
        impact = plan([], policy, baseline=True, base=commit, head=commit, candidate=commit)
        prepared = path.parent / path.stem / "prepared"
        if record["status"] == "queued":
            prepared.parent.mkdir(mode=0o700, exist_ok=True)
            save_record(path, record, "preparing", prepared=str(prepared))
            fetch_objects(args, dict(base=commit, head=commit, candidate=commit), policy["repository"], (commit,))
            prepare.prepare(SimpleNamespace(source=args.source, inputs=args.inputs, output=prepared,
                                            policy=args.policy, execution_devices=execution, gh=args.gh,
                                            baseline=True, base=commit, head=commit, candidate=commit), api=api)
            save_record(path, record, "prepared")
        prepared_scope(prepared, execution, impact)
        if record["status"] == "prepared":
            save_record(path, record, "staging")
            prepare.stage(SimpleNamespace(prepared=prepared, gh=args.gh), password=None)
            save_record(path, record, "staged")
        prepared_scope(prepared, execution, impact)
        save_record(path, record, "dispatching")
        run = prepare.dispatch(SimpleNamespace(prepared=prepared, gh=args.gh), api=api)
        save_record(path, record, "running", run=run, dispatched_at=time.time())
    except Exception as error:
        save_record(path, record, "interrupted", error=type(error).__name__ + ": " + str(error),
                    no_automatic_retry=True)
    return True


def fetch_aggregate(api, policy, run_id, output):
    """The hosted report of one parent run, from its exact artifact."""
    listing = api.call("repos/" + policy["controller_repository"] + "/actions/runs/" + str(run_id)
                       + "/artifacts?per_page=100")
    name = "selected-aggregate-" + str(run_id) + "-1"
    matches = [item for item in listing.get("artifacts", []) if item.get("name") == name and item.get("expired") is False]
    if len(matches) != 1:
        raise ValueError("hosted report artifact missing or duplicated")
    raw = api.raw("repos/" + policy["controller_repository"] + "/actions/artifacts/" + str(matches[0]["id"]) + "/zip")
    expected = matches[0].get("digest")
    if expected is not None and expected != "sha256:" + hashlib.sha256(raw).hexdigest():
        raise ValueError("hosted report artifact digest differs")
    selection_dispatch.unpack(raw, output)
    return load(Path(output) / "aggregate.json")


def ingest_baseline(args, api, policy, path, record):
    """Add the verified cells of a completed baseline job to the history; no retry."""
    try:
        prepared = Path(record["prepared"])
        envelope = load(prepared / "envelope.json")
        report = fetch_aggregate(api, policy, record["run"]["id"], prepared.parent / ("aggregate-" + uuid.uuid4().hex))
        if report.get("envelope_sha256") != fingerprint(envelope):
            raise ValueError("hosted report belongs to another prepared job")
        finished = record.get("completed_at")
        measured_at = (finished if isinstance(finished, str) and MEASURED_AT.fullmatch(finished)
                       else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        stored, skipped = [], []
        root = history.store_root(args.inputs)
        for device in envelope["execution_devices"]:
            records, missed, characteristics = history.records_from_report(
                report, device=device, job_id=envelope["jobs"][device]["job_id"], commit=record["commit"],
                run_id=record["run"]["id"], run_attempt=1, measured_at=measured_at)
            # The characteristics file first: every record names it, and later PR jobs seal it.
            if characteristics is not None:
                history.write_characteristics(root, device, characteristics)
            stored += [dict(device=device, model=item["model"], tier=item["tier"],
                            result=history.write(root, item)) for item in records]
            skipped += missed
        save_record(path, record, "ingested", stored=stored, skipped=skipped)
    except Exception as error:
        save_record(path, record, "ingest-failed", error=type(error).__name__ + ": " + str(error),
                    no_automatic_retry=True)


def tick(args, api=None):
    policy = policy_read(load(args.policy))
    api = api or prepare.Gh(args.gh)
    records = [
        (path, load(path))
        for path in sorted(args.state.glob("*.json"))
        if re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
    ]
    active = False
    for path, record in records:
        if record["status"] in ("preparing", "staging", "dispatching"):
            saved = Path(record["prepared"]) / "dispatched-run.json"
            if (
                record["status"] == "dispatching"
                and saved.is_file()
                and not saved.is_symlink()
            ):
                save_record(
                    path,
                    record,
                    "running",
                    run=load(saved),
                    dispatched_at=record["updated_at"],
                )
            else:
                save_record(
                    path,
                    record,
                    "interrupted",
                    no_automatic_retry=True,
                    error="preparation/staging/dispatch was interrupted; preserve evidence and inspect",
                )
        if record["status"] in ("running", "cancelling"):
            monitor(args, api, policy, path, record)
            active |= record["status"] in ("running", "cancelling")
        if record.get("kind") == "baseline" and record["status"] == "completed":
            ingest_baseline(args, api, policy, path, record)
    prepare.access(api, policy)
    pending = []
    for pr in pages(api, f"repos/{policy['repository']}/pulls?state=open"):
        number = pr["number"]
        reviews = pages(api, f"repos/{policy['repository']}/pulls/{number}/reviews")
        runs = [
            r
            for r in reviews
            if marker(r) == "run"
            and r.get("commit_id") == pr["head"]["sha"]
            and writable(api, policy["repository"], r.get("user", {}).get("login"))
        ]
        if not runs:
            continue
        review = max(runs, key=lambda r: r["id"])
        path = args.state / (key(policy["repository"], number, review) + ".json")
        if path.exists():
            continue
        try:
            value = authorization(
                api,
                policy,
                number,
                review["id"],
                pr["head"]["sha"],
                max_age=args.max_authorization_age_s,
                require_ci=False,
            )
        except (Stale, Waiting) as error:
            pending.append(dict(pr=number, review_id=review["id"], reason=str(error)))
            continue
        record = dict(
            schema_version=1,
            **value,
            created_at=time.time(),
            inputs_sha256=digest(args.inputs),
            policy_sha256=digest(args.policy),
        )
        save_record(path, record, "queued")
        records.append((path, record))
    head = revision(api.call("repos/" + policy["repository"] + "/branches/main")["commit"]["sha"])
    path = args.state / (baseline_key(policy["repository"], head) + ".json")
    if not path.exists():
        record = dict(schema_version=1, kind="baseline", repository=policy["repository"], commit=head,
                      created_at=time.time())
        needed, measured = baseline_needed(args, policy, head)
        if needed:
            save_record(path, record, "queued")
            records.append((path, record))
        else:
            # No performance-relevant file differs from the measured commit: no job.
            save_record(path, record, "not-needed", measured_commit=measured)
    if not active:
        # One queue: PR and baseline jobs start in the order they were queued.
        for path, record in sorted(records, key=lambda item: (item[1].get("created_at", 0), item[0].name)):
            if record["status"] in ("queued", "prepared", "staged"):
                step = advance_baseline if record.get("kind") == "baseline" else advance
                if step(args, api, policy, path, record) is not False:
                    break
    return dict(
        active_parent=active, pending_source_reviews=pending, records=len(records), main_head=head
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "inputs", "state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path(__file__).with_name("policy.json")
    )
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--fetch-remote", default="ssdash")
    parser.add_argument("--poll-interval-s", type=int, default=30)
    parser.add_argument("--max-authorization-age-s", type=int, default=86400)
    parser.add_argument("--run-timeout-s", type=int, default=14400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        not 10 <= args.poll_interval_s <= 300
        or not 60 <= args.run_timeout_s <= 14400
        or not 60 <= args.max_authorization_age_s <= 604800
    ):
        parser.error("finite polling/authorization/run limits required")
    args.state = args.state.resolve()
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = args.state.lstat()
    if (
        info.st_uid != os.getuid()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_mode & 0o077
    ):
        parser.error(
            "state must be a private directory owned by this preparation account"
        )
    descriptor = os.open(
        args.state / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                observed = tick(args)
                atomic(
                    args.state / "poll-status.json",
                    dict(status="observed", timestamp=time.time(), **observed),
                )
            except Exception as error:
                atomic(
                    args.state / "poll-status.json",
                    dict(
                        status="blocked",
                        timestamp=time.time(),
                        error=type(error).__name__ + ": " + str(error),
                    ),
                )
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()
