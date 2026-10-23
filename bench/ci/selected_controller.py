#!/usr/bin/env python3
"""Exact reviewed PR -> root-staged selected children -> terminal dedicated App check."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import HASH, load, policy_read, revision, save
from ci.impact import fingerprint, plan as selection, selected, validate
from ci.private_controller import API, admission, bind_context, check_body, context, current_ci, source_manifest, writable
from ci.selected_aggregate import aggregate
from ci.selection_contract import JOB
from ci import selection_dispatch


def changed_paths(api, pr):
    paths, count = [], 0
    for page in range(1, 32):
        rows = api.call(f"pulls/{pr['number']}/files?per_page=100&page={page}")
        count += len(rows)
        for row in rows:
            paths.append(row["filename"])
            if row.get("previous_filename"):
                paths.append(row["previous_filename"])
        if len(rows) < 100:
            break
    else:
        raise ValueError("incomplete changed-file API listing")
    if pr.get("changed_files") != count:
        raise ValueError("complete PR changed-file count differs")
    return paths


def jobs_checked(jobs, impact):
    devices = {cell["device"] for cell in selected(impact)}
    if not isinstance(jobs, dict) or set(jobs) != devices:
        raise ValueError("root-staged jobs must exactly cover selected devices")
    fields = {"device", "job_id", "source_revision", "config_sha256", "selection_plan_sha256",
              "control_revision", "control_manifest_sha256"}
    for device, job in jobs.items():
        if (not isinstance(job, dict) or set(job) != fields or job["device"] != device
                or not JOB.fullmatch(job["job_id"]) or job["source_revision"] != impact["candidate"]
                or job["selection_plan_sha256"] != fingerprint(impact)):
            raise ValueError("root-staged job/source/selection identity differs")
        revision(job["control_revision"])
        if any(not HASH.fullmatch(job[key]) for key in ("config_sha256", "control_manifest_sha256")):
            raise ValueError("root-installed config/controller hashes required")
    if len({job["job_id"] for job in jobs.values()}) != len(jobs):
        raise ValueError("each selected physical device needs a fresh distinct job")
    return jobs


def environment(args, env):
    policy = policy_read(load(args.policy))
    event = load(env["GITHUB_EVENT_PATH"])
    identity = context(policy, env, event)
    return policy, event, identity


def preflight(args, env=os.environ):
    policy, event, identity = environment(args, env)
    private = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    metadata = private.call("")
    branch = private.call("branches/main")
    if (metadata.get("private") is not True or metadata.get("default_branch") != "main"
            or not writable(private, identity["actor"]) or branch.get("protected") is not True
            or branch.get("commit", {}).get("sha") != identity["workflow_sha"]
            or env.get("INFERSSD_CI_ENABLED") != "true"):
        raise ValueError("private protected controller activation/permission/source gate failed")
    inputs = event["inputs"]
    number = inputs["pull_request"]
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("positive PR required")
    api = API(policy["repository"], env["SOURCE_READ_TOKEN"])
    pr = api.call("pulls/" + number)
    admitted = admission(api, pr, policy, revision(inputs["head_sha"]))
    admitted["normal_ci"] = current_ci(api, admitted, policy)
    impact = selection(changed_paths(api, pr), policy, **{key: admitted[key] for key in ("base", "head", "candidate")})
    again = admission(api, api.call("pulls/" + number), policy, admitted["head"])
    if any(again[key] != admitted[key] for key in ("base", "head", "candidate", "authorization")):
        raise ValueError("PR or independent exact-SHA authorization changed during planning")
    from ci.stage_session import unique_object
    jobs = jobs_checked(json.loads(inputs["staged_jobs"], object_pairs_hook=unique_object), impact)
    admitted.update(selection=impact, selection_plan_sha256=fingerprint(impact), jobs=jobs,
                    controller=identity, control_revision=source_manifest(args.control),
                    policy_sha256=fingerprint(policy))
    save(args.plan, admitted)
    return 0


def execute(args, env=os.environ):
    policy, _, _ = environment(args, env)
    admitted = load(args.plan)
    bind_context(admitted, policy, env)
    impact = validate(admitted["selection"], policy)
    if admitted["policy_sha256"] != fingerprint(policy) or admitted["control_revision"] != source_manifest(args.control):
        raise ValueError("admitted trusted control/policy changed")
    jobs = jobs_checked(admitted["jobs"], impact)
    # Source read credentials are not used to dispatch or sent to device runners.
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"], version="2026-03-10")
    args.artifacts.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, selection_plan_sha256=fingerprint(impact), children={}, errors={})
    dispatched = {}
    parent = f"{admitted['controller']['run_id']}-{admitted['controller']['run_attempt']}"
    deadline = time.monotonic() + 10800
    previous = {}

    def interrupted(number, frame):
        raise KeyboardInterrupt("selected controller interrupted")

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, interrupted)
    try:
        for device, job in jobs.items():
            try:
                dispatched[device] = selection_dispatch.start(api, job, admitted["controller"]["workflow_sha"], parent)
                save(args.artifacts / (device + "-dispatch.json"), dispatched[device])
            except Exception as error:
                state["errors"][device] = f"{type(error).__name__}: {error}"
        for device, child in dispatched.items():
            try:
                run = selection_dispatch.wait(api, child, timeout_s=max(1, deadline - time.monotonic()))
                state["children"][device] = run
                selection_dispatch.collect(api, run, jobs[device], args.artifacts / device, parent=parent)
            except Exception as error:
                state["errors"][device] = f"{type(error).__name__}: {error}"
    finally:
        save(args.artifacts / "dispatch-state.json", state)
        try:
            cancel_children(args, env)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return 1 if state["errors"] else 0


def cancel_children(args, env=os.environ):
    """Also called by the hosted always step after a killed execute process."""
    policy, _, identity = environment(args, env)
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    results = {}
    if args.artifacts.exists():
        for path in sorted(args.artifacts.glob("*-dispatch.json")):
            child = load(path)
            parent = f"{identity['run_id']}-{identity['run_attempt']}"
            if (child.get("controller_sha") != identity["workflow_sha"]
                    or child.get("inputs", {}).get("selection_parent") != parent):
                raise ValueError("refuse cancellation of another controller's child")
            try:
                results[path.name] = selection_dispatch.cancel(api, child)
            except Exception as error:
                results[path.name] = dict(error=repr(error), cleanup_confirmed=False)
        save(args.artifacts / "cancellation-requests.json", results)
    return 0  # Request success is never a device acceptance or cleanup verdict.


def finalize(args, env=os.environ):
    policy, event, identity = environment(args, env)
    api = API(policy["repository"], env["DEVICE_CHECK_TOKEN"])
    # Even failed admission gets a terminal failure for the explicitly requested
    # source PR/head. This fallback authorizes no device work or positive result.
    number, head = event["inputs"]["pull_request"], revision(event["inputs"]["head_sha"])
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("positive PR required")
    pr = api.call("pulls/" + number)
    if (pr.get("head", {}).get("repo", {}).get("full_name") != policy["repository"]
            or pr.get("base", {}).get("repo", {}).get("full_name") != policy["repository"]):
        raise ValueError("terminal check cannot target an unrelated repository")
    admitted = dict(head=head, controller=identity)
    report = dict(schema_version=1, purpose="selected-ci", passed=False, formal_samples=0)
    try:
        admitted = load(args.plan)
        bind_context(admitted, policy, env)
        impact = validate(admitted["selection"], policy)
        if (admitted["head"] != head or admitted["pr"] != int(number)
                or admitted["policy_sha256"] != fingerprint(policy)
                or admitted["control_revision"] != source_manifest(args.control)):
            raise ValueError("admission/control/request binding differs")
        state = load(args.artifacts / "dispatch-state.json")
        if state["selection_plan_sha256"] != fingerprint(impact) or state["errors"]:
            raise ValueError("one or more selected dispatches failed or are missing")
        private = API(policy["controller_repository"], env["GITHUB_TOKEN"])
        for device, child in state["children"].items():
            latest = private.call("actions/runs/" + str(child["id"]))
            selection_dispatch.run_identity(latest, child["workflow_id"], identity["workflow_sha"], admitted["jobs"][device]["job_id"])
            if latest.get("status") != "completed" or latest.get("conclusion") != "success":
                raise ValueError("selected child state changed before App publication")
        report = aggregate(impact, policy, args.artifacts, admitted["jobs"], state["children"])
        current = admission(api, api.call("pulls/" + number), policy, head)
        if any(current[key] != admitted[key] for key in ("base", "head", "candidate", "authorization")):
            raise ValueError("source or independent approval changed before publication")
        if current_ci(api, admitted, policy) != admitted["normal_ci"]:
            raise ValueError("latest normal CI changed before publication")
        again = selection(changed_paths(api, pr), policy, **{key: admitted[key] for key in ("base", "head", "candidate")})
        if again != impact:
            raise ValueError("complete changed paths changed before publication")
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    save(args.output, report)
    admitted.update(head=head, controller=identity)
    body = check_body(admitted, report, int(env["DEVICE_CHECK_APP_ID"]))
    body["output"]["title"] = "Selected device gates passed" if report["passed"] else "Selected device gate blocked"
    body["output"]["summary"] = json.dumps(report, ensure_ascii=False, allow_nan=False)[:60000]
    expected_app = body.pop("_expected_app_id")
    check = api.call("check-runs", body)
    if (check.get("app", {}).get("id") != expected_app or check.get("head_sha") != head
            or check.get("external_id") != body["external_id"]):
        raise ValueError("terminal check is not bound to the dedicated App/exact requested head")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "execute", "cancel_children", "finalize"))
    for name in ("control", "policy", "plan", "artifacts", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("control", "policy", "plan"))
    args = parser.parse_args()
    raise SystemExit(globals()[args.mode](args))
