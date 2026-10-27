#!/usr/bin/env python3
"""Exact reviewed PR -> root-staged selected children -> terminal dedicated App check."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import HASH, digest, load, policy_read, revision, save
from ci.impact import fingerprint, plan as selection, selected, validate
from ci.private_controller import API, admission, bind_context, check_body, context, current_ci, source_manifest, writable
from ci.selected_aggregate import aggregate, operator_aggregate
from ci.selection_contract import JOB, execution_scope, FUNCTIONAL_MODE
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


def jobs_checked(jobs, impact, execution_devices=None, validation_mode=FUNCTIONAL_MODE):
    devices = {cell["device"] for cell in selected(impact)}
    if execution_devices is not None:
        scope = execution_scope(dict(validation_mode=validation_mode, execution_devices=execution_devices))
        if not set(scope["execution_devices"]) <= devices:
            raise ValueError("operator scope contains an unaffected device")
        devices = set(scope["execution_devices"])
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


OPERATOR_PURPOSE = "operator-selected-report-only"
MAX_ENVELOPE_BYTES = 60 * 1024


def operator_access(policy, event, identity, env):
    """Only private-repository metadata is queried at runtime; no source token."""
    if identity["run_attempt"] != 1:
        raise ValueError("operator jobs are single-use; prepare fresh jobs instead of rerunning")
    api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    metadata, branch = api.call(""), api.call("branches/main")
    triggering = env.get("GITHUB_TRIGGERING_ACTOR", "")
    sender_id = event.get("sender", {}).get("id")
    if (metadata.get("private") is not True or metadata.get("default_branch") != "main"
            or branch.get("protected") is not True
            or branch.get("commit", {}).get("sha") != identity["workflow_sha"]
            or not triggering or type(sender_id) is not int or sender_id <= 0
            or not writable(api, identity["actor"]) or not writable(api, triggering)
            or env.get("INFERSSD_CI_ENABLED") != "true"):
        raise ValueError("operator identity/permission/protected controller activation gate failed")
    return dict(login=identity["actor"], id=sender_id, triggering_actor=triggering)


def operator_envelope(value, policy, identity, actor):
    fields = {"schema_version", "purpose", "repository", "base", "head", "candidate",
              "selection", "selection_plan_sha256", "execution_devices", "validation_mode",
              "prepared_by", "controller_repository", "controller_workflow_sha", "policy_sha256", "jobs"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("complete sealed operator envelope required")
    impact = validate(value["selection"], policy)
    scope = execution_scope(value)
    prepared = value["prepared_by"]
    if (value["schema_version"] != 1 or value["purpose"] != OPERATOR_PURPOSE or scope is None
            or value["repository"] != policy["repository"]
            or value["controller_repository"] != policy["controller_repository"]
            or value["controller_workflow_sha"] != identity["workflow_sha"]
            or value["policy_sha256"] != fingerprint(policy)
            or value["selection_plan_sha256"] != fingerprint(impact)
            or any(value[key] != impact[key] for key in ("base", "head", "candidate"))
            or not isinstance(prepared, dict) or prepared.get("login") != actor["login"]
            or type(prepared.get("id")) is not int or prepared["id"] != actor["id"]):
        raise ValueError("operator envelope source/actor/controller/policy identity differs")
    jobs_checked(value["jobs"], impact, scope["execution_devices"], scope["validation_mode"])
    return value


def read_operator_envelope(event):
    from ci.stage_session import unique_object
    raw = event.get("inputs", {}).get("prepared_envelope")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise ValueError("complete operator envelope exceeds the bounded dispatch input")
    def nonfinite(_):
        raise ValueError("operator envelope contains a nonfinite number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=nonfinite)


def operator_job_controls(args, jobs):
    revision = source_manifest(args.control)
    manifest_sha256 = digest(args.control / "control-source.json")
    if any(job["control_revision"] != revision
           or job["control_manifest_sha256"] != manifest_sha256 for job in jobs.values()):
        raise ValueError("operator jobs must use this exact reviewed controller export")


def operator_preflight(args, env=os.environ):
    policy, event, identity = environment(args, env)
    actor = operator_access(policy, event, identity, env)
    envelope = operator_envelope(read_operator_envelope(event), policy, identity, actor)
    operator_job_controls(args, envelope["jobs"])
    admitted = dict(envelope, controller=identity, operator=actor,
                    envelope_sha256=fingerprint(envelope), control_revision=source_manifest(args.control))
    save(args.plan, admitted)
    return 0


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
    operator = admitted.get("purpose") == OPERATOR_PURPOSE
    if operator:
        _, event, identity = environment(args, env)
        actor = operator_access(policy, event, identity, env)
        envelope = read_operator_envelope(event)
        operator_envelope(envelope, policy, identity, actor)
        if (admitted.get("envelope_sha256") != fingerprint(envelope)
                or any(admitted.get(key) != value for key, value in envelope.items())
                or admitted.get("operator") != actor):
            raise ValueError("admitted operator envelope changed")
    jobs = jobs_checked(admitted["jobs"], impact, admitted["execution_devices"] if operator else None,
                        admitted["validation_mode"] if operator else FUNCTIONAL_MODE)
    if operator:
        operator_job_controls(args, jobs)
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
            collection_attempted = False
            try:
                run = selection_dispatch.wait(api, child, timeout_s=max(1, deadline - time.monotonic()))
                state["children"][device] = run
                collection_attempted = True
                selection_dispatch.collect(api, run, jobs[device], args.artifacts / device, parent=parent)
            except Exception as error:
                state["errors"][device] = f"{type(error).__name__}: {error}"
                if not collection_attempted:
                    # A complete inconclusive pair exits nonzero. Retain its raw
                    # artifact once so the always report can recompute metrics;
                    # cancellation/unknown state still cannot become a pass.
                    try:
                        run = api.call("actions/runs/" + str(child["id"]))
                        selection_dispatch.run_identity(run, child["workflow_id"], child["controller_sha"], child["job_id"])
                        if run.get("id") != child["id"] or run.get("status") != "completed":
                            raise ValueError("failed child has no exact terminal run")
                        state["children"][device] = run
                        selection_dispatch.collect(api, run, jobs[device], args.artifacts / device, parent=parent)
                        state.setdefault("retained_failed_children", {})[device] = run["id"]
                    except Exception as retained_error:
                        state["errors"][device] += "; raw retention: " + type(retained_error).__name__ + ": " + str(retained_error)
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
    uncertain = False
    if args.artifacts.exists():
        original = args.artifacts / "cancellation-requests.json"
        prior = load(original) if original.exists() else None
        prior_sha = digest(original) if prior is not None else None
        for path in sorted(args.artifacts.glob("*-dispatch.json")):
            child = load(path)
            parent = f"{identity['run_id']}-{identity['run_attempt']}"
            if (child.get("controller_sha") != identity["workflow_sha"]
                    or child.get("inputs", {}).get("selection_parent") != parent):
                raise ValueError("refuse cancellation of another controller's child")
            try:
                claim = args.artifacts / ("cancellation-" + path.stem + "-attempt.json")
                expected_claim = dict(schema_version=1, parent=parent, child_sha256=fingerprint(child),
                                      run_id=child["id"], cleanup_confirmed=False)
                attempted = prior is not None or claim.exists()
                if not attempted:
                    try:
                        # Claim before the first possible POST. An interrupted
                        # or ambiguous request is never an automatic retry.
                        save(claim, expected_claim)
                    except FileExistsError:
                        attempted = True
                if attempted:
                    if claim.exists() and load(claim) != expected_claim:
                        raise ValueError("prior cancellation claim identifies another child")
                    run = api.call("actions/runs/" + str(child["id"]))
                    selection_dispatch.run_identity(run, child["workflow_id"], child["controller_sha"], child["job_id"])
                    previous = prior.get(path.name) if isinstance(prior, dict) else None
                    unknown = (not isinstance(previous, dict) or previous.get("run_id") != child["id"] or "error" in previous
                               or previous.get("prior_outcome_unknown") is True)
                    results[path.name] = dict(run_id=child["id"], status=run.get("status"),
                        conclusion=run.get("conclusion"), prior_attempt=True, prior_outcome_unknown=unknown,
                        cancellation_requested=False, cleanup_confirmed=False)
                    uncertain |= unknown or run.get("status") != "completed"
                else:
                    results[path.name] = selection_dispatch.cancel(api, child)
                    uncertain |= results[path.name].get("cancellation_requested") is not False
            except Exception as error:
                results[path.name] = dict(error=repr(error), cleanup_confirmed=False)
                uncertain = True
        if prior is None:
            save(original, results)
        else:
            save(args.artifacts / ("cancellation-recheck-" + str(time.time_ns()) + ".json"),
                 dict(schema_version=1, parent=f"{identity['run_id']}-{identity['run_attempt']}",
                      prior_requests_sha256=prior_sha, results=results, cleanup_confirmed=False))
    # Closed Actions runs are not a physical cleanup verdict; only retained
    # root guardian/unit/process evidence can establish that independently.
    return 1 if uncertain else 0


def summary_text(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def metric_text(metric, key):
    value = metric.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        return "unavailable"
    return f"{value:+.3f}%" if key == "change_percent" else f"{value:.4f}"


def operator_summary(stream, report):
    """Render verified pair values, never turn missing/inconclusive into green."""
    status = report.get("status", "passed" if report["passed"] else "failed")
    stream.write("## Operator-authorized device report\n\n")
    stream.write("Declared device scope: **" + summary_text(status) + "**. ")
    stream.write("Full impact coverage: **" + str(report["full_impact_coverage_complete"]).lower() + "**.\n\n")
    stream.write("Linux scope: **" + summary_text(report.get("linux_scoped_status", "not-verified"))
                 + "**. Mac scope: **" + summary_text(report.get("mac_scoped_status", "not-verified")) + "**.\n\n")
    stream.write("PR publication is not enabled. These results identify sealed source; they do not attest the current PR head.\n\n")
    for key in ("base", "head", "candidate"):
        if key in report:
            stream.write(key + ": `" + summary_text(report[key]) + "`\n\n")
    stream.write("| Device | Model | Functional | Numerical | Performance | Outcome |\n|---|---|---|---|---|---|\n")
    pairs = []
    for row in report["cells"]:
        performance = row.get("performance", row["status"])
        if isinstance(performance, list):
            pairs.extend((row, pair) for pair in performance)
            performance = ", ".join(pair["tier"] + ": " + pair["status"] for pair in performance) or "not-measured"
        deferred = row.get("performance_deferred_tiers", [])
        if deferred:
            performance += "; deferred tiers: " + ", ".join(deferred)
        values = [row.get("device"), row.get("model"), row.get("functional", row["status"]),
                  row.get("numerical", row["status"]), performance, row["status"]]
        stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
    if pairs:
        stream.write("\n### 512 prompt / 256 decode: two decode segments\n\n")
        stream.write("| Device | Model | Tier | 0–128 base tok/s | PR tok/s | Change | 128–256 base tok/s | PR tok/s | Change | Outcome |\n")
        stream.write("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row, pair in pairs:
            metrics = pair.get("metrics", {})
            first, second = metrics.get("decode_first128_tps", {}), metrics.get("decode_second128_tps", {})
            values = [row["device"], row["model"], pair["tier"],
                      *(metric_text(first, key) for key in ("base", "candidate", "change_percent")),
                      *(metric_text(second, key) for key in ("base", "candidate", "change_percent")), pair["status"]]
            stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
        stream.write("\n| Device | Model | Tier | Metric | Base | PR | Change | Better direction |\n|---|---|---|---|---:|---:|---:|---|\n")
        for row, pair in pairs:
            for key, label in (("decode_tps", "Whole decode tok/s"), ("ttft_s", "Request TTFT seconds"),
                               ("server_prefill_tps", "Native prefill tok/s")):
                metric = pair.get("metrics", {}).get(key, {})
                values = [row["device"], row["model"], pair["tier"], label,
                          *(metric_text(metric, field) for field in ("base", "candidate", "change_percent")),
                          "lower" if key == "ttft_s" else "higher"]
                stream.write("| " + " | ".join(summary_text(value) for value in values) + " |\n")
        stream.write("\nChange is (PR/base − 1) × 100; lower TTFT is better. TTFT starts at the request, not model loading. ")
        stream.write("One sample per arm: observed slowdown or different generated tokens is inconclusive and does not pass.\n\n")
        stream.write("<details><summary>Pair identities and retained raw evidence</summary>\n\n")
        for row, pair in pairs:
            identity = pair.get("measurement_identity", {})
            stream.write("- " + summary_text(row["device"] + "/" + row["model"] + "/" + pair["tier"]) + ": ")
            stream.write("prompt SHA `" + summary_text(identity.get("prompt_sha256", "unavailable")) + "`; ")
            stream.write("template IDs SHA `" + summary_text(identity.get("prompt_ids_sha256", "unavailable")) + "`; ")
            stream.write("conditions/settings SHA `" + summary_text(identity.get("conditions_sha256", "unavailable")) + "`.\n")
            for side in ("base", "candidate"):
                raw = pair.get("evidence", {}).get(side, {}).get("result.json", {})
                path = "/".join((row["device"], row.get("job_id", "unknown-job"), raw.get("path", "unavailable")))
                stream.write("  - " + side + " source `" + summary_text(pair.get("sources", {}).get(side, "unavailable"))
                             + "`: `" + summary_text(path) + "`, SHA `" + summary_text(raw.get("sha256", "unavailable")) + "`.\n")
        stream.write("\n</details>\n")
    if report.get("error"):
        stream.write("\nFailure: " + summary_text(report["error"]) + "\n")
    for error in report.get("errors", []):
        stream.write("\nFailure: " + summary_text(error) + "\n")
    stream.write("\nDevices outside this declared scope and any 4K deferred coverage remain separate. ")
    if "local-mbp" in report.get("execution_devices", []):
        stream.write("Mac memory protection monitors worker RSS/process count and host reserve; it does not disable swap or provide a Linux memory.max hard cap. ")
    stream.write("Formal benchmark samples: 0/16. Full comparison values and hashes are in aggregate.json; complete raw and cleanup evidence are in the retained selected-control artifacts.\n")


def operator_report(args, env=os.environ):
    """A terminal Actions artifact/summary, with no source status/check writes."""
    report = dict(schema_version=1, purpose=OPERATOR_PURPOSE, passed=False, status="failed", execution_scoped_status="failed",
                  linux_scoped_status="not-verified", mac_scoped_status="not-verified",
                  full_impact_coverage_complete=False, pr_publication="not-enabled",
                  source_identity="sealed-source-not-current-pr", formal_samples=0,
                  cells=[], errors=[])
    try:
        policy, event, identity = environment(args, env)
        actor = operator_access(policy, event, identity, env)
        envelope = operator_envelope(read_operator_envelope(event), policy, identity, actor)
        report.update(base=envelope["base"], head=envelope["head"], candidate=envelope["candidate"],
                      operator=actor, controller=identity, envelope_sha256=fingerprint(envelope),
                      **execution_scope(envelope))
        admitted = load(args.plan)
        bind_context(admitted, policy, env)
        if (any(admitted.get(key) != value for key, value in envelope.items())
                or admitted.get("envelope_sha256") != fingerprint(envelope)
                or admitted.get("operator") != actor
                or admitted.get("control_revision") != source_manifest(args.control)):
            raise ValueError("admitted operator envelope/control changed")
        impact = envelope["selection"]
        scope = execution_scope(envelope)
        jobs = jobs_checked(envelope["jobs"], impact, scope["execution_devices"], scope["validation_mode"])
        operator_job_controls(args, jobs)
        state = load(args.artifacts / "dispatch-state.json")
        if state["selection_plan_sha256"] != fingerprint(impact):
            raise ValueError("operator dispatch selection plan differs")
        if set(state["children"]) != set(jobs):
            raise ValueError("selected operator children are missing or unexpected")
        api = API(policy["controller_repository"], env["GITHUB_TOKEN"])
        children = {}
        for device, child in state["children"].items():
            latest = api.call("actions/runs/" + str(child["id"]))
            selection_dispatch.run_identity(latest, child["workflow_id"], identity["workflow_sha"], jobs[device]["job_id"])
            if latest.get("id") != child["id"] or latest.get("status") != "completed" or latest.get("conclusion") not in ("success", "failure"):
                raise ValueError("operator child is not an exact completed single attempt")
            children[device] = latest
        report = operator_aggregate(impact, policy, args.artifacts, jobs, children, scope)
        if state["errors"]:
            inconclusive_devices = {row["device"] for row in report["cells"] if row["status"] == "inconclusive"}
            retained = state.get("retained_failed_children", {})
            complete = all(retained.get(device) == children[device]["id"] for device in state["errors"] if device in children)
            if (not set(state["errors"]) <= inconclusive_devices or report.get("status") != "inconclusive"
                    or not complete):
                report.update(passed=False, status="failed", execution_scoped_status="failed", full_impact_coverage_complete=False)
                for platform, devices in (("linux", {"cloud-5090", "dgx-spark"}), ("mac", {"local-mbp"})):
                    if devices & set(state["errors"]):
                        report[platform + "_scoped_status"] = "failed"
            report["child_workflow_errors"] = state["errors"]
        report.update(base=envelope["base"], head=envelope["head"], candidate=envelope["candidate"],
                      operator=actor, controller=identity, envelope_sha256=fingerprint(envelope))
    except Exception as error:
        report.update(passed=False, status="failed", execution_scoped_status="failed", full_impact_coverage_complete=False,
                      error=f"{type(error).__name__}: {error}")
    save(args.output, report)
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            operator_summary(stream, report)
    return 0 if report["passed"] else 1


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
    parser.add_argument("mode", choices=("preflight", "operator_preflight", "execute", "cancel_children", "finalize", "operator_report"))
    for name in ("control", "policy", "plan", "artifacts", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("control", "policy", "plan"))
    args = parser.parse_args()
    raise SystemExit(globals()[args.mode](args))
