#!/usr/bin/env python3
"""Private exact-SHA admission and App-only publication; never execute a candidate."""

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.compare import aggregate
from ci.contract import (
    DEVICES,
    HASH,
    classify,
    digest,
    load,
    policy_read,
    positive,
    revision,
    save,
)


class API:
    def __init__(self, repository, token):
        self.repository, self.token = repository, token

    def call(self, path, data=None, method=None):
        request = urllib.request.Request(
            "https://api.github.com/repos/" + self.repository + "/" + path,
            None if data is None else json.dumps(data).encode(),
            {
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)


def context(policy, env, event):
    repo = policy["controller_repository"]
    workflow = (
        repo + "/" + policy["controller_workflow"] + "@" + policy["controller_ref"]
    )
    if (
        env.get("GITHUB_REPOSITORY") != repo
        or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or env.get("GITHUB_REF") != policy["controller_ref"]
        or env.get("GITHUB_WORKFLOW_REF") != workflow
        or event.get("repository", {}).get("private") is not True
        or event.get("repository", {}).get("full_name") != repo
    ):
        raise ValueError(
            "physical admission requires the protected private controller workflow/ref"
        )
    sha = revision(env.get("GITHUB_WORKFLOW_SHA"))
    if revision(env.get("GITHUB_SHA")) != sha:
        raise ValueError("controller workflow/source revision mismatch")
    actor = env.get("GITHUB_ACTOR", "")
    if not actor or actor != event.get("sender", {}).get("login"):
        raise ValueError("dispatch actor is missing or changed")
    return {
        "repository": repo,
        "workflow": workflow,
        "workflow_sha": sha,
        "run_id": positive(int(env["GITHUB_RUN_ID"]), "run id"),
        "run_attempt": positive(int(env["GITHUB_RUN_ATTEMPT"]), "attempt"),
        "actor": actor,
    }


def writable(api, login):
    return api.call(
        "collaborators/" + urllib.parse.quote(login, safe="") + "/permission"
    ).get("permission") in ("admin", "maintain", "write")


def admission(api, pr, policy, expected_head):
    head = revision(expected_head)
    if (
        pr.get("state") != "open"
        or pr.get("draft") is not False
        or pr.get("head", {}).get("repo", {}).get("full_name") != policy["repository"]
        or pr.get("base", {}).get("repo", {}).get("full_name") != policy["repository"]
        or pr.get("base", {}).get("ref") != "main"
        or pr.get("head", {}).get("sha") != head
        or not pr.get("head", {}).get("ref", "").startswith(policy["staging_prefix"])
        or pr.get("mergeable") is not True
    ):
        raise ValueError(
            "requires reviewed current SHA on a trusted staging PR; forks stay hosted-only"
        )
    base, candidate = revision(pr["base"]["sha"]), revision(pr.get("merge_commit_sha"))
    branch = api.call("branches/" + urllib.parse.quote(pr["head"]["ref"], safe=""))
    if (
        branch.get("protected") is not True
        or branch.get("commit", {}).get("sha") != head
    ):
        raise ValueError(
            "staging branch must be protected and still identify the reviewed SHA"
        )
    if [p.get("sha") for p in api.call("commits/" + candidate).get("parents", [])] != [
        base,
        head,
    ]:
        raise ValueError(
            "candidate must merge precisely the current base and reviewed head"
        )
    reviews = []
    for page in range(1, 32):
        rows = api.call(f"pulls/{pr['number']}/reviews?per_page=100&page={page}")
        reviews.extend(rows)
        if len(rows) < 100:
            break
    else:
        raise ValueError("incomplete review listing")
    latest = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        if review.get("state") != "COMMENTED":
            latest[review.get("user", {}).get("login")] = review
    for login, review in latest.items():
        if (
            login
            and review.get("state") == "CHANGES_REQUESTED"
            and writable(api, login)
        ):
            raise ValueError(
                "an outstanding maintainer request for changes blocks physical admission"
            )
    approved = []
    for login, review in latest.items():
        if (
            login
            and login != pr.get("user", {}).get("login")
            and review.get("commit_id") == head
            and review.get("state") == "APPROVED"
            and ("[device-ci-approved:" + head + "]")
            in review.get("body", "").splitlines()
            and writable(api, login)
        ):
            approved.append(
                {
                    "id": positive(review["id"], "review id"),
                    "reviewer": login,
                    "head": head,
                }
            )
    if not approved:
        raise ValueError("a different maintainer must approve this exact head SHA")
    return {
        "schema_version": 1,
        "purpose": "ci",
        "repository": policy["repository"],
        "pr": positive(pr["number"], "PR"),
        "head": head,
        "base": base,
        "candidate": candidate,
        "authorization": {"staging_ref": pr["head"]["ref"], "reviews": approved},
    }


def current_ci(api, plan, policy):
    workflow = api.call("actions/workflows/" + Path(policy["upstream_workflow"]).name)
    runs = api.call(
        "actions/workflows/"
        + str(workflow["id"])
        + "/runs?event=pull_request&head_sha="
        + plan["head"]
        + "&per_page=100"
    )["workflow_runs"]
    if not runs:
        raise ValueError("normal hosted CI has not completed for the reviewed SHA")
    run = api.call("actions/runs/" + str(max(runs, key=lambda r: r["id"])["id"]))
    prs = run.get("pull_requests", [])
    if (
        run.get("workflow_id") != workflow["id"]
        or run.get("path") != policy["upstream_workflow"]
        or run.get("event") != "pull_request"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("head_sha") not in (plan["head"], plan["candidate"])
        or len(prs) != 1
        or prs[0].get("number") != plan["pr"]
        or prs[0].get("head", {}).get("sha") != plan["head"]
        or prs[0].get("base", {}).get("sha") != plan["base"]
    ):
        raise ValueError(
            "normal CI is not the successful latest attempt for this exact head/base"
        )
    return {
        "id": positive(run["id"], "CI run"),
        "attempt": positive(run["run_attempt"], "CI attempt"),
        "workflow_id": workflow["id"],
        "head": plan["head"],
        "base": plan["base"],
    }


def source_manifest(control):
    manifest = load(control / "control-source.json")
    revision(manifest["revision"])
    if not manifest.get("files"):
        raise ValueError("missing exported control manifest")
    for relative, expected in manifest["files"].items():
        path = control / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or digest(path) != expected
        ):
            raise ValueError("exported control source differs from reviewed manifest")
    return manifest["revision"]


def preflight(args, env=os.environ):
    policy = policy_read(load(args.policy))
    event = load(env["GITHUB_EVENT_PATH"])
    identity = context(policy, env, event)
    controller = API(policy["controller_repository"], env["GITHUB_TOKEN"])
    metadata = controller.call("")
    if (
        metadata.get("private") is not True
        or metadata.get("default_branch") != "main"
        or not writable(controller, identity["actor"])
    ):
        raise ValueError("private controller metadata/dispatch permission is invalid")
    if env.get("INFERSSD_CI_ENABLED") != "true":
        raise ValueError(
            "physical activation disabled until real isolation and native cleanup gates pass"
        )
    inputs = event.get("inputs", {})
    text = inputs.get("pull_request", "")
    if not text.isdecimal() or int(text) < 1:
        raise ValueError("positive explicit PR number required")
    expected = revision(inputs.get("head_sha"))
    api = API(policy["repository"], env["SOURCE_READ_TOKEN"])
    plan = admission(api, api.call("pulls/" + text), policy, expected)
    plan["normal_ci"] = current_ci(api, plan, policy)
    paths, count = [], 0
    for page in range(1, 32):
        rows = api.call(f"pulls/{text}/files?per_page=100&page={page}")
        count += len(rows)
        for row in rows:
            paths.append(row["filename"])
            if row.get("previous_filename"):
                paths.append(row["previous_filename"])
        if len(rows) < 100:
            break
    else:
        raise ValueError("incomplete changed file listing")
    current = api.call("pulls/" + text)
    if current.get("changed_files") != count:
        raise ValueError("incomplete changed file listing")
    again = admission(api, current, policy, expected)
    if any(again[k] != plan[k] for k in ("head", "base", "candidate", "authorization")):
        raise ValueError("PR/authorization changed during admission")
    hashes = json.loads(env.get("INFERSSD_CI_DEVICE_CONFIG_HASHES", "{}"))
    if set(hashes) != set(DEVICES) or any(
        not isinstance(v, str) or not HASH.fullmatch(v) for v in hashes.values()
    ):
        raise ValueError("all device broker/config identities must be provisioned")
    plan.update(
        classify(list(dict.fromkeys(paths)), policy),
        controller=identity,
        control_revision=source_manifest(args.control),
        policy_sha256=digest(args.policy),
        device_config_sha256=hashes,
    )
    save(args.plan, plan)
    with open(env["GITHUB_OUTPUT"], "a") as output:
        output.write(
            "candidate="
            + plan["candidate"]
            + "\ncode="
            + str(plan["code"]).lower()
            + "\n"
        )
        output.write(
            "matrix="
            + json.dumps(
                {
                    "include": [
                        {
                            "device": d,
                            "labels": [
                                "self-hosted",
                                "inferssd-private",
                                "inferssd-" + d,
                            ],
                        }
                        for d in DEVICES
                    ]
                }
            )
            + "\n"
        )


def bind_context(plan, policy, env):
    if plan.get("controller") != context(policy, env, load(env["GITHUB_EVENT_PATH"])):
        raise ValueError("plan belongs to another controller run/attempt/ref/actor")


def check_body(plan, report, app_id):
    identity = plan["controller"]
    external = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "name": "InferSSD device performance",
        "head_sha": plan["head"],
        "status": "completed",
        "conclusion": "success" if report["passed"] else "failure",
        "external_id": f"{identity['repository']}:{identity['run_id']}:{identity['run_attempt']}:{external}",
        "details_url": f"https://github.com/{identity['repository']}/actions/runs/{identity['run_id']}/attempts/{identity['run_attempt']}",
        "output": {
            "title": "Three device gates passed"
            if report["passed"]
            else "Device gate blocked",
            "summary": "Exact reviewed head/base/candidate; one sample per side/cell. Observed slowdown blocks acceptance as inconclusive; no automatic retries. Formal samples 0.\n"
            + report.get("error", ""),
        },
        "_expected_app_id": positive(app_id, "device check App id"),
    }


def finalize(args, env=os.environ):
    policy = policy_read(load(args.policy))
    policy["_sha256"] = digest(args.policy)
    plan = load(args.plan)
    bind_context(plan, policy, env)
    if plan.get("policy_sha256") != policy["_sha256"] or plan.get(
        "control_revision"
    ) != source_manifest(args.control):
        raise ValueError("plan/control/policy identity differs")
    api = API(policy["repository"], env["DEVICE_CHECK_TOKEN"])
    report = {"purpose": "ci", "passed": False, "formal_samples": 0}
    try:
        if plan["code"]:
            report = aggregate(
                [args.artifacts / ("device-" + d) for d in DEVICES], plan, policy
            )
        else:
            report.update(
                passed=True, reason="documentation-only; no physical execution"
            )
        current = admission(
            api, api.call("pulls/" + str(plan["pr"])), policy, plan["head"]
        )
        if any(
            current[k] != plan[k]
            for k in ("base", "head", "candidate", "authorization")
        ):
            raise ValueError("head/base/candidate or exact-SHA review changed")
        if current_ci(api, plan, policy) != plan["normal_ci"]:
            raise ValueError("normal CI run/attempt changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(passed=False, error=str(error))
    save(args.output, report)
    body = check_body(plan, report, int(env["DEVICE_CHECK_APP_ID"]))
    expected = body.pop("_expected_app_id")
    check = api.call("check-runs", body)
    if (
        check.get("app", {}).get("id") != expected
        or check.get("head_sha") != plan["head"]
        or check.get("external_id") != body["external_id"]
    ):
        raise ValueError("check was not emitted by the dedicated App for this run/head")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "finalize"))
    for key in ("control", "policy", "plan"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(preflight(args) if args.mode == "preflight" else finalize(args))
