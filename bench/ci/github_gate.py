#!/usr/bin/env python3
"""Trusted GitHub-hosted preflight/finalizer; never execute an artifact as code."""

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.compare import aggregate
from ci.contract import DEVICES, authorize, classify, digest, load, policy_read, save


class GitHub:
    def __init__(self, repository):
        self.repository = repository

    def call(self, path, data=None, method=None):
        request = urllib.request.Request(
            "https://api.github.com/repos/" + self.repository + "/" + path,
            json.dumps(data).encode() if data is not None else None,
            {
                "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as stream:
            return json.load(stream)


def outputs(values):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(
                key
                + "="
                + (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, separators=(",", ":"))
                )
                + "\n"
            )


def preflight(args):
    policy = policy_read(load(args.policy))
    repository = os.environ["GITHUB_REPOSITORY"]
    if repository != policy["repository"]:
        raise ValueError("workflow repository differs from trusted InferSSD policy")
    if os.environ.get("INFERSSD_CI_ENABLED") != "true":
        outputs(
            {"enabled": "false", "reason": "device runners/configuration not activated"}
        )
        return
    event = load(os.environ["GITHUB_EVENT_PATH"])
    observed = None
    if os.environ["GITHUB_EVENT_NAME"] == "workflow_run":
        run = event["workflow_run"]
        if (
            run.get("event") != "pull_request"
            or run.get("conclusion") != "success"
            or len(run.get("pull_requests", [])) != 1
        ):
            outputs(
                {"enabled": "false", "reason": "not one successful current PR CI run"}
            )
            return
        number, observed = run["pull_requests"][0]["number"], run["head_sha"]
    elif os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        text = event.get("inputs", {}).get("pull_request", "")
        if not text.isdecimal() or int(text) < 1:
            raise ValueError("manual execution requires a positive PR number")
        number = int(text)
    else:
        raise ValueError("unsupported trusted workflow event")
    api = GitHub(repository)
    pr = api.call("pulls/" + str(number))
    plan = authorize(pr, repository, policy, observed_sha=observed)
    paths, files_seen = [], 0
    for page in range(1, 32):
        rows = api.call(f"pulls/{number}/files?per_page=100&page={page}")
        files_seen += len(rows)
        for row in rows:
            paths.append(row["filename"])
            if row.get("previous_filename"):
                paths.append(row["previous_filename"])
        if len(rows) < 100:
            break
    else:
        raise ValueError("too many changed files to classify safely")
    if type(pr.get("changed_files")) is not int or files_seen != pr["changed_files"]:
        raise ValueError("GitHub changed-file listing is incomplete")
    current = authorize(
        api.call("pulls/" + str(number)), repository, policy, observed_sha=observed
    )
    if any(current[k] != plan[k] for k in ("base", "head", "candidate")):
        raise ValueError("PR changed during preflight")
    plan.update(
        classify(list(dict.fromkeys(paths)), policy),
        policy_sha256=digest(args.policy),
        control_revision=subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
    )
    hashes = json.loads(os.environ.get("INFERSSD_CI_DEVICE_CONFIG_HASHES", "{}"))
    from ci.contract import HASH

    if set(hashes) != set(DEVICES) or any(
        not isinstance(v, str) or not HASH.fullmatch(v) for v in hashes.values()
    ):
        raise ValueError(
            "every device needs its reviewed config SHA-256 before activation"
        )
    plan["device_config_sha256"] = hashes
    check = api.call(
        "check-runs",
        {
            "name": "InferSSD device performance",
            "head_sha": plan["head"],
            "status": "in_progress" if plan["code"] else "completed",
            **({"conclusion": "success"} if not plan["code"] else {}),
            "output": {
                "title": "Three-device CI" if plan["code"] else "Documentation-only PR",
                "summary": plan["reason"]
                + "; CI samples are separate from formal benchmark samples.",
            },
        },
    )
    plan["check_id"] = check["id"]
    save(args.plan, plan)
    outputs(
        {
            "enabled": "true" if plan["code"] else "false",
            "plan": str(args.plan),
            "check_id": str(check["id"]),
            "head": plan["head"],
            "candidate": plan["candidate"],
            "control": plan["control_revision"],
            "matrix": {
                "include": [
                    {
                        "device": device,
                        "labels": ["self-hosted", "inferssd-ci", "inferssd-" + device],
                    }
                    for device in DEVICES
                ]
            },
        }
    )


def finalize(args):
    policy = policy_read(load(args.policy))
    policy["_sha256"] = digest(args.policy)
    plan = load(args.plan)
    if (
        plan.get("repository") != os.environ["GITHUB_REPOSITORY"]
        or plan["policy_sha256"] != policy["_sha256"]
    ):
        raise ValueError("finalizer plan/policy belongs to another workflow")
    api = GitHub(plan["repository"])
    report = {"purpose": "ci", "passed": False, "formal_samples": 0}
    try:
        current = authorize(
            api.call("pulls/" + str(plan["pr"])), plan["repository"], policy
        )
        if any(current[k] != plan[k] for k in ("base", "head", "candidate")):
            raise ValueError("PR head/base moved while device CI was running")
        report = aggregate(
            [args.artifacts / ("device-" + d) for d in DEVICES], plan, policy
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(passed=False, error=str(error))
    save(args.output, report)
    title = (
        "All three real-device gates passed"
        if report["passed"]
        else "Device gate blocked"
    )
    limits = policy["thresholds"]
    summary = (
        f"One sample per side/cell. Provisional decode{100 * limits['decode_regression_fraction']:g}%/TTFT{100 * limits['ttft_regression_fraction']:g}% policy; no statistical claim. "
        "Raw evidence and JSON comparison are in private workflow artifacts. Formal samples:0.\n\n"
    )
    if "error" in report:
        summary += report["error"]
    for device in report.get("devices", []):
        for row in device["rows"]:
            summary += f"\n- {row['device']} / {row['model']} / {row['tier']}: {'pass' if row['passed'] else 'blocked'}"
            for metric, data in row.get("metrics", {}).items():
                summary += f"; {metric} {data['base']:.4f} → {data['candidate']:.4f} ({data['ratio']:.4f}×)"
            if row.get("error"):
                summary += "; " + row["error"]
    api.call(
        "check-runs/" + str(plan["check_id"]),
        {
            "status": "completed",
            "conclusion": "success" if report["passed"] else "failure",
            "output": {"title": title, "summary": summary},
        },
        method="PATCH",
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "finalize"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise SystemExit("legacy public workflow admission is disabled; use the private exact-SHA controller")
    raise SystemExit(preflight(args) if args.mode == "preflight" else finalize(args))
