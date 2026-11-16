#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Label new issues and pull requests and ask their owner to pick them up.

An issue form already sets its type label (bug, app, model-request, enhancement, perf
or question). This adds the labels its answers name and assigns the issue to the owner
of the first matching rule. A pull request gets one label per area in its title and
asks that area's owner for a review. When an issue's answers or a pull request's title
change, the labels the old text named and the new one does not are removed; labels
added by hand stay.

Run by .github/workflows/triage.yml with the event from GITHUB_EVENT_NAME and
GITHUB_EVENT_PATH. With --dry-run it prints the calls instead of making them.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Form answers that become labels: the question's heading, then each option's label.
# An option mapped to None adds none.
FIELDS = {
    "Operating system": {"Linux": "linux", "macOS": "macos", "Windows": "windows"},
    "Backend": {"CUDA": "cuda", "Metal": "metal", "Vulkan": "vulkan", "CPU only": "cpu", "Not sure": None},
    "Model storage": {"One drive": None, "Several drives or a RAID volume": "multi-drive"},
    "Kind": {"Problem": "bug", "Request": "enhancement"},
}
# An issue goes to the owner of the first label it carries here, else to FALLBACK.
ISSUE_OWNERS = (("app", "DearJohnn"), ("model-request", "Bi666666"), ("multi-drive", "ustc-sunny"),
                ("perf", "YOURENXIAKE"))
# A pull request asks the owner of the first title area here, else FALLBACK.
PULL_OWNERS = (("model", "Bi666666"),)
FALLBACK = "Chikati"


def load_title_rules():
    spec = importlib.util.spec_from_file_location(
        "check_pr_title", Path(__file__).with_name("check-pr-title.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TITLE_RULES = load_title_rules()


def answers(body):
    """An issue form's answers by heading; GitHub renders each as '### heading' then the answer."""
    found = {}
    text = "\n" + (body or "").replace("\r\n", "\n")
    for part in re.split(r"\n### ", text)[1:]:
        heading, _, answer = part.partition("\n")
        found[heading.strip()] = answer.strip()
    return found


def issue_labels(body):
    """The labels an issue form's answers name."""
    labels = set()
    for heading, answer in answers(body).items():
        options = FIELDS.get(heading)
        if options is None:
            continue
        for choice in answer.split(", "):
            label = options.get(choice.strip())
            if label:
                labels.add(label)
    return labels


def title_areas(title):
    """The known areas a pull request title names, in title order."""
    match = TITLE_RULES.FORM.fullmatch((title or "").strip())
    if not match:
        return []
    areas = []
    for area in match.group("areas").split(", "):
        if area in TITLE_RULES.AREAS and area not in areas:
            areas.append(area)
    return areas


def issue_owner(labels):
    return next((owner for label, owner in ISSUE_OWNERS if label in labels), FALLBACK)


def pull_owner(areas):
    return next((owner for area, owner in PULL_OWNERS if area in areas), FALLBACK)


def names(labels):
    return {label["name"] for label in labels or ()}


def plan(event_name, event):
    """The calls the event needs, as (method, path, body) relative to the repository."""
    action = event.get("action")
    calls = []
    if event_name == "issues":
        issue = event["issue"]
        number = issue["number"]
        new = issue_labels(issue.get("body"))
        current = names(issue.get("labels"))
        if action == "edited":
            if "body" not in event.get("changes", {}):
                return calls
            old = issue_labels(event["changes"]["body"].get("from"))
            for label in sorted((old - new) & current):
                calls.append(("DELETE", f"/issues/{number}/labels/{urllib.parse.quote(label, safe='')}", None))
        if new - current:
            calls.append(("POST", f"/issues/{number}/labels", {"labels": sorted(new - current)}))
        if action == "opened" and not issue.get("assignees"):
            calls.append(("POST", f"/issues/{number}/assignees", {"assignees": [issue_owner(current | new)]}))
    elif event_name == "pull_request_target":
        pull = event["pull_request"]
        number = pull["number"]
        new = set(title_areas(pull.get("title")))
        current = names(pull.get("labels"))
        if action == "edited":
            if "title" not in event.get("changes", {}):
                return calls
            old = set(title_areas(event["changes"]["title"].get("from")))
            for label in sorted((old - new) & current):
                calls.append(("DELETE", f"/issues/{number}/labels/{urllib.parse.quote(label, safe='')}", None))
        if new - current:
            calls.append(("POST", f"/issues/{number}/labels", {"labels": sorted(new - current)}))
        if action in ("opened", "reopened", "ready_for_review") and not pull.get("draft"):
            owner = pull_owner(title_areas(pull.get("title")))
            requested = {user["login"] for user in pull.get("requested_reviewers") or ()}
            if owner != pull["user"]["login"] and owner not in requested:
                calls.append(("POST", f"/pulls/{number}/requested_reviewers", {"reviewers": [owner]}))
    return calls


def call(api, repository, token, method, path, body):
    request = urllib.request.Request(
        f"{api}/repos/{repository}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        # A label someone removed meanwhile is already gone.
        if not (method == "DELETE" and error.code == 404):
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the calls instead of making them")
    args = parser.parse_args()
    event_name = os.environ["GITHUB_EVENT_NAME"]
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
        event = json.load(handle)
    for method, path, body in plan(event_name, event):
        print(method, path, json.dumps(body) if body is not None else "")
        if not args.dry_run:
            call(os.environ.get("GITHUB_API_URL", "https://api.github.com"), os.environ["GITHUB_REPOSITORY"],
                 os.environ["GH_TOKEN"], method, path, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
