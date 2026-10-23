"""Dispatch only the protected integration workflow; retain its exact child run.

One POST per fresh root-staged job. A lost/ambiguous response is a failure, not an
invitation to repeat a measurement. Tokens never follow artifact redirects.
"""

import io
from pathlib import Path, PurePosixPath
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from ci.contract import digest, positive, save

WORKFLOW = ".github/workflows/integration-smoke.yml"


def run_identity(run, workflow_id, controller_sha, job_id):
    if (run.get("workflow_id") != workflow_id or run.get("path") != WORKFLOW
            or run.get("event") != "workflow_dispatch" or run.get("head_branch") != "main"
            or run.get("head_sha") != controller_sha or run.get("run_attempt") != 1
            or run.get("display_title") != "integration-" + job_id):
        raise ValueError("child workflow/ref/revision/job/attempt identity differs")
    return run


def start(api, job, controller_sha, parent):
    workflow = api.call("actions/workflows/integration-smoke.yml")
    if workflow.get("path") != WORKFLOW or workflow.get("state") != "active":
        raise ValueError("reviewed integration workflow is not active at its fixed path")
    inputs = {key: job[key] for key in ("device", "job_id", "source_revision", "config_sha256", "selection_plan_sha256")}
    inputs["selection_parent"] = parent
    result = api.call("actions/workflows/" + str(workflow["id"]) + "/dispatches",
                      dict(ref="main", inputs=inputs, return_run_details=True), method="POST")
    # Current GitHub REST supports returning the exact run. Old/ambiguous 204 is
    # rejected without another POST; no heuristic search for a convenient pass.
    if not isinstance(result, dict):
        raise ValueError("dispatch returned no exact run ID; retain and stop without retry")
    run_id = positive(result.get("workflow_run_id"), "child run")
    return dict(id=run_id, workflow_id=workflow["id"], controller_sha=controller_sha,
                job_id=job["job_id"], inputs=inputs)


def wait(api, dispatched, *, timeout_s=21600, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout_s
    while clock() < deadline:
        run = api.call("actions/runs/" + str(dispatched["id"]))
        run_identity(run, dispatched["workflow_id"], dispatched["controller_sha"], dispatched["job_id"])
        if run["status"] == "completed":
            if run.get("conclusion") != "success":
                raise ValueError("selected child run did not complete successfully")
            return run
        sleep(min(30, max(0, deadline - clock())))
    # Cancellation initiates the existing disconnect guardian. Never declare
    # cleanup based on the cancellation API response; missing receipt is red.
    api.call("actions/runs/" + str(dispatched["id"]) + "/cancel", {}, method="POST")
    raise TimeoutError("selected child deadline; cleanup must be independently confirmed")


def cancel(api, dispatched):
    """Request cancellation of the exact child, never claim worker cleanup."""
    run = api.call("actions/runs/" + str(dispatched["id"]))
    run_identity(run, dispatched["workflow_id"], dispatched["controller_sha"], dispatched["job_id"])
    if run.get("status") != "completed":
        api.call("actions/runs/" + str(dispatched["id"]) + "/cancel", {}, method="POST")
    return dict(run_id=dispatched["id"], cancellation_requested=run.get("status") != "completed",
                cleanup_confirmed=False)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download(api, artifact):
    request = urllib.request.Request(
        "https://api.github.com/repos/" + api.repository + "/actions/artifacts/" + str(positive(artifact["id"], "artifact")) + "/zip",
        headers={"Authorization": "Bearer " + api.token, "Accept": "application/vnd.github+json"})
    try:
        urllib.request.build_opener(NoRedirect()).open(request, timeout=30)
    except urllib.error.HTTPError as response:
        if response.code != 302:
            raise
        location = response.headers["Location"]
    else:
        raise ValueError("artifact API did not return its signed download redirect")
    url = urllib.parse.urlsplit(location)
    if url.scheme != "https" or url.username or url.password or not url.hostname:
        raise ValueError("invalid artifact signed URL")
    # This request has no GitHub credential header.
    with urllib.request.urlopen(location, timeout=120) as response:
        raw = response.read((256 << 20) + 1)
    if len(raw) > 256 << 20:
        raise ValueError("artifact compressed size exceeds bound")
    import hashlib
    expected = artifact.get("digest")
    if expected is not None and expected != "sha256:" + hashlib.sha256(raw).hexdigest():
        raise ValueError("GitHub artifact digest differs")
    return raw


def unpack(raw, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist()) > 20000:
            raise ValueError("too many artifact members")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (not item.filename or path.is_absolute() or "\\" in item.filename
                    or any(p in ("", ".", "..") for p in item.filename.rstrip("/").split("/"))
                    or item.filename in seen or stat.S_ISLNK(mode)):
                raise ValueError("unsafe or duplicate artifact member")
            seen.add(item.filename)
            total += item.file_size
            if total > 512 << 20 or item.file_size > 64 << 20:
                raise ValueError("artifact uncompressed size exceeds bound")
            target = output.joinpath(*path.parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(item))


def collect(api, child, job, output, fetch=download, *, parent=None):
    listing = api.call("actions/runs/" + str(child["id"]) + "/artifacts?per_page=100")
    name = f"integration-smoke-{job['device']}-{child['id']}-1"
    matches = [item for item in listing["artifacts"] if item.get("name") == name and item.get("expired") is False]
    if len(matches) != 1:
        raise ValueError("selected child artifact missing or duplicated")
    unpack(fetch(api, matches[0]), output)
    from ci.contract import load
    dispatch = load(Path(output) / "dispatch.json")
    if any(dispatch.get(key) != job[key] for key in (
            "device", "job_id", "config_sha256", "selection_plan_sha256", "control_revision", "control_manifest_sha256")):
        raise ValueError("root-installed controller/job/plan artifact binding differs")
    if dispatch.get("revision") != job["source_revision"]:
        raise ValueError("child source differs from dispatched candidate")
    if parent is not None and dispatch.get("selection_parent") != parent:
        raise ValueError("selected child belongs to another parent run/attempt")
    save(Path(output) / "child-run.json", child)
    return dict(artifact_id=matches[0]["id"], dispatch_sha256=digest(Path(output) / "dispatch.json"))
