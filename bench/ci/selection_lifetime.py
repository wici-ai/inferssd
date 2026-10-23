"""Keep a selected child subordinate to its exact live hosted controller.

The read-only private Actions credential stays in this trusted coordinator UID.
Candidate stages never inherit it. A failed observation shuts down the leased
socket; only the existing broker guardian owns worker termination and the lock.
"""

import re
import threading


PARENT = re.compile(r"([1-9][0-9]*)-([1-9][0-9]*)")
WORKFLOW = ".github/workflows/device-performance.yml"


def active_parent(api, parent, sha):
    match = PARENT.fullmatch(parent)
    if not match:
        raise ValueError("invalid selected parent run/attempt")
    run_id, attempt = map(int, match.groups())
    workflow = api.call("actions/workflows/device-performance.yml")
    run = api.call("actions/runs/" + str(run_id))
    if (workflow.get("path") != WORKFLOW or workflow.get("state") != "active"
            or run.get("workflow_id") != workflow.get("id")
            or run.get("path") != WORKFLOW or run.get("event") != "workflow_dispatch"
            or run.get("head_branch") != "main" or run.get("head_sha") != sha
            or run.get("run_attempt") != attempt or run.get("status") != "in_progress"
            or run.get("conclusion") is not None):
        raise ValueError("selected parent is no longer the exact active controller")
    # The publisher may still run after control was cancelled. It must not keep
    # an independently dispatched device child alive during that interval.
    listing = api.call(f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
    jobs = listing["jobs"]
    if listing.get("total_count") != len(jobs) or len(jobs) >= 100:
        raise ValueError("incomplete parent job listing")
    control = [job for job in jobs if job.get("name") == "control"]
    if (len(control) != 1 or control[0].get("status") != "in_progress"
            or control[0].get("conclusion") is not None):
        raise ValueError("selected hosted control job is no longer active")


class ParentWatch:
    def __init__(self, api, parent, sha, interval=60):
        self.api, self.parent, self.sha, self.interval = api, parent, sha, interval
        self.session, self.error = None, None
        self.stopping = threading.Event()
        self.mutex = threading.Lock()
        self.thread = None

    def check(self):
        with self.mutex:
            if self.error is not None:
                raise RuntimeError(self.error)
        active_parent(self.api, self.parent, self.sha)

    def start(self):
        self.check()  # Refuse before opening any device lease.
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def attach(self, session):
        with self.mutex:
            self.session = session
            error = self.error
        if error is not None:
            session.abort()
            raise RuntimeError(error)

    def _run(self):
        while not self.stopping.wait(self.interval):
            try:
                self.check()
            except Exception as error:
                with self.mutex:
                    self.error = type(error).__name__ + ": " + str(error)
                    session = self.session
                if session is not None:
                    session.abort()
                return

    def stop(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(35)  # Three parent API calls, each bounded to 10s.
        with self.mutex:
            if self.error is not None:
                raise RuntimeError(self.error)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("parent observer did not terminate")
