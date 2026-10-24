# Review-authorized two-Linux operator entry

The source PR maintainer submits a comment-type review whose complete body is
`/ci linux`. GitHub's review `commit_id` binds the exact current head. A current
writable maintainer may authorize their own PR; no special source branch,
second-person approval, App or PAT is needed in this operator/report-only mode.
An external fork author need not have source write access: a source maintainer's
review authorizes that exact head, while the separate preparation service account
uses its own existing permissions. Reviewer need not have controller-repository
write permission or equal `prepared_by`. Author-provided commands do not count
unless that author independently has current source-maintainer permission.
Ordinary issue comments, labels, quoted code/examples and prose are not commands.
The test authorization does not approve a merge or publish a source PR check.

The poller runs on the trusted preparation host with its existing gh login and
reviewed source Git repository and two-device profiles. It does not run on or
depend on the offline tested M5. Its persistent state, exact command and process
supervisor must be deployed on the chosen always-online preparation host; this
source change alone does not establish that the background process is running.

```
python3 -I bench/ci/review_operator.py \
  --source /ABSOLUTE/TRUSTED/SOURCE_REPO --fetch-remote ssdash \
  --inputs /ABSOLUTE/REVIEWED/profiles.json \
  --state /ABSOLUTE/PRIVATE/review-operator-state --gh /path/to/existing/gh
```

Default polling is30 seconds; `--once` is a diagnostic, not sustained activation.
The private state directory is owner-only. A held local flock excludes duplicate
pollers. Each repo/PR/review_id/head tuple has one durable record. Status files and
atomic records are fsynced, so a stopped host queues authorization in GitHub and
resumes observing it later. `poll-status.json` exposes blocked API/authorization
and source-merge readiness, and records retain normal-CI/profile waits. An offline
or blocked preparation host is not a successful CI attempt.

The entry reuses `operator_prepare.prepare`, `stage` and `dispatch`. It fetches
missing exact commit objects only from a configured remote whose URL identifies
the fixed source repository's `refs/pull/N/head` and `refs/pull/N/merge`; it never
fetches a fork URL, pushes/copies a branch, checks out or executes candidate code on
the preparation host. Before preparation and again after root staging, it checks
the PR is open/nondraft with the fixed source base repository, the review remains present/authorized on
the exact head, current base/merge identity and latest normal hosted CI success.
Profile/policy changes cannot be silently applied to an existing authorization.

Background staging accepts only already reviewed root transport or the fixed
narrow passwordless root helper. It never prompts, reads, stores or forwards a
personal password. The separately authorized one-time operator bootstrap must
install that exact helper/policy; the poller neither edits sudoers nor grants
arbitrary Python/root commands. Missing unattended staging permission stays
queued with an explicit reason.

After dispatch it monitors only the returned exact parent run ID, workflow,
controller SHA, actor and attempt1. A new head/base, edited/dismissed authorization,
later current-head maintainer `/ci cancel` review or request for changes, lost
authorization or finite deadline requests cancellation of that exact parent.
The existing parent's cancellation/always-run cleanup and child guardian chain
remain responsible for physical cleanup; a cancel API response never proves it.
An API/transport cancellation failure is retained with its attempt/error and
retried on the next poll against the same freshly verified exact parent until
acknowledgement or completion. Cancellation is idempotent; this never repeats a
dispatch or measurement. A successful acknowledgement is not posted again.
The record retains the Actions result and points to its summary/artifacts.

Preparation/staging interruption and ambiguous dispatch are retained without
automatic replay. A saved, already validated `dispatched-run.json` allows a
restart to resume monitoring without another POST. Otherwise it requires
inspection and a fresh review authorization; no guessed run, repeated model
sample or best result selection exists. Queue authorization expires after24h
(configurable bounded lifetime); parent monitoring is bounded by4h, matching
the existing workflow limit. Original completed attempts remain recorded after
a source update and cannot satisfy a new head.

Activation still requires deploying the final trusted controller/wheel, exact
root policy/source/model staging and current job before/after probes on both
Linux devices, then one real review→prepare→dispatch→Actions result plus a
finite cancellation/cleanup case. Mac remains deferred and formal samples0/16.
This module's contract tests execute none of those physical operations.
