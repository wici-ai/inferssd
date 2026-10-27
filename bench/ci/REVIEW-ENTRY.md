# Review-authorized device test entry

The source PR maintainer submits a comment-type review whose complete body is
`/ci test`. GitHub's review `commit_id` binds the exact current head. A current
writable maintainer may authorize their own PR; no special source branch,
second-person approval, App or PAT is needed in this operator/report-only mode.
An external fork author need not have source write access: a source maintainer's
review authorizes that exact head, while the separate preparation service account
uses its own existing permissions. Reviewer need not have controller-repository
write permission or equal `prepared_by`. Author-provided commands do not count
unless that author independently has current source-maintainer permission.
Ordinary issue comments, labels, quoted code/examples and prose are not commands.
The test authorization does not approve a merge or publish a source PR check.

The impact plan selects models and checks. Reviewed execution profiles may select
cloud-5090, DGX Spark and local-mbp; each omitted device stays explicitly deferred.
Required performance uses
the unchanged Azure 512-tier prompt and decode 256, once per base and candidate
on the same device/model/settings. It does not add 4K or formal benchmark samples.
Functional-only success cannot satisfy a required performance comparison.

The removed Linux-specific command is not an alias. Existing review/run records
are retained without replay; upgrading the operator never extends an old review
to a new performance run. Submit a fresh `/ci test` Comment review to authorize
the current head. `/ci cancel` remains the cancellation command.

The poller runs on the trusted preparation host with its existing gh login and
reviewed source Git repository and explicit device profiles. Its service must
remain available independently of the tested M5. Its persistent state, exact command and process
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

Background staging accepts already reviewed root transport or the fixed
stager's restricted local socket client. An existing permitted narrow
passwordless sudo helper is also supported. The socket service checks the
connecting UID and runs only the trusted receiver under its root-owned policy,
original device lease and finite byte/time limits. It does not expose a shell
or candidate execution. The separately authorized one-time operator bootstrap
installs this CI service; it must preserve the host's vendor access policy.
The poller never prompts, reads, stores or forwards a personal password and
does not edit sudoers. Missing unattended staging permission stays queued
with an explicit reason.

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

Open the exact parent run in the private `inferssd-device-ci` repository, then
read its Actions Summary. For each required executed model it shows base and PR
0–128/128–256 decode token/s and `(PR / base - 1) * 100%`, plus TTFT, prefill
and whole decode. A positive throughput change means faster; a lower TTFT is
better. Missing, failed and inconclusive performance stays non-passing. Device
deferrals and full impact coverage are reported separately from each execution scope.
The `selected-aggregate-RUN-ATTEMPT` artifact contains `aggregate.json`;
`selected-control-RUN-ATTEMPT` retains the raw two-sided requests, timestamps,
settings/source/prompt identities, observations and cleanup evidence.

These request metrics exclude model/vocabulary loading before server readiness.
Each profile establishes only its own hardware, models, resource constraints and
cache settings. Mac uses the root executor's sampled RSS/process/free-memory
guards; its receipts do not assert Linux cgroup hard UMA or swap isolation.

Preparation/staging interruption and ambiguous dispatch are retained without
automatic replay. A saved, already validated `dispatched-run.json` allows a
restart to resume monitoring without another POST. Otherwise it requires
inspection and a fresh review authorization; no guessed run, repeated model
sample or best result selection exists. Queue authorization expires after24h
(configurable bounded lifetime); parent monitoring is bounded by4h, matching
the existing workflow limit. Original completed attempts remain recorded after
a source update and cannot satisfy a new head.

Activation requires deploying the final trusted controller/wheel, exact root
policy/source/model staging and current job before/after probes on each enabled
device, then one real review→prepare→dispatch→Actions result plus a finite
cancellation/cleanup case. Mac uses the fixed SSH entry and recovery contract in
[RECOVERY.md](macos-sandbox/RECOVERY.md). Code, a runner's online state and existing
short readiness receipts alone do not establish a new performance run. Formal samples0/16.
This module's contract tests execute none of those physical operations.
