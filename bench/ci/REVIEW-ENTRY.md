# Review-authorized device test entry

Device checks are authorized by a maintainer for PRs from branches in the
source repository. The maintainer submits a comment-type review whose complete
body is `/ci test`. GitHub's review `commit_id` binds the exact current head. A
maintainer with current write access may authorize their own PR; no special
source branch, second-person approval, App or PAT is needed in this
operator/report-only mode. The preparation service account uses its own existing
permissions, and the reviewer does not need controller-repository write
permission. Commands from anyone without current source write permission do not
count. Ordinary issue comments, labels, quoted code/examples and prose are not
commands. The test authorization does not approve a merge or publish a source PR
check.

External fork PRs do not currently run on physical devices. Authorization also
requires the latest successful hosted CI run for the exact head to be associated
with exactly that PR (`pull_requests` must contain one entry), and GitHub does not
associate workflow runs from forks with the PR. A `/ci test` review on a fork PR
therefore stays waiting until it expires. To test an external contribution on
devices, a maintainer can open a PR from a branch in the source repository with
the reviewed change and authorize that head.

The impact plan selects models and checks. Reviewed execution profiles may select
cloud-5090, dgx-spark and local-mbp; each omitted device stays explicitly deferred.
Required performance uses the fixed 512-token CI prompt
(`bench/prompts/prompt-512.txt`) and decode 256: three PR runs and one base run
(PR, base, PR, PR) on the same device, model and launch request, each side
launching with the settings its own strategy derives, decided against the base
commit's baseline history on that device (see
[performance acceptance](PERFORMANCE.md)). It does not add 4K or formal benchmark
samples. Functional-only success cannot satisfy a required performance comparison.

The poller also watches `main`. Each new head is queued as one baseline job
that builds that commit once and runs each model four times on every enabled
device; no review authorizes it, and nothing else triggers one. PR and baseline
jobs run one at a time in the order they were queued, so a PR queued after a
merge finds its base measured. When a baseline job completes, the poller
downloads its hosted report and adds the verified cells to the
`baseline-history` directory next to the device profiles. A PR whose base has no
history still runs; its cells are `needs-review` ("no baseline history").

Review and run records are retained without replay; upgrading the operator
never extends an old review to a new performance run. Submit a fresh `/ci test`
Comment review to authorize the current head. `/ci cancel` is the cancellation
command.

The poller runs on the trusted preparation host with its existing gh login and
reviewed source Git repository and explicit device profiles. Its service must
remain available independently of the tested devices. Its persistent state,
exact command and process supervisor must be deployed on an always-online
preparation host; this code alone does not establish that the background
process is running.

```
python3 -I bench/ci/review_operator.py \
  --source /ABSOLUTE/TRUSTED/SOURCE_REPO --fetch-remote origin \
  --inputs /ABSOLUTE/REVIEWED/profiles.json \
  --state /ABSOLUTE/PRIVATE/review-operator-state --gh /path/to/existing/gh
```

`--fetch-remote` names a remote of that local repository whose URL identifies the
fixed source repository. Default polling is 30 seconds; `--once` is a diagnostic,
not sustained activation. The private state directory is owner-only. A held local flock excludes duplicate
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
The baseline history is not part of that binding: each job seals the records
it uses when it is prepared.

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
controller SHA, actor and attempt 1. A new head/base, edited/dismissed authorization,
later current-head maintainer `/ci cancel` review or request for changes, lost
authorization or finite deadline requests cancellation of that exact parent.
The existing parent's cancellation/always-run cleanup and child guardian chain
remain responsible for physical cleanup; a cancel API response never proves it.
An API/transport cancellation failure is retained with its attempt/error and
retried on the next poll against the same freshly verified exact parent until
acknowledgement or completion. Cancellation is idempotent; this never repeats a
dispatch or measurement. A successful acknowledgement is not posted again.
The record retains the Actions result and points to its summary/artifacts.

Hosted API reads and the local `gh` adapter share `read_retry.py`'s error
classification and finite backoff policy. Only a GET without a request body may
recover from the listed temporary HTTP/transport failures, with at most three
attempts and no cached authorization. Hosted calls share one monotonic scheduling
budget (normally 30 seconds), each socket timeout is at most 10 seconds, and child
metadata reads also retain their existing 10-second observation deadline.
Socket inactivity timeouts do not establish a hard wall limit against a peer
that continually delivers bytes. The local adapter reserves bounded process
draining within its request budget. Permission denials, certificate failures and
unknown errors stop immediately. Transport retries never repeat a dispatch,
cancellation or model request. Fixed metadata diagnostics retain failures and
recovery without URLs, credentials or response bodies.

Results are visible to maintainers with access to the private controller
repository, `wici-ai/inferssd-device-ci`. Open the exact parent run there, then
read its Actions Summary. For each required executed cell it lists, per metric
(0–128/128–256 and whole decode token/s, TTFT and native prefill), the four
base history values, the in-run base, the three PR runs, the change of the PR
geometric mean over the base estimate (`(PR / base - 1) * 100%`), the noise
threshold and the result (`within noise`, `improved`, `regressed` or the reason
for review), followed by the cell decision, the history size and degrees of
freedom, and its reasons. A positive throughput change means faster; a lower
TTFT is better. Missing, failed, regressed and needs-review performance stays
non-passing; `improved` labels a metric and never changes the cell status. A baseline job's summary
lists the four values per metric and the noise estimate and its degrees of
freedom after adding them. Device
deferrals and full impact coverage are reported separately from each execution scope.
The `selected-aggregate-RUN-ATTEMPT` artifact contains `aggregate.json`;
`selected-control-RUN-ATTEMPT` retains every run's raw requests, timestamps,
machine profile, derived launches, settings/source/prompt identities,
observations and cleanup evidence.

These request metrics exclude model/vocabulary loading before server readiness.
Each profile establishes only its own hardware, models, resource constraints and
cache settings. Mac uses the root executor's sampled worker physical footprint,
process count and host free-memory guards; worker RSS is diagnostic only. Its receipts
do not assert Linux cgroup hard UMA or swap isolation.

Before each Mac smoke, short preflight and timed run launch, the coordinator
takes exactly two host snapshots separated by a fixed 2-second wait. Collection,
validation or quiet-policy failure refuses launch and retains `host-quiet.json`
with raw observations. There is no automatic retry. Existing swap settings are
preserved; the check does not disable swap, close applications or clear caches.
The fixed quiet policy is:

- Available memory in both snapshots must be at least the smaller of the pinned
  worker memory budget and 75% of the recorded physical RAM. Available memory is
  free plus file-backed pages, as the kernel counts it: a model the previous run
  mapped stays cached on the active queue until pressure ages it, and the system
  reclaims it without compression or swap.
- Swap-in plus swap-out counter growth must be zero; compression plus
  decompression growth must be at most 64 MiB, using the observed page size.
- The memory-pressure level must be available and equal to 1 in both snapshots.
- Non-CI CPU use must be at most 0.5 cores in total and 0.25 cores per process;
  each non-CI process must use at most 512 MiB RSS. CPU uses cumulative CPU-time
  differences over the observation interval. CI UIDs 62001/62002, each observer
  and its recorded parent chain are excluded. Existing services are assessed
  by these measurements, not rejected merely for being present.

Snapshots before/after the request and after cleanup retain timed
swap, compression, pressure and process activity. Timed swap activity, abnormal
pressure or excess non-CI CPU flags that run; compression above 64 MiB with
abnormal pressure is also flagged, while compression alone is not. Timed checks
do not apply the quiet free-memory or RSS thresholds. A difference above 1 GiB
between any two of a cell's final prelaunch available-memory samples also
flags the cell. A flag makes an otherwise passing cell `needs-review` (and keeps
a baseline cell out of the history); it never hides a `regressed` result, and
measured slowdowns remain visible.
Swap or pressure may be induced by the model; these observations do not
automatically establish external interference or justify a retry. This is
resource sampling, not strict isolation or proof of identical cache conditions.

Preparation/staging interruption and ambiguous dispatch are retained without
automatic replay. A saved, already validated `dispatched-run.json` allows a
restart to resume monitoring without another POST. Otherwise it requires
inspection and a fresh review authorization; no guessed run, repeated model
sample or best result selection exists. Queue authorization expires after 24 h
(configurable bounded lifetime); parent monitoring is bounded by 4 h, matching
the workflow limit. Original completed attempts remain recorded after
a source update and cannot satisfy a new head.

Activation requires deploying the final trusted controller/wheel, exact root
policy/source/model staging and current job before/after probes on each enabled
device, then one real review→prepare→dispatch→Actions result plus a finite
cancellation/cleanup case. Mac uses the fixed SSH entry and recovery contract in
[RECOVERY.md](macos-sandbox/RECOVERY.md). Code, a runner's online state and existing
short readiness receipts alone do not establish a new performance run. These runs
add no formal benchmark samples. This module's contract tests execute none of
those physical operations.
