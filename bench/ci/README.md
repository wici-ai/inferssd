# Private physical-device CI admission

## Delivery scope

The current private manual path uses `StageSession` and the independently
provisioned per-stage broker. `controller/integration-smoke.workflow.yml` builds
one reviewed source and makes an eight-token request.
`controller/checkpoint.workflow.yml` builds the fixed f5 baseline and reviewed
candidate, then records one Muse 512/256 request per side under the same lease.
See [the checkpoint instructions](controller/CHECKPOINT.md). Its actual
`checkpoint.json` determines success, failure or inconclusive performance;
installing this code or running the smoke is not a performance result.

The full three-device/two-model PR workflow remains an **unactivated deployment
template**. Its admission/finalization contract and the current stage broker
still require full end-to-end integration and native gates on all devices. Keep
`INFERSSD_CI_ENABLED` unset or `false`; do not require or publish its App check
merely because the manual checkpoint or this aggregate code change succeeds.

The public InferSSD repository must have **no physical self-hosted runner**.
Its device workflow is informational and GitHub-hosted only. A different
workflow could otherwise request the same labels and bypass this module's
preflight. Runner labels and fork-workflow approval are not access controls.
GitHub documents persistent compromise and credential risks on self-hosted
runners, including after environment approval. See [secure use](https://docs.github.com/en/actions/reference/security/secure-use#hardening-for-self-hosted-runners).

Physical execution belongs only to the separately private
`wici-ai/inferssd-device-ci`, using a protected `main` and a private manual
workflow. `controller/device-performance.yml` is a deployment template, not an
active workflow in the public repository. `INFERSSD_CI_ENABLED` stays unset or
`false` until actual OS isolation, original-lock/STOP/guardian cleanup, and
all three device configurations have passed their deployment gates. This code
installs no runner and changes no host or GitHub settings.

## Exact source authorization

A maintainer supplies both a PR number and the full SHA. Admission requires:

- An open non-draft same-repository PR into `main`, from a protected
  `device-ci/approved/` staging branch, still at the supplied SHA.
- An independent maintainer's latest non-comment review explicitly approving
  that SHA, with a standalone review-body line
  `[device-ci-approved:FULL_40_CHARACTER_SHA]` explicitly authorizing physical
  execution. An ordinary code review is insufficient. Dismissal, requested changes, lost write permission, or a different
  commit blocks execution. Author association alone grants no permission.
- A current merge candidate whose parents are exactly the frozen base and head.
- The latest normal CI run/attempt for that head and base, with the expected
  workflow ID/path, completed successfully. An older success is not selected.
- A private controller dispatch on the protected workflow/ref, with current
  actor permission. Controller workflow SHA, run ID and attempt, source origin,
  review IDs, base/head/candidate and complete changed-file listing enter the plan.

Forks and unreviewed code get hosted-only checks. Version one never runs a fork
head directly on physical hardware. A maintainer can import a reviewed patch
into a protected internal staging branch, preserving its authorship and origin,
then obtain a new independent exact-SHA review and dispatch. This transfers trust
explicitly; reviewing code does not make arbitrary code harmless.

The public `workflow_run` entry is removed. GitHub notes that this trigger can
receive secrets/write tokens independently of the preceding run; it is not a
trust upgrade for preceding code. Manual dispatch can also target non-default
refs, so both job conditions and the controller verify the protected ref.
See [workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

## OS execution boundary

No candidate is checked out on a runner. Hosted preparation exports immutable
Git objects as `source.bundle`, validates both candidate parents, and binds its
hash into the current plan. The runner checks out only the protected private
controller at its exact workflow SHA. Source-read credentials stay on the hosted
preparer. Device jobs have only the controller's read-only Actions/content token;
no publishing App key or token reaches them.

`isolated_device.py` invokes an operator-pinned, root-owned broker client with a
fixed JSON protocol. The broker/client, proof file and all their ancestors must
be root-owned, not symlinks, and not group/world writable. Whole-file hashes
must match the reviewed per-host config. A fresh nonce binds describe/run replies.
The proof and describe response must identify separate controller/execution UIDs,
root supervisor, sandboxed execution stages, and every required negative gate:

- Candidate cannot read controller HOME, runner/SSH credentials or host service
  sockets; cannot modify control or evidence; cannot signal/trace the coordinator.
- No sudo/privilege escalation, Docker socket, external network or escape through
  user systemd/DBus. Model/source inputs are read-only; scratch is bounded.
- Disconnect cleanup, original lock inode, STOP quarantine and verified reset.

The trusted coordinator must remain outside each candidate stage's sandbox.
Running the entire driver and candidate as one worker UID permits evidence
forgery and is refused by the protocol. MemoryMax, swap controls, process-group
cleanup, inherited flock descriptions and ephemeral runner registration address
resources/lifecycle; they are not malicious-code isolation. macOS physical Metal
needs actual per-stage OS isolation too. A disposable runner registration does
not reset the host or isolate a personal account.

`controller/broker-unprovisioned.py` deliberately exits 78. It is not a sandbox.
A config boolean or synthetic test result cannot replace native negative gates.
Backend implementation/provisioning is a separate reviewed module; if any host
cannot prove the boundary, leave its runner and overall activation disabled.

The full-gate `isolated_device.py` contract expects `describe` and `run`; it is
not the current manual checkpoint protocol. The staged checkpoint uses
`job.open`, `stage.start/poll/probe/client/output/stop`, `job.freeze` and
`job.finish` on one persistent connection. Wiring the full-gate contract to
those stages remains pending. In that planned contract, `run` receives the exact
plan,
device/config path, source bundle, pinned control root and output directory;
it accepts no arbitrary shell command from the job. It must validate and copy
inputs before candidate access, enforce the per-stage boundary, hold the existing
lease until all owned work is gone, and emit the complete original device result
plus protected isolation/cleanup evidence. Its daemon must survive controller
loss and quarantine uncertain cleanup rather than release the lease.

## Tests and comparison

Inside that boundary the existing driver algorithm is retained: build exact base
and candidate once; run the complete operator-reviewed small numerical/lifecycle
CTest set on both; compare tiny greedy IDs and pre-sampling top-five records;
then measure DeepSeek/Muse 512/256 once per side, plus 4K/256 for resource/executor/
benchmark changes. Both builds precede timing. Same model shards, toolchain,
templates, typed settings and real resource policy are required. Raw SSE, input
IDs, image hashes, CTest XML and cleanup are revalidated, including the broker
receipt for a private-controller plan. Missing/failed/partial data cannot pass.

These are `purpose=ci` samples, separate from the formal 0/16 ledger. All five
metrics, including native prefill, must show no observed slowdown. Any valid
slowdown is `inconclusive` and blocks acceptance pending bounded independent
diagnosis; invalid evidence is `failed`. Neither a 5% decode nor a 10% TTFT
regression is permitted. A passing one-sample comparison is only an observation,
not proof of statistical equivalence. See [performance acceptance](PERFORMANCE.md)
for outcomes, diagnosis limits and the old-main reference-adapter prerequisite.
There are no hardware results in this deployment template. Local synthetic tests
do not prove OS isolation.

## Publication and deployment

Only the GitHub-hosted publisher mints the dedicated check App's short-lived
installation token. It has checks-write and source read permissions for InferSSD,
not access to device hosts. It rechecks current head/base/candidate, review and
normal-CI attempt **after** evidence processing. Old controller runs/attempts,
changed sources or incomplete broker evidence cannot be replayed. The check is
published on the exact head with a run/attempt/plan-derived external ID.

Require `InferSSD device performance` specifically from this dedicated App,
not from “any source” or the GitHub Actions App: another workflow can produce an
identically named Actions check. Require branches up to date, protect the staging
and controller branches, and disallow bypass. The finalizer's last check cannot
prevent a future base update by itself. See [required-check sources and strict
status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).

Deploy control bytes from a reviewed commit, never a dirty checkout:

```sh
python3 bench/ci/export_controller.py --source /path/to/inferssd \
  --revision FULL_REVIEWED_40_CHARACTER_SHA --output /path/to/new-controller-export
```

The final structure already contains `tools/inferssd`; export it directly.
Do not pass `--registry-revision` when that package is present; that option is
only for the older control branch that needed a separately pinned registry.

Review and commit that export into the private controller repository. The export
contains a source-origin/file-hash manifest; the private workflow is separately
protected. Provision read-only source credentials only for hosted preparation,
the dedicated App only for hosted publication, and the immutable device configs.
Where supported, environment required reviewers with self-review disabled
supplement the explicit exact-SHA review marker. GitHub Team does not provide
that extra reviewer rule for a private environment; this version relies on the
independent maintainer marker plus the separately permission-checked private
dispatch. It does not claim an environment review occurred. See
[deployment protection availability](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

Organization runner groups can restrict selected repository/workflow refs when
available, but this design does not rely on that feature. Never add public-repo
access. See [runner-group access](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/manage-runners/self-hosted-runners/manage-access).

After actual isolation and native cleanup gates pass on every intended device,
review all receipts before enabling `INFERSSD_CI_ENABLED` or requiring the App
check. Do not publish, install, or activate merely by merging this module.
