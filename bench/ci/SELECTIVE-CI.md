# Impact-selected physical CI

This branch provides the trusted orchestration implementation. It has not been
deployed or dispatched. Local tests validate protocol and failure handling; they
are not physical-device or performance evidence. Formal benchmark samples: 0.
The original guarded per-stage executors, device lease, STOP/quarantine, source
freezer, exact PID/birth/listener and disconnect guardians remain authoritative.

## Plan and scope

`impact.py` rebuilds a deterministic plan from the complete PR file API listing,
including old names on renames, and exact base/head/merge-candidate SHAs. Unknown
and shared runtime changes require all three allowlisted devices and both models.
CUDA-only changes select the two CUDA devices; Metal/Mac changes select the Mac.
Explicit model graph files select that model. Current operator/format cases are
listed in the fixed `CASES` catalogue. Model-adapter and runtime-policy changes
also require their frozen descriptor/policy equivalence executables.

A selected cell requires a real model request and its related numerical gates.
A successful ADD row cannot satisfy MXFP4/Q8_0/Q4_K/Q5_K, normalization or SWIGLU.
Unsupported/zero executed rows fail. CUDA tier cases retain their normal-provider
CPU-reference oracles and original thresholds; no synthetic replacement runs on
the device. Test/control-only changes still select functional and numerical gates;
changes affecting runtime/placement/resources require base/candidate performance.
Ordinary performance uses the fixed Azure 512 prompt with decode 256. Only the
explicit compressed long-context rule adds 4K. No automatic completion retry or
best-of selection exists. Each cell records TTFT separately, native prefill and
first/second 128-token and whole decode rates. The existing n=1 policy marks any
observed slowdown inconclusive and blocks acceptance.

## Trusted PR entry

The source repository workflow provides an unprivileged hosted preview only. Its
plan is not authorization. The private `device-performance.yml` independently
reads the real PR, complete changed-file pages, current normal CI and reviews.
The unchanged admission contract requires an open, nondraft same-repository PR
from a protected `device-ci/approved/` branch, exact merge parents `[base, head]`,
and a different writable maintainer's latest approval on that exact head with a
standalone `[device-ci-approved:FULL_SHA]` line. An outstanding maintainer request
for changes blocks admission. The controller's own main must also be protected.

The hosted coordinator dispatches only the active private
`.github/workflows/integration-smoke.yml` on main using Actions-write permission.
The request asks GitHub REST for the exact returned run ID. An ambiguous response
fails without a second POST. The child is bound by workflow path/ID, main SHA,
fresh job ID, run name and attempt 1. Required selected-plan input is checked
against the root-owned job configuration and root broker manifest before opening
the lease. A plan-less readiness run cannot satisfy a selected child.

Only the hosted publisher receives the dedicated check App token. It runs after
failed admission/dispatch as well, and publishes a completed failure for a valid
requested source PR/head when artifacts or admission are missing. It rechecks
source, exact-head approval, complete changed paths and the latest hosted CI
before publishing success. The stable check is `InferSSD device performance`.
There are no required workflow path filters. Unselected cells are explicitly
`not-required`; selected missing, failed, unclean or inconclusive cells are red.

GitHub's [dispatch REST contract](https://docs.github.com/en/rest/actions/workflows)
supports `return_run_details`; [GITHUB_TOKEN dispatch exceptions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
allow this fixed child dispatch. [Required check semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
explain the always-run external App aggregate. No `pull_request_target` candidate
execution is used; see [GitHub's security guidance](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target).

## Operator preparation (no deployment command)

1. Freeze source base/head/candidate and the reviewed control commit. Export
   control and both private workflow templates with `export_controller.py`, then
   review/install through the existing root operator process. Read the device's
   actual installed `control-source.json` revision and whole-file hash. The
   private workflow commit is a separate identity from installed device control.
2. Obtain complete changed paths as JSON (for an operator diagnostic,
   `git diff --name-only --no-renames BASE HEAD` includes both sides of renames).
   Run `python3 -I bench/ci/impact.py --paths paths.json --policy
   bench/ci/policy.json --base BASE --head HEAD --candidate CANDIDATE --output
   plan.json`. The private PR entry recomputes this plan itself.
3. Prepare a fresh root job using the existing reviewed source/model staging
   mechanism. Preserve admitted readonly model identities; do not duplicate
   hundreds of GB. The manifest's `source_revisions` must exactly match required
   base/candidate sources. No runner can install or edit this manifest.
4. Prepare a local operator template with `device`, `host` (actual hardware, OS,
   resources), `control_revision`, `control_manifest_sha256`, `manifest_path`,
   `numerical` (actual CUDA0/MTL0 and cwd), and `sides.base/candidate`. Each side
   contains `revision`, `paths`, the existing `staged_build.run_build` build spec,
   optional sandboxed `alias`, and `models` keyed by the exact selected models.
   Each model contains the existing eight-token `smoke` spec, `settings` and
   `template` (`host_path`, `runtime_path`, `sha256`), `model_identity.shards`
   (`host_path`, `runtime_path`, historical hash, `stat` with device/inode/bytes/
   mtime_ns/ctime_ns), and `benchmarks` keyed by selected tier. Each benchmark
   uses the existing `staged_benchmark.benchmark` argv/environment/cwd/port,
   timeout_s/request_timeout_s and honest os_page_cache_treatment contract.
   Candidate targets AND sealed image_relatives must cover every selected
   numerical executable. Both arms must use the same resource cap, placement,
   model distribution, template and semantic typed settings. Native snapshots,
   real runtime image mappings and root-observed resources are checked.
5. `python3 -I bench/ci/stage_selection.py --template template.json --manifest
   manifest.json --plan plan.json --policy bench/ci/policy.json --output fresh`
   produces three reviewable local files only. The generator derives the Linux
   stage allowlist from fixed trusted operations. The manifest carries the plan
   and config payload hash; the config carries the exact final manifest hash,
   avoiding a circular hash. Install these only through the existing root path.
6. A single operator-authorized integration dispatch can use
   `dispatch-inputs.json` without the PR parent. It validates one device's selected
   stages but does not publish an App check or claim PR admission. A full parent
   run consumes the per-device dispatch-input objects as `staged_jobs` and fails
   if any required device/model is absent. Do not pass an operator diagnostic off
   as an independently approved PR attempt.

macOS control must include the read-only `resource_policy` probe field added in
this branch, so the compared envelope is the actual continuously enforced root
executor policy. Old probes fail closed. An unresolved model admission cannot be
converted to a skip, an alternate model, a smaller budget or a green result.

## Activation boundary observed on 2026-10-24

The real private controller main was `769dde5b9b37b8b462c3f65bbfce62331b548511`;
its branch API reported `protected=false`. The source repository had no open PR
and therefore no exact-head approval for this batch. Those are actual activation
blocks, not permission to relax the contract. Existing root-stage readiness runs
use a different manual authorization path. Cloud DeepSeek and Spark Muse model
admission gaps also remain explicit; inventory alone is not model admission.
No settings, protection rule, runner installation or dispatch was changed here.
