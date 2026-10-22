# Test gates

The `unit` CTest label contains only model-free tests that require neither a GPU
nor cgroup setup. Build the targets before running `ctest -L unit`; a missing
executable, skipped test or zero selected tests is not a passing gate. CI uses an
explicit subset to keep the default check small. GPU compilation is a separate,
manual check while this repository is private; it does not establish GPU runtime
correctness. Device runs must record both the selected and actually executed tests.

The Linux CPU sink suite also exercises io_uring, O_DIRECT, affinity, cgroup memory
limits and experimental scheduling paths. It is not interchangeable with the
portable unit gate. `kimi_runtime_routing` initializes the Linux sink and is
registered only on Linux under `device`, even though it generates its own model. Do not interpret a full-suite run on a small hosted runner as
a production device acceptance test. GPU and cgroup suites need dedicated launchers
and explicit resource prerequisites before they can become automated gates.

## Known failures retained for investigation

The following 12 inherited cases are disabled and carry the `known-failure` label:

- `kimi_cpu_sink_schedule_t{2,12,16}_d{1,8}`
- `kimi_cpu_sink_expert_job_t{2,12,16}_d{1,8}`

They exercise the experimental expert-job scheduling path. The recorded numerical
failure is `max_abs=0.024879`; the production settings leave `EXPERT_JOB` off. The
first-call SIGFPE was separately reproduced with two threads and no cache:
expert-job dispatch left the active participant count at zero. The dispatch now
initializes it, and `kimi_cpu_sink_expert_first_call` compares fresh two- and
four-thread sinks with the ordinary phase path. This does not resolve the recorded
numerical failure. The twelve cases remain disabled pending numerical validation
or removal with the unused implementation during code organization. Track the
decision and validation in
[issue #1](https://github.com/wici-ai/ssdash/issues/1) and resolve it before v0.1.0.

Other previously reported device failures are not allowlisted: their full names,
reproduction conditions and outcomes still need to be collected on the current
hardware. In particular the GB10 MXFP4 parity test needs a separately documented
numeric expectation; it must not silently become a generic skip.

## Linux resource prerequisites

The registered Linux sink/prefill I/O fixtures carry `device`; cache-admission
cases additionally carry `cgroup`. The metadata-only prefill graph and cache
compute-policy tests carry `unit`. Existing manual CUDA workloads remain manual;
adding labels does not authorize a device run.

A fixture returns 77 with a reason for insufficient CPU affinity (compute and I/O
participants are reported separately), unavailable io_uring setup, unsupported
aligned O_DIRECT on its actual input filesystem, or a missing required finite
cgroup. The finite-cgroup probe does not infer a payload budget: fixtures may
change reserves or intentionally exercise pressure. Their explicit size and
owned-scope rules remain authoritative, including the load-room minimum check.
Use the intended storage directory through `TMPDIR`; probing another mount does
not establish that this fixture's filesystem supports direct I/O.

Fixed-buffer tests use the actual `IORING_REGISTER_BUFFERS` result, including the
real ring geometry and concurrently live rings. EPERM or ENOMEM there records a
skip and the current `RLIMIT_MEMLOCK`; a tiny successful registration is not
proof of the full pin budget. Negative memlock tests still deliberately lower
the child limit and assert rejection. Malformed settings, missing files, short
reads, ordinary allocation errors, registration EINVAL/EFAULT, and numerical
mismatches remain failures. No runtime fallback is added.

CTest preserves 77 as skipped in JUnit. The device gate compares every status with
an explicit reviewed baseline and rejects an all-skipped selection. ASan cannot
establish the prefill cache's process-RSS release assertion: that lifecycle test
executes its other assertions but returns 77, requiring a separate normal-build
pass for physical-release acceptance. Current device allowlists and hosted
runner capability measurements must be collected; they are not inferred from
historical failure totals.
