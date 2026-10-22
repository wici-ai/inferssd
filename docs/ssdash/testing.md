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
failure is `max_abs=0.024879`; the production settings leave `EXPERT_JOB` off. A
small-runner investigation also recorded a SIGFPE for `schedule_t2`. Disabling
these tests does not fix either defect. They must be repaired with numerical and
resource validation or removed together with that unused implementation during
code organization. Track the decision and validation in
[issue #1](https://github.com/wici-ai/ssdash/issues/1) and resolve it before v0.1.0.

Other previously reported device failures are not allowlisted: their full names,
reproduction conditions and outcomes still need to be collected on the current
hardware. In particular the GB10 MXFP4 parity test needs a separately documented
numeric expectation; it must not silently become a generic skip.
