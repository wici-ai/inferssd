# Short device comparisons

These tools coordinate one **ABBA cycle: four arms, in that exact order**. They
start only commands explicitly supplied by the operator. They do not download
models, choose settings, publish results, retry failures, discard slow values or
start long-prompt runs. The default per-arm timeout is 300 seconds. A timeout is
a failed arm retained in the record, not an outlier to remove.

The model client and actual prompt assets are still being consolidated. Until
those audited assets are imported, use an existing verified short-request arm
command with the result contract below. Do not treat this harness alone as a
reproducible public model benchmark or as a completed device acceptance gate.

## One ABBA cycle

Create a JSON config containing `protocol` and exactly `arms.A` and `arms.B`:

```json
{
  "protocol": {
    "input_tokens": 614,
    "output_tokens": 96,
    "prompt_sha256": "REPLACE_WITH_THE_64_HEX_DIGIT_SHA256_OF_THE_ACTUAL_PROMPT"
  },
  "arms": {
    "A": {"revision": "FULL_BASELINE_COMMIT", "argv": ["bash", "/path/to/arm-a.sh"]},
    "B": {"revision": "FULL_CANDIDATE_COMMIT", "argv": ["bash", "/path/to/arm-b.sh"]}
  }
}
```

Commands are argument arrays; no shell expression is evaluated by the driver.
Paths in command arguments are interpreted from the directory where the driver
runs. Record the exact engine revision and verify that the binary was built from
it. `revision` is operator-supplied provenance, not an attestation by this tool.
Both arms must use the same prompt bytes, tokenizer, output length, cache policy,
memory limit and sampling protocol. Every arm starts its own fresh server and
runs one request. Cache eviction and server ownership belong in that arm command;
the driver does not change OS settings or infer ownership from a listening port.

```sh
python3 bench/run_abba.py --config comparison.json --output bench/runs/comparison-001 \
  --lock /path/to/shared-device.lock --stop /path/to/STOP
```

Choose the device's existing shared lock and STOP paths. The lock protocol is
compatible with `exec 9>device.lock; flock -n 9`: a nonblocking exclusive flock
is held for the complete cycle. An occupied lock is refused immediately. The
shared file's contents and inode are preserved; the driver never removes it,
steals it or unlocks another process's descriptor. Do not use a different lock
path to work around an occupied device. The output directory must not exist.

The driver supports 1–1024 actual input tokens and 1–512 output tokens. Longer
protocols need an explicitly reviewed extension. It sets these environment values
for each command:

- `SSDASH_ARM`: `A` or `B`.
- `SSDASH_RUN_ID`: the cycle's unique ID.
- `SSDASH_RUN_DIR`: that arm's new artifact directory.
- `SSDASH_RESULT_PATH`: the required JSON result file.

The command must exit successfully and write the actual token counts and prompt
SHA-256 from `protocol`, plus a nonempty `metrics` mapping of finite nonnegative
numbers. Metric names and units must agree across all four arms. For example,
`ttft_s`, `prefill_tps` and `decode_tps` may describe a fixed short protocol. A
96-token completion must not be labeled as a completed 128-token decode window.
Keep raw stream events and numerical comparison evidence alongside `result.json`.

`manifest.json` records the exact declared config. `arms.jsonl` always records
four rows on a controlled completion, failure or STOP: unstarted arms are marked
`not_run`. Every started arm retains its elapsed time, return code, stdout and
result path. `summary.json` is complete only if all four arms succeed. Only then
it reports all raw values, arithmetic means and `B_over_A` ratios. A ratio's
meaning depends on the metric: larger throughput and smaller latency are useful
in opposite directions. No noise threshold or speedup claim is inferred.

STOP and termination signals stop the current arm's newly created process group
and prevent subsequent arms. An arm that exits successfully but leaves descendants
in its process group is still failed: the driver cleans up that group while
holding the device lock, then records the remaining arms as `not_run`.
Arm commands that create systemd scopes or otherwise
move server processes outside that group must trap termination and stop their
own exact scope, as the adaptive arm launcher does. No process-name or port-wide
kill is used. Unexpected machine/process death can leave an incomplete run;
never combine partial runs or fill in missing measurements silently.

## Device test gate

Capture CTest JUnit XML (for example with `ctest --output-junit results.xml`) or
supply normalized JSON of this form:

```json
{"tests": [{"name": "test-example", "status": "passed"}]}
```

A reviewed baseline must enumerate the entire intended test set:

```json
{"tests": {"test-example": {"expect": "passed"}}}
```

Each expected failure or skip requires an explicit `reason`, normally a tracked
issue and its device/resource condition. No actual device failure sets are
invented here; they must be collected and reviewed before a device gate runs.

```sh
bench/device/gate.sh --results results.xml --baseline reviewed-baseline.json \
  --output gate.json
```

The gate rejects missing, duplicate, unlisted and zero test results, unexpected
failures or skips, and a selection with no executed test. An expected failure
that now passes also requires a baseline update in the same change. The report
includes counts, every difference and the baseline file's SHA-256. A disabled or
unavailable test is not reported as passing. CI's portable unit subset remains
separate from GPU, cgroup and model-bearing device acceptance.

Run harness tests with `python3 -m pytest -q bench/tests`. These use short stub
commands and temporary lock files; they never acquire a real device lock. Review
run artifacts for local paths and private data before copying them into the
repository; benchmark directories are local evidence by default.
