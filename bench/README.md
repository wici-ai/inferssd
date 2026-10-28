# Device measurements

For final performance acceptance, use the separate [formal benchmark](FINAL.md):
the fixed 512 and 4K prompts once each, with 256 output tokens and no ABBA or
automatic retry. The tools below retain their original short diagnostic bounds.

These tools coordinate one **ABBA cycle: four arms, in that exact order**. They
start only commands explicitly supplied by the operator. They do not download
models, choose settings, publish results, retry failures, discard slow values or
start long-prompt runs. The default per-arm timeout is 300 seconds. A timeout is
a failed arm retained in the record, not an outlier to remove.

The client below accepts an operator-supplied plaintext prompt; the fixed
benchmark prompts are in [prompts](prompts/). This harness alone is not a
reproducible public model benchmark or a device acceptance gate.

## Record one short chat

Start an owned server with the intended model, template and device settings, then
record one request:

```sh
python3 bench/stream_chat.py --url http://127.0.0.1:8080 \
  --prompt prompt.txt --input-tokens 614 --output-tokens 256 \
  --output run/result.json
```

The client calls `/apply-template` with one user message and thinking disabled.
It requires that endpoint's authoritative token IDs, which preserve the distinction
between user text and template control tokens. The rendered token count must equal
the declared input count. It sends those exact IDs to native `/completion` with greedy top-k
sampling, a fixed output count, EOS ignored, prompt caching disabled, and five
pre-sampling probabilities. It rejects a truncated request, an incomplete stream,
missing token IDs, invalid top-five records or mismatched final counts. It accepts
1--1024 input and 1--512 output tokens; the default whole-operation deadline is
300 seconds. Each output path must be new.

Alongside `result.json`, the client retains `result.protocol.json`, timestamped
decoded `result.events.jsonl`, and the original SSE response bytes in
`result.events.sse`. Partial evidence and a failed result survive request errors.
Exact token IDs, unrounded parsed log probabilities, prompt hashes and request
settings remain available for review. These files may contain private prompt text
and generated content; inspect them before publication.

TTFT starts immediately before the completion HTTP request. Decode throughput
counts intervals between received token events: the first 128-token window has
127 intervals and the second has 128. These are client-observed timings, not
server kernel timings. The separate prefill rate uses the server's prompt time.
Transport buffering and evidence recording can affect client-observed intervals.

```sh
python3 bench/compare_outputs.py run-a/result.json run-b/result.json \
  --output comparison.json
```

Comparison requires complete validated records with identical prompt hashes,
rendered IDs and token counts. It reports exact token/probability equality and
the first differing token, without applying a numerical tolerance. Once greedy
outputs diverge, later positions have different contexts; positional agreement
does not substitute for a teacher-forced numerical comparison. The client does
not establish binary provenance or device isolation; record those in the arm.

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

- `INFERSSD_ARM`: `A` or `B`.
- `INFERSSD_RUN_ID`: the cycle's unique ID.
- `INFERSSD_RUN_DIR`: that arm's new artifact directory.
- `INFERSSD_RESULT_PATH`: the required JSON result file.

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
Arm commands that create systemd scopes or otherwise move server processes
outside that group must trap termination and stop their own exact scope. No
process-name or port-wide kill is used. Unexpected machine/process death can
leave an incomplete run; never combine partial runs or fill in missing
measurements silently.

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
