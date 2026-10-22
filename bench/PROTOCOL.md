# Benchmark protocol

Every result identifies a concrete model, build, machine and workload. A result
from one combination does not establish performance on another. Keep diagnostic
experiments separate from the final device acceptance record.

## Inputs and provenance

Record these fields alongside the raw output:

- Full source revision, clean/modified tree state, compiler and build options;
  binary and loaded backend-library hashes. Separate build directories must also
  load their own libraries, including on macOS where binaries may embed paths.
- CPU, available affinity, physical memory, GPU and device memory, storage device,
  filesystem, operating system, kernel and driver. Record active power mode and
  host tuning without assuming advertised peak specifications were achieved.
- Model repository/revision, quantization and SHA-256 of every GGUF shard;
  tokenizer, chat template and template arguments. Expert packages or alias files
  must identify their underlying model files and hashes.
- Exact prompt SHA-256, actual rendered input token IDs/count, requested output
  count, context/batch sizes and sampling settings. Prompt filenames are labels,
  not authoritative token counts. Fixed prompt bytes are in [prompts](prompts/).
- Effective settings and resource limits, including host memory/swap limits,
  expert-cache budget, GPU reserve and any deliberate transfer throttling.
  Declare direct GGUF reads and unrestricted transfers when that is the protocol.

Use one user message, thinking disabled, temperature zero/top-k one, a fixed
output count, EOS ignored and prompt caching disabled.
The template endpoint's authoritative token IDs preserve message provenance;
re-tokenizing flattened text can interpret literal user text as control tokens.
The client verifies counts before generation and keeps the exact request.

## Runs and clocks

Final performance acceptance runs the fixed 512 and 4K prompt files exactly once
each per supported device/model on the final source, generating all 256 output
tokens. Each tier starts a fresh owned server and records its first completion.
Record the actual authoritative template token count: the 4K label can exceed
4096 tokens. Do not change prompt bytes to equalize token counts between models.
Keep every failed attempt; no automatic retries, averaging or best-run selection.
See [the formal driver contract](FINAL.md) for deadlines and required receipts.

ABBA remains an optional short diagnostic comparison with an explicit baseline
and candidate. Preserve its A1, B1, B2 and A2 separately and label it as a
diagnostic; it does not replace either final performance sample.

A fresh server may make initialization or warm-up reads before the measured
request. It does not imply an empty application cache, and it never establishes
that the OS file cache is cold. A repeated request on the same server is a
separate warm condition. Record any explicit OS-cache treatment and apply it
equally to comparison arms. Do not run copying, hashing, compilation or another
benchmark concurrently on the measured device.

TTFT is elapsed monotonic time from sending the completion request to receiving
its first token event. Decode rate uses intervals between token events: a
128-token first window has 127 intervals; tokens 129--256 span 128 intervals
starting at token 128. A shorter request does not establish either full window.
Server-reported prefill/compute timings are separate metrics and must retain
their own count and time definitions.

The separate short driver accepts 1--1,024 input and 1--512 output tokens, with an explicit
per-arm timeout. An archived longer prompt does not expand these bounds or
schedule a run. STOP, timeout, OOM, early EOS, truncation, protocol mismatch and
incomplete output remain failures in the record; do not discard them as outliers
or silently substitute a retry. Cleanup stops only the run's owned processes.

## Correctness and interpretation

Save original stream bytes, timestamped parsed events, token IDs and complete
pre-sampling top-five probabilities. Compare only runs with matching prompt IDs,
settings and token counts. Exact equality is stronger than matching text. Once
greedy generations diverge, their later contexts differ: positional agreement
does not establish teacher-forced numerical agreement. Device-specific numerical
tolerances and repeated-reference variation must be declared before applying
them; the exact-output comparison tool does not invent a tolerance.

For bottleneck analysis, distinguish logical expert bytes from physical device
reads, and total request time from read-active time. Summed read-service times
can exceed wall time when reads overlap; a wait counter nested within acquire
time must not be added to it again. Likewise, RSS, process footprint, Metal
allocation sizes, file-backed pages and system memory are not disjoint quantities.
Track compression/decompression and swap changes when evaluating larger caches.

A physical-bandwidth limit divided by required bytes per token is only a
read-only throughput bound; compute, layer dependencies and synchronization can
lower achieved throughput. Report an optimization as measured on the tested
configuration, with rejected alternatives and remaining limits. Do not infer an
absolute performance optimum from a short pair or a sequential disk probe.

## Acceptance and publication

The device test gate compares the entire intended test set with a reviewed
baseline. Missing, additional, failed or skipped tests require an explicit
explanation; an unavailable or disabled test is not a pass. A fully skipped
selection cannot establish acceptance. Keep unit, GPU, resource-dependent and
real-model evidence distinct.

Inspect artifacts before publication. Remove credentials and private operational
paths while retaining enough hardware, model, source and protocol provenance to
reproduce the result. Published tables link their exact commands, settings and
raw records. Historical measurements and candidate experiments do not replace
validation of the final source being released.
