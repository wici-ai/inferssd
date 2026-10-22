# Mac expert-source experiments

`measure.py` runs one fresh server for a specified model and configuration.
The first request starts with an empty application expert cache; later requests
reuse that cache but disable prompt/KV reuse. No OS page-cache purge is performed.
Record the physical disk counters rather than calling the first request a cold SSD test.

The experimental `SSD_LLAMA_MAC_EXPERT_READ_AHEAD` setting accepts 0, 1, 2, or 4.
Zero retains synchronous reads. Nonzero values bound the number of outstanding
reads for experts already selected by the current operator. Staging is additionally
limited to 64 MiB per operator. Expert arithmetic, execution order, cache heat and
admission order remain unchanged. This setting exists for controlled experiments;
a production default requires numerical, memory and end-to-end validation.

`GGML_MOE_EXPERT_GGUF_ALIAS_TSV` selects the same index format as the Linux reader.
The index is generated from GGUF metadata with `scripts/kimi-build-gguf-expert-alias-tsv.py`.
It is mutually exclusive with `SSD_LLAMA_MAC_EXPERT_PACK`. GGUF slices can be unaligned
and end exactly at EOF. Alignment 1 reads only the logical bytes. Alignment 4096
or 16384 includes surrounding bytes within the same source file, stops at EOF and
returns the original logical subrange to the operator. Cache accounting includes
the whole allocation. The 64 MiB staging bound applies per operator batch; callers
running multiple contexts must include each batch in their memory budget.

Counters are cumulative. `read_call_ms` sums individual pread service times across
workers and is not elapsed I/O time or a device bandwidth denominator. `read_wait_ms`
measures demand-side blocking; `compute_ms` measures the ggml lead thread's work and
`barrier_ms` its synchronization waits, including waits for other computing threads.
`acquire_ms` includes cache policy, staging setup and read waits. Lookup, policy and
allocation counters are nested diagnostics and must not all be added as disjoint
request phases. Physical bytes divided by the corresponding phase wall time give
observed device throughput. Memory sampling records RSS, compression and swap.

With `--trace`, four files are written on normal shutdown: the legacy request list,
`.requests.tsv` with demand timestamps and hit flags, and `.reads.tsv` with actual
source extents, read timestamps and call durations, and `.phases.tsv` with cumulative
phase counters. These optional traces accumulate in memory until shutdown, so use
short requests when collecting them. They allow exact per-request
miss-byte accounting and bounded replay of real misses without duplicating weights.

The source test uses small synthetic files, including multiple sources, unaligned
ranges, EOF, runtime truncation, duplicate keys, pinned entries and cache eviction.
Depth 0/1/2/4 must return identical bytes and cache/read counters. Compile the test
and reader as separate translation units with ASan/UBSan or TSan.
