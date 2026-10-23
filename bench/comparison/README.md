# Independent release comparison

This directory prepares and measures a local release comparison. It does not
register a runner, claim a device CI gate, or populate the full FINAL table.
The first target is the local RTX 5090 and the shared DeepSeek-1017 GGUF files.

## Frozen reference

The reference is SSD-llama `v0.1.13`, commit
`08157d66c8702b29988be83e7c54fc51aa91b26e`, tree
`58965a04306cb236fd938254ff5265bbf6165805`. The 8,021 source blobs were checked
against that Git tree. Its archive SHA-256 and the before/after hashes of all six
changed files are in `reference/compatibility-manifest.json`.

`reference/v0.1.13-minimal-gguf-direct.patch` adapts only logical GGUF ranges,
unaligned direct reads, complete payloads ending before page padding, and the
corresponding cache buffer/ABI handling. The source is the reviewed subset of
`6c6c81ceb51cb3abfb2dc421625060e30793bd69`; later compute kernels, worker scheduling,
and reusable host bounce buffers are excluded. Report the reference as
**v0.1.13 + GGUF compatibility**, never the unmodified release package.

`native_range_check.cpp` reuses that release's numerical fixture to check an
unaligned MXFP4 payload, complete logical EOF, a cache hit, and rejection of a
truncated weight. It is compiled separately and does not change the reference
server or use model weights.

## Fixed comparison contract

Use the same ten GGUF files, expert index, external chat template, 512 prompt,
actual template tokens, greedy/seed/reasoning settings, context, batch/ubatch,
dense GPU placement, and initial fresh-server condition. Decode exactly 256;
retain raw SSE and monotonic token timestamps. Full-file model SHA verification
must not be claimed when only stable path/device/inode/size/timestamps were
verified. Each arm rechecks the actual files and expert ranges.

For the current local comparison, RAM is 13,976,514,560 bytes and swap is zero.
Both sides use 10 total CPU participants / 9 decode compute participants, no
additional CPU pinning, prefetch depth 2, a resource-sized RAM cache,
count-decay/uniform admission, and a 512 MiB cache guard. The separate controller
host-memory guard is 1 GiB. GGUF reads and H2D transfers are unrestricted.

The fixed launch uses context 1024, batch/ubatch 512, and 10 batch threads.
The prompt label does not prescribe its tokenizer count; retain the actual
authoritative input IDs and count.

Both sides explicitly use Engram prefetch 1 and HC fusion 1. The typed registry's
Engram fallback is 0, while the production DeepSeek launch profile already
selects 1; these are different layers of defaults. The original release launcher
has separate base/elastic VRAM safety values 1024/64 MiB. The current typed
margin controls both, so this comparison explicitly uses 64/64 MiB on both sides,
with candidate allocator reserve 0. The reference low VRAM watermark is
also explicitly aligned from 16 to 0 MiB; both high watermarks are 64 MiB. This is a comparison at equal effective
settings, not release-default versus current-default behavior. Record all such
launch changes and native effective evidence; do not modify algorithms to make
the comparison run.

## Execution and interpretation

Install the frozen controller wheel in a private environment. `owned_command.py`
uses the existing `OwnedProcess` guardian, existing device lock inode, finite
systemd scope, zero swap, deadlines, STOP handling and cleanup receipts for each
explicit preparation command. It never starts a model itself.

`pair.py --plan PLAN.json --output NEW_DIRECTORY` runs `arm.py` once per side
under one existing device lease. The plan names the frozen `base` and `candidate`
JSON profiles and `attempt: initial`. Each profile contains explicit argv/env,
source/build receipt, hashes of server/libraries/template/index/settings, GGUF
file identities, scope/resource policy and a reviewed `effective_contract`.
Both profiles must have the same semantic contract. Native `/proc` evidence,
loaded image hashes/inodes, source-bound build inputs, native settings and
before/after resources are retained. Only `final_chat.py` emits the single
completion; `run_final.read_result` reconstructs its result from raw artifacts.

Report prefill from native prompt timing, TTFT, and decode intervals 127/128/255.
Any observed slowdown remains **inconclusive**, with no numerical allowance.
A pass means only that this fixed pair showed no slowdown. Output divergence is
reported separately. No automatic rerun, average, replacement, or best-sample
selection is performed. If a specific offline diagnosis warrants it, admit at
most one reverse candidate/base diagnostic with `attempt: reverse-diagnostic`,
a retained `initial_pair` reference and a concrete `hypothesis`; retain both
pairs independently. No 4K, other model or other device is added to this task.

The legacy cache budget is a literal byte count: `UINT64_MAX` is **not** an
automatic budget in v0.1.13. The first local pair exposed this mapping error and
is retained as `invalid-resource-match`, irrespective of its measured speeds.
The corrected profile sets the legacy budget to 12,905,668,608 bytes (686 physical
slots), the current engine's observed initial capacity under the same 13 GiB
limit. The current engine retains its automatic policy. Both arms must report
that initial capacity, successful mapping/admission and nonzero cache hits;
native cache allocation/reclaim lifecycles are recorded as implementation
differences. A configuration correction is a separately identified pair, never
a replacement of the invalid original. Disk counters and I/O pressure accompany
the measurement; no historical disk-rate normalization is applied.

At the end, archive the exact old server, all project DSOs and preloads, source,
patch, build flags and raw evidence off the device. Verify recovery and hashes
before deleting only the reviewed cleanup-manifest paths. Preserve the current
engine, shared GGUF files and unrelated user files.
