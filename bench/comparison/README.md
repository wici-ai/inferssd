# Independent release comparison

This directory prepares and measures one fixed base/candidate pair on a single
local Linux device: the predecessor codebase's `v0.1.13` release against the
current engine, using the same DeepSeek V4.1 GGUF files and the fixed 512/256
protocol.
It does not register a runner, claim a device CI gate, or populate the formal
benchmark described in [FINAL.md](../FINAL.md).

## Reference build

The reference is the predecessor codebase's `v0.1.13` release, commit
`08157d66c8702b29988be83e7c54fc51aa91b26e`, tree
`58965a04306cb236fd938254ff5265bbf6165805`. Its archive SHA-256, the verified
source blob count and the before/after hashes of every changed file are in
`reference/compatibility-manifest.json`.

`reference/v0.1.13-minimal-gguf-direct.patch` adapts only logical GGUF ranges,
unaligned direct reads, complete payloads ending before page padding, and the
corresponding cache buffer/ABI handling. It is a reviewed subset of a later
commit, `6c6c81ceb51cb3abfb2dc421625060e30793bd69`; later compute kernels, worker
scheduling and reusable host bounce buffers are excluded. Report the reference
as **v0.1.13 + GGUF compatibility**, never as the unmodified release package.

`native_range_check.cpp` reuses that release's numerical fixture to check an
unaligned MXFP4 payload, complete logical EOF, a cache hit, and rejection of a
truncated weight. It is compiled separately and does not change the reference
server or use model weights.

## Comparison contract

Use the same GGUF files, expert index, external chat template, 512 prompt,
actual template tokens, greedy/seed/reasoning settings, context, batch/ubatch,
dense GPU placement, and initial fresh-server condition. Decode exactly 256;
retain raw SSE and monotonic token timestamps. Do not claim full-file model SHA
verification when only stable path/device/inode/size/timestamps were verified.
Each arm rechecks the actual files and expert ranges.

Both sides use the same host memory limit with zero swap, the same CPU
participants and placement, prefetch depth, RAM-cache admission policy and
guards. GGUF reads and H2D transfers are unrestricted. The prompt label does
not prescribe its tokenizer count; retain the actual authoritative input IDs
and count.

Several defaults differ between the release launcher and the typed registry, so
set them explicitly on both sides:

- Engram prefetch 1 and HC fusion 1. The typed registry's Engram fallback is 0,
  while the production DeepSeek launch profile already selects 1; these are
  different layers of defaults.
- VRAM safety margins 64/64 MiB. The release launcher has separate base/elastic
  values (1024/64 MiB), while the typed margin controls both. The candidate's
  allocator reserve is 0, the reference's low VRAM watermark is aligned from 16
  to 0 MiB, and both high watermarks are 64 MiB.

This is a comparison at equal effective settings, not release-default versus
current-default behavior. Record all such launch changes and native effective
evidence; do not modify algorithms to make the comparison run.

The legacy cache budget is a literal byte count: `UINT64_MAX` is **not** an
automatic budget in v0.1.13. Set the legacy budget to the current engine's
observed initial cache capacity under the same memory limit, and keep the
current engine's automatic policy. Both arms must report that initial capacity,
successful mapping/admission and nonzero cache hits. Native cache
allocation/reclaim lifecycles are recorded as implementation differences.

## Execution and interpretation

Install the pinned framework wheel in a private environment. `owned_command.py`
uses the existing `OwnedProcess` guardian, existing device lock inode, finite
systemd scope, zero swap, deadlines, STOP handling and cleanup receipts for each
explicit preparation command. It never starts a model itself.

`pair.py --plan PLAN.json --output NEW_DIRECTORY` runs `arm.py` once per side
under one existing device lease. The plan names the reviewed `base` and
`candidate` JSON profiles and `attempt: initial`. Each profile contains explicit
argv/env, source/build receipt, hashes of server/libraries/template/index/settings,
GGUF file identities, scope/resource policy and a reviewed `effective_contract`.
Both profiles must have the same semantic contract. Native `/proc` evidence,
loaded image hashes/inodes, source-bound build inputs, native settings and
before/after resources are retained. Only `final_chat.py` emits the single
completion; `run_final.read_result` reconstructs its result from raw artifacts.

Report prefill from native prompt timing, TTFT, and decode intervals 127/128/255.
Any observed slowdown remains **inconclusive**, with no numerical allowance.
A pass means only that this fixed pair showed no slowdown. Output divergence is
reported separately. No automatic rerun, average, replacement or best-sample
selection is performed. If a specific offline diagnosis warrants it, admit at
most one reverse candidate/base diagnostic with `attempt: reverse-diagnostic`,
a retained `initial_pair` reference and a concrete `hypothesis`; retain both
pairs independently.

A configuration correction is a separately identified pair, never a replacement
of an earlier invalid pair. Disk counters and I/O pressure accompany the
measurement; no historical disk-rate normalization is applied. Archive the
reference server, project libraries and preloads, source, patch, build flags
and raw evidence with the result.
