<!-- SPDX-License-Identifier: Apache-2.0 -->
# Experimental multi-NVMe reads

This opt-in Linux experiment lets the CPU expert reader choose among identical
expert payloads on independent NVMe drives without RAID. Two drives are the
primary configuration. Four drives use the same mechanism; there is no separate
four-drive implementation. The source representation accepts 2 to 8 replicas;
device validation here covers 2 and 4 only.

The default remains the existing source mapping. The experiment adds:

- Replica selection using outstanding bytes and observed completion throughput.
- Disjoint, 4 KiB-aligned range reads, with completion fan-in before an expert is
  ready. The planner considers widths 1, 2, 4 and 8, bounded by available replicas.
- A context-local load pool shared by demand reads and asynchronous prefetch, so
  their reservations influence the same routing decision.
- Optional forced asynchronous io_uring submission with bounded io-wq workers.
- Separately enabled CPU scheduling that groups workers by L3 topology and
  assigns row chunks locally. This is not required to open two replica sources.

These are experimental interfaces, including environment switches that have not
yet been integrated into the supported typed reader policy. They do not establish
Mac/Windows support, general multi-disk scaling, or numerical equivalence of full
model generation. The adaptive planner still uses initial throughput and fan-in
cost heuristics; further measurement and API review are needed before enabling it
by default.

## Source preparation and use

Each replica must resolve to a single regular payload file with the same tensor
layout, offsets, sizes and **identical bytes**. Full expert-pack copies are
required on separate drives; this consumes one full pack's space per drive.
The engine checks layout, distinct file identity and equal size, but does not hash
payload contents or establish that paths are on different physical drives.

Verify every complete pack before starting, and keep it immutable for the entire
reader lifetime. For example, the following hashes must be equal:

```sh
sha256sum /mnt/nvme-a/model.expert-pack /mnt/nvme-b/model.expert-pack
```

Run the project's built-in profiling on the target machine before inference:

```sh
python3 -m inferssd profile --reprofile --build build-linux-cuda \
  --model /mnt/nvme-a/model-00001-of-00010.gguf \
  --model /mnt/nvme-b/model-00001-of-00010.gguf \
  --output machine.json
```

Add to an `inferssd-settings 1` file:

```text
expert.source = ["/mnt/nvme-a/model.expert-pack"]
expert.replicas = ["/mnt/nvme-a/model.expert-pack", "/mnt/nvme-b/model.expert-pack"]
io.range_max_experts = 2
```

The logical source may also be a static expert/role map, as in the benchmark.
Every logical payload source must belong to the declared replica set. The
`io.range_max_experts` bound limits demand splitting when many experts already
offer independent reads.

For the measured adaptive I/O mode, set these variables in the inference process:

```sh
export INFERSSD_REPLICA_POLICY=adaptive
export INFERSSD_IO_POOL=1
export INFERSSD_IO_POOL_SPLIT=1
export INFERSSD_IO_FORCE_ASYNC=1
export INFERSSD_PREFETCH_FORCE_ASYNC=1
```

The additional L3 mode uses `INFERSSD_CPU_L3_LOCAL=1`, an explicit comma-separated
`INFERSSD_CPU_L3_CPUS` list matching the compute participants, and
`INFERSSD_CPU_L3_UP_ROWS=64` / `INFERSSD_CPU_L3_DOWN_ROWS=64`. Select CPUs from the
machine's measured topology; the CPU numbers in the benchmark are not portable
defaults. Unset the experiment variables and clear `expert.replicas` to return to the
default source and scheduling policy.

## Ownership and validation boundary

The source index owns all replica descriptors and lends an immutable view to the
reader. The reader owns its shared routing pool, range-completion bookkeeping,
I/O and prefetch workers. Buffers remain owned until all pieces have completed;
shutdown drains and joins readers before destroying borrowed sources. The pool
is per reader/context, not a machine-wide coordinator. Async worker limits are
per ring and do not constitute a process-wide I/O admission budget.

The device tests cover two/four replica ranges, adaptive/shared routing, byte
accounting, short reads and drain, plus existing lookahead, prefetch and worker
lifecycle cases. Synthetic range outputs are compared bit for bit with the CPU
reference. Existing pread-interposition prefetch fault injection does not
intercept io_uring; that subtest explicitly skips in async mode and remains a
coverage gap. Full-model greedy output can diverge and change expert routing, so
the performance samples below are complete-configuration comparisons rather
than a fixed-token numerical acceptance test.

Payload verification should become an enforceable source contract; experiment
switches need typed-policy integration, and async failure coverage and
cross-platform builds remain work before promotion from draft.

## Two-drive measurements

Hardware: AMD Ryzen Threadripper 9960X (24 cores / 48 threads), **32 GB physical RAM (2 x 16 GB DDR5-5600)**, one RTX 5090 (32 GB), and ZHITAI TiPro9000 1 TB drives, each PCIe 5.0 x4, independent XFS, no RAID. The inference scope is capped at **24 GiB RAM with swap disabled**; the initial expert VRAM budget is **16 GiB**, with the existing elastic policy retained.

Compared untouched main `f79ec88` with native feature commit `eb8ebbc` after rebuilding and running the project's built-in profiling for both. DeepSeek-V4.1-Flash-1017 Q8_0/MXFP4, repository 4K prompt (**4104 actual prompt tokens**), 256 naturally generated greedy tokens, context 8192, one request slot, prompt cache off. **All reported decode rates use the last 128 decoding steps: `128 / (T256 - T128)`.** Three interleaved repetitions per mode; arithmetic means below. All modes use the same verified packs, 16 decode participants in a 17-thread pool, 12 prefill threads, matched memory budgets and explicit placement costs.

| Two-drive configuration | Prefill (tok/s) | Decode: last 128 steps (tok/s) |
| --- | ---: | ---: |
| Two drives, main static mapping | 264.06 | 10.375 |
| Two drives, adaptive I/O | 270.07 | 10.581 |
| Two drives, adaptive I/O + L3 scheduling | 266.13 | 10.585 |


Using the last 128 decoding steps, the combined two-drive mode changes throughput by +2.0% versus the same two-drive main baseline; I/O alone changes it by +2.0%. The sample ranges overlap: these three runs do not demonstrate a stable two-drive speedup, and L3 scheduling adds no clear benefit here.

For additional four-drive context, an earlier replica I/O + L3 run reached **approximately 12.5 tok/s** over the last 128 decoding steps (recorded: 12.486790 tok/s, based on `bd2cd9f`). This is a historical single-run observation, separate from the three-run current-main two-drive table. Its token timings and provenance are included in the results attachment.

Fresh server for every sample; POSIX_FADV_DONTNEED on model files before launch. This is not a guarantee that drive/controller caches are cold. All four complete expert packs were SHA256-verified before profiling and kept read-only. The common placement costs were retained from the preceding main calibration, with CPU costs interpolated for 12 prefill threads; CPU 0-16 and I/O CPU 20-23 masks are machine-specific. Natural outputs may diverge and change routing. Per-run timings, token IDs, min/max/standard deviations, configuration, build hashes, validation and telemetry receipts are in the [machine-readable results](benchmarks/multi-nvme-20261113.json). Generated sequences differed in 153 of 153 pairwise sample comparisons; all prompt token IDs matched. Synthetic read tests provide bitwise reference checks, while these full-model timing runs do not establish numerical equivalence.
