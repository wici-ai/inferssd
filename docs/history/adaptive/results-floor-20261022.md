# The floor at request boundaries on unified memory (2026-10-22)

On unified memory the engine publishes a floor (the host-memory ledger's: file reserve + margin + explicit reserves)
and the GPU tier grows until MemAvailable keeps the floor, the RAM cache's demand and 16-64 MiB (its watermarks).
In the n2 ABBAs on the Spark ysy9 (GB10, DeepSeek-V4.1-Flash-1017 Q8_0) single 4 Hz samples of the harness's memfloor
sampler fell under the floor at request boundaries, by up to 426 MiB. This branch (uma-floor, on 151c871) finds why
and fixes it. The server-level gate is pending (the Muse download shares the drive until MUSE_DL_DONE).

## What took the memory

Every under-floor sample of the three n2 ABBAs (`m1-n2`, `by-fx`, `by-m3`, 32 arms; `~/ssdl/floor-probe/floor-arms.py`, times from the
sampler's first sample):

Settings: the derived ones at 2906 (`y9-fw-dsv-ysy9-5c2fc47.settings`) in `m1-n2` and `by-fx`, at 414 in `by-m3`.

| arm | build | when | margin | MemAvailable - MemFree - Cached step | what the engine did |
|---|---|---|---|---|---|
| `m1-n2-p4096-b-7` | 2826e92 | 141.6 s, request 2, decode tokens 2-3 | -409 MiB | -480 MiB (MemFree -13, Cached -175) | decode_grow +224 slots, watermark +12, +14 |
| `m1-n2-p4096-a-8` | a485f71 | 140.8 s, request 2, decode token 2 | -139 MiB | -319, then -175 MiB | decode_grow +250 slots, then watermark_shrink with free_before=0 |
| `by-fx-p4096-a-1` | 2826e92 | 141.3 s, request 2, decode token 2 | -166 MiB | -354, then -171 MiB | decode_grow +224 slots, watermark +10 |
| `by-m3-p4096-a-1` | 0cc0fcb | 112.0 s, request 2, prefill start | -314 MiB | -503 MiB (MemFree -586, Cached +417) | compute buffers admitted (229 slots given back), graph prefill begins |
| `by-m3-p4096-b-3` | 151c871 | 141.6 s, request 2, decode tokens 2-3 | -426 MiB | none at the dip; +532 / -533 MiB at the next two samples | decode_grow +224, watermark +13; at token 3422 watermark found 648 MiB free (0.25 s earlier 122 MiB) and grew +32 slots (573 MiB) |
| `by-fx-p512-b-3` | 0cc0fcb | 108.5 s, request 2, 10 s into the server's first graph prefill | -15 MiB | none (MemFree -104) | `pinned staging: enabled, 16 slots of 5.98 MiB`, then the watcher's floor reclaim |

No dip came at a checkpoint save (5 MiB, through the ledger), a reply reuse, a seed copy (into slots the tier holds),
KV growth (allocated at load) or the prefill's compute buffers themselves (the ledger and `ggml_cuda_reserve_memory`
admit them first).

**Five of the six: the kernel's watermark boost.** MemAvailable is not a count of free pages. `si_mem_available()`
(mm/show_mem.c, v6.11) keeps back `min(file LRU / 2, wmark_low)` and `min(reclaimable / 2, wmark_low)`, and `wmark_low`
sums `low_wmark_pages(zone) = _watermark[WMARK_LOW] + watermark_boost` (include/linux/mmzone.h). The kernel raises a
zone's boost when an allocation falls back to another migratetype (mm/page_alloc.c `boost_watermark`), up to
`high * vm.watermark_boost_factor / 10000`, and kswapd takes it back once the zone is balanced. On ysy9
(`~/ssdl/floor-probe/memavail.py` on ysy9, read only): factor 15000, the Normal zone's low 42934 and high 74638 pages without the
boost, so its largest boost is 111957 pages (437 MiB); the kernel's formula reproduces MemAvailable to the MiB
(3 samples, diff 0); with the reclaimable half (about 290 MiB) above the unboosted low watermark, MemAvailable moves by
about 560 MiB between no boost and the largest, with nothing allocated or freed.

- A real allocation moves MemFree or Cached: `~/ssdl/floor-probe/uma-probe.cu` on ysy9 (12 cudaMalloc of 287 MiB:
  MemAvailable -279 to -297 MiB each at once, a later memset or H2D copy 0; cudaFree +3507 MiB at once;
  cudaMallocHost 287 MiB: Shmem +286 MiB). The dips' steps move MemAvailable alone, by 319-533 MiB.
- The large allocations at a boundary into a nearly full DRAM are what raise the boost: the prefill's compute buffers
  (3.8 GB), the tier's regrowth at decode token 2 (4.0-4.5 GB in 130-320 ms). The tier measured its room while
  kswapd had just dropped the boost, grew into it, and the boost came back.
- Arms with a step of 300 MiB or more after load: 7 of 24 at p4096; all five boundary dips are among them, none of
  the 17 others went under the floor (lowest margins 0-41 MiB).
- The RAM cache paid too. Its guard reads MemAvailable; below the floor it gives slots back, and a slot given back stays
  disabled. In request 2 the five arms with dips lost 48, 12, 24, 1 and 47 of 240 slots (`m1-n2-p4096-b-7`, `-a-8`,
  `by-fx-p4096-a-1`, `by-m3-p4096-a-1`, `-b-3`); the two other arms with a step lost 24 (`by-fx-p4096-a-5`) and 0
  (`by-fx-p4096-b-2`); the 17 arms without a step lost 0-7. The tier's watermark grew into what they released. (The
  message of f625314 lists "47, 48, 24, 24 and 12" as the arms with steps: that leaves out two step arms and counts
  `by-fx-p4096-a-5`, which had no dip.)
- Request 1 has the same steps (`by-m3-p4096-a-1` at 73.9 s: +525 then -523 MiB) but the RAM cache's demand (1.3-1.7 GB
  of uncommitted slots, which the tier keeps free) absorbed them. In request 2 the demand is 0. The n2 protocol (a
  second request on the same server) is why the dips grew from the warm-up's tens of MiB.

**The sixth: pinned memory outside the ledger.** The moe_stream_batch pinned staging (16 x 5.98 MiB = 96 MiB,
`cudaHostAlloc`) is made at a server's first graph prefill. In the p512 ABBA that was request 2: MemAvailable fell
104 MiB with MemFree, the floor watcher gave a tier block back within its 100 ms. Device allocations asked
`ggml_cuda_reserve_memory` first; pinned host memory, from the same DRAM, did not.

## The fix

1. **`f625314` unified memory: count MemAvailable at the kernel's full watermark boost.** `ggml/src/kimi-memory-boost.h`
   computes, from /proc/meminfo, /proc/zoneinfo (which prints low and high with the boost added) and
   vm.watermark_boost_factor, how far MemAvailable can still fall as every zone's boost grows to its largest, with the
   kernel's own formula. Whoever takes memory up to the floor counts MemAvailable less that:
   - ggml's free device memory on a unified GPU (`ggml_cuda_memory_info`): the tier's decode growth, watermark and
     reclaim, and `ggml_cuda_reserve_memory`;
   - the ledger's `make_room` (allocations outside the caches), the floor watcher, and the margin's intervals (a boost
     moving within a decode token is not untracked growth).

   The tier's room no longer depends on the boost: a dropped boost adds as much to the headroom as to MemAvailable.
   The RAM cache's per-layer guard keeps reading MemAvailable as it is (one /proc/meminfo read): it commits only the
   slots the tier keeps free as its demand. No knob, no constant: factor 0, no boost line, a hugetlbfs pool, or every
   zone at its largest boost keep nothing back. On ysy9 the tier keeps up to about 560 MiB more free (about 30 of
   ~5500 slots) while the boost is low, nothing more while it is at its largest (437 MiB now, with the download).
   Reading /proc/zoneinfo costs 11.5 us there (`~/ssdl/floor-probe/readcost.cpp`), a few times a token.
2. **`c35d217` ggml-cuda: pinned host memory on unified memory makes its room first.** `ggml_cuda_admit_host_alloc`
   runs `ggml_cuda_reserve_memory` for the current device when it is unified; the CUDA host buffer type and both object
   transports' stage slots call it before they allocate. It does not refuse: with nothing to give back (a live layer)
   the allocation proceeds as before, because the callers' fallbacks (pageable buffers, direct H2D) would cost the
   prefill.

## Unit tests (ysy9, eng-floor = c35d217)

`~/ssdl/eng-floor` is 151c871's source (`src-151c871.tgz`) with the branch's diff applied; its tree hashes equal to
git's (8105 files). `LIBURING_PREFIX=~/ssdl/deps BUILD_TESTS=1 tools/adaptive/build.sh`: BUILD_RC=0, test targets
TESTS_BUILD_RC=0 (`~/ssdl/ab/floor-build.sh`, under the lock, each test in an 8 GiB scope without swap; build.log
and `ut-*.out` in `~/ssdl/eng-floor/build-cuda130/bin`). Every test passed:

| test | result |
|---|---|
| test-kimi-cpu-sink (default: host ledger and synthetic) | HOST_LEDGER_OK, CPU_SINK_TEST_OK |
| test-kimi-cpu-sink --prefetch-tests | CPU_SINK_PREFETCH_OK |
| test-kimi-vram-elastic | REAL_CUDA_ELASTIC_PASS assertions=127 unified=1 |
| test-cuda-memory-reserve (GGML_CUDA_MEMORY_RESERVE_MIB=576) | MemAvailable 118092 MiB, headroom 2 MiB (the boost was at its largest), ggml free 118090 MiB; the pinned buffer had 512 MiB reclaimed first: both PASS |
| test-kimi-vram-reclaim, -pool, -deferred | PASS (54, 32309, 31 assertions) |
| test-moe-object-engine stage, host (the stage slots now admitted) | rc=0 |

- test-kimi-cpu-sink (host ledger): the headroom with ysy9's zones (521 MiB with no boost, 613 pages with Normal at
  its largest, 0 at every zone's largest, without a factor, boost lines or inputs, with hugetlbfs); `make_room` counts
  it and has the tier make room, the RAM cache's guard does not; a boost raised within a decode token is not untracked
  growth, 700 MiB behind a dropped boost is; a discrete GPU keeps nothing back.
- test-cuda-memory-reserve (unified device, no large allocation): ggml's free memory is MemAvailable less the headroom;
  with a reserve leaving 8 MiB, a 256 MiB CUDA host buffer has a stand-in cache (512 MiB) given back first and is
  still pinned. (The discrete part would fill the DRAM on unified memory and does not run there.)

## Gate (pending MUSE_DL_DONE)

`~/ssdl/floor-probe/floor-gate.sh` (refuses to start without MUSE_DL_DONE): the n2 ABBA at p4096, 2 cycles, the lock
held across the 8 arms, A = eng-m3 (151c871), B = eng-floor, the derived settings (414); then a boundary stress, A B B
A under one lock, each server serving request 1 and six new conversations (128 tokens each). A side sampler records
the Normal zone's boost and the headroom at 4 Hz (`~/ssdl/ab/floor-boost.txt`). The bar: no 4 Hz sample under the
published floor in any B arm, decode and TTFT within the protocol's noise; A should show steps and, in some arms, dips.
