# final1 on the Spark ysy9 (2026-10-22)

final1 = f6ce81b (dev: unified-placement merged) + bytes-minors + uma-floor. The device is the Spark ysy9 (GB10
unified memory, 20 Arm cores, 121 GiB). Models: DeepSeek-V4.1-Flash-1017 Q8_0 and Muse Spark Open UD-Q4_K_XL (new
download). A = `~/ssdl/eng-m3` (151c871). B = `~/ssdl/eng-final1` (5df2e7f; see section 1 for eca0496).

Summary:

- The review had no blockers or majors. Three minors are fixed in their branches (cb4239d, 2b544c8, 57720ed) and
  merged into final1 again (a8e44d2, 5df2e7f). One harness minor is fixed in the gate's own scripts. Two are listed.
- Unit tests on ysy9 pass, except test-kimi-vram-q2's new host-jobs case from unified-placement, which asked the
  tier too early. The test fix is eca0496 on final1 (tests only); with it the test passes on ysy9 (section 1).
- Every kimi CTest: 124 of 136 pass; the 12 failures are the known `max_abs=0.024879` ones.
- The Muse download finished: rc=0, all 12 shards passed sha256 (shard 5 on its third try).
- Floor: no B arm went under the published floor: 4 n2 arms and 2 stress arms on DeepSeek, 2 Muse arms, and 2
  attribution arms. The stress arms include a moving watermark boost. A (eng-m3) went under it in 3 of its 6
  DeepSeek arms and 1 of its 2 Muse arms (worst -325 MiB), and its RAM cache gave back up to 71 slots. B gave back
  none.
- Decode: B is 1.5-2.8 % slower than A on DeepSeek in three of the four windows (every B arm below every A arm in
  request 2's two windows; request 1's 1-128 is equal, +0.05 %), and 2.3 % slower on Muse at 129-256. On DeepSeek
  the attribution ABBA (dev f6ce81b against final1) puts this on the unified-placement merge, not on bytes-minors or
  uma-floor (section 3.7). Muse had no attribution run. The plan moves fewer experts to the GPU
  than the greedy it replaced (15.1-15.4 against 18.9-20.0 a token on DeepSeek) and is limited by its candidates,
  not by its prices. By unified-placement's own Spark bar (decode not worse beyond the arms' spread; the plan
  replaces the greedy in place there; `wf-unify/design.md:438-439`) this fails.
- TTFT: equal within 0.5 % on both models.
- Teacher-forced: DeepSeek decode 62/64 with the base's two misses (50, 82); prefill 63/64 (one new near tie).
  Muse decode 60/64, prefill 61/64 against 233's llama.cpp reply (no Muse base on ysy9).
- The prefetch depth re-derived from final1's spill is 5 in all 4 B arms; deployed 6. One rank apart: no extra ABBA.

## 0. The review's minors

| # | minor | where | status | evidence |
|---|---|---|---|---|
| 1 | make_room counts MemAvailable at the full boost, but its RAM cache action stopped at the raw MemAvailable | uma-floor | fixed, `2b544c8`: `check_room` and `reclaim_for_guard` take `full_boost`; make_room's action passes it | g++ -fsyntax-only; test-kimi-cpu-sink default (HOST_LEDGER_OK, CPU_SINK_TEST_OK), --cache-load-tests, --elastic-cache-tests pass on ysy9. No dedicated unit test: the cache's ledger reads the real /proc, so the system term cannot be made to bind with a known boost |
| 2 | the read ring was still made resident at load for an automatic budget without partial reclaim or elastic | bytes-minors | fixed, `cb4239d`: `reserve_cache` returns before the ring, from the same predicate as `reserve_at_load` (`gives_room_back`) | `case=load_gate_ring status=PASS resident_growth=0 slot=53489664 depth=2` (f1-load.out); negative control in section 1 |
| 3 | test-cuda-memory-reserve's full-boost check cannot tell old from new at the largest boost | uma-floor | listed | the run again had headroom 2 MiB (`unified: MemAvailable 117304 MiB, boost headroom 2 MiB, ggml free 117301 MiB`); the ledger unit tests tell them apart |
| 4 | results-floor and f625314's message list the wrong arms for the slots given back | uma-floor | doc fixed, `57720ed` (the commit message stays: no history rewrite) | floor-arms.py over the 24 p4096 arms: dips 48, 12, 24, 1, 47; the other step arms 24 and 0; the rest 0-7 |
| 5 | floor-gate.sh checked eng-m3's COMMIT only; by-abba.sh has no per-arm check | harness | fixed in the gate's own scripts: `~/ssdl/wf-m4/f1-abba.sh` checks the engine's COMMIT before every arm and stops on a mismatch | every arm's meta.txt names its COMMIT (`eng=eng-final1 5df2e7f`) |
| 6 | by-125.sh / by-233.sh log a STOP during unit tests into bin/ | 125/233 chains | listed, not touched: those machines run their own gates now | - |

Merges: `a8e44d2` (bytes-minors) and `5df2e7f` (uma-floor) into final1, --no-ff, clean. e1374aa and earlier are unchanged.

## 1. Build and unit tests on ysy9

`~/ssdl/eng-final1`: `src-151c871.tgz` + the 151c871..5df2e7f diff. Its tree hash equals git's (b31d976c,
`~/ssdl/wf-m4/f1-build.log` 11:08:54). `LIBURING_PREFIX=~/ssdl/deps BUILD_TESTS=1 tools/adaptive/build.sh`:
BUILD_RC=0 (11:12:23); the test targets TESTS_BUILD_RC=0. Each test ran once in an 8 GiB user scope without swap,
under the lock (`~/ssdl/wf-m4/f1-build.sh`; outputs `build-cuda130/bin/f1-*.out`).

| test | result |
|---|---|
| test-kimi-host-lock | KIMI_HOST_LOCK_OK |
| test-kimi-prefill-cost | KIMI_PREFILL_COST_OK |
| test-kimi-vram-pool | PASS, 32309 assertions |
| test-kimi-prefetch-tier | PREFETCH_TIER_OK |
| test-kimi-cpu-sink (default) | HOST_LEDGER_OK, CPU_SINK_TEST_OK |
| --prefetch-tests | CPU_SINK_PREFETCH_OK (spill_twice, spill_last PASS) |
| --cache-load-tests | CACHE_LOAD_OK; load_regrow_owed `before_slots=4 after_slots=5 of=6 owed_slots=1`; load_gate_ring resident growth 0 |
| --elastic-cache-tests | ELASTIC_CACHE_OK |
| test-kimi-vram-elastic | REAL_CUDA_ELASTIC_PASS, 127 assertions, unified=1 |
| test-kimi-vram-reclaim | PASS, 54 assertions |
| test-kimi-vram-deferred | PASS, 31 assertions |
| test-kimi-vram-group | KIMI_VRAM_GROUP_PASS checks=88, worst NMSE 3.14e-4 |
| test-cuda-memory-reserve (RESERVE_MIB=576) | both PASS (headroom 2 MiB: the boost at its largest) |
| test-moe-object-engine stage, host | rc=0 |
| test-kimi-vram-q2 at 5df2e7f | rc=1: `REAL_GPU_Q2_TIER_FAIL: costs: no copy measured before the probe` (see below) |
| every kimi CTest (`ctest -R "kimi\|dsv41"`, 136) | 124 pass; 12 fail = schedule/expert_job t{2,12,16}_d{1,8}, all `max_abs=0.024879` (the known 12); 0 Not Run; kimi_host_lock, kimi_prefill_cost, kimi_dense_stream, kimi_cpu_sink_prefetch, kimi_cpu_sink_cache_load pass |

**test-kimi-vram-q2.** The host-jobs case (`tests/kimi-host-jobs-test.inc`, from unified-placement) reads
`costs2().in_place` right after it creates its tier. The tier reads the device (`detect_unified`, vram-tier.cpp:378)
at its first layer (vram-tier.cpp:1393, 1480), so every tier reports in_place=0 before that, also on GB10. The test
took the discrete path, and its first check failed on a tier that had since read the device. The engine reads
costs2 only in decode, after the tier's layers (kimi-cpu-sink.cpp:2832, 2989); the servers here logged
`[kimi_host_register] reason=decided enabled=0 in_place=1`. Fix `eca0496` (final1, tests only): ask a throwaway
tier after one layer. eng-final1 then got `tests/kimi-host-jobs-test.inc` at eca0496 (md5 4705848b, equal to
git's) and the two tests that include it were rebuilt; the engine libraries and llama-server have the same md5 before
and after (26 files, `~/ssdl/wf-m4/libs-{before,after}.md5`). test-kimi-vram-q2: `HOST_JOBS_PASS type=q8_0
in_place=1 ... slots=0`, `REAL_GPU_Q2_TIER_PASS checks=146` (`f1e-q2.out`). test-kimi-vram-mxfp4 fails earlier, at
its MXFP4 parity self-check (`cpu=0x1.242ab8p+12 gpu=0x1.242ab6p+12 ... the GPU kernel does not reproduce this
host's ggml_vec_dot_mxfp4_q8_0 path`), the same on eng-m3's binary: a GB10 limit of that test, not of this branch.
It is not a CTest. So the MXFP4 host-jobs case did not run on GB10.

**Negative control** for `cb4239d`: final1's test-kimi-cpu-sink `--cache-load-tests` against eng-minors' libraries
(a19d154: 079198e without cb4239d) fails at load_gate_ring: `resident +106987520 bytes, a slot 53489664` (the ring's
two slots; `f1e-control-load.out`).

## 2. The Muse download

`~/ssdl/dl/status.txt`: `2026-10-22_11:13:49 MUSE_DL_DONE rc=0`. `log-muse.txt`: 12 distinct files VERIFIED
sha256, no GIVING UP. Shard 5 failed twice with the same wrong hash (2148d13a..., 10:19:24 and 11:00:12; the
downloader fetched it again itself) and passed at 11:13:44. The 12 files under
`/media/golden-root/var/tmp/<user>/models/Muse-Spark-Open-816B-A42B-GGUF/UD-Q4_K_XL` sum to 505,502,231,288 bytes,
the manifest's total. Dirty + Writeback were 0 MiB at 11:16:18, before the first timed run.

## 3. DeepSeek

Settings: `y9-fw-dsv-ysy9-derived.settings` (threshold 414, prefetch 6). Preflight (`f1-gate.log` 11:16:18): user
slices without MemoryMax/SwapMax, no LD_PRELOAD, no llama-server, no GPU app. One lock over the whole chain
(`~/ssdl/wf-m4/f1-gate.sh`), STOP checked before each step and each arm, watchdog 3072 MiB (it fired in no run),
the memfloor sampler in every arm, and a 4 Hz side sampler of the Normal zone's boost (`f1-boost.txt`).

### 3.1 n2 ABBA at p4096, 2 cycles

Request 1 (4104 tokens) and a new conversation (3420 tokens) on the same server, 256 tokens each.
Runs `~/ssdl/ab/runs/1022-1116..1138-f1-dsv-p4096-*`. "When" is estimated from load_s, TTFT and the decode rates.

| arm | engine | under floor (n, worst MiB, when) | lowest margin MiB | lowest MemAvailable MiB | r1 TTFT s | r1 decode 1-128 / 129-256 | r2 TTFT s | r2 decode 1-128 / 129-256 | GPU host experts a token | tier hits a token | RAM slots given back (r2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 111623-f1-dsv-p4096-a-1 | eng-m3 | 2: -295 at 90 s (~request 1 decode token 78); -4 at 167 s (~request 2 decode token 152) | -295 | 4255 | 32.84 | 5.912 / 7.283 | 29.32 | 6.974 / 7.271 | 18.98 | 219.6 | 21 (3) |
| 111935-f1-dsv-p4096-b-2 | eng-final1 | 0 | 148 | 4834 | 32.36 | 6.245 / 7.073 | 29.04 | 6.914 / 7.142 | 15.36 | 219.4 | 0 (0) |
| 112244-f1-dsv-p4096-b-3 | eng-final1 | 0 | 153 | 4703 | 32.32 | 6.253 / 7.073 | 29.13 | 6.927 / 7.130 | 15.11 | 219.5 | 0 (0) |
| 112553-f1-dsv-p4096-a-4 | eng-m3 | 0 | 21 | 4572 | 32.44 | 6.375 / 7.289 | 29.03 | 7.025 / 7.433 | 19.33 | 219.7 | 0 (0) |
| 112900-f1-dsv-p4096-a-5 | eng-m3 | 0 | 15 | 4565 | 32.35 | 6.317 / 7.069 | 29.18 | 7.016 / 7.229 | 20.04 | 218.7 | 1 (0) |
| 113208-f1-dsv-p4096-b-6 | eng-final1 | 0 | 164 | 4714 | 32.29 | 6.252 / 7.042 | 29.01 | 6.890 / 7.026 | 15.36 | 219.2 | 0 (0) |
| 113517-f1-dsv-p4096-b-7 | eng-final1 | 0 | 155 | 4705 | 32.64 | 6.214 / 7.092 | 29.00 | 6.918 / 7.112 | 15.29 | 219.5 | 0 (0) |
| 113825-f1-dsv-p4096-a-8 | eng-m3 | 0 | 17 | 4567 | 32.40 | 6.348 / 7.258 | 29.04 | 7.048 / 7.282 | 18.91 | 219.7 | 1 (1) |

| request, window | A mean (range) | B mean (range) | B against A (cycle 1, 2) | every B below every A |
|---|---|---|---|---|
| r1 TTFT s | 32.51 (32.35-32.84) | 32.40 (32.29-32.64) | -0.32 % (-0.92, +0.29) | no |
| r1 decode 1-128 | 6.238 (5.912-6.375) | 6.241 (6.214-6.253) | +0.05 % (+1.72, -1.57) | no |
| r1 decode 129-256 | 7.225 (7.069-7.289) | 7.070 (7.042-7.092) | -2.14 % (-2.92, -1.35) | no |
| r2 TTFT s | 29.14 (29.03-29.32) | 29.05 (29.00-29.13) | -0.32 % (-0.30, -0.34) | no |
| r2 decode 1-128 | 7.016 (6.974-7.048) | 6.912 (6.890-6.927) | -1.48 % (-1.13, -1.82) | yes |
| r2 decode 129-256 | 7.304 (7.229-7.433) | 7.103 (7.026-7.142) | -2.76 % (-2.94, -2.57) | yes |

A's arm 1 has the low r1 decode 1-128 (5.912): by the estimate, its -295 MiB dip fell in that window.

### 3.2 The floor (uma-floor's bars)

- (a) B: 0 samples under the published floor in all 4 arms; lowest margin 148-164 MiB. A: arm 1 went under twice
  (-295 MiB at about request 1 decode token 78, -4 MiB in request 2); its other arms kept 15-21 MiB.
- (b) Time: see 3.1 and 3.7. The decode deficit is unified-placement's, not the floor's.
- (c) B's RAM cache gave back 0 slots (all arms, both requests). A: 21 (3 in request 2), 0, 1, 1.
- (d) The boost moved only during A's arm 1 (Normal boost 0-437 MiB, headroom 2-707 MiB). From arm 2 to arm 8 it
  stayed at 304 MiB (headroom 135 MiB) in every sample, so these 7 arms do not test a moving boost. The stress
  (3.3) does.
- No watchdog, no `[kimi_host_ledger] event=refused` in any arm. B's tier kept the headroom free: its largest size
  was 5494-5503 slots against A's 5507-5542 (135 MiB is 7 slots); tier hits a token were equal (219.2-219.5 against
  218.7-219.7).

### 3.3 Boundary stress (A B B A, one lock)

Request 1 at p4096 and six new conversations on each server (1523, 3420, 2936, 4104, 3420 and 523 tokens), 128
tokens each. Runs `1022-1141..1200-f1-dsvst-p4096-*`.

| arm | engine | under floor (n, worst MiB, when) | lowest margin MiB | Normal boost MiB | headroom MiB | decode 1-128 of the 7 requests | GPU host experts a token | RAM slots given back |
|---|---|---|---|---|---|---|---|---|
| 114132-f1-dsvst-p4096-a-1 | eng-m3 | 1: -18 at 277 s (~request 5 decode token 122) | -18 | 133-437 | 2-359 | 6.37, 6.65, 6.69, 6.86, 6.60, 7.04, 6.77 | 24.94 | 18 |
| 114742-f1-dsvst-p4096-b-2 | eng-final1 | 0 | 162 | 157-437 | 2-282 | 6.18, 6.42, 6.55, 6.64, 6.41, 6.84, 6.68 | 19.26 | 0 |
| 115356-f1-dsvst-p4096-b-3 | eng-final1 | 0 | 293 | 0-157 | 282-595 | 6.20, 6.40, 6.50, 6.59, 6.38, 6.84, 6.70 | 19.13 | 0 |
| 120012-f1-dsvst-p4096-a-4 | eng-m3 | 2: -199 at 209 s (~request 4 prefill 27.4 s in); -325 at 210 s (~request 4 prefill 27.7 s in) | -325 | 0-437 | 2-595 | 6.37, 6.59, 6.50, 6.74, 6.58, 6.91, 6.86 | 25.28 | 71 |

The boost moved in all four arms. B kept the floor (margins 162 and 293 MiB) while the headroom went from 2 to 595
MiB. A went under it three times (worst -325 MiB, at request 4's prefill) and its RAM cache gave back 18 and 71
slots. Decode 1-128 of B is 1.0-3.2 % below A in each of the 7 requests (every B arm below every A arm in 6 of 7).

### 3.4 The decode plan (unified-placement) on the Spark

B arms' `[kimi_decode_place] reason=destroy` (510 decode tokens each):

| arm | GPU experts a token | planned / candidates | refused, wanted | error p50 / p90 % | predicted / measured / CPU only, s | CPU expert us | host / resident us an expert |
|---|---|---|---|---|---|---|---|
| b-2 | 15.42 | 7864 / 8107 | 0, 0 | 16 / 100 | 32.48 / 32.43 / 36.24 | 1675 | 297 / 183 |
| b-3 | 15.17 | 7735 / 8039 | 0, 0 | 15 / 100 | 32.56 / 32.65 / 36.18 | 1577 | 309 / 191 |
| b-6 | 15.42 | 7862 / 8149 | 0, 0 | 16 / 100 | 32.87 / 33.04 / 36.55 | 1576 | 329 / 187 |
| b-7 | 15.35 | 7829 / 8076 | 0, 0 | 16 / 100 | 32.55 / 32.73 / 36.20 | 1665 | 301 / 187 |

- `[kimi_vram_host] in_place=1`, jobs = the planned moves, copies 0. `[kimi_host_register] reason=decided enabled=0
  in_place=1`: nothing is page-locked on unified memory.
- The greedy in A moved more: `[kimi_vram_host_sink] gpu_host_experts` 18.9-20.0 a token (n2) and 24.9-25.3
  (stress), against B's 15.1-15.4 and 19.1-19.3.
- The plan moved 96-97 % of its candidates. So on the Spark it is limited by its candidate set (host-ready RAM hits
  and misses by landing), not by its prices. The design's replay predicted more moves than the greedy (DSV 33 a
  token) and +2.8 ms a token. Measured: fewer moves, and request 2's decode 129-256 at 140.8 ms a token against A's
  136.9 ms (+3.9 ms).

### 3.5 bytes-minors' spill counters

| | A (4 arms) | B (4 arms) |
|---|---|---|
| spill_us at destroy | 2.32-2.43 s | 2.17-2.29 s |
| spills | 11272-11531 | 10892-11209 |
| experts started (sum of started_r) | 11819-12289 | 12059-12291 |
| issued / skip_vram / hits | 13206-13593 / 104123-104686 / 4829-5094 | 13318-13669 / 104136-104508 / 4901-4912 |

- The last MoE layer's run now opens the window, and B's spill time is 0.14 s lower a server (stress: 4.53-4.57 s
  against 4.94-5.01 s). Issued, skip_vram and hits stay within the arms' spread.
- Each periodic report (at 8192 and 16384 layers; 4 in the stress arms) is below the destroy report's spill_us, in
  every arm. This was item 2a's only check.
- Seeds: planned = published in every arm (A 5184-5232, B 5184-5200), none cancelled.

### 3.6 Teacher-forced, and the prefetch depth

- `probe-tf-decode.py` at 414 on eng-final1 against 233's p512 reference (`runs/1022-120624-f1-tfd-eng-final1`):
  decode 62/64 with the base's misses at 50 and 82 (the same tokens, log-probabilities within 0.11 of eng-m3's
  `1022-090108-m3-tfd-eng-m3`); prefill 63/64, one new miss at 29 (top-1 ' ' -0.611 against ' operates' -0.784, a
  near tie; base 64/64). The arithmetic is not bitwise across these engines (the plan computes other experts on
  the GPU), so a near tie can flip.
- `by-derive.py` on the bootstrap `fw/runs/dsv-ysy9`, with each B arm's prefetch report in place of the bootstrap's
  (`wf-m4/der/d-b{2,3,6,7}.txt`): depth 5 in all four (min(k_pay 5, k_mem 6); spill 2.17-2.29 s over 12059-12291
  experts read). Deployed: 6. One rank apart, so no extra ABBA; the bytes results already found 5 and 6 equal in an
  ABBA (`results-bytes-20261022.md`, d).

### 3.7 Attribution: dev (f6ce81b) against final1

The gate's deficit could come from any of the three merges. `~/ssdl/eng-f6ce` = f6ce81b (src-151c871.tgz + diff,
tree hash dd995453 equal to git's; server build only), then an n2 ABBA at p4096, one cycle, the lock held, A =
eng-f6ce, B = eng-final1 (`~/ssdl/wf-m4/f1-after.sh`, runs `1022-1233..-f1-f6-p4096-*`).

| arm | engine | under floor (n, worst MiB, when) | lowest margin MiB | lowest MemAvailable MiB | r1 TTFT s | r1 decode 1-128 / 129-256 | r2 TTFT s | r2 decode 1-128 / 129-256 | GPU host experts a token | tier hits a token | RAM slots given back (r2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 123350-f1-f6-p4096-a-1 | eng-f6ce | 0 | 16 | 4568 | 32.67 | 6.100 / 7.095 | 29.09 | 6.894 / 7.144 | 14.78 | 219.9 | 46 (0) |
| 123700-f1-f6-p4096-b-2 | eng-final1 | 0 | 27 | 4626 | 32.28 | 6.252 / 7.091 | 29.02 | 6.958 / 7.122 | 15.09 | 219.6 | 0 (0) |
| 124007-f1-f6-p4096-b-3 | eng-final1 | 0 | 562 | 5141 | 32.40 | 6.255 / 7.088 | 29.02 | 6.920 / 7.120 | 15.08 | 219.6 | 0 (0) |
| 124317-f1-f6-p4096-a-4 | eng-f6ce | 0 | 12 | 4562 | 32.34 | 6.271 / 7.103 | 29.04 | 6.790 / 7.121 | 14.96 | 219.8 | 47 (44) |

| request, window | A = eng-f6ce, mean (range) | B = eng-final1, mean (range) | B against A |
|---|---|---|---|
| r1 TTFT s | 32.51 (32.34-32.67) | 32.34 (32.28-32.40) | -0.52 % |
| r1 decode 1-128 | 6.185 (6.100-6.271) | 6.253 (6.252-6.255) | +1.10 % |
| r1 decode 129-256 | 7.099 (7.095-7.103) | 7.090 (7.088-7.091) | -0.13 % |
| r2 TTFT s | 29.06 (29.04-29.09) | 29.02 (29.02-29.02) | -0.17 % |
| r2 decode 1-128 | 6.842 (6.790-6.894) | 6.939 (6.920-6.958) | +1.42 % |
| r2 decode 129-256 | 7.133 (7.121-7.144) | 7.121 (7.120-7.122) | -0.16 % |

- dev and final1 decode alike: within -0.16 % at 129-256 in both requests, and final1 is 1.1-1.4 % faster at 1-128
  (one cycle). dev's own level (r2 129-256: 7.12-7.14) is final1's in the gate (7.03-7.14), not eng-m3's
  (7.23-7.43). So the gate's deficit is the unified-placement merge's; bytes-minors and uma-floor cost nothing
  measurable.
- dev has the plan too: 14.8-15.0 GPU host experts a token (final1 15.1). Planned 7565-7658 of 7788-7911
  candidates (0.97).
- dev has no uma-floor: its RAM cache gave back 46 and 47 slots (44 in request 2 of arm 4); final1 none. Neither
  went under the floor in these four arms (dev's margins 12 and 16 MiB, final1's 27 and 562). dev's tier grew to
  5546-5556 slots, final1's to 5500.
- The boost sampler had stopped with the gate, so these arms have no boost record.

## 4. Muse

`muse-spark-gb10.settings` for both arms (graph prefill, prefetch 4, CONTEXT 8192, no checkpoint, no prompt
cache), `models/muse-spark.conf`, the launcher `run-ssd-llama.sh`. One unified-protocol turn at p4096 (4221 tokens,
256 tokens, `f1-muse-launch.sh` = muse-launch.sh plus COMMIT, cool-down and the memfloor sampler). One A B B A cycle
under the gate's lock. Runs `1022-1208..1224-f1-muse-p4096-*`.

| arm | engine | load s | under floor | lowest margin MiB | lowest MemAvailable MiB | TTFT s | decode 1-128 / 129-256 | GPU host experts a token | tier hits a token | spill s (spills) | seeds planned / published |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 120807-f1-muse-p4096-a-1 | eng-m3 | 76 | 0 | 16 | 5183 | 52.50 | 2.035 / 2.284 | 63.23 | 383.1 | 2.62 (5380) | 2581 / 2581 |
| 121224-f1-muse-p4096-b-2 | eng-final1 | 70 | 0 | 309 | 5476 | 52.38 | 2.047 / 2.209 | 64.04 | 377.8 | 2.60 (5459) | 2565 / 2565 |
| 121636-f1-muse-p4096-b-3 | eng-final1 | 72 | 0 | 310 | 5477 | 52.56 | 2.050 / 2.214 | 63.09 | 377.9 | 2.53 (5335) | 2581 / 2581 |
| 122049-f1-muse-p4096-a-4 | eng-m3 | 68 | 1: -12 at 170 s (~request 1 decode token 101) | -12 | 5155 | 52.46 | 2.043 / 2.245 | 65.60 | 381.2 | 2.51 (5292) | 2581 / 2581 |

| window | A mean (range) | B mean (range) | B against A | every B below every A |
|---|---|---|---|---|
| TTFT s | 52.48 (52.46-52.50) | 52.47 (52.38-52.57) | -0.01 % | no |
| decode 1-128 | 2.039 (2.035-2.043) | 2.048 (2.047-2.050) | +0.47 % | no (every B above) |
| decode 129-256 | 2.264 (2.245-2.284) | 2.212 (2.209-2.214) | -2.34 % | yes |

- No attribution run for Muse: its -2.3 % at 129-256 is final1 against eng-m3 only. Two things differ there: the
  plan, and B's 10 fewer tier slots with fewer tier hits (below).
- Floor: B 0 samples under it (margins 309-310 MiB); A arm 4 went under once (-12 MiB at about decode token 101).
  The boost stayed at 136 MiB (headroom 303 MiB) in all four arms. B's tier was 10-12 slots smaller (2850-2851
  against 2861-2862), and it hit 377.8-377.9 tier experts a token against 381.2-383.1.
- The plan: 63.3-64.3 GPU experts a token, planned 16151-16394 of 23102-23190 candidates (0.70), refused 0, error
  p50 13 % / p90 36 %, predicted 73.7-73.8 s against 83.7 s measured. The greedy in A moved 63.2-65.6 a token.
- Against the old Spark (spark-b9a5, `results-spark-20261021.md`): p4096 TTFT 49-51 s there, 52.5 s here; decode
  1.72-1.80 / 2.04-2.14 there, 2.04-2.05 / 2.21-2.28 here.
- Teacher-forced on eng-final1 (`runs/1022-122459-f1-tfm-eng-final1`): decode 60/64, prefill 61/64. No Muse
  reference from the old Spark is on ysy9 or in the scratchpad (its llama.cpp turn was
  `~/bench/runs/20261020-005656-muse-fit7g-p512` on spark-b9a5). The reference used is 233's llama.cpp turn on the
  same GGUF (`bl233/runs/20261019-233051-muse-default-p512`, 669 prompt tokens, a full 256-token reply), copied to
  `~/ssdl/ab/ref-muse-p512-233.json`. Six of the seven misses are near ties (log-probability gaps 0.0002-0.31);
  prefill position 101 is not (' core' -0.43 against ' service' -1.26). There is no eng-m3 run against this
  reference, so no base to compare.

## 5. Open

- unified-placement's decode plan is slower than the greedy it replaced on the Spark (3.4, 3.7). Its candidate set
  there is narrower than what the greedy moved. This is in dev already (f6ce81b), so final1 inherits it.
- Minors 3 and 6 (section 0).
- dev moved again during this work: adaptive-derived-settings = 76768b3 (warm-discrete, e7a277a: the vram tier's
  young-table counts). final1 merges into it cleanly (`git merge-tree --write-tree 76768b3 final1`: tree bea9546a,
  no conflict). That change was not gated here.
- `~/ssdl/eng-final1` is left in place. COMMIT = eca0496 (final1's HEAD); COMMIT-NOTE says the engine was built at
  5df2e7f and that eca0496 changes only the host-jobs test. The gate ran with COMMIT = 5df2e7f. `~/ssdl/eng-f6ce`
  (f6ce81b, server only) is also left there.
