# Fewer drive bytes a token: three levers (2026-10-22)

The offline analysis of 125 (RTX 5090, DeepSeek-V4.1-Flash-1017) found decode drive-bound: about 70 % of a token is
the demand read path. It ranked the levers by size and by how sure they are on every traced machine. This branch
builds the three that help or are neutral everywhere:

- **a.** The next-layer prefetch asks the GPU tier what it holds now (`7c4dbf1`, `cab8379`; after review `f124650`
  a seed is held from its publication, `3dc8939` skips the tier gives back are counted, `5fd7c1a` the reads' spill
  past their window is measured; after the merge review `6e94d58` a spilling read counts once, `1154d5f` a run
  without a hint opens the window).
- **d.** The framework's prefetch depth can be 0: a read pays or is not made (`8303a2b`; after review `dc146f8` each
  rank's spill comes from the engine's measure).
- **R1.** Off unified memory the RAM expert cache takes its arena and pages at load (`5c3c06a`, `331999c`; after
  review `76983a7` its page-in threads cannot terminate the process, `2dc88a1` it prices the model's top-k and runs
  without a GPU tier, `0cc0fcb` the first decode takes back what the first prefill gave, `bea3923` the receipt's load
  time and first touches by window; after the merge review `fec1a7c` the slots the load step owes take their room
  first, `079198e` the load step only for a cache that can give room back).

Not built (they wait for measured a and d and for unified placement): a landing pool for prefetches, preemptible
reads, one RAM pool with one count, demotion of tier victims.

The merge with dev (151c871) was checked on ysy9, and the fixes after its review are on the branch bytes-minors:
both are at the end.

The device was the Spark ysy9 (GB10, unified memory, DeepSeek-V4.1-Flash-1017 Q8_0). 125 and 233 were busy: their
gates are listed at the end with the analysis's predictions. Builds on the Spark: eng-m1 (dev 2826e92, the
reference), eng-bytes (the branch: first rule a alone, `d51c549` = the code of `cab8379`; after review the branch at
`0cc0fcb`, the engine of the final commit), eng-bytes-r1 (the code of `331999c`, before review). Every build, unit
test and server ran under `~/ssdl/spark.lock`, one server at a time, watchdog floor 3072 MiB; the scripts are
`~/ssdl/ab/by-*`.

## a. The prefetch asks the tier

**Before.** `prefetch_schedule` skipped an expert only if its last selection at the next layer had been a tier hit
(`last_vram`). That flag goes stale both ways: the tier admits experts after their last selection, and evicts
others. At decode start it is empty, so every seed of the cold tier was read again at its first selection.

**After.**
- The slot table keeps, per layer and expert, how many slots hold it: a slot valid with it, or reserved for a
  deferred upload of it that the layer's next prepare collects first. A seed counts from its publication, which no
  prepare does: a prefill's planned seeds are published when a decode phase completes the prefill, Q2's cold-prefill
  admission right after its own copy. Every change of a slot's key, valid, uploading or seed goes through one
  function, so the answer is a lookup.
- Residency ABI 2 adds `held(layer, experts, count, out)`: no lease, any phase, under the tier's lifecycle mutex.
- The prefetch skips a hinted expert the tier holds (`skip_vram`), and it is not novel in the statistics.
- `prefetch_drain`'s `needed()` counts only experts the run computes on the CPU. A read whose expert the tier came to
  hold before its run is counted (`vram_late`) and is not read urgently. The converse, an expert skipped as held that
  the tier gave back before the next run selects it (a host or OOM reclaim between the two runs), is counted too
  (`skip_vram_lost`).

**Unit tests** (eng-bytes at `0cc0fcb`, `~/ssdl/ab/by-fx1.log`):

| test | result |
|---|---|
| test-kimi-vram-pool: `holds()` against a scan of the slots through admissions, evictions, deferred uploads, seeds (not held while copied, held once published, never once cancelled, freed unused), a global shrink, native cancel, abort | pass, 32309 assertions |
| test-kimi-vram-reclaim: the ABI on a deferred upload in flight (held), after its block is given back (not held) | pass, 54 assertions |
| test-kimi-prefetch-tier: the sink against a stand-in tier: a held hint is skipped and not novel; a read the tier comes to hold is not needed; an expert the tier gives back before the run is lost; a read inside its window does not spill | pass: skip_vram 1, issued 1, vram_late 1, skip_vram_lost 1, activated 0, novel_r 1,0,1,0, started_r 1,0,0,0, spills 0 |
| test-kimi-cpu-sink (default), --prefetch-tests (only the unselected expert's reads may spill: spills 0 here), --elastic-cache-tests, --cache-load-tests; test-kimi-vram-deferred, -elastic, -q2 | pass |
| test-kimi-vram-exclusive | fails at the first GPU hit, as on dev; see below |
| every kimi CTest (`ctest -R "kimi\|dsv41"`, 135) | 123 pass; the 12 `kimi_cpu_sink_schedule_t*` and `kimi_cpu_sink_expert_job_t*` fail with "job_cold diagnostic numerical tolerance exceeded; max_abs=0.024879", and dev 2826e92 (`~/ssdl/by-ref`) fails the same 12 with the same max_abs (`by-ctref.out`) |

**Counters** (one traced arm each, the p4096 protocol turn, `by-ctr.py`): `by-ctr-eng-m1` (dev), `by-ctr-eng-bytes`
(rule a alone) and `fx-ctr-eng-bytes` (the branch after review). The settings were `y9-fw-dsv-ysy9-5c2fc47.settings`
(threshold 2906, prefetch depth 6), not the current derived file `y9-fw-dsv-ysy9-derived.settings` (414), which differs
from it only in the threshold. At this prompt the routes are the same at both thresholds: 2 tokens to the sink, 4104 to
the graph.

| a token, unless named | eng-m1 | rule a | the branch | change, rule a |
|---|---|---|---|---|
| prefetch reads issued (run) | 11152 | 6761 | 6743 | -39 % |
| of them used from RAM by the next layer | 18.8 % (2096) | 34.8 % (2354) | 34.5 % (2329) | used reads +12 % |
| residual: novel, selected and not used, over issued | 27.0 % | 1.2 % | 1.2 % | |
| skip_vram / vram_late / skip_vram_lost | 46660 / - / - | 52051 / 0 / - | 52027 / 0 / 0 | |
| prefetch bytes | 0.678 GB | 0.438 GB | 0.436 GB | -35 % |
| tier hits, 1-128 / 129-256 | 215.62 / 223.35 | 215.48 / 223.20 | 215.52 / 223.26 | -0.06 % / -0.07 % |
| RAM hits | 10.41 / 6.67 | 11.16 / 8.82 | 11.04 / 8.68 | |
| demand misses | 13.98 / 9.98 | 13.35 / 7.98 | 13.45 / 8.06 | -4.5 % / -20.0 % |
| demand bytes | 0.263 / 0.188 GB | 0.251 / 0.150 GB | 0.253 / 0.152 GB | |
| drive reads, 1-128 / 129-256 (the drive's counter) | 1.16 / 0.63 GB | 0.79 / 0.52 GB | 0.76 / 0.55 GB | -32 % / -18 % |
| drive reads, the whole decode (the drive's counter) | 0.90 GB | 0.65 GB | 0.65 GB | -27 % |
| new RAM-cache slots in 1-128 (the prefetch's staging) | 4.459 GB | 4.515 GB | 4.515 GB | same |
| prefetch reads' spill past their windows (run) | - | - | 1.16 s, 5362 spill events | |

The drive rows are the drive's own reads. The launcher samples `/proc/diskstats` every second (`monitor.txt`, column
3: the MB the drive read in that second). The seconds whose stamp falls between token 1 and token 128 are summed and
divided by 127 tokens; 128 to 256 by 128; the whole decode (1 to 256) by 255. An independent checker computed them
(`drive.py`, `drive2.py` in the workflow's scratchpad `vm3/`). One-second samples blur the window edges: moving both
windows by a second either way changes a window's figure by up to 0.05 GB (the branch's 1-128: 0.75-0.79 GB). So the
whole-decode figure is the firm one: 0.90 -> 0.65 GB a token, -27 %, for rule a and for the branch. The branch's split
is -35 % / -13 %. An earlier version of these rows was not measured per window. It added the run's average prefetch
bytes (the readers' `reader_bytes` over 255 tokens) to each window's demand bytes: 0.940 / 0.865 GB for dev and 0.689
/ 0.588 GB for both builds. But the prefetch's reads are front-loaded. The engine's periodic report at hinted layer
8192 (about token 210 of 255) has dev's readers at 154604636608 of their 172765383104 bytes (89 %), and the branch's
at 95149509632 of 111159311360 (86 %). So that split gave 1-128 too few bytes and 129-256 too many. The spill row
counts events: this engine counted a read once for every window it ran past. After the merge review a read counts once
(below).

The residual row is what the analysis called the tier-served share: an inference from the statistics (novel and
routed less used, over issued, ranks within the depth), not a count of reads the tier served. The direct numbers are
`skip_vram`, `issued` and `vram_late`. Every seed was published before decode in these runs (5232 planned, 5232
published), so the seed rule changes nothing here.

**ABBA**, the n2 protocol: A B B A twice, eng-m1 (A) against eng-bytes (B), the lock held across the 8 arms, each a
cold server at `y9-fw-dsv-ysy9-5c2fc47.settings` (threshold 2906; not the current 414, see the counters). Request 1 is
the protocol turn after load; request 2 a new conversation on the same server (`work-4turn.json` entry 2, 3420
tokens), 256 tokens each, EOS ignored (`w2r-eval.py`).

| ABBA, request | A: TTFT, decode 1-128 / 129-256 | B | cycle 1 | cycle 2 |
|---|---|---|---|---|
| rule a alone (`by-a2-p4096-*`), 1, p4096 | 32.24 s, 6.114 / 7.098 | 32.22 s, 6.419 / 7.316 | -0.06 %, +4.50 % / +3.09 % | -0.04 %, +5.47 % / +3.05 % |
| rule a alone, 2, new text | 28.90 s, 6.795 / 6.912 | 28.86 s, 7.050 / 7.436 | -0.08 %, +3.97 % / +7.68 % | -0.21 %, +3.56 % / +7.47 % |
| the branch (`by-fx-p4096-*`), 1, p4096 | 32.21 s, 6.128 / 7.021 | 32.20 s, 6.420 / 7.294 | -0.06 %, +5.65 % / +5.04 % | 0.00 %, +3.90 % / +2.76 % |
| the branch, 2, new text | 28.95 s, 6.684 / 6.933 | 28.91 s, 7.069 / 7.387 | -0.41 %, +5.33 % / +7.58 % | +0.17 %, +6.19 % / +5.51 % |
| the branch (`by-fx-p512-*`), 1, p512 | 16.50 s, 5.407 / 7.089 | 16.49 s, 5.700 / 7.313 | -0.12 %, +5.62 % / +3.95 % | +0.06 %, +5.21 % / +2.38 % |
| the branch, 2, new text | 29.04 s, 6.580 / 7.002 | 29.06 s, 6.972 / 7.428 | -0.01 %, +6.35 % / +7.33 % | +0.16 %, +5.56 % / +4.88 % |

- In all three ABBAs every B arm is faster than every A arm in both decode windows of both requests (the branch at
  p4096, request 1: A 6.073-6.176 and 6.838-7.111, B 6.385-6.473 and 7.269-7.307; at p512: A 5.385-5.456 and
  7.045-7.127, B 5.658-5.776 and 7.271-7.436; `by-arms.py`).
- Over both cycles the branch reads +4.8 % / +3.9 % at p4096 and +5.4 % / +3.2 % at p512 in request 1, and +5.8 % /
  +6.6 % and +6.0 % / +6.1 % in the new conversation. TTFT does not move (-0.12 % to +0.08 %).
- The p512 prompt prefills on the sink (523 tokens under the threshold 2906), so it has no graph seeds; p4096 has
  5232, all published before decode. At the current threshold 414 the p512 prompt would go to the graph, so p512 at
  414 is not measured here. The p4096 runs route every pass the same way at both thresholds (2 tokens to the sink,
  4104 and 3420 tokens to the graph).
- Load 40-41 s in every arm. The engine's floor on unified memory (`w2-memfloor.py`): every arm stayed above it at
  every one-second sample but two, each once: dev's arm `by-fx-p4096-a-1` 166 MiB under it, the branch's
  `by-fx-p512-b-3` 15 MiB under it; the lowest margins of the others were 0-31 MiB. The watchdog's 3072 MiB floor was
  never reached.

**Teacher-forced: not decidable on this host.** `probe-tf-decode.py` (threshold 414, 64 requests, 128 positions)
agrees with the reference at decode 62/64 in every run of either build, with the same two misses (positions 50 and
82, near ties). But no two runs of one build are bit for bit equal on this GB10, base included, so equality with the
reference cannot be shown or refuted here:
- The runs at the derived settings (`m1-tfd-eng-m1`, `by-tfd-eng-m1`, `by-tfd-eng-bytes`, before review): base
  against base had the same top-1 at 127 of 128 positions, identical top-5 log-probabilities at 1, the top-1
  log-probability up to 0.22 apart; the branch against the two base runs 126 and 127, 1, 0.32 and 0.39.
- With placement pinned (`fx-tfpin-*`: a fixed 48 GiB tier of 2741 slots, no elastic growth, no adaptive admission;
  eng-m1, eng-bytes, eng-bytes, eng-m1 under one lock; `by-tfbit.py`): base against base identical top-10 at 1 of 128
  positions (the first, the graph prefill of the prompt; the first decode step already differs), the reference
  token's log-probability up to 0.33 apart; branch against branch 1, 0.26; branch against base 1, 0.20-0.38.
- With every timing-dependent admission removed as well (`fx-tfpin2-*`: uploads synchronous, no async and no
  deferred, and no tier in prefill, so no seeds): base against base still identical at 1 of 128, 0.32 apart; branch
  against branch 1, 0.39; branch against base 1-9, 0.30-0.36. Every pair has the same top-1 at all 128 positions, and
  every run reads prefill 64/64, decode 62/64.

So on GB10 the decode path is not bit-reproducible from run to run even where nothing the levers touch varies, and
the branch falls inside that spread. 125's and 233's chains repeat the strictly pinned check and decide it only if
dev agrees with itself there.
Lever a changes no arithmetic by construction: it changes which reads go into RAM slots, and a RAM hit and a drive
read give the same bytes.

**test-kimi-vram-exclusive** fails at its first GPU hit ("chain changed output"): the Q3_K test2 engine on this
GB10 does not reproduce the CPU bit for bit. Dev 2826e92 fails the same way here (`by-ref`), so it is not this
branch's. VRAM_EXCLUSIVE, the only user of the residency ABI before (ABI 2 is now required for every tier), therefore
has no hardware test with rule a yet; 233's chain runs it on a discrete GPU.

## d. The prefetch depth can be 0

**Before.** `depth()` returned `k_win >= 1` even with no idle window, so the framework could never derive "no
prefetch".

**After.** A rank's read pays when the share of it the next layer uses (`p_j`) times the demand time a used read
saves exceeds what it costs: its spill into the next layer's demand reads, plus its own transfer when it starts
after the drive's idle window. The depth is `min(k_pay, k_mem)`: 0 (prefetch off) when the drive has no idle window,
or when the spill outweighs what a read saves. No new knob. The inputs:
- the demand time a read saves: the bootstrap's layer fit, a read expert less a ready one (`[nm] - [nr]`);
- the spill, measured by the engine (after review): each reader tracks the read it runs; when a run's drain closes the
  window, a running read spills unless it is urgent or the run takes its expert, until it completes or a new window
  opens. The report gives `spill_us` (wall time with at least one spilling read), `spills` (the reads that spilled,
  each once after the merge review), `spill_us_r` (each spilling read's own time past the close, by the rank of its
  expert in the hint) and `started_r` (the reads that began, by rank). A rank's price is `spill_us` shared out by
  `spill_us_r`, over its `started_r`; a rank the bootstrap did not read costs the most a read rank does, plus the
  transfer past the window.
- A report without the measure (an engine before it: every stored bootstrap) keeps the first estimate, the readers'
  busy time beyond the windows' total, over the reads. That is a lower bound: zero whenever the readers are busy for
  less than all windows together, however many reads overrun their own layer's window. The receipt says so.

**Stored runs** (`tools/adaptive/tests/test_settings.py`, and `regress.sh` on the stored bootstraps): every value of
every settings.env is unchanged, pending ones included.

| bootstrap | depth | why |
|---|---|---|
| 233c4 (DeepSeek, old engine, no rank statistics) | none | nothing to derive from |
| Spark with Muse (stored spark-run1) | 5 | the lower bound: no spill (readers 44.4 s, windows 104 s); rank 6 starts after the window |
| Spark with DeepSeek (stored spark-dsv1) | 6 | the lower bound: no spill (10.4 s against 59.9 s) |
| Spark ysy9 with DeepSeek (`fw/runs/dsv-ysy9`) | 6 | the lower bound: no spill |
| 125 with DeepSeek (placement all) | 2 | the lower bound: 649 us a read, below what ranks 1 and 2 save (p 0.92 and 0.77 of 1.29 ms); rank 3 starts after the 700 us window |

**On the Spark from the branch's measured spill.** The stored ysy9 bootstrap with its prefetch report replaced by
`fx-ctr-eng-bytes`'s (rule a, depth 6; `bytes-ut/der-fx-eng-bytes`): 1.16 s of spill over 6067 experts read, an
expert's read of ranks 1-6 spilling 20, 61, 158, 180, 244 and 279 us. With 1.11 ms saved a used read (1550 us an
expert's read), ranks 1-5 pay (rank 5: 0.224 x 1.11 ms = 249 us against 244 us) and rank 6 does not (0.179 x 1.11 ms
= 199 us against 279 us): **depth 5**, against the deployed 6. The aggregate estimate had given 0 us and 6.

What the rule predicts, from the same numbers: depth 5 against 6 saves rank 6's net cost, 80 us on 7.6 reads a token,
about 0.6 ms a token (0.4 %, under an ABBA's resolution); depth 5 against 0 (prefetch off) gains ranks 1-5's net
saving, about 5.7 ms a token (4 % of a 129-256 token). Two ABBAs on eng-bytes (the protocol turn, p4096, 2 cycles,
the lock held across each set of 8 arms, `y9-fw-dsv-ysy9-5c2fc47.settings` (2906) with the depth overridden,
`by-fx2.log`):

| ABBA (`by-d65-*`, `by-d50-*`) | A: TTFT, decode 1-128 / 129-256 | B | cycle 1 | cycle 2 |
|---|---|---|---|---|
| depth 6 (A) against 5 (B) | 32.20 s, 6.434 / 7.301 | 32.25 s, 6.405 / 7.271 | +0.26 %, +0.01 % / -0.20 % | +0.10 %, -0.93 % / -0.62 % |
| depth 0 (A) against 5 (B) | 32.26 s, 6.038 / 6.768 | 32.18 s, 6.417 / 7.257 | -0.29 %, +5.71 % / +6.97 % | -0.25 %, +6.84 % / +7.47 % |

- 6 against 5: -0.46 % / -0.41 % over both cycles, the arms overlapping in both windows (A 6.392-6.464 and
  7.290-7.317, B 6.392-6.429 and 7.230-7.295). Time cannot tell the two depths apart, as predicted. The counters moved
  as the per-rank prices said: depth 5 issued 4927-4978 reads against 6737-6874, read 84-85 GB against 111-112 GB a
  run, spilled 0.80-0.84 s against 1.14-1.20 s (-0.37 s; rank 6's price predicted -0.42 s), and lost 340-360 used
  reads (rank 6 had 332). The two cancel: about 1.5 ms a token each way. The deployed DeepSeek depth stays 6
  (PENDING); this ABBA does not accept 5 over it.
- 0 against 5: every depth-5 arm beats every depth-0 arm in both windows (A 5.937-6.086 and 6.747-6.806, B
  6.384-6.448 and 7.208-7.308): +6.3 % / +7.2 %, TTFT -0.27 %. The rule's sign holds; its size is under-predicted
  (10 ms a token measured in 129-256, 5.7 ms predicted).

## R1. The RAM cache's storage at load

**Before.** The RAM cache learned its geometry at the first decode token, mapped its arena then, and committed each
slot at its first admission. On 125 the first touch of 28.4 GB of new slots cost 13.8 ms a token over decode tokens
1-128, 1.77 s a cold start.

**After.** Off unified memory the load-time hook (`kimi_cpu_sink_vram_reserve` -> `reserve_tier`, warm2's) also gives
the RAM cache, when it is kept across requests:
- its geometry from the expert tensors' strides, each role as a decode request reserves it (its bytes and a page
  prefix, in whole pages), so the first decode token finds the same geometry and keeps it; the frequency nodes'
  guard priced for the model's experts a token (`n_expert_used`, 6 here), as the first decode layer prices it (the
  first version used 16, two slots fewer on 125);
- with an automatic budget, the read ring first, as the first decode token charges it before sizing the cache;
- its arena, and the slots the ledger's guard admits now committed through the same accounting as an admission and
  paged in by `MADV_POPULATE_WRITE` (else a write a page) from new threads, as many as the sink's compute threads
  plus one (not the sink's pinned pool); their storage is reserved first, so a failure to start one leaves its part
  to the calling thread; 4 KiB pages stay, so partial reclaim still gives single slots back.
- llama calls the hook whenever the sink has a RAM cache or a GPU tier (before: only with a VRAM budget, so a machine
  without a tier kept paying the first touches). After the merge review the sink takes the load step only for a
  cache that can give room back to the first prefill's ledger requests: partial reclaim, or elastic.

The first prefill may give slots back through the ledger (context checkpoints, compute buffers), as every later
request's prefill does. Without elastic, a given-back slot stays disabled for good, while a cache sized at the first
decode token would size itself after those allocations. So the first decode token after the load step takes back the
given-back slots as far as the room allows then (`load_regrow`): with an automatic budget the surplus less the
metadata room geometry keeps, over geometry's price of a slot; with a fixed budget all of them. After the merge review
the enabled slots the load step did not commit (owed: the room it read was smaller than the room geometry sized from)
take their room first. They are paged in at their admission, as every slot was without the load step. A later
request's give-back stays, as before. What the first prefill keeps (on 125 and 233 the server grew by about 0.4 GiB
during it, 299 MiB of that the ledger's grants for checkpoints) does not come back, and a cache sized at the first
decode token would not have it either.

Receipts: `[kimi_cpu_sink_cache] reason=load slots of slot_bytes committed_bytes us threads pages`, and
`load_regrow=1 before_slots after_slots of owed_slots surplus_bytes committed_bytes` (owed_slots after the merge
review). The framework's receipt (insitu.json,
insitu.txt) carries the load step, the new slots decode committed in tokens 1-128 and after, and the server's load
time (`arm.sh` writes `load_s`: from the launch to the health check). On unified memory nothing changes: the RAM cache
there only lands the prefetch, out of the demand warm2's tier reserve leaves at load, and takes its pages at the
first decode token.

**Unit test** (`test-kimi-cpu-sink --cache-load-tests`, the branch at `0cc0fcb`, `by-fx1.log`): pass. Against a sink
without the load step: the same capacity (12 slots of 98304 bytes), the same hits and misses, bit-identical outputs;
no new slot in decode (the reference commits 6, 589824 bytes); the arena kept through a prefill and an epoch it never
saw. An elastic prefill's shrink gives 8 load-time slots back, and decode admissions refault them. New: with partial
reclaim and no elastic, under a fixed and under an automatic budget, an allocation larger than the cgroup during the
first prefill takes all 12 load-time slots back through the ledger; the first decode enables all 12 again
(`load_regrow=1 before_slots=0 after_slots=12 of=12`) and pages them in at admission (3 slots admitted, 3 paged in);
a later prefill's give-back stays given back.

**On the Spark (unified memory): nothing changes.** The pre-review pair (`by-r1-p4096-*`, rule a against rule a +
d + R1): identical launcher lines, no `reason=load` line, the same RAM cache receipt (240 slots) and ledger, the tier
grown at load alike, memfloor margins 19-44 MiB, times within the 3 % noise. The ABBA above runs the branch with R1
in it.

R1 has not run where it acts. On 125 and 233 its gate checks capacity before time (below).

## Pending on 125 and 233

Both machines were busy. The chains are written and not run: `bytes-ut/by-125.sh` and `bytes-ut/by-233.sh` in the
workflow's scratchpad, with `bytes-ut/by-srcA.sh` for their sources. After the merge review they were regenerated
for the branch bytes-minors (the merge 151c871 and the review's fixes, below); the versions for bytes 74c2483 are in
`bytes-ut/old-74c2483`. Each builds three trees from `git archive`: eng-d391 (dev 391c905, the reference: edge-b's
runtime code is not reached for DeepSeek or Muse, see the merge check below), eng-mn-a (the branch with R1's engine
commits reversed, newest first: the review's R1 fixes `079198e fec1a7c`, then `0cc0fcb 2dc88a1 76983a7 5c3c06a`;
lever a with its fixes) and eng-mn (the branch), checks each one's COMMIT before every arm, holds the box's lock once
over every arm of an ABBA, runs the unit tests (now with the prefetch's spill cases, and test-kimi-vram-group, whose
device-wide memory checks apply on these discrete GPUs) and every kimi CTest, and samples the server scope's
`memory.current`, `memory.peak` and `memory.events` every second. `by-eval.py`
reports per arm the per-token medians of 1-128 and 129-256, the stall tokens (more than 100 ms over their window's
median: the ~210 ms single-read stalls show as such tokens, and they are larger than the effects tested), the RAM
cache's slots at load, at the first decode and at destroy, its shrinks, rejections and pauses, the ledger's
refusals, and memory.events. Their run labels carry `mn-`; `by-eval.py` also reads load_regrow's owed slots.

| lever | 125 (RTX 5090), predicted | 233 (RTX 5060 Ti), predicted | gate |
|---|---|---|---|
| a | the residual 25.7 % -> ~0; used/issued 0.60 -> ~0.8; misses a token in 129-256 48.99 -> 44.7-46.0; tier hits within 1 %; drive bytes -0.098 / -0.069 GB a token; decode +2 to +6 % (1-128), +3 to +5.5 % (129-256) | DeepSeek: the residual 3.6 % -> 0, bytes -1.3 to -1.9 % / -0.6 to -0.8 %, below an ABBA's resolution: gated on counters, timed for no regression only; Muse: neutral | counters (dev / a / branch), then ABBA dev / a, 2 cycles, p4096 and p512 (233: p4096), decided on the medians with the stall tokens beside them |
| d | the stored bootstrap keeps 2 (the lower bound, 649 us a read); re-derived from the branch's measured spill on `fw/runs/dsv-125` | DeepSeek: re-derived on `fw/runs/dsv1017-233a`; Muse: no bootstrap on 233 yet, one must come first | an ABBA of the deployed and the derived depth, 2 cycles, only where they differ |
| R1 | cache_new_slot_bytes in 1-128 28.4 GB -> the regrown slots only (a few, under 0.2 GB); RAM slots at the first decode within 2 of a's; decode 1-128 +7 to +10 % (13.8-15.6 ms a token); load time up by the page-in (about 4.3 s of CPU over the threads); TTFT not worse; memory.events max/oom 0 | DeepSeek: 0.20 s a cold start; Muse: none visible | the reason=load and load_regrow receipts and the slot counts first, then ABBA a / branch, 2 cycles |
| TF | bit for bit with placement pinned and every timing-dependent admission off, if dev agrees with itself there | the same, DeepSeek | `by-tfbit.py`: identical top-10 at every position |

233's chain also runs test-kimi-vram-exclusive on a discrete GPU, the gate for VRAM_EXCLUSIVE with rule a.

## Merging with unified-placement

`git merge-tree` of this branch with unified-placement (`18bc4ec`): the only conflicts are the two in
`vram-tier.cpp`'s destroy report and stream teardown, byte for byte the ones dev 2826e92 (warm2) has with
unified-placement; none comes from this branch. The merged tree's tools/adaptive tests pass (74). Where the two meet:
- **R1's arena and the decode registrar.** The registrar page-locks RAM cache slots once the arena exists and unlocks
  them in `before_discard`, which `release()` and `shrink_to` call. With R1 the arena exists from load, so the
  registrar can start at the first decode plan; the load step maps a fresh arena (nothing registered yet) and its
  geometry's `release()` runs before any mapping. `regrow_adopted` only enables slots, so it discards nothing.
  Page-locking every slot at load instead of in the drive's shadow is possible on top of R1, not done here.
- **The host jobs and `held`.** On a discrete GPU a host job's copy into the expert's admission reservation is
  published at finish through `vram_slot_table::publish`, which keeps the held count; a host job into a scratch slot
  is not an admission and holds nothing.
- **ABIs.** Residency ABI 2 (`held`) and host ABI 2 (`register_host`, `costs2`) are separate structs and exports.
- **The load hook's signature** (`experts_used`) and llama's call site are files unified-placement does not touch.

## The merge with dev on ysy9 (151c871, M3)

On 10-22 (08:29-09:12 CST) ysy9 built and checked the merge 151c871. It joins dev 391c905 (edge, warm-up and edge-b)
and the bytes branch 74c2483 (levers a and d, R1). The question was whether the merge breaks anything on the Spark with
DeepSeek-V4.1-Flash-1017 Q8_0, and what edge-b's tests show on GB10. The usual rules held: one lock
(`~/ssdl/spark.lock`) for every build, test and server; `~/ssdl/STOP` checked before each step and each arm; one
server at a time; the 3 GiB watchdog, which never fired. The reference is `eng-bytes` (0cc0fcb = d51c549 +
by-fx1.patch; its tree differs from 74c2483 only in tools/adaptive). The merge is `eng-m3`, built from
`git archive 151c871` (the tar's md5 matched on both sides). Scripts: `~/ssdl/ab/m3-chain.sh`, `m3-after.sh`,
`m3-arms.py`.

**Build.** `LIBURING_PREFIX=~/ssdl/deps BUILD_TESTS=1 tools/adaptive/build.sh`: `BUILD_OK` in 3 min 17 s, liburing
static, architectures 121a, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16+bf16`. The log ends with
`build.sh: liburing ~/ssdl/deps/lib/liburing.a`. No errors. 252 warnings against eng-bytes' 213, of the
same kinds (missing declarations, missing field initializers, -Waddress, unused variables); the extra ones are in
edge-b's `moe_stream_mm.cu` and the two test targets that `BUILD_TESTS=1` now builds (test-kimi-vram-mm, -group).

**Unit tests** (each in a user scope of 8 GiB with no swap, under the lock; outputs
`~/ssdl/eng-m3/build-cuda130/bin/m3-*.out`):

| test | result |
|---|---|
| test-kimi-vram-pool | PASS, 32309 assertions (eng-bytes: 32309) |
| test-kimi-prefetch-tier | PASS: skip_vram 1, vram_late 1, skip_vram_lost 1, issued 1, activated 0, novel_r 1,0,1,0, started_r 1,0,0,0, spills 0 (as eng-bytes) |
| test-kimi-cpu-sink; `--prefetch-tests`; `--cache-load-tests`; `--elastic-cache-tests` | PASS (CPU_SINK_TEST_OK; CPU_SINK_PREFETCH_OK; CACHE_LOAD_OK, 12 of 12 slots, pages populated; ELASTIC_CACHE_OK) |
| test-kimi-vram-elastic / -reclaim / -deferred | PASS: 127 assertions unified=1; 54 unified=1; 31, uploads 5, published 3 |
| test-kimi-vram-q2 | PASS, 140 checks, worst rel L2 0.0157312 |
| test-kimi-dense-stream / test-server-text-reuse | PASS (stage 327680) / PASS |
| test-kimi-vram-exclusive | rc=1, `REAL_GPU_EXCLUSIVE_FAIL: exclusive/async/hidden chain changed output`. All 19 lines equal dev 2826e92's (`~/ssdl/by-ref/.../by-ut-exclusive.out`) and eng-bytes' apart from the scope name and times. |
| test-kimi-vram-mxfp4 | rc=1, `check=up_gate_c512/scales input=0 row=0 cpu=0x1.242ab8p+12 gpu=0x1.242ab6p+12`. All 27 lines equal eng-m1's (dev 2826e92) apart from the scope name. |
| every kimi CTest (`ctest -R "kimi\|dsv41"`, 135) | 123 pass, 12 fail: the same 12 tests and the same status for all 135 as eng-bytes (`by-fx1-ctest.out`). Each of the 12 `kimi_cpu_sink_schedule_t*` / `kimi_cpu_sink_expert_job_t*` prints `max_abs=0.024879`, as dev 2826e92 does (`by-ctref.out`). (`m3-ctest2.out`. The first pass, `m3-ctest.out`, had 19 "Not Run" because the chain had not built their executables.) |
| test-kimi-vram-group (edge-b, numerics) | rc=1 on GB10, at its memory check, not at its numerics (below) |

Compared with eng-bytes' outputs of the same tests, line counts are equal. After times are masked, what still differs
is memory sizes, temp file names, and in -q2 how the host lane batched the same 4 jobs (1 batch against 2).

**test-kimi-vram-group on unified memory.** All three runs fail at case 2 (t1-norm) with "release did not give back
what reserve took". The device's free memory after release is 458752 bytes (448 KiB) below its level before
reserve, the same number in every run. The test compares `cudaMemGetInfo` free sizes exactly. On GB10 that is the
system's memory, and it counts host pages too. The numbers it has before failing match 233's: case t1, engine
1.31e-4 and CPU lane 3.75e-4 against the exact layer.
- It is not lazy kernel loading. With `CUDA_MODULE_LOADING=EAGER`, case 1 keeps 480 KiB instead of 35-39 MiB, and case
  2 still keeps 448 KiB.
- A shim logged the real free size and returned a fixed one to the test (`m3-group-memlog.out`, EAGER). In Muse's
  fixture, cases 2-15 each keep 448 KiB from before reserve to after release. t33-host-norm keeps 496 KiB,
  t300-host-clamp keeps 0, and both hidden-768 cases keep 480 KiB. In the rows-1408 fixture the free size moves by
  -48 MiB to +1.39 GB inside a case. That is the system's free memory changing under the test. Between cases, free
  memory moves by tens of MiB with the test's own allocations. So this trace cannot show whether the 448 KiB adds up
  over cases.
- The engine's own count is 0 after every release (`require(device_bytes == 0)` passes). Its release frees every
  buffer that reserve allocates (`group-compute-adapter.cu:217-232, 290-295`; B1's context release,
  `moe_stream_mm.cu:76-82`, frees its scratch and replaces its CUDA context).
- With the shim all 22 cases pass: `KIMI_VRAM_GROUP_PASS checks=88 worst_token_nmse_against_exact=0.000314`. That
  includes the two cases that read the experts in place from pageable host memory (t33-host-norm: engine 1.73e-4, CPU
  3.91e-4; t300-host-clamp: 1.61e-4, 2.99e-4). The engine is at most 2.2e-4 from the exact layer outside the split
  cases, and 2.39e-4 and 3.14e-4 in them. The CPU lane is 1.68e-4 to 5.08e-4. 233 gave 3.15e-4.

So the numerics of B2 hold on GB10. The test's exact free-memory equality does not fit unified memory, where
`cudaMemGetInfo` counts the whole system. Where the 448 KiB comes from is not found. It is the same number in every
run, and lazy kernel loading does not explain it. Candidates are per-case CUDA resources that the driver holds in host
memory, such as the test's own upload stream, which copies inside the window and is destroyed after it. It is edge-b's
test and edge-b's engine; the bytes merge does not touch either file. This needs a fix in the test (for example,
compare the engine's own counters on unified memory) or a found cause.

**B2 lane timing on GB10** (`~/ssdl/ab/m3-group-time.txt`). These are the defaults: Muse's shape (latent 4096, hidden
4096, Q4_K/Q5_K, B_e = 30408704 bytes), 24 experts a layer, the CPU lane on 19 participants plus the I/O thread. The
second run adds `--drive-gbps 12.14`. That rate is ysy9's drive read rate from the framework's bootstrap:
`~/ssdl/fw/runs/dsv-ysy9/probe/device.json`, field `ssd[ssd_knee_qd]` with `ssd_knee_qd` = 3, `gibs` 11.307 GiB/s
(runs 11.279, 11.335) = 12.14 GB/s. The framework's prefill model reads the same field (`tools/adaptive/prefill.py:83`).
Microseconds per expert, fixed + per route, and the value at 150 routes, run 1 / run 2:

| lane | idle | busy |
|---|---|---|
| GPU, weights in device memory | 173.3 + 2.275 (514.5) / 184.2 + 2.022 (487.5) | 226.7 + 2.372 (582.4) / 212.0 + 2.527 (591.0) |
| GPU, in place from pageable host memory | 179.4 + 2.009 (480.7) / 180.4 + 2.017 (483.0) | 180.4 + 2.060 (489.4) / 173.6 + 2.186 (501.6) |
| CPU lane | 1184.9 + 102.4 (16544) / 1617.4 + 100.7 (16722) | 1266.3 + 116.8 (18787) / 1045.9 + 118.1 (18755) |
| one expert's copy from page-locked memory | 515.2 us, 59.0 GB/s / 516.6 us, 58.9 GB/s | 514.7 us / 517.0 us |

- The points behind the GPU lines: 8, 26, 150 and 300 routes take 230, 237, 432 and 896 us from device memory, idle,
  in run 1. In place it is 212, 237, 439 and 802 us.
- Gate (run 2): bound 0.8 x B_e / R = 2003.9 us. Device memory busy 591.0 us: pass, 3.4x margin. In place busy
  501.6 us: pass, 4.0x margin. The copy takes 517.0 us. **In place is chosen**
  (`in_place=1`): 501.6 us is no more than the longer of the copy and the device chain (591.0 us).
- Against edge-b's predictions for the Spark: device memory 0.3-0.5 ms predicted, 0.49 idle and 0.58-0.59 busy
  measured. In place 0.35-0.65 ms predicted, 0.48-0.50 measured. Copy 0.58 ms predicted, 0.52 measured. In place
  adopted, as predicted. The CPU lane takes 101-118 us a route, 31-38x the GPU's time at 150 routes.

**Does edge-b change DeepSeek on the Spark? No.** Nothing of B1 or B2 runs for DeepSeek's experts (MXFP4) or for
anything else.
- Code (151c871). The B2 engine's only entry, `kimi_vram_group_compute_get_api`
  (`ggml/src/ggml-cuda/kimi-vram-tier/group-compute-adapter.cu:544`), has no caller outside
  `tests/test-kimi-vram-group.cpp` (git grep over ggml, src, common, tools). B1's `ggml_cuda_moe_stream_mm_*`
  (`moe_stream_mm.cu:193-397`) are reached only from the B2 engine (`group-compute-adapter.cu:140-148`, by dlsym)
  and from `tests/test-kimi-vram-mm.cpp`. `ggml_cuda_moe_stream_mmq` (`mmq.cu:251`) and `ggml_cuda_moe_stream_mmvq`
  (`mmvq.cu:1363`) are called only at `moe_stream_mm.cu:357,364`, and the quantized-sum quantizers only at
  `moe_stream_mm.cu:318,323`. The tier picks DeepSeek's expert engine at `vram-tier.cpp:1425-1440`: MXFP4 up and down
  take `kimi_vram_format_compute_get_api` (`mxfp4-compute-adapter.cpp:433`) and any other format takes the Q2 engine.
  The grouped engine is not among them.
- What edge-b did change on DeepSeek's path is arithmetic-neutral. The default quantizers (`quantize.cu:482`,
  `:527`) now instantiate a template with `quantized_sums=false`. They sum the same float values in the same reduction
  (`quantize.cu:214`, `:414`), only later in the kernel. This applies to the graph's dense Q8_0 products; MXFP4 on
  Blackwell quantizes with `quantize_mmq_mxfp4_cuda`, which is unchanged. The shared `kimi_vram_activate` serves the Q2
  engine only.
- Logs. In all 8 p4096 arms (4 of them eng-m3), the server logs
  `[kimi_vram_bridge] format=mxfp4 latent=5120 hidden=2304` and `[kimi_vram_mxfp4_batch] status=ready`. The model
  loads q8_0 332, bf16 65 and mxfp4 120 tensors. There are 0 lines with `kimi_vram_group`; the engine prints
  `[kimi_vram_group_engine] status=reserved` at every reserve (`group-compute-adapter.cu:263`). There are 0 lines with
  `moe_stream_mm`. `libkimi-vram-tier.so` exports the symbol, but nothing calls it.

**Regression: n2 at p4096** (`by-m3-p4096-*`, `~/ssdl/ab/m3-p4096.json`, `m3-arms.txt`). The protocol is the n2
protocol of the bytes workflow: two ABBA cycles, the lock held over all 8 arms, request 1 and then a new
conversation (`work-4turn.json` entry 2, 3420 tokens), 256 tokens each with EOS ignored. A is eng-bytes, B is
eng-m3. The settings file is `y9-fw-dsv-ysy9-derived.settings` (threshold 414). The bytes run `by-fx-p4096` itself
used `y9-fw-dsv-ysy9-5c2fc47.settings` (threshold 2906; `by-abba-fx.log` line 27). The two files differ only in the
threshold. Both send the same passes the same way: 2 tokens to the sink, and 4104 and 3420 tokens to the graph
(`[kimi_prefill_route]`).

| arm | request 1: TTFT, decode 1-128 / 129-256 | request 2 | issued | skip_vram | seeds planned/published | floor margin (MiB) | reply parts from arm 1 at char (req 1, req 2) |
|---|---|---|---|---|---|---|---|
| a-1 eng-bytes | 32.267, 5.984 / 7.328 | 28.942, 7.086 / 7.364 | 13115 | 104792 | 5216 / 5216 | -314 (1 sample) | - , - |
| b-2 eng-m3 | 32.135, 6.378 / 7.235 | 28.856, 6.989 / 7.436 | 13161 | 104627 | 5200 / 5200 | +9 | 1111, 123 |
| b-3 eng-m3 | 32.211, 6.328 / 7.101 | 28.901, 6.852 / 7.324 | 13607 | 104355 | 5200 / 5200 | -426 (1 sample) | 591, 222 |
| a-4 eng-bytes | 32.263, 6.395 / 7.326 | 28.886, 7.047 / 7.306 | 13234 | 104646 | 5200 / 5200 | +41 | -, - |
| a-5 eng-bytes | 32.250, 6.354 / 7.288 | 28.938, 7.054 / 7.375 | 13066 | 104766 | 5184 / 5184 | +38 | 973, 222 |
| b-6 eng-m3 | 32.234, 6.399 / 7.317 | 28.982, 7.103 / 7.297 | 13236 | 104706 | 5200 / 5200 | +31 | -, 123 |
| b-7 eng-m3 | 32.183, 6.378 / 7.274 | 28.928, 7.043 / 7.273 | 13259 | 104551 | 5216 / 5216 | +11 | 706, - |
| a-8 eng-bytes | 32.276, 6.359 / 7.263 | 28.911, 7.075 / 7.324 | 13355 | 104539 | 5216 / 5216 | +27 | -, - |

| B against A | request 1: TTFT, 1-128 / 129-256 | request 2 |
|---|---|---|
| cycle 1 | -0.29 %, +2.64 % / -2.17 % | -0.12 %, -2.07 % / +0.61 % |
| cycle 2 | -0.17 %, +0.50 % / +0.27 % | +0.11 %, +0.12 % / -0.88 % |
| all | -0.23 %, +1.56 % / -0.95 % | -0.01 %, -0.97 % / -0.13 % |

- vram_late and skip_vram_lost were 0 in every arm. No seeds were cancelled or freed unused. The reads' spill took
  2.26-2.38 s per server, against 2.28-2.36 s in bytes' own B arms. Load took 40-42 s, and decode_grow ran 3 times in
  every arm.
- The bar was bytes' own B arms: 6.385-6.473 / 7.269-7.307 in request 1. eng-m3 read 6.328-6.399 / 7.101-7.317, so 3
  of 4 arms are under it in 1-128 (by at most 0.057, 0.9 %) and 2 of 4 in 129-256. But eng-bytes itself, rerun in
  the same ABBA, also misses that 1-128 range in 3 of 4 arms (5.984, 6.354, 6.359). Today's session sits a little
  lower than the bytes session. Inside this ABBA the two builds overlap: 1-128 is 6.354-6.395 for eng-bytes (without
  a-1) and 6.328-6.399 for eng-m3.
- The two low 129-256 values follow the text, as in M1. b-3's reply parts from the others at character 591 (about
  token 121), so all of 129-256 decodes another text: 7.101. b-2 parts at 1111 (about token 228): 7.235. The one
  eng-m3 arm with the common text, b-6, reads 6.399 / 7.317. The eng-bytes arms with that text read 6.395 / 7.326,
  6.359 / 7.263 and 5.984 / 7.328. Parting points 591, 706 and 973 are known near ties from M1; 973 and 222 also occur
  in eng-bytes here.
- a-1 (eng-bytes) read 5.984 in request 1's 1-128, the slowest of all 8 arms. It started 1 s after the B2 timing runs
  (GPU and 20 CPU threads at load), and its request 2 is normal (7.086 / 7.364). The cause is not established.
- Floor (4 Hz): one arm of each build had one sample under the published floor. a-1 was 314 MiB under it 119 s after
  the start, b-3 426 MiB under it 149 s after the start. Both were in the second request; M1 saw the same thing at
  about 148 s. The others stayed 9-41 MiB above it. The lowest 4 Hz MemAvailable was 4124 MiB, about 1.0 GiB above
  the 3 GiB watchdog. This is the warm-up's open item again, in both builds.

**Correctness through decode** (`m3-tfd-eng-m3`, `probe-tf-decode.py`, threshold 414, 64 requests): decode 62/64
and prefill 64/64. The decode misses are base's two: position 50 (top 1 ' from' -0.582 against ',' -1.235) and 82
(' retains' -0.441 against ' keeps' -1.032). The log shows that decode ran and used the tier: `decode_active=1` 64
times, tier hits 21995 and uploads 4256 (eng-m1 in M1: 21966-22224 and 4282-4297). Seeds: 5184 planned and
published, 4986 freed unused. The 524-token prompt went to the graph (`n_tokens=524 route=graph max_sink_tokens=414`),
and the 63 one-token passes went to the sink. Prefetch: issued 4587, skip_vram 10007, vram_late 0, skip_vram_lost 0.

**Verdict.** The merge breaks nothing these checks can see on the Spark:
- The build is clean.
- Every unit test that passes on eng-bytes passes. The failing CTests, test-kimi-vram-exclusive and
  test-kimi-vram-mxfp4 fail with dev's exact text and values.
- edge-b's code does not run for DeepSeek.
- The ABBA shows no regression: cycle 2 is +0.50 % / +0.27 %, and the cycle-1 differences follow one slow reference
  arm and the reply text.
- Teacher-forced decode is 62/64 with base's two misses.

These were edge-b's first runs on GB10:
- The B2 numerics pass in all 22 cases, in place included, with a worst NMSE of 3.14e-4. This needed a shim that
  neutralizes the free-memory check.
- The lanes are timed. In place is chosen, and it is 4.0x under the bound.
- The unit test fails on unified memory at an exact free-memory check. The gap is 448 KiB and deterministic; whether
  it adds up over cases is not shown. It needs a fix in the test or a found cause. (After the merge review: the
  cause is the driver's memory for newly mapped address ranges, and the test checks the engine's own count there;
  see the next section.)

Not covered here: discrete GPUs (125, 233), Muse, drive homes.

## After the merge review (bytes-minors)

The review of the merge 151c871 found no blockers and six minor points. The branch bytes-minors fixes each in a commit
of its own, with a unit test where the behaviour can be tested. A download was writing to ysy9's drive, so nothing ran
on a server and nothing was timed. The build is `~/ssdl/eng-minors` (the code of `09d0a4a`; this document's commit
changes only this file): `build.sh` with BUILD_TESTS=1, BUILD_OK in 3 min 30 s. Scripts and logs are in
`~/ssdl/minors-diag/` (`minors-chain.sh` and its log; the outputs `~/ssdl/eng-minors/build-cuda130/bin/mn-*.out`).

| | commit | change | unit test on ysy9 |
|---|---|---|---|
| 1 | `6e94d58` | `spills` counted spill events: a read that spilled, stopped at the next window's open and still ran at the next close counted twice. A flag per reader, reset when it picks up a read, counts a read once. `spill_us` and `spill_us_r` still add every interval. | test-kimi-cpu-sink `--prefetch-tests`: the test's own `pread` holds the three role reads of an expert hinted for layer 2 and not selected there. They spill at layer 2's close, stop when that run's own read ends, and spill again at the next token's drain: spills=3 in each of 3 runs (`started_r` 1,1; `spill_us` 300.6-300.9 ms, the drain's wait) |
| 2 | `1154d5f` | The last MoE layer's run has no hint. With no demand reads it returned before `prefetch_open_gate`, so the reads kept spilling across the token boundary. It now opens the window before it looks for a hint. The periodic report adds the spill interval still open. | test-kimi-cpu-sink `--prefetch-tests`: layer 2's run takes only the prefetched expert (no reads, no hint); the reads are held 300 ms after it: spills=3 and `spill_us` 13-55 us in 3 runs (`spill_us_r` 0,39 to 0,165) |
| 3 | `fec1a7c` | `regrow_adopted` priced only the given-back slots. The enabled slots the load step did not commit (the room it read was smaller than geometry's) need room at admission too; they now take it first (`owed_slots` in the receipt). | test-kimi-cpu-sink `--cache-load-tests`: 51 MiB experts, an automatic budget; the test's `pread` makes the process 1.5 slots larger after the ledger's first reading: the load step commits 5 of 6 slots; the first prefill gives 2 resident slots back; the first decode sees room for 2 slots past the metadata (surplus 134.2-134.4 MB, a slot 53.5 MB) and 1 owed: `before_slots=4 owed_slots=1 after_slots=5`, in 3 runs |
| 4 | `392c7eb` | Comments: a seed counts from its publication, which comes at `phase_complete` for a prefill's planned seeds and right after its copy for Q2's cold-prefill admission. | - |
| 5 | `079198e`, `cb4239d` | Without a GPU tier the sink now exists from load, so the first prefill's ledger requests go through it. An arena committed at load without partial reclaim cannot give room back to them. The load step runs only for a cache that can (partial reclaim, or elastic). `cb4239d` (after review): the read ring an automatic budget made resident at load first waits for the first decode too. | test-kimi-cpu-sink `--cache-load-tests`: without partial reclaim and elastic, no `reason=load` line and nothing mapped at load, and the first decode pages its 2 admissions in; elastic alone takes the load step (12 slots) and its prefill keeps 4; an automatic budget with 51 MiB experts: no receipt, nothing mapped, resident memory grows by less than half a slot over the load hook (`case=load_gate_ring`) |
| 6 | `09d0a4a` | test-kimi-vram-group checks the engine's own count, and the device-wide free size only where it counts the device's own memory (below). | test-kimi-vram-group on GB10: PASS in 2 runs (88 checks, worst NMSE 3.14e-4, as with the shim); `device_free_checked=0`, and 12 of the 21 printed cases keep 458752 bytes after release |

**Negative controls.** The branch's test binary run against the merge's libraries (`~/ssdl/eng-m3`, the same C API;
`minors-control.sh`): the spill case reads `spills=6` (the branch: 3) and fails; the owed case's first decode takes
both given-back slots back (`before_slots=4 after_slots=6`, the branch: 5) and fails. Each control stops at its first
failing case, so the last-layer case and the load gate's case have no control run; by its code the merge's engine
opens no window in a last-layer run without reads (the reads spill until they complete, here the 300 ms hold), and
commits at load without partial reclaim.

**test-kimi-vram-group on GB10: where the 448 KiB goes.** On GB10 `cudaMemGetInfo`'s free size is the system's free
memory (it follows MemFree). A shim that logged every call with the process's and the system's counters
(`memlog3.c`) found the 448 KiB in none of them: not in RSS, anon or the glibc heap, not in Slab, PageTables,
VmallocUsed, KernelStack or Shmem. The CUDA stack limit stayed 1024 bytes. Not every case keeps it: some keep 0, and
a later case's pass sometimes gives 448 KiB back. A standalone probe (`gb10-free.cu`, `gb10-free2.cu`) then showed:
- a 4 MiB cudaMalloc/cudaFree keeps 448 KiB on its first cycle only, and 0 over the next 7 (0 over all 8 in a
  second run);
- with a 3 MiB buffer held before each cycle, so that every 6 MiB buffer lands on an address range not mapped
  before, 6 of 8 cycles keep 416-928 KiB, in each of two runs;
- 400 cycles of random sizes (1-64 MiB) do not add up: -78.9 and -12.6 MiB at the end of two runs;
- streams, 16 events, pinned host memory and VMM mappings keep 0 in 7 or 8 of 8 cycles (one events cycle 444 KiB,
  while the download ran); cuBLAS's first handle keeps 262 MB (its kernels), and a kernel with a large stack frame
  grows the stack limit to 8192 and keeps 529 MB, both once.
So the driver keeps memory for an address range a buffer mapped, after cudaFree, and the test's own allocations
between cases (resident experts, the arena, pinned sources) push the engine's buffers onto new ranges. Other
processes and the page cache move the same number: during the download it moved by tens of MB within 500 ms. The
rule now (the test's header comment): the engine must give back what it took, so that is checked from what it owns,
`device_bytes` (no growth after reserve, 0 after release; it already was). The device-wide checks apply only where
the free size counts the device's own memory, a discrete GPU the test owns (`cudaDevAttrIntegrated` 0). On an
integrated GPU they are printed (`device_free_checked=0`), not checked. 125's and 233's chains run it with the checks
on.

| unit test on ysy9 (`09d0a4a`) | result |
|---|---|
| test-kimi-vram-pool | PASS, 32309 assertions |
| test-kimi-prefetch-tier | PASS: skip_vram 1, vram_late 1, skip_vram_lost 1, issued 1, spills 0 |
| test-kimi-cpu-sink (default) | CPU_SINK_TEST_OK |
| test-kimi-cpu-sink `--prefetch-tests`, 3 runs | CPU_SINK_PREFETCH_OK in each (the cases above) |
| test-kimi-cpu-sink `--cache-load-tests`, 3 runs | CACHE_LOAD_OK in each: the existing cases (12 of 12 slots at load, pages populated), the owed slots and the load gate (above) |
| test-kimi-cpu-sink `--elastic-cache-tests` | ELASTIC_CACHE_OK |
| test-kimi-vram-elastic / -reclaim / -deferred | PASS: 127 assertions unified=1; 54 unified=1; 31, uploads 5, published 3 |
| test-kimi-vram-q2 | PASS, 140 checks, worst rel L2 0.0157312 |
| test-kimi-vram-group, 2 runs | PASS (rc=1 on the merge) |
| every kimi CTest (`ctest -R "kimi\|dsv41"`, 135, `mn-ctest.out`) | 123 pass; 12 fail, the `kimi_cpu_sink_schedule_t*` and `kimi_cpu_sink_expert_job_t*` with max_abs=0.024879, as on dev and the merge. kimi_cpu_sink_prefetch, kimi_cpu_sink_cache_load and kimi_dense_stream pass |
| tools/adaptive (`pytest`, on the VM) | 72 passed |

**Server-level checks once the download is done** (they need a server; none is timed for a decision except the
ABBA):
- ysy9 (unified memory; fixes 3 and 5 do not act there): one traced p4096 arm each of eng-m3 and eng-minors at the
  derived settings (414), under the lock. The prefetch report: `spills` at most the role reads started; `spill_us`
  lower than eng-m3's (2.26-2.38 s a server in M3) by the last MoE layer's open window; issued, skip_vram and RAM
  hits within the arms' spread. The periodic report, which a 256-token turn reaches at about token 210: its
  `spill_us`, the open interval included, at most the destroy report's. Then the depth re-derived from
  eng-minors' spill (`by-derive.py` on `fw/runs/dsv-ysy9`), and an ABBA of the deployed and the derived depth (2
  cycles) only where they differ. No teacher-forced run: no arithmetic changed.
- 125 and 233 (discrete GPUs; fixes 3 and 5 act there): the regenerated chains (`bytes-ut/by-125.sh`,
  `by-233.sh`, above). Their unit tests run test-kimi-vram-group with the device-wide checks on and the owed slots'
  case in their scopes (8 and 4 GiB). Their counters read `reason=load` and `load_regrow` with `owed_slots`, and
  the RAM slots at the first decode. Then ABBA a and R1 (2 cycles), the pinned teacher-forced check, and d.
- The deployed launch sets CACHE_PARTIAL_RECLAIM=1 (`run-ssd-llama.sh`'s profile), so the load step still runs there.
  A launch without partial reclaim or elastic now sizes the RAM cache at the first decode token: no `reason=load`
  line.
