# The adaptive framework from zero on the Spark (2026-10-21)

The framework ran from nothing on the Spark with Muse Spark Open UD-Q4_K_XL: the device probe, one bootstrap,
in-situ costs, the plan and `settings.py`. Its `settings.env` was then set against the best manual configuration,
`models/examples/muse-spark-gb10.settings`, in an ABBA of two cycles at the 512 and 4096 protocol prompts. The
framework also served the protocol's 4-turn conversation (its `validate` step). DeepSeek-V4.1-Flash-1017 was
measured by hand on both routes; the framework's own run on it was queued and could not be read (see
[DeepSeek-V4.1-Flash-1017](#deepseek-v41-flash-1017)). That afternoon a second Spark, ysy9, ran DeepSeek from zero
with the code after the review, its ABBA against the manual best, the bare launch under stress and the warm-up
measurement (see [Spark ysy9](#spark-ysy9)). In the evening ysy9 ran the warm-up itself, a fix for the floor during a
prefill and the server's reuse of reply tokens (see
[Warm-up, the floor in a prefill and reply tokens on ysy9](#warm-up-the-floor-in-a-prefill-and-reply-tokens-on-ysy9)).

## Summary

| prompt | manual best: TTFT, decode 1-128 / 129-256 | framework from zero | difference | gate |
|---|---|---|---|---|
| 512 (669 tokens) | 38.57 s, 1.735 / 2.099 tok/s | 38.59 s, 1.727 / 2.109 tok/s | +0.1 %, -0.5 % / +0.5 % | pass |
| 4096 (4221 tokens) | 50.87 s, 1.802 / 2.137 tok/s | 50.31 s, 1.797 / 2.139 tok/s | -1.1 %, -0.3 % / +0.1 % | pass |

Means of four arms each (A B B A A B B A). The gate is TTFT within 3 % and decode not worse; the noise on the Spark
is 1.5 s of TTFT at 512 and 3 % of decode.

- **The framework reaches the manual best from zero.** It takes 17.4 minutes from nothing to its decision.
  - The two configurations start the server with the same environment except five values.
  - The prefill threshold (295 against 0) sends both protocol prompts to the GPU graph, as the manual best does.
  - The other four are prefetch depth 5 (4), context and ubatch 6400 (8192), and one checkpoint (none).
  - Every value the manual best set by hand now has a source: derived, decided or self-tuned. Two of them are
    rules rather than measurements of this run: the tier from one block and the 4 GiB margin (no run learned a
    margin). After the review the launcher holds both on unified memory (see [After the review](#after-the-review)).
- **The protocol turn cannot tell most of the derived values from the manual ones.** Any threshold from 32 to 668
  sends both prompts the same way, prefetch 4 to 8 tie, and context, ubatch and checkpoint have no measurable
  effect on one turn. So the ABBA shows that the framework lands in the flat region around the manual best, not
  that each value is the best one.
- **Three framework bugs were found and fixed on the way.** One value of the manual best was never written, the
  workload's follow-ups were counted wrong, and the loop's feedback was corrupted on Muse. See
  [Fixes](#fixes-made-on-the-way).
- **The plan predicted the 4-turn conversation at 1100.6 s; the closed loop served it in 1101.2 s.** The parts'
  errors cancel (see [The validation](#the-validation)): the first prompt's graph pass is predicted 13 % fast, the
  follow-ups' prefills about 13 % fast, and decode 5-10 % slow.
- **DeepSeek-V4.1-Flash-1017 by hand: the graph route and prefetch 6.** At 4104 tokens the graph takes 31.3 s to the
  first token against 40.1 s on the sink; at 523 they tie (17.2 and 17.7 s). Depth 6 decodes 6-7 % faster than 2.
  llama.cpp PR #28696 takes 13x (512) and 48x (4096) longer to the first token and decodes 3-4x slower.
- **Offline, the review's model still sent DeepSeek's 4104 prompt to the sink; fixed.** On the stored logs (no
  device), the reviewed model priced both routes on the warm tier, where DeepSeek's sink reads 40 % less.
  - A first fix weighed the tier states by turns (74a6e49). For the protocol's conversation (the 4096 prompt and
    three short follow-ups) it derived 3703, which would send any first prompt of 314-3703 tokens, the 512 prompt
    among them, to the cold sink.
  - The threshold now gives the workload its least time (4f98401): n* = 313 for the protocol's conversation, of
    one turn or four. Both protocol prompts take the graph, the manual best's route. k* = 6, the manual best's depth.
  - The model reproduces its in-sample points within 0.5 %. Its one point out of sample, the sink at 523 tokens, is
    +0.4 %, but it takes its union from the 523-token graph prefill, rebuilt from kept numbers. Without that
    prefill the 523-token predictions are 7-14 % low.
  - Muse keeps both protocol prompts on the graph: n* = 411, after the sink's compute is priced at Muse's 488
    routes a token (efe942d).
  - None of the code after the review has run on a device (see
    [What the Spark did not test](#what-the-spark-did-not-test)).

## Machine and build

| | |
|---|---|
| machine | DGX Spark (spark-b9a5): NVIDIA GB10, 20 Arm cores (10 X925 + 10 A725), 128 GB LPDDR5x unified (MemTotal 121.7 GiB), no swap |
| drive | Samsung MZALC4T0HBL1 4 TB, PCIe Gen5 x4; the models on its own ext4 (`/media/golden-root/var/tmp`) |
| software | Ubuntu 24.04, kernel 6.17.0-1029-nvidia, driver 580.173.02, CUDA 13.0, gcc 13.3 |
| engine | branch batch2 (base 2fb29a5): the engine at 40b99d6, built fresh in `~/ssdl/eng-b2` by `tools/adaptive/build.sh` (`-march=armv8.2-a+dotprod+i8mm+fp16+bf16`, CUDA arch 121a, liburing static) |
| framework | `tools/adaptive` and `run-ssd-llama.sh` at this commit's parent (the later commits of the branch change no engine code) |
| model | Muse Spark Open 816B-A42B, unsloth UD-Q4_K_XL (62 layers x 256 experts, top-8, Q4_K/Q5_K experts of 30.4 MB read from the GGUF) |

The machine ran nothing else. Every run kept a 3 GiB watchdog on MemAvailable; it fired in none of the runs read
here.

## The run

```sh
ROOT=~/ssdl/fw MODEL_CONF=tools/adaptive/models/muse-spark.conf \
MODEL_DIR=/media/golden-root/var/tmp/<user>/models/Muse-Spark-Open-816B-A42B-GGUF/UD-Q4_K_XL \
BOOT_PROMPT=prompts/p4096.json WORK=prompts/conv4.json WORKLOAD=known WATCHDOG_MIN_AVAIL_MB=3072 \
  tools/adaptive/adaptive.sh serve|validate ~/ssdl/fw/runs/muse-spark-v1
```

- The bootstrap text is the 4096 protocol prompt. The workload is the protocol's 4-turn conversation: that prompt,
  then three short follow-up questions (`rt-mt.py`'s). Both stay within the unified protocol; the framework's own
  `boot-en-doc.json` and `work-4turn.json` have other lengths.
- `serve` was stopped at `DECISION_READY`. The ABBA used that `settings.env`. Then `validate` ran on the same run
  directory, which keeps the steps it has.
- Nothing was given but the 3 GiB watchdog: no reserve, no margin, no setting.
- The harness is in `~/ssdl/ab` on the Spark: `v1-fw.sh` (this command under the machine's lock), `v1-abba.sh`
  (the ABBA through `muse-launch.sh`), `v1-eval.py` (the tables below), `v1-chain.sh` (the order of the steps).

| step | time | what it measured |
|---|---|---|
| 1 device probe | 20 s | the drive at the queue depths a decode layer issues: knee at QD 3, 11.76 GiB/s; unified memory |
| 2 bootstrap (`all`) | 17.0 min | the prompt on the CPU sink twice (cold 333.2 s, warm 331.7 s), 512 decode tokens at prefetch depth 2 (1.90 tok/s over 129-512), the trace and profile |
| 3 in-situ costs | 1 s | per-layer decode costs, the sink's prefill compute, the hint's statistics per rank, the drive's idle window |
| 4 plan and `settings.py` | 2 s | 4 tier sizes priced over the workload; the derived values |

## What the framework decided

`~/ssdl/fw/runs/muse-spark-v1/settings.env`, each line with its receipt (shortened here):

| setting | framework | manual best | source | from |
|---|---|---|---|---|
| GPU tier | whole (2545 slots at the bootstrap) | whole | decided | the plan: 1100.6 s for the workload, against 1250-1841 s for 3/4 down to 1/4 of it; writes no line |
| CPU placement | `all` | `all` | decided | the conf's only placement; writes no line |
| `GGML_KIMI_CPU_PREFILL_MAX_TOKENS` | 295 | 0 | derived | n* where the sink's compute meets the graph's reads, `n k c = U(n) B_e / R_io`: c = 156.6 us a route (the warm prefill), U(295) = 153.5 experts a layer (the decode trace), B_e = 30.4 MB, R_io = 12.63 GB/s (the probe's knee) |
| `UBATCH` | 6400 | 8192 (legacy) | derived | the largest graph pass up to the context; a pass of 6400 tokens fits next to the 20107 MiB of dense weights, the KV and the compute buffers |
| `GGML_KIMI_CPU_SINK_PREFETCH` | 5 | 4 | derived | min(top-k 8, k_win 5, k_mem 8): the idle window 3123 us a hinted layer holds 1.30 expert reads of 2409 us; novel reads per rank 0.196, 0.237, 0.283, 0.325, 0.369, ... reach 1.41 at rank 5; every rank saves more reads (0.10-0.18) than a tier slot serves (0.046 a token) |
| `CONTEXT` | 6400 | 8192 | derived | the workload: 4217, 28, 22 and 23 new prompt tokens and 512 decode tokens a turn, up to a multiple of 256 |
| `CTX_CHECKPOINTS` | 1 | 0 | derived | the server says it saves an earlier state after each prompt batch (at multiples of 1 token), so a checkpoint splits no prompt |
| `GGML_KIMI_CPU_SINK_VRAM_BYTES` | auto | auto | derived | unified memory: the tier starts from one block and grows after the first decode token into what the host-memory ledger leaves (a rule; after the review the launcher's own there) |
| `GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB` | 4096 | 4096 | self-tuned | the largest margin the ledger learned in the framework's runs, at least 4096 MiB; no run learned one (`margin_updates=0`), so this was the floor, a constant (after the review the launcher's own start there, and `settings.py` writes a margin only where a run learned one) |

What follows from these in the launcher is the same in both: actual-shape graphs on both routes, the widened
batched reads of the 47,616 unaligned expert ranges (the graph route is reachable, 295 < 6400), the legacy profile
for the rest. The dry runs of the two configurations differ only in the five lines of the table whose values
differ.

## Against the manual best

Arms in order, each a cold server and one unified-protocol turn (no disk prompt cache). A = the manual best
(`muse-spark-gb10.settings`), B = the framework's `settings.env`. Runs: `~/ssdl/ab/runs/1021-0944..1047-v1-muse-spark-v1-*`.

| run | arm | TTFT s | decode 1-128 | decode 129-256 | tier slots | lowest MemAvailable MiB |
|---|---|---|---|---|---|---|
| p512 1021-094448 | A | 39.05 | 1.747 | 2.133 | 2832 | 5210 |
| p512 1021-094852 | B | 38.02 | 1.745 | 2.128 | 2800 | 4887 |
| p512 1021-095255 | B | 38.83 | 1.722 | 2.103 | 2800 | 4897 |
| p512 1021-095700 | A | 38.71 | 1.743 | 2.090 | 2832 | 5200 |
| p512 1021-100107 | A | 37.84 | 1.708 | 2.094 | 2832 | 5180 |
| p512 1021-100515 | B | 38.24 | 1.727 | 2.098 | 2800 | 4886 |
| p512 1021-100923 | B | 39.28 | 1.712 | 2.108 | 2800 | 4887 |
| p512 1021-101331 | A | 38.67 | 1.742 | 2.079 | 2832 | 5200 |
| p4096 1021-101739 | A | 50.58 | 1.796 | 2.137 | 2832 | 5208 |
| p4096 1021-102152 | B | 50.76 | 1.784 | 2.127 | 2784 | 4897 |
| p4096 1021-102609 | B | 49.86 | 1.787 | 2.137 | 2784 | 4873 |
| p4096 1021-103021 | A | 51.10 | 1.812 | 2.177 | 2832 | 5194 |
| p4096 1021-103435 | A | 50.68 | 1.801 | 2.118 | 2832 | 5203 |
| p4096 1021-103847 | B | 50.34 | 1.820 | 2.149 | 2784 | 4897 |
| p4096 1021-104257 | B | 50.28 | 1.796 | 2.145 | 2784 | 4883 |
| p4096 1021-104710 | A | 51.13 | 1.799 | 2.116 | 2832 | 5199 |

B against A per cycle (arms 1-4 and 5-8):

| prompt | cycle | TTFT | decode 1-128 | decode 129-256 |
|---|---|---|---|---|
| 512 | 1 | -1.2 % | -0.7 % | +0.2 % |
| 512 | 2 | +1.3 % | -0.3 % | +0.8 % |
| 4096 | 1 | -1.0 % | -1.0 % | -1.2 % |
| 4096 | 2 | -1.2 % | +0.4 % | +1.4 % |

- Every arm took the routes of the manual best: the 2-token warm-up on the sink, the prompt in one graph pass
  (669 and 4221 tokens are above 295).
- No arm came near the watchdog. The ledger's floor was 5173 MiB in A and 4861 MiB in B (its file reserve, measured
  at start, was 1077 and 765 MiB), and MemAvailable stayed 7-37 MiB above it.
- The drive read 8.8-11.3 GB/s in the 4 x 1 GiB check before every arm.
- Greedy replies are not compared: replies of the same configuration differ from run to run too (the first
  difference among the A arms is at character 142-419 at 512 and 391-874 at 4096), because the tier's contents
  decide which experts the GPU computes and which the CPU does.

### What differs from the manual best, and why

| value | effect on the protocol turn | why the framework's value is right for its workload |
|---|---|---|
| threshold 295 (0) | none: both prompts are above 295 | a prompt of 32-295 tokens takes the sink, whose compute then hides under the same reads the graph would do, and saves the graph's own time beyond its reads (about 3 s a pass); with 0 it would take the graph. The follow-ups of the conversation (21-28 tokens) take the sink under both, because the GPU does not offload a ubatch below 32 tokens |
| prefetch 5 (4) | none measurable: the staging of the fifth rank costs 62 tier slots | the ABBA of 5 against 4 on this build came out equal (batch2: 1.798/2.137 against 1.800/2.156 at 4096) |
| context and ubatch 6400 (8192) | 434 MiB less KV; at load 710 MiB less GPU and 312 MiB less host compute buffer | the conversation needs 6338 tokens; one graph pass still holds every protocol prompt |
| one checkpoint (none) | 120 MiB at 512, 368 MiB at 4096, during the request | a conversation's next turn restores it instead of prefilling the prompt again |
| tier | 32 slots fewer at 512 and 48 at 4096 | the sum of the above: the fifth staging rank (62 slots) and the checkpoint less what the smaller context and the lower floor give back |

The manual best is the measured best of the values tried by hand. The framework's values differ where its workload
differs (a conversation, not one turn) or where the measurement cannot tell them apart (prefetch 4 or 5).

## The validation

`validate` serves the workload, the protocol's 4-turn conversation with 512-token replies, on a fresh server with
an empty disk prompt cache: once under the closed loop and once static, both from the plan's configuration (Muse has
one placement, so there is no other one to start from).

| run | conversation | predicted | measured |
|---|---|---|---|
| `muse-spark-v1` (with the fixes) | loop from the plan | 1100.6 s | 1101.2 s |
| `muse-spark-v1` | static | 1100.6 s | not read: access to the Spark ended during it (see below) |
| `muse-spark-v1-run0` (before fixes 2 and 3) | loop from the plan | 1151.1 s | 1097.6 s |
| `muse-spark-v1-run0` | static | 1151.1 s | 1082.3 s |

Per turn in the first run (static): the 4218-token prompt took the graph in 51.2 s (predicted 44.5 s), the
follow-ups of 27, 21 and 21 new tokens took the sink in 11.1, 9.9 and 10.4 s (predicted 19.4-19.6 s at the wrongly
counted 155-161 tokens), and decode over tokens 129-512 ran at 2.05-2.15 tok/s (predicted 1.95). With the count fixed
the plan prices the follow-ups at 8.6-9.8 s, and its total moved from 1151.1 to 1100.6 s. Neither conversation switched
configuration in the first run: there is one placement, and a smaller tier is slower.

The fixed run's total matches by errors that cancel. Against the first run's measurements (the fixed run's turns
could not be read), per part:

| part | predicted | measured | error |
|---|---|---|---|
| the 4218-token graph pass | 44.5 s | 51.2 s | -13 % |
| the three follow-ups' prefills (sink) | 8.6-9.8 s each | 9.9-11.1 s each | about -13 % |
| decode, tokens 129-512 of each turn | 1.95 tok/s | 2.05-2.15 tok/s | 5-10 % slow, about +50 s over 4 x 384 tokens |

The first prompt's graph pass is predicted 11-13 % fast (44.4-44.5 s against 50.1-51.2 s). The model prices a graph
pass as its reads at the probe's rate (12.6 GB/s) plus the time outside the experts. Two things it leaves out: with
actual-shape graphs the expert copy read at 10.9-11.2 GB/s at 669 tokens, against 11.5-12.1 GB/s without
(route-per-ubatch gates, 10-21), and the graph's own time beyond its reads is about 3 s a pass on the Spark
(README, Limits).

**The static conversation of the fixed run was not read, and no copy of it exists.** The plan's 1100.6 s for it stays
unmeasured. The validation directory the review stored on the VM is not that run: it is a fixture built offline to
exercise `compare_online.py`'s route check, the graph arm's log (the fx-f3b prefills of 669 and 4221 tokens) as one
server's log and two hand-written turns of 300 s each. Its "static plan predicted 317.9 s" is batch2's one-turn
workload against 600 s of written turns, and means nothing. Its route check, rerun at 4f98401, reads both prefills
from the log as graph passes and prices them: 36.5 s against 36.7 s at 669 tokens (-0.6 %) and 49.9 s against 49.6 s
at 4221 (+0.5 %), the graph route 86.4 s against 86.4 s. Those are the arm's own prefills, in sample: the check
works on a stored log, which says nothing yet about the model on a served conversation.

## DeepSeek-V4.1-Flash-1017

### By hand

DSV-1017 Q8_0 (MXFP4 experts read from the GGUF; 40 layers x 384 experts, top-6, 18.8 MB an expert) was measured
by hand on both routes before the Muse runs: the engine at 7153eac (branch eng-route, the route per ubatch), the
launcher's DeepSeek profile with `VRAM_BYTES=auto`, a 4096 MiB margin and `CTX_CHECKPOINTS=0` (context 131072,
ubatch 8192). Each arm is a cold server and one unified-protocol turn. Most run directories were lost with the
Spark; their numbers were copied before. Three are stored: s1-dsv-sink-p512, s1-dsv-sink-p4096 and
s2-pf2-prof-p4096.

| run | route | prompt | TTFT s | prefill tok/s | decode 1-128 / 129-256 | the graph's expert copy |
|---|---|---|---|---|---|---|
| 1021-015422-s1-dsv-sink-p512 | sink (threshold unset) | 523 | 17.71 | 29.53 | 4.242 / 6.093 | |
| 1021-015622-s1-dsv-graph-p512 | graph (threshold 0) | 523 | 17.20 | 30.41 | 4.222 / 6.157 | 29256 role slices, 183.35 GB in 14.60 s (12.56 GB/s) |
| 1021-015811-s1-dsv-sink-p4096 | sink | 4104 | 40.09 | 102.39 | 4.167 / 6.286 | |
| 1021-020023-s1-dsv-graph-p4096 | graph | 4104 | 31.26 | 131.35 | 4.210 / 6.406 | 38784 role slices, 243.06 GB in 19.45 s (12.50 GB/s) |

- At 523 tokens the routes tie (0.5 s apart). At 4104 tokens the graph is 8.8 s (22 %) faster.
- The graph's copy reads the union of the prompt's experts at 12.5 GB/s, the probe's knee rate (12.0 GB/s) or a
  little above.
- Decode does not depend on the route.

The prefetch depth, on the graph at 4104 tokens:

| arms | TTFT s | decode 1-128 / 129-256 |
|---|---|---|
| s2, one arm each: depth 2 (profiled), 4, 6, 2 | 31.13, 31.45, 31.29, 31.17 | 4.120 / 5.949, 4.424 / 6.460, 4.470 / 6.539, 4.269 / 6.368 |
| s3 ABBA, 2 cycles: depth 2, 4 arms | 31.21-31.37, mean 31.29 | 3.997-4.255 / 5.809-6.284, mean 4.077 / 6.079 |
| s3 ABBA, 2 cycles: depth 6, 4 arms | 31.27-31.38, mean 31.35 | 4.307-4.423 / 6.394-6.543, mean 4.366 / 6.460 |

Depth 6 decodes 7.1 % (1-128) and 6.3 % (129-256) faster than depth 2 in the ABBA, at the same TTFT. Depths 4 and 6
tie within noise: one arm each, 1.0-1.2 % apart, against 3 % of decode noise. 6 is the model's top-k.

- **Correctness.** Teacher-forced top-1 at 512 on the graph: 62/64 against the reply of llama.cpp PR #28696 on 233,
  61/64 against the same PR's reply on the Spark. The extra miss is the Spark PR reply's ' Transit' at position 17:
  ours and 233's PR both choose ' rail'.
- **Profile** (s2-pf2-prof-p4096: graph, depth 2). Tokens 129-256: of 240 routed experts a token, 0.919 were in the
  GPU tier, 0.013 in the RAM cache and 0.068 read; 0.31 GB read a token; the sink took 98 ms a token, 57 ms of it
  neither computing nor waiting on the drive. Tokens 1-128: 0.791 in the tier, 0.75 GB read a token. The tier
  reached 5610 slots.
- **Baseline.** llama.cpp PR #28696 on the Spark:

| prompt | TTFT s | prefill tok/s | decode 1-128 / 129-256 | ssd-llama on the graph, depth 2 |
|---|---|---|---|---|
| 512 | 219 | 2.39 | 1.406 / 1.453 | TTFT 12.7x shorter, decode 3.0x / 4.2x |
| 4096 | 1503 | 2.73 | 1.320 / 1.608 | TTFT 48x shorter, decode 3.2x / 4.0x |

The manual best on the Spark is therefore the graph route (threshold 0), prefetch 6, the tier from one block and a
4 GiB margin, no checkpoint.

### The framework on the Spark, before the review

batch2's run from zero (`/media/golden-root/var/tmp/<user>/fw-b2/dsv1`, its probe, bootstrap and plan stored)
predates step 2g: its bootstrap ran the 4096 protocol prompt on the sink only (cold 40.0 s for 4104 tokens; warm
33.8 s for 4616 tokens, 5418 of the 13193 experts it touched in the tier; decode 5.93 tok/s over tokens 129-512 at
depth 2). The plan chose the whole tier (5637 slots), `all`. With no graph prefill it priced the graph like the
sink and derived n* = 4568, which keeps the 4104 prompt on the sink (40.1 s against 31.3 s), and k* = 6. DeepSeek's
conf holds every derived value as pending, so its `settings.env` kept the deployment's. Its loop conversation (one
turn of the 4096 prompt and 512 decode tokens) took 141.5 s against 126.3 s predicted; the static one was not read.

### The framework's queued run: not read

The same steps for DeepSeek, followed by an ABBA of the derived values against the manual best, were queued to
start after the Muse validation. They ran the framework of this document's first version
(`~/ssdl/eng-b2/tools/adaptive`), from before the review: no graph arm, and the graph priced like the sink. Access
to the Spark ended (the login is refused since about 11:25 CST) before they could be read. If they ran, expect what
batch2's run derived: n* about 4568, so the 4104 prompt on the sink, written as pending. The scripts stop at
`~/ssdl/STOP` like the others. The results would be in `~/ssdl/fw/runs/dsv-spark-v1` (log
`~/ssdl/ab/v1-chain-dsv-spark-v1.log`) and `~/ssdl/ab/runs/*-v1-dsv-spark-v1-p{512,4096}-{ex,fw}-{1..8}` (read with
`python3 ~/ssdl/ab/v1-eval.py v1-dsv-spark-v1-p`).

The framework after the review needs a run of its own: these commands with a checkout of this branch, built by
`build.sh`, in place of `~/ssdl/eng-b2` (here and in `v1-fw.sh`):

```sh
cd ~/ssdl/ab && bash v1-fw.sh dsv-spark-v1 validate ~/ssdl/eng-b2/tools/adaptive/models/dsv41.conf \
    /media/golden-root/var/tmp/<user>/models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0
sed -E 's/^# pending //' ~/ssdl/fw/runs/dsv-spark-v1/settings.env > v1-fw-dsv-spark-v1-derived.settings
bash v1-abba-dsv.sh dsv-spark-v1 ~/ssdl/ab/v1-fw-dsv-spark-v1-derived.settings prompt-512.txt prompt-4096.txt
```

What the code after the review derives from batch2's stored bootstrap is replayed offline below; no device has
run it.

## Fixes made on the way

Each fix is a commit of its own on this branch. The first was found in batch2's run from zero
(`/media/golden-root/var/tmp/<user>/fw-b2/run1`), the other two in a first run of this session
(`~/ssdl/fw/runs/muse-spark-v1-run0`). The run above is the next one, with all three.

1. **`settings.py` did not write `VRAM_BYTES=auto` on unified memory** (709564d). The manual best takes the tier from
   one block (its measured best, with the 4 GiB margin), and the design lists it among the values `settings.py`
   writes. Without it the framework's servers kept the legacy 21 GB base: in batch2's validation the tier started
   at 690 slots and grew to 2802-2834 at the second decode token. A smaller tier is now capped by
   `ELASTIC_MAX_SLOTS` alone on unified memory. Discrete GPUs are unchanged (the stored 233 run gives the same
   `settings.env`).
2. **The workload's later turns were counted as conversations of their own** (4461f5b). Each follow-up counted the
   template's preamble too: 161/155/156 new tokens where the server prefilled 27/21/21. The context came out at 6912
   instead of 6400, and the plan priced each follow-up at about 20 s of prefill instead of about 10 s. A turn is now
   the conversation rendered through it (empty replies) less the conversation before it: 28/22/23.
3. **The loop merged two requests' decode tokens** (3c63d9b). The engine numbers a decode token by its epoch and
   position, and the epoch does not change between requests. Muse's replies through the chat endpoint are its
   analysis, which the history drops, so each follow-up decodes positions the previous turn decoded. `logparse`
   summed the two turns' profiles: before the last turn the loop read 3.48x the CPU compute, 2.70x the drive's time
   and -539 ms a token outside the sink. Recomputed from the same logs with the fix: 1.14, 0.88 and 139 ms, the same
   as after the first turn. A decode layer at a position not past the last one's now starts a run of its own. The
   stored 233 and Spark bootstraps give the same `insitu.json` and `model.json` as before.

## After the review

The commits after this document change what the framework derives; a run from zero with them waits for a
device. Expected, from the stored bootstrap and the code (replayed in
[Offline replay](#offline-replay-on-the-stored-logs)):

| value | at this document | after the review |
|---|---|---|
| tier from one block, 4 GiB margin | written by `settings.py` (derived, self-tuned) | the launcher's own on unified memory, for every server; `settings.py` writes a margin only where a run learned one |
| threshold | 295: the graph priced with the sink's time a token, the probe's rate, U(n) from decode tokens | from the graph's own prefills (a new bootstrap step: a cut of the text and the whole, on the graph), with the sink's compute at the routes a prompt token takes and the workload's least time (efe942d, 4f98401); with two graph prefills of the protocol prompts on the Spark, 411 for the one-turn and the 4-turn workload. Both protocol prompts still take the graph |
| checkpoints | 1 | 1 for the 4-turn workload; none for a workload of one turn |
| context | 6400 | the same for a known workload (this run's); a learned one keeps the launcher's |
| prefetch depth | 5 | 5: each rank now pays its own slot (rank 1 none on unified memory); ranks 2-8 cost 0.044-0.055 hits a token against nu p of 0.10-0.19 |

The validation also sets each static prefill against the prefill model by route, and flags a route more than 15 %
off (design 3.3).

## Offline replay on the stored logs

Everything in this section ran on a VM after the Spark rental ended, on copies of run directories stored before. No
server ran. It is steps 3 and 4 of the framework at 4f98401 (`insitu.py` with `--graph`, `plan.py --known`,
`settings.py`) on a bootstrap the Spark recorded and a graph arm built from the stored manual runs. Step 2g, the
graph arm, came with the review and never ran on a device: it prefills a cut of the bootstrap text (at most 512
tokens) and the whole text on one fresh server at the threshold 0.

### DeepSeek

The bootstrap is batch2's (above). The graph arm:

- (a) the 4104-token graph prefill only: the real log of s2-pf2-prof-p4096 (its load lines, its
  `[kimi_cpu_prefill_graph]` and `[moe_prefill_copy]` lines; prompt 31.12 s);
- (b) the same, plus the 523-token graph prefill of s1-dsv-graph-p512. Its log is lost, so its three lines are
  written from the kept numbers (29256 slices of 6266960 bytes, the slice size of the 4104 line, 183.35 GB; read
  14.60 s; prompt 17.20 s from 30.41 tok/s) and marked as synthesised in the log. Step 2g's cut would be a part of
  the 4096 text of at most 512 tokens, not the 512 protocol prompt on its own server.

The framework as at 2ed3d6c still sent the 4104 prompt to the sink: n* = 4384 (a) and 4358 (b). It priced both
routes on the warm tier, the state of a serving server. DeepSeek's sink computes the tier's experts on the GPU and
does not read them: on the warm tier it reads 40 % less (146 GB at 4616 tokens against 244 GB at 4104), and its
time a token besides drops from 4.81 to 2.81 ms. The graph copies every expert it touches, whatever the tier holds
(the scheduler's copy reads the expert source; 243 GB at 4104 tokens, the whole union). So at 4104 tokens the model
gives the warm sink 30.1 s and the graph 31.1 s, while the first turn of a fresh server, which the protocol measures
and which the plan prices on a cold tier, takes 40.0 s on the sink (measured 40.1 s).

Two commits fix it.

- 74a6e49 weighed the sink's time over the tier states by turns: one conversation on a fresh server, its first
  turn on a cold tier and the others on the warm one, so 1/turns cold. That is exact only for turns of one size.
  The protocol's conversation is the 4096 prompt and three follow-ups of 22-28 tokens, which the GPU does not
  offload and which stay on the sink. Their warm weight still pulled n* to 3703. Any first prompt of 314 to 3703
  tokens would then take the cold sink, which the model prices up to 8 s slower than the graph (2000 tokens: 28.0
  against 23.9 s; 523 tokens: 17.8 against 17.2 s).
- 4f98401 makes the threshold give the workload its least prefill time, as the plan prices it. The time changes
  only where the threshold crosses one of the workload's pass sizes. Between them the threshold is the break-even
  over the tier states the workload's passes of at least 32 tokens meet: the cold one alone for the protocol's
  conversation, 1 : 3 for four turns of one size. Without a workload it is the warm break-even, as before. A unit
  test (`TierStates` in `tests/test_settings.py`) holds it on these numbers.

| | (a) the 4104 prefill only | (b) 523 and 4104 |
|---|---|---|
| the graph, priced from its prefills | 12.53 GB/s; 2.86 ms a token beyond its reads | 12.54 GB/s; 1.27 s a pass and 2.55 ms a token beyond its reads |
| n* at 2ed3d6c (the warm tier only) | 4384 | 4358 |
| n* at 74a6e49 (1/turns cold): one turn of 4104 tokens | 0 | 313 |
| the same: the protocol's conversation (4104, 28, 22, 23) | 3648 | 3703 |
| the same: four turns of 4104 (`serve`'s learned workload) | 3648 | 3703 |
| n* at 4f98401 (the least time): one turn of 4104 tokens | 0 | 313 |
| the same: the protocol's conversation | 0 | 313 |
| the same: four turns of 4104 | 3648 | 3703 |
| UBATCH, CONTEXT, CTX_CHECKPOINTS: one turn; the protocol's conversation | 4864, 4864, 0; 6400, 6400, 1 | the same |
| prefetch depth k* | 6 | 6 |

- With (b), 523 and 4104 tokens both take the graph for one turn and for the protocol's conversation, the manual
  best's route for both protocol prompts. A threshold of 313 instead of 0 sends prompts of 32-313 tokens to the
  sink, where the model prices the sink faster.
- With (a), one graph prefill cannot split the graph's time beyond its reads into a part per pass and a part per
  token. All of it goes per token, the graph's copy reads faster than the probe's knee (12.53 against 12.01 GB/s),
  and the graph wins at every size on a cold tier: n* = 0. The engine keeps passes below 32 tokens on the sink
  anyway, and so does the model now.
- `serve`'s learned workload, four turns the size of the bootstrap text, weighs a cold tier once in four turns
  (3703), while a serving server meets it once per start (README, Limits).
- k* = min(top-k 6, k_win 6, k_mem 6): the drive's idle window is 2927 us a hinted layer, 1.77 expert reads of
  1658 us (18.8 MB at the probe's QD-3 rate); the novel shares of ranks 1-6 (0.114-0.288) sum to 1.18, so even
  the full width does not fill the window; each rank saves 0.060-0.100 reads a layer and token (nu p), its tier slot
  would serve 0.000-0.008. Measured: 6 decodes 6-7 % faster than 2 (ABBA); 4 and 6 tie within noise.
- With the derived values adopted (they stay pending for DeepSeek), the plan predicts the one-turn workload at
  117.3 s (prefill 31.1 s) against 126.2 s on the sink (prefill 40.0 s), and the protocol's conversation at 383.7 s
  against 392.6 s.

Each route's prefill against the model, cold (the state of every manual arm), TTFT in seconds:

| route | prompt | measured | model, (b) | model, (a) |
|---|---|---|---|---|
| sink | 523 | 17.71 | 17.79 (+0.4 %), its union from the rebuilt 523 graph prefill | 16.45 (-7.1 %) |
| sink | 4104 | 40.09 | 39.99 (-0.3 %), in sample | 39.99 (-0.3 %), in sample |
| graph | 523 | 17.20 | 17.22 (+0.1 %), in sample | 14.84 (-13.7 %) |
| graph | 4104 | 31.26 | 31.10 (-0.5 %), in sample | 31.12 (-0.4 %), in sample |

- In sample: the bootstrap's cold prefill was the same 4104 prompt (40.04 s), and the graph's points are the arm's
  own.
- The one point of (b) out of sample is the sink at 523 tokens. Its time model is new (the sink's compute and its
  time a token besides, from the 4104 prefill), but its union U(523) is the one the rebuilt 523-token graph prefill
  of the same prompt gives. So it checks the time model, not the union.
- With (a) the 523-token pass touches U(n) of the decode trace, 222 experts a layer against the 244 the graph's
  copy read: decode tokens touch about 9 % fewer experts than prompt tokens here (7 % for Muse), so both routes
  come out 7-14 % fast.
- The warm states were never measured on the Spark. The model gives the warm sink 30.1 s at 4104 tokens and 10.5 s
  at 523. The bootstrap's warm prefill re-prefills its own text after decode, so its tier holds more of the
  prompt's experts than a new prompt would find: the warm sink's reads are a lower bound.

### Muse

The bootstrap is batch2's run from zero (above, `fw-b2/run1`). The graph arm is the real lines of fx-f3b-p512 and
fx-f3b-p4096 (graph route, 669 and 4221 tokens, prompts 36.73 and 49.64 s) in one log. They come from two fresh
servers rather than step 2g's one, and from the example settings of that day: `VRAM_BYTES` 21e9 and a 16 GiB
margin, not the launcher's start on unified memory after the review (one block, 4 GiB), which step 2g would use.

| workload | n* | UBATCH | CONTEXT | CTX_CHECKPOINTS | k* |
|---|---|---|---|---|---|
| batch2's: one turn of 4217 tokens | 411 | 4864 | 4864 | 0 | 5 |
| V1's: 4217, 28, 22 and 23 tokens | 411 | 6400 | 6400 | 1 | 5 |
| V1 as run (before the review) | 295 | 6400 | 6400 | 1 | 5 |

- At 2ed3d6c n* was 403 for both workloads. Muse's sink takes no expert from the tier in prefill, and its cold and
  warm prefills differ by 0.3 ms a token besides compute (2.67 against 2.37 ms) and not in compute (156.6 against
  156.9 us a route), so the tier states barely matter.
- The rise to 411 is the sink's compute priced at Muse's 488 routes a prompt token (efe942d). Its last layer
  computes the output token only, and k L = 496 had priced the sink's compute 1.6 % high.
- Both protocol prompts take the graph under all three. The context, ubatch and checkpoint follow the workload;
  prefetch depth is V1's.

| route | prompt | measured TTFT s | model |
|---|---|---|---|
| graph | 669 | 36.74 (fx-f3b) | 36.50 (-0.6 %), in sample |
| graph | 4221 | 49.66 (fx-f3b) | 49.91 (+0.5 %), in sample |
| graph | 669 | 38.57 (the V1 ABBA's A arms: one block, 4 GiB) | 36.50 (-5.4 %) |
| graph | 4221 | 50.87 (the same) | 49.91 (-1.9 %) |
| sink | 669 | 49.26-49.89 (3 runs, 10-20 and 10-21) | 52.90 (+6 to +7 %) |
| sink | 4221 | 332.79, 337.67 (2 runs, 10-20) | 333.77 (+0.3 %, -1.2 %), in sample |
| sink, follow-ups of the T=500 4-turn run (fx-f3c) | 29, 23 and 24 | 11.88, 10.87, 11.33 | 10.99, 9.73, 9.95 (-8 to -12 %) |

- The sink at 4221 tokens is in sample: the bootstrap's cold prefill was the same prompt (4218 tokens, 333.5 s).
- Against the configuration the post-review launcher serves (the A arms), the graph model is 5.4 % and 1.9 % fast.
  Fitted to the A arms instead, the graph's time beyond its reads is 4.70 s a pass and 2.67 ms a token, and
  n* = 441. Both prompts stay on the graph.

## What the Spark did not test

The Spark ran the framework from zero on Muse with the code before the review, and both models by hand. These wait
for a device (the second Spark, ysy9, has since run DeepSeek from zero, its ABBA, part of U1 and the warm-up C0:
see [Spark ysy9](#spark-ysy9)):

- **The post-review code end to end on unified memory.** The review's 11 commits and the four after it (74a6e49,
  efe942d, 680ac68, 4f98401) changed the bootstrap (step 2g, the graph arm), the prefill model (the graph priced
  from its own prefills; the sink's compute at the routes a token takes; passes below the offload minimum on the
  sink; the threshold as the workload's least time over the tier states its turns meet), k_mem (the RAM cache
  behind the tier replayed with the engine's policy), the launcher's unified-memory defaults (`VRAM_BYTES=auto`,
  the margin max(`MEMORY_HEADROOM_MIB`, 4096)) and the sink's prefetch depth up to the hint's width. They ran on
  stored logs (above) and in unit tests only. Step 2g has never started a server.
- **The bare launch under stress (design U1).** The launcher now starts unified memory from its own defaults. The
  stress run that should come first did not run: the 4096 prompt, follow-ups long enough to take the graph after
  the tier has grown, a 32K prompt, checkpoints on, once at 4096 MiB and once at a learned margin. Every Spark run
  here had a 3 GiB watchdog on MemAvailable, and none came near it.
- **DeepSeek from zero on a device**: the framework's run with step 2g, its validation, and the ABBA of its derived
  values against the manual best (graph, depth 6). For the planned run (`prompts/conv4.json`, `WORKLOAD=known`)
  the replay expects n* = 313, depth 6, context and ubatch 6400 and one checkpoint, so both protocol prompts on the
  graph, as in the manual best. Before 4f98401 the same workload derived 3703, which sends the 512 prompt to the
  sink. At 523 tokens the two routes measured 17.7 and 17.2 s, 3 % apart, so a route error there sits at the 3 %
  TTFT gate. The prices of both routes at 523 tokens rest on a rebuilt graph prefill (above).
- **The warm state.** What the sink and the graph take at 4104 tokens on a server that has served before was never
  measured (the model: 30.1 against 31.1 s). The bootstrap's warm state is its own text again, an upper bound on
  what the tier holds for a new prompt.
- **The read rate of the graph's copy with actual-shape graphs.** At 669 tokens (Muse) the copy read 10.9-11.2 GB/s
  with actual shape and 11.5-12.1 GB/s without, about 2 s of TTFT. The cause is unknown. The prefill model takes the
  rate the bootstrap's own graph prefills read, so it predicts what the engine does, not what it could do.
- Also not run: prompts near n* on both routes ([ask]: about 150 and 600 tokens for DeepSeek, 200 and 800 for Muse),
  the framework's own `boot-en-doc.json` and `work-4turn.json`, and V1's static conversation (above).

## Open items

- **Not read: the static conversation of the final validation, and the framework's DeepSeek run** (above): access
  to the Spark ended at about 11:25 CST. The manual DeepSeek arms were read before.
- **176 and 233** were not reachable. Their from-zero runs, and the ABBAs D1-D4 that adopt DeepSeek's derived values
  there, wait. The discrete-GPU paths of the fixes are covered by the unit tests and the stored runs only.
- **Muse's reasoning is not turned off through the chat endpoint** (design M13). The framework's client sends chat
  turns with `--reasoning off`; Muse still writes its analysis, which the server returns as reasoning. So the
  bootstrap's warm prefill is the prompt without a reply (4218 tokens), and a conversation's history does not carry
  the replies. Decode is the same work either way.
- **The plan prices decode at the bootstrap's prefetch depth** (2), not at the derived one (README, Limits).
- **The start-up probe of larger ubatches reads nothing on unified memory**: the tier at the first request is its base
  there, whatever the ubatch (DSV41's 16384 and 32768 both read 1116 slots, its legacy 21 GB base). The memory plan
  (design M6) replaces it. Fixed on ysy9 (b029cdd): the probe now reads the tier after its first decode growth.
- **No margin below 4096 MiB can be learned** by a run that starts at 4096 (batch2's open item).
- **batch2's U1 decode gate is met within noise only**: tokens 129-256 averaged 2.147 tok/s over 10 protocol arms
  against the 2.194 asked (2.137 in this ABBA's A arms), 1-128 and 1-256 met it. The reference build read 2.168 the
  same day, and eng-b2 read 1.6 % below it with the same sign in both cycles of a 2-cycle ABBA. An ABBA of the
  2fb29a5 base (eng-route) against eng-b2 at the tier from one block and 4 GiB would tell whether the 2fb29a5 base
  costs that, and where its 615 MiB higher file reserve comes from.

## Spark ysy9

The rental of spark-b9a5 ended at about 11:25 CST. ysy9, another DGX Spark, ran the open tasks on the afternoon of
10-21 (16:36-18:14 CST), and the runs a review of them asked for at 18:54-19:50. Nothing measured on spark-b9a5 ran
again. Only DeepSeek-V4.1-Flash-1017 Q8_0 is on ysy9; Muse was not copied there.

- **The framework from zero on DeepSeek beats the manual best in decode and ties it in TTFT.** ABBA of two cycles,
  B with the threshold 414: TTFT -0.5 % at both prompts, decode +3.3 % / +6.5 % at 512 and +3.5 % / +11.2 % at
  4096 (tokens 1-128 / 129-256), the same sign in every cycle. Gate (TTFT within 3 %, decode not worse): pass.
  - The plan chose the `split` CPU placement over `all`, from bootstraps that decoded at 6.80 and 5.92 tok/s.
  - The derived values put both protocol prompts on the graph, as the manual best does: threshold 414, prefetch
    depth 6, context and ubatch 6400, one checkpoint.
  - **414 was wrong for the workload as served** (found by the review): a follow-up that prefills its reply again
    (534-539 tokens, warm) took the graph, 15.6-16.4 s against 7.1-7.7 s on the warm sink. The threshold now prices
    those passes (5c2fc47) and is 2906. At 2906 the 523 prompt takes the cold sink, which on ysy9 is faster than
    the graph: the p512 ABBA at 2906 gives TTFT -3.4 % and decode +3.4 % / +8.5 %. Gate: pass.
- **Two framework bugs were found and fixed** (396687a, b029cdd). Before them the plan chose a 16-slot tier priced
  at a negative time. The review found four more framework issues, fixed after it: the threshold above (5c2fc47),
  a route check that flagged only totals (71994ff), a plan that would choose a configuration priced below zero
  (f12f9f8), and a ubatch probe that compared two different quantities (59cf072).
- **U1, the bare launch under stress, is not met.** The sink half passed (no kill, MemAvailable at least 4747 MiB,
  never below the ledger's floor). The graph half, run after the review, had no kill and stayed 1.3 GiB above the
  3 GiB watchdog, but MemAvailable fell below the ledger's published floor: by up to 230 MiB when a graph prefill
  took its workspace from a full tier, and by up to 1.2 GiB for about three minutes of a 32K prompt's prefill, as
  the floor rose with refaulted file pages.
- **Seeding the tier from the prefill clears its bar** (warm-up C0). With the true prefill counts it saves
  2.77-2.87 s of decode reads at 4096 and 3.13-3.18 s at 512 over A+B. All three roles of a layer are exposed at the
  measured 110 GB/s copy rate, 0.94 s. Net 1.83-1.93 s and 2.19-2.24 s, above the 1.3 s the design asks; with
  fewer seeds than the full quota it nets more.
- **Correctness: 61/64 teacher-forced, one short of the protocol's 62/64.** The manual best, the framework's settings
  and depth 2 give the same three misses with the same log-probabilities, as they must: the check runs only the
  graph prefill and one-token sink passes. Two misses are near ties. On the decode path, the ABBAs' greedy replies
  of A and B part from the reference only at near ties or at the shared miss; see
  [Correctness](#correctness-on-ysy9).

### Machine differences

| | spark-b9a5 (morning) | ysy9 (afternoon) |
|---|---|---|
| machine | DGX Spark, GB10, 20 Arm cores, MemTotal 121.7 GiB, no swap | the same |
| software | Ubuntu 24.04, kernel 6.17.0-1029-nvidia, driver 580.173.02, CUDA 13.0, gcc 13.3 | Ubuntu 24.04.3, kernel 6.11.0-1014-nvidia, driver 580.82.09, CUDA 13.0, gcc 13.3 |
| drive | Samsung MZALC4T0HBL1 4 TB, ext4 at `/media/golden-root/var/tmp` | the same model (firmware NXHB202Q), the same mount |
| drive probe (DeepSeek's expert reads) | knee QD 3, 12.0 GB/s | knee QD 3, 11.31 GiB/s (12.14 GB/s) |
| engine | eng-route (7153eac) for the manual DeepSeek arms | eng-dev: dev 6333f49 built by `build.sh` (liburing static), unit tests passed; the framework at b029cdd |
| CPU governor | not kept | performance on all 20 cores |

The machines measure alike. The manual best's arms read the same on both: TTFT 17.11 s and 31.34 s here against
17.20 s (depth 2) and 31.35 s there; decode at depth 6 and 4096 4.448 / 6.500 tok/s here against 4.366 / 6.460. The
cold sink prefill of 4104 tokens took 39.66 s here and 40.0 s there. The graph's expert copy read 12.32-12.47 GB/s
here and 12.50-12.56 there.

### What ran

Everything ran under `flock ~/ssdl/spark.lock`, one server at a time, with a 3 GiB watchdog on MemAvailable. It never
fired. The harness is in `~/ssdl/ab` on ysy9, the `y9-*` scripts, adapted from spark-b9a5's `v1-*` and `b2-*`.

| step | script | runs |
|---|---|---|
| the framework from zero, with its validation | `y9-fw.sh dsv-ysy9 validate` | `~/ssdl/fw/runs/dsv-ysy9` (the first, wrong decision is kept in `first-decision/`) |
| teacher-forced checks | `y9-tf.sh` | `~/ssdl/ab/runs/1021-*-y9-tf-dsv-ysy9-*` (three) |
| ABBA, 2 cycles, p512 and p4096 | `y9-abba-dsv.sh` | `~/ssdl/ab/runs/1021-17*-y9-dsv-ysy9-p{512,4096}-{ex,fw}-{1..8}` |
| U1, the bare launch under stress | `y9-u1.sh`, `y9-stress-client.sh`, `y9-mt.py` | `~/ssdl/ab/runs/1021-180202-y9-u1-dsv-ysy9-stress` |
| warm-up C0 | `y9-c0.sh`, `y9-d2d.cu` | `~/ssdl/ab/runs/1021-180743-y9-c0-dsv-ysy9-p4096`, `1021-180954-y9-c0-dsv-ysy9-p512`, `~/ssdl/ab/c0/` |
| after the review: four conversations with B's settings, the p512 ABBA at the new threshold, U1's graph half | `y9-rev.sh`, `y9-rev2.sh`, `y9-conv.py`, `y9-conv2.py`, `y9-conv*-client.sh`, `y9-u1g2-client.sh` | `~/ssdl/ab/runs/1021-1{85357,90030,90707,91418}-y9-rev-conv-*`, `1021-19*-y9-rev-p512-{ex,fw}-{1..8}`, `1021-193602-y9-rev-u1g-stress`; log `y9-rev.log` |

The order was `y9-chain.sh dsv-ysy9` (the teacher-forced checks, the ABBA, U1, C0), then `y9-after.sh` (one more
teacher-forced check). `y9-mt.py` is `rt-mt.py` with DeepSeek's rendering (the server's template with
`enable_thinking` false, no raw suffix).

### The framework from zero

```sh
ROOT=~/ssdl/fw MODEL_CONF=tools/adaptive/models/dsv41.conf \
MODEL_DIR=/media/golden-root/var/tmp/<user>/models/DeepSeek-V4.1-Flash-1017-GGUF/Q8_0 \
BOOT_PROMPT=prompts/p4096.json WORK=prompts/conv4.json WORKLOAD=known WATCHDOG_MIN_AVAIL_MB=3072 \
  tools/adaptive/adaptive.sh validate ~/ssdl/fw/runs/dsv-ysy9
```

The bootstrap text is the 4096 protocol prompt. The workload is the protocol's 4-turn conversation. Nothing else was
given.

| step | time | what it measured |
|---|---|---|
| 1 device probe | 15 s | knee at QD 3, 11.31 GiB/s; unified memory; host-to-device 55.2 GB/s |
| 2 bootstrap `all` | 3.5 min | cold sink prefill 39.66 s (4104 tokens), warm 33.53 s (4616); 512 decode tokens at depth 2, 5.92 tok/s over 129-512 |
| 2 bootstrap `split` | 3.4 min | cold 40.92 s, warm 36.85 s; 6.80 tok/s over 129-512 |
| 2g the graph arm | 1.5 min | a cut of 491 tokens: 179.7 GB copied in 14.41 s (12.47 GB/s), prompt 16.72 s; the whole 4104: 243.6 GB in 19.69 s (12.37 GB/s), prompt 30.52 s |
| 2u ubatch probes | 2.0 min | the tier after its first decode growth: 5585 slots at 16384, 5350 at 32768 |
| 3, 4 in-situ costs, plan, `settings.py` | 22 s | |

From nothing to the decision: 10.9 minutes (steps 1 to 2g of the first run, 520 s, and steps 2u to 4 of the
second, 135 s).

**Step 2g ran on a device for the first time.** It started one server at the threshold 0 and prefilled a cut of
the text and the whole text, each from scratch. `insitu.py` read both. The graph is priced at 12.42 GB/s, 1.16 s a
pass and 2.36 ms a token beyond its reads.

#### Two bugs

The first run decided `16,all@16384`: a 16-slot tier with a 16384-token ubatch, predicted at -373 s for the
workload. Two bugs made it.

1. **The ubatch probe read one block on unified memory** (b029cdd). The probe's two-token request read the tier the
   engine allotted at the first request. On unified memory the launcher starts the tier from one block (16 slots)
   and grows it only after the first decode token. So both larger ubatches looked as if they left 16 slots, and
   `insitu.py` added `all@16384` and `split@16384` with 16 slots. The probe now decodes four tokens, and
   `ubprobe.py` reads the most the tier held: 5585 slots at 16384 and 5350 at 32768 against 5646 at 8192. (This was
   an open item of this document: the probe "reads nothing on unified memory".) The review found that this compares
   two different quantities, and that the probe still misses most of what a larger ubatch costs on unified memory.
   Both are in [After the review on ysy9](#after-the-review-on-ysy9).
2. **The layer table's linear fit was singular and the solver divided by rounding errors** (396687a). Classes the
   bootstrap saw too rarely are priced by a fit on (1, tier hits, ready, read, any read). The three counts add up to
   the top-k, so the columns are dependent. The solver skipped a pivot only below an absolute 1e-12. For `all` it
   gave -16.8 ms for the class (6, 0, 0) that measured 1.87 ms, and -6 ms for an all-miss layer. So a 16-slot tier
   decoded at -4.84 tok/s, and the quarter tiers came out non-monotone (1411 slots faster than 2823). The solver now
   gives no coefficient to a column the earlier ones span. It prices (6, 0, 0) at 1.872 ms and an all-miss layer at
   12.50 ms, as the pseudo-inverse does.

Both have unit tests (`LayerFit`, `UbatchProbe` in `tests/test_settings.py`; 56 tests pass on the VM). The engine is
unchanged. The second run kept steps 1-2g and redid 2u-4 with the fixes.

#### What the framework decided

`~/ssdl/fw/runs/dsv-ysy9/settings.env`. DeepSeek's conf holds five derived values as pending (they wait for an ABBA
on 125, formerly 176). The ABBA below applied them by stripping the `# pending ` prefix.

| setting | framework | manual best | source | from |
|---|---|---|---|---|
| GPU tier | whole (5624 slots at the bootstrap) | whole | decided | the plan: 358.0 s for the workload against 390.4 s for `all` (5646 slots), 359.2-397.5 s with a larger ubatch and 408.6-626.4 s with a smaller tier; writes no line |
| CPU placement | `split`: `THREADS=20`, `COMPUTE_CPUS` the 19 cores but core 9, `IO_CPUS=9`, `DECODE_PARTICIPANTS=9` | `all` | decided | the bootstraps: the sink took 81.8 ms a token on `split` against 100.2 ms on `all` (51.8 against 64.9 ms neither computing nor waiting on the drive) |
| `GGML_KIMI_CPU_PREFILL_MAX_TOKENS` | 414 (pending); 2906 after the review (5c2fc47) | 0 | derived | the least time for turns of 4104, 22, 16 and 17 new tokens (one cold pass of at least 32 tokens): the sink's compute 23.2 us a route, 240 routes a token, 4.41 ms a token besides; U(414) = 226.2 of 384 experts a layer; the graph as above. 5c2fc47 also prices each follow-up as the reply before it prefilled again (534, 528, 529 tokens, warm), 1/2 each: 1 cold pass against 1.5 warm, 2906 |
| `UBATCH` | 6400 (pending) | 8192 (legacy) | derived | the largest graph pass up to the context |
| `CONTEXT` | 6400 (pending) | 131072 (legacy) | derived | the workload: 4104, 22, 16 and 17 new tokens and 512 decode tokens a turn |
| `CTX_CHECKPOINTS` | 1 (pending) | 0 | derived | the server saves an earlier state after each prompt batch (at multiples of 2 tokens) |
| `GGML_KIMI_CPU_SINK_PREFETCH` | 6 (pending) | 6 | derived | min(top-k 6, k_win 6, k_mem 6): the idle window 2461 us a hinted layer, 1.59 expert reads of 1550 us; novel per rank 0.108-0.287; each rank's slot would serve 0.000-0.008 hits a token |
| `VRAM_BYTES`, margin | the launcher's: from one block, 4096 MiB | auto, 4096 MiB | launcher | no run learned a margin (`margin_updates=0`), so `settings.py` writes none |

- The offline replay at 6333f49 expected n* = 313 for `all` on spark-b9a5's bootstrap. Here n* is 357 for `all` and
  414 for `split`, whose sink computes 23.2 us a route against 18.4. Both protocol prompts (523 and 4104 tokens)
  stay above it, on the graph. Prefetch 6, context and ubatch 6400 and one checkpoint are as the replay expected.
- 414 fits the workload as counted, not as served: a follow-up that prefills its reply again (534-539 tokens, warm)
  takes the graph at 414, about 16 s against 7 s on the warm sink. After the review the threshold is 2906, which
  keeps the 4104 prompt on the graph and sends the 523 prompt and the re-prefills to the sink (see
  [After the review on ysy9](#after-the-review-on-ysy9)). The ABBA below ran B at 414.
- The workload's follow-ups counted 22, 16 and 17 tokens (DeepSeek's template; the replay had used Muse's 28, 22
  and 23). The served turns prefilled one more each (23, 17, 18): the last reply token is not in the cache.

#### The validation

The four conversations ran the deployment's values (the pending ones held): every prefill on the sink, depth 2,
context 131072, 8 checkpoints, 512 decode tokens a turn, each on a fresh server.

| conversation | measured (first request to last token) | prefill + 512 / rate | the static plan predicted |
|---|---|---|---|
| static-plan (`split`, 5624) | 366.3 s | 348.5 s | 358.0 s |
| loop-plan (`split`, stayed) | 373.3 s | 351.7 s | |
| static-other (`all`, 5646) | 409.9 s | 394.8 s | 390.4 s |
| loop-other (`all`, stayed) | 427.4 s | 406.5 s | |

- The plan ranks the placements right, and the decode rates show it: `split` decoded 6.68-7.13 tok/s, `all`
  5.95-6.40 (static). The static totals differ by 10.6 % (predicted 8.3 %), but that gap also holds a different
  re-prefill pattern (below): static-other prefilled three replies again and static-plan one, so their prefills took
  61.5 and 51.8 s. Decode alone took 296.6 s on `split` and 333.4 s on `all`, 11.0 % apart; the two extra re-prefills
  of static-other add about 11 s.
- Decode: `split` 6.68-7.13 tok/s against 6.59 predicted; `all` 5.95-6.40 against 5.95.
- Neither loop switched. From `all` the loop priced `split` with a restart and a cold tier, and the three turns left
  did not pay for them.
- **Some follow-ups prefill the previous reply again.** A greedy reply's tokens are not always the tokens its text
  gives back: in 7 of the 12 follow-ups the rendered history parted from the cache inside the previous reply (for
  example at position 4947, 309 tokens into a 512-token reply). The server then restores its latest checkpoint,
  near the end of the previous prompt (4634 of its 4638 tokens), and prefills 534-539 tokens instead of 17-23
  (6.9-7.6 s instead of 1.5-1.8 s). The workload's count does not know this. One checkpoint restores the same point
  as eight.
- The route check (`compare_online.py`, every prefill on the sink): the 4104-token cold prefills -0.5 % and -0.7 %;
  the sink in total +6.8 % (static-plan) and +14.8 % (static-other), under the 15 % flag. That flag on a route's
  total let the accurate cold 4104-token prefill hide the warm ones. Per turn, the warm sink is predicted slow at a
  few hundred tokens: +35 % to +49 % at 535-539 tokens, +28 % to +39 % at 17-18. Per turn, as the check now flags it
  (71994ff), all 6 warm prefills of the two static conversations are more than 15 % off: the warm sink's price is
  wrong at these sizes (see What remains).

### Against the manual best

Each arm is a cold server and one unified-protocol turn (`dsv-launch.sh`, no disk prompt cache). A = the manual best
of spark-b9a5 (graph route `MAX_TOKENS=0`, prefetch 6, `VRAM_BYTES=auto`, a 4096 MiB margin, `CTX_CHECKPOINTS=0`;
the launcher's context 131072 and ubatch 8192). B = `y9-fw-dsv-ysy9-derived.settings`, the framework's
`settings.env` with the pending values applied (threshold 414; the p512 ABBA at 2906 is in
[After the review on ysy9](#after-the-review-on-ysy9)).

| run | arm | TTFT s | decode 1-128 | decode 129-256 | tier slots | lowest MemAvailable MiB | ledger floor MiB |
|---|---|---|---|---|---|---|---|
| p512 1021-173001 | A | 17.13 | 4.537 | 6.616 | 5504 | 4753 | 4729 |
| p512 1021-173154 | B | 17.01 | 4.667 | 7.157 | 5552 | 4569 | 4551 |
| p512 1021-173345 | B | 17.06 | 4.694 | 7.111 | 5552 | 4576 | 4551 |
| p512 1021-173537 | A | 17.09 | 4.523 | 6.708 | 5488 | 4764 | 4729 |
| p512 1021-173731 | A | 17.13 | 4.562 | 6.731 | 5488 | 4754 | 4729 |
| p512 1021-173924 | B | 17.04 | 4.704 | 7.160 | 5552 | 4578 | 4551 |
| p512 1021-174115 | B | 17.02 | 4.668 | 7.048 | 5552 | 4571 | 4551 |
| p512 1021-174307 | A | 17.11 | 4.505 | 6.688 | 5504 | 4749 | 4729 |
| p4096 1021-174500 | A | 31.30 | 4.422 | 6.414 | 5472 | 4761 | 4729 |
| p4096 1021-174709 | B | 31.20 | 4.638 | 7.160 | 5520 | 4586 | 4551 |
| p4096 1021-174914 | B | 31.25 | 4.592 | 7.258 | 5536 | 4580 | 4551 |
| p4096 1021-175120 | A | 31.27 | 4.476 | 6.586 | 5488 | 4760 | 4729 |
| p4096 1021-175330 | A | 31.46 | 4.452 | 6.460 | 5472 | 4765 | 4729 |
| p4096 1021-175541 | B | 31.16 | 4.600 | 7.259 | 5536 | 4577 | 4551 |
| p4096 1021-175747 | B | 31.17 | 4.594 | 7.230 | 5536 | 4576 | 4551 |
| p4096 1021-175952 | A | 31.31 | 4.443 | 6.539 | 5488 | 4772 | 4729 |

| prompt | A: TTFT, decode 1-128 / 129-256 | B | B against A, cycle 1 | cycle 2 | all | gate |
|---|---|---|---|---|---|---|
| 512 (523 tokens) | 17.12 s, 4.532 / 6.686 | 17.03 s, 4.683 / 7.119 | -0.4 %, +3.3 % / +7.1 % | -0.5 %, +3.4 % / +5.9 % | -0.5 %, +3.3 % / +6.5 % | pass |
| 4096 (4104 tokens) | 31.34 s, 4.448 / 6.500 | 31.20 s, 4.606 / 7.227 | -0.2 %, +3.7 % / +10.9 % | -0.7 %, +3.4 % / +11.5 % | -0.5 %, +3.5 % / +11.2 % | pass |

- Every arm took the same routes: the 2-token load pass on the sink, the prompt in one graph pass (523 and 4104 are
  above 414 and 0).
- B differs from A in the placement, the threshold, the context and ubatch, and one checkpoint. Only the placement
  changes the protocol turn's work: neither prompt changes route, and DeepSeek's checkpoint is 5 MiB. The
  bootstraps measured `split` 14.7 % faster than `all` over tokens 129-512 at depth 2; the ABBA gives B 6.5-11.2 %
  over 129-256 at depth 6.
- B's tier is 32-64 slots larger (the smaller context and ubatch). Its ledger floor is 178 MiB lower (its file
  reserve at start). MemAvailable stayed 18-43 MiB above the floor in every arm.
- The drive read 8.4-11.0 GB/s in the 4 x 1 GiB check before each arm. Its hottest sensor reached 56 C at most.

### Correctness on ysy9

Teacher-forced top-1 over the first 64 tokens of llama.cpp PR #28696's greedy reply on 233 (`probe-tf.py`, the 523-
token prompt), one cold server each:

| configuration | run | agreement | misses: position, ours against the reference (log-probabilities) |
|---|---|---|---|
| B, the framework's settings | 1021-172624-y9-tf-dsv-ysy9-fw | 61/64 | 5: ' Or' -0.672 vs ' storage' -0.758; 8: ' the' -0.642 vs ' replaced' -0.758; 50: ' from' -0.565 vs ',' -1.197 |
| A, the manual best | 1021-172812-y9-tf-dsv-ysy9-ex | 61/64 | the same, to the last digit |
| A at depth 2 | 1021-181157-y9-tf-dsv-ysy9-ex-pf2 | 61/64 | the same |

- All three configurations give the same log-probabilities, and they must: the check cannot tell them apart. Each
  position is a request of its own for one token, with the prompt cache on: the prompt takes the graph and every
  later position is a one-token prefill on the sink (63 such passes in each log). No request decodes, so the GPU
  tier is never allocated (`[kimi_vram_tier] reason=destroy ... capacity_bytes=0 slots=0`), and the CPU computes
  every routed expert the same way whatever the placement or depth. The check covers the graph prefill and the
  one-token sink pass only. The served decode path, where the tier serves most experts and the placement matters,
  is the decode-path check below.
- Two misses are near ties (0.09 and 0.12 nats). Position 50 is not (0.63 nats); 233 missed it too, with eng-m8
  (10-21, 62/64, 0.33 nats).
- spark-b9a5 read 62/64 on eng-route (7153eac); which positions it missed was not kept. eng-route is not on ysy9,
  so the build and the machine were not separated. The source does not explain the third miss: from 7153eac to
  6333f49 the engine changes the prefetch hint's width (the model's top-k, 6, the depth both TF runs of the manual
  best used), the prefetch statistics, the decode_grow receipt and the launcher's unified-memory defaults, which
  arm A sets anyway. None of these is in a forward pass. So the miss more likely comes from the build (`build.sh`'s
  flags against eng-route's) or the machine.

**Decode-path check.** The ABBA's eight p512 arms decoded the same prompt greedily with the GPU tier serving. Each
reply against the reference, up to its first divergence (first 64 reference tokens; the teacher-forced
log-probabilities at that position):

| arms | reference tokens before the first divergence | where they part |
|---|---|---|
| A (ex-1, ex-5, ex-8) | 8 | ' the' for ' replaced' (teacher-forced: -0.642 vs -0.758, 0.12 nats) |
| A (ex-4) | 21 | ' operates' for ' runs' (-0.828 vs -0.812, 0.02 nats; teacher-forced top-1 is ' runs') |
| B (fw-2, fw-3, fw-6) | 50 | ' from' for ',' (-0.565 vs -1.197: the teacher-forced miss both share, also 233's) |
| B (fw-7) | 8 | ' the' for ' replaced', as A |

- No arm parts from the reference before position 8, and every arm parts at a near tie or at position 50, which
  the teacher-forced check misses for A and B alike. In each cycle B's earliest divergence is no earlier than A's.
- The served path is not deterministic at near ties: A's four arms part at 8 three times and at 21 once, and B's
  at 50 three times and at 8 once. It also differs from the teacher-forced path: every ABBA arm decodes ' storage'
  at position 5, where the one-token passes put ' Or' first (0.09 nats); the C0 run with A's settings and the trace
  on decoded ' Orrin' there.
- So on this evidence B's decode is as correct as A's: four greedy replies each, all parting at near ties or at the
  shared miss. It is not a token-level proof; that would need teacher-forced decode with a grown tier.

### The bare launch under stress (U1): the sink half

`y9-u1.sh dsv-ysy9`: no SETTINGS at all. The launcher's unified-memory defaults: the tier from one block, a
4096 MiB margin, 8 checkpoints, context 131072, ubatch 8192, every prefill on the sink, depth 2. The client is the
protocol's 4-turn conversation with the prompt cache on (256 tokens a turn), then the 4096 prompt as a new
conversation on the same server. Watchdog 3 GiB.

| turn | new tokens (cached) | route | TTFT s | decode 1-128 / 129-256 |
|---|---|---|---|---|
| 1 | 4104 (0) | sink | 39.84 | 4.176 / 6.453 |
| 2 | 283 (4100) | sink | 5.01 | 5.888 / 6.340 |
| 3 | 279 (4378) | sink | 4.65 | 6.096 / 6.230 |
| 4 | 18 (4912) | sink | 1.34 | 6.769 / 6.709 |
| 5, the 4096 prompt again | 4104 (0) | sink | 29.17 | 5.851 / 6.487 |

- **The sink half passes; this run alone is not U1's gate.** No kill. The lowest MemAvailable was 4747 MiB (1 Hz
  monitor; 4756 MiB in the client's sampler), above 3 GiB. The gate asks for more (design section 7): follow-ups of
  1.4-3K tokens that take the graph after the tier has grown, the 32K prompt, a second run at a learned margin, and
  the protocol's decode. Every prefill here took the sink, and the risky case on unified memory, the graph's
  workspace taken back from a full tier, did not occur. The graph half ran after the review (see
  [U1: the graph half](#u1-the-graph-half)).
- **MemAvailable stayed above the floor the ledger published at each moment.** The floor was 4729-4731 MiB in turns
  1-4 (MemAvailable at least 4747 MiB then) and 4792 MiB in turn 5 (at least 4820 MiB). The floor rose 61 MiB in
  turn 5: the second cold prefill refaulted file pages, and the ledger counts them into its file reserve.
- No ledger refusal, no allocation failure. The tier gave memory back 9 times (`host_reclaim`, 1-121 slots) and once
  reactively at turn 5's prefill. Five checkpoints of 5 MiB each. The tier grew from 16 to 5616 slots after the
  first decode token, in 3.50 s; later turns regrew 2-160 slots.
- **The first measurement of the warm sink at 4104 tokens:** turn 5 prefilled in 29.16 s. The model gives the warm
  sink 29.81 s (+2.2 %); the cold graph takes 31.2-31.3 s. The cold sink of turn 1, 39.83 s, is predicted at 39.64 s.
- Turns 2 and 3 prefilled 283 and 279 tokens for follow-ups of 17-22: the same re-prefill of a reply as in the
  validation. The model gives the warm sink at 283 tokens 8.10 s against 4.99 and 4.63 s measured (+62 % and
  +75 %).
- Only the sink route ran here.

### Warm-up C0

`y9-c0.sh dsv-ysy9` on dev 6333f49 with no code change: cold p4096 and p512 on the graph route with A's settings,
`GGML_SCHED_MOE_LOG=1`, `GGML_KIMI_CPU_SINK_TRACE=1`, `GGML_KIMI_CPU_SINK_PROFILE=1` and
`GGML_MOE_TRACE_LAYER_GAPS` (written at exit to `~/ssdl/ab/c0/dsv-ysy9-p*-gaps.*.csv`), then a device-to-device copy
probe. The replay used the design's scripts (`wf-warm/replay.py`, `parse_trace.py`, `decompose.py` and
`wf-warm-review/seed_gate.py`'s engine rule) with two new ones in the session's scratchpad: `wf-ysy9/c0_counts.py`
(the true prefill counts and the split layout from the `moe_copy` lines) and `wf-ysy9/c0_replay.py` (the go rule).

| | p4096 | p512 |
|---|---|---|
| TTFT | 31.22 s | 17.11 s |
| decode 1-128 / 129-256 | 4.281 / 6.627 | 4.400 / 6.488 |
| the graph's expert copy | 38868 slices, 243.6 GB in 19.77 s (12.32 GB/s), no fallback | 29262 slices, 183.4 GB in 14.71 s (12.47 GB/s) |
| the prefill's union (true counts) | 12956 experts, 291-361 a layer | 9754, 205-315 a layer |
| routes in a layer's 137 most-routed experts | 83.5-96.3 %, mean 92.7 % | 81.8-97.1 %, mean 92.9 % |
| split layout | each layer's gate, up and down inputs in three splits of their own (120 expert splits) | the same |
| the prefill graph's splits | 30.24 s: the 120 expert splits' inputs 29.13 s, their compute launches 0.22 s; 3 CPU splits 0.59 s; 3 other GPU splits 0.30 s | 16.68 s: inputs 16.23 s, compute 0.10 s; CPU 0.15 s; other GPU 0.20 s |
| decode deficit (steps 1-127 against the 128-255 mean) | 10.50 s: growth stall 3.37, extra reads 5.47, step 1 0.49, the rest 1.17 | 9.29 s: 3.42, 4.41, 0.36, 1.11 |

The device-to-device probe on GB10 (`y9-d2d.cu`, `cudaMemcpyAsync` timed by events, 10 repeats): 114.6 GB/s for
1 GiB; 110.1 GB/s for 140 copies of one role (6266880 bytes, 57 us each), a layer's seed quota of one role.
Host-to-device from pinned memory: 59.0 and 58.0 GB/s.

The inputs of the expert splits take 29.13 s at 4104 tokens, but their reads only 19.77 s. The other 9.4 s is
waiting for the GPU: the scheduler reads a layer's routing from the GPU before it copies the layer's experts
(`ggml_backend_synchronize(ids_backend)`), so the copy starts only after all the GPU work before it. A seed copy
fenced before a split's compute is therefore on the critical path. With three splits a layer, all three roles are
exposed, not one as the design assumed.

The go rule (design C0): the reads C saves over A+B, minus the exposed copy, at least 1.3 s on DeepSeek. Per-layer
rolling quota of the free slots, the run's grown tier (5482 at p4096, 5532 at p512: the most the decode tier held in
these A-setting runs) as a stand-in for mechanism B's tier at load, which is not measured. During the prefill the
graph's workspace holds about 400 of those slots. Reads saved in seconds over steps 1-255 against the measured run,
priced with the run's own fit (126 + 164.7 ms/GB at p4096, 124 + 158.6 at p512):

| seed ranking | p4096: C over A+B, design / engine rule | copy | net | p512: C over A+B | copy | net |
|---|---|---|---|---|---|---|
| A+B, no seed: its reads saved | 0.77 / 0.77 | | | 0.78 / 0.78 | | |
| the union, no counts | 1.65 / 1.26 | 0.94 | 0.72 / 0.32 | 2.06 / 1.86 | 0.94 | 1.12 / 0.91 |
| **the true prefill counts** | **2.87 / 2.77** | 0.94 | **1.93 / 1.83** | **3.18 / 3.13** | 0.94 | **2.24 / 2.19** |
| the same prompt's bootstrap decode (proxy) | 3.83 / 3.92 | 0.94 | 2.89 / 2.98 | | | |
| the oracle (the reply's own steps 1-127) | 6.40 / 6.40 | 0.93 | 5.47 / 5.47 | 6.72 / 6.72 | 0.92 | 5.80 / 5.80 |

- **Seeding clears its bar:** 1.83-1.93 s net at 4096 and 2.19-2.24 s at 512, against 1.3 s. C4-C6 may go ahead.
- The verdict does not rest on the full quota. With 400 or 800 fewer seeds a layer-quota (the decode tier kept full),
  the true counts net 2.09 and 2.27 s at 4096 and 2.54 and 2.59 s at 512 (design and engine rule agree there): the
  copy shrinks faster than the saved reads.
- The copy is priced at the measured 110.1 GB/s for every seeded expert's three roles, 0.94 s at a full quota, an
  upper bound: it leaves out any overlap with the split's own GPU work. The design had assumed one role a layer at
  27-44 GB/s (0.8-1.3 s).
- The design's replay ran at depth 2 (95.6 GB of decode reads over steps 1-127). At depth 6, the manual best, they
  are 55.8 GB, and the true counts land between no counts and the same-prompt proxy, as the design bracketed.
- The growth stall (3.37-3.42 s, B's part) is not in these reads.
- Muse was not measured (not on ysy9). The design's gate asks for both models.

### After the review on ysy9

A review of the runs above found that the threshold 414 fits the workload as counted, not as served; that U1 had
run its sink half only; and some smaller overclaims. Four framework commits fix what can be fixed in the framework
(5c2fc47, 71994ff, f12f9f8, 59cf072; 60 unit tests pass). `y9-rev.sh` and `y9-rev2.sh` then ran on ysy9
(18:54-19:50 CST, the same lock, STOP check, one server and 3 GiB watchdog as above): four conversations, an ABBA
at p512 and U1's graph half.

#### The threshold and the follow-ups as served

The workload counted follow-ups of 22, 16 and 17 tokens, all below the 32-token offload minimum. So only the cold
4104-token prompt voted, and the threshold was the cold break-even, 414. In the validation, though, 7 of 12
follow-ups prefilled the reply before them again, 534-539 tokens on a warm tier (above). With the threshold at 414
those passes take the graph. No run had applied B's threshold to a conversation, so nothing had caught it.

5c2fc47 prices each follow-up in two forms of weight 1/2: its own new tokens, and the 512-token reply before it
prefilled again with them (the server's cache parts inside a reply whose text renders back in other tokens, and it
restores the checkpoint at the end of the previous prompt). The workload's passes then weigh 1 cold against 1.5
warm. The threshold for the same run is 2906 for `split` (3152 for `all`): the 4104 prompt stays on the graph, and
the re-prefills go to the warm sink. The model gives a 539-token re-prefill 10.3 s on the warm sink and 17.1 s on
the graph. Nothing else in `settings.env` changes (`settings-5c2fc47.env` in the run directory). The stored 233,
Spark Muse and Spark DeepSeek runs derive the same values.

Four conversations, each the protocol's 4-turn conversation on a fresh server with B's settings applied, 512
tokens a turn (`y9-conv.py`, the server's template, prompt cache on, EOS ignored):

| run | threshold | history | follow-up prefills: tokens, route, s | prefill, 4 turns | decode 129-512 |
|---|---|---|---|---|---|
| 1021-185357-y9-rev-conv-b414 | 414 | the replies as generated | 23, 17, 18: sink, 1.7, 1.5, 1.4 | 35.9 s | 6.79-7.28 |
| 1021-190030-y9-rev-conv-b2906 | 2906 | as generated | 23, 17: sink, 1.7, 1.5; 534: sink, 7.0 | 41.7 s | 6.82-7.21 |
| 1021-190707-y9-rev-conv-p414 | 414 | each reply altered at its start | 539, 535, 536: graph, 16.4, 16.0, 15.6 | 79.5 s | 6.95-7.35 |
| 1021-191418-y9-rev-conv-p2906 | 2906 | each reply altered at its start | 539, 535, 537: sink, 7.7, 7.1, 7.2 | 53.3 s | 7.00-7.08 |

- With the replies as generated, the two runs re-prefilled once between them (b2906's last turn). Whether a reply
  renders back in its own tokens depends on the reply, and the replies here differ from the validation's (another
  route and depth for the first prompt). To measure the misroute every time, the `p` runs put "Reply: " before each
  reply in the history (`y9-conv2.py --alter-replies`). The server then restores the checkpoint at the end of the
  previous prompt (cached 4100, 4634, 5164 tokens) and prefills the reply again: the same pass.
- **At 414 every re-prefill took the graph, 15.6-16.4 s; at 2906 the warm sink, 7.1-7.7 s.** 8.7 s a re-prefill,
  26.2 s a conversation. The first prompt took the graph in all four runs (31.2-31.5 s).
- The model prices the warm sink at these sizes 34-45 % slow (10.3 s), the known error (What remains), and the
  graph 4-10 % slow (17.1 s). Neither moves the threshold out of 539-4103, where every choice gives the served
  workload the same time.
- Each graph re-prefill took its workspace from the full tier: 146-155 slots (5595 to 5440 in turn 2) by
  `host_reclaim reason=host_reserve`, which decode refilled. Decode was not slower for it: 6.95-7.35 tok/s over
  129-512, against 7.00-7.08 with the re-prefills on the sink.
- Memory: no watchdog, no refusal, no allocation failure in any of the four. The lowest MemAvailable was 4562-4573
  MiB in the three runs without graph re-prefills, 11-18 MiB above the ledger's highest floor (4551-4554 MiB). In
  conv-p414, after three graph re-prefills, it touched 4552 and 4554 MiB for two seconds of turn 4's decode (4549
  in the client's sampler), while the floor had risen to 4556.8 MiB with refaulted file pages: 3-8 MiB under the
  floor, 1.5 GiB above the watchdog. No reclaim was logged at that moment.

#### The p512 ABBA at the new threshold

At 2906 the 523-token prompt takes the sink, cold, where B at 414 and the manual best take the graph. The model
priced that 0.24 s slower (17.24 against 17.00 s), and spark-b9a5 had measured the sink there 0.5 s slower (`all`,
depth 2). So the p512 ABBA ran again, A against B at 2906 (`y9-fw-dsv-ysy9-5c2fc47.settings`), two cycles,
`~/ssdl/ab/runs/1021-19*-y9-rev-p512-{ex,fw}-{1..8}` (`y9-rev-abba512.json`):

| run | arm | TTFT s | decode 1-128 | decode 129-256 | routes | lowest MemAvailable MiB | ledger floor MiB |
|---|---|---|---|---|---|---|---|
| 1021-192104 | A | 17.10 | 4.510 | 6.443 | 2 sink, 523 graph | 4751 | 4729 |
| 1021-192258 | B | 16.59 | 4.712 | 7.144 | 2 sink, 523 sink | 4584 | 4551 |
| 1021-192449 | B | 16.55 | 4.712 | 7.112 | 2 sink, 523 sink | 4586 | 4551 |
| 1021-192640 | A | 17.12 | 4.553 | 6.758 | 2 sink, 523 graph | 4747 | 4729 |
| 1021-192834 | A | 17.16 | 4.606 | 6.528 | 2 sink, 523 graph | 4759 | 4729 |
| 1021-193027 | B | 16.55 | 4.677 | 7.044 | 2 sink, 523 sink | 4569 | 4551 |
| 1021-193217 | B | 16.54 | 4.689 | 7.148 | 2 sink, 523 sink | 4582 | 4551 |
| 1021-193409 | A | 17.16 | 4.507 | 6.487 | 2 sink, 523 graph | 4763 | 4729 |

| prompt | A | B at 2906 | B against A, cycle 1 | cycle 2 | all | gate |
|---|---|---|---|---|---|---|
| 512 (523 tokens) | 17.14 s, 4.544 / 6.554 | 16.56 s, 4.697 / 7.112 | -3.2 %, +4.0 % / +8.0 % | -3.6 %, +2.8 % / +9.0 % | -3.4 %, +3.4 % / +8.5 % | pass |

- **The cold sink at 523 tokens is 0.58 s faster than the graph on ysy9** with `split` and depth 6, against 0.24 s
  slower in the model: the model's cold sink is 4 % slow at this size (17.24 against 16.56 s), as its warm sink is
  at a few hundred tokens. B at 414 had taken the graph (17.03 s in the first ABBA). The gate passes, with a margin
  now on both sides.
- The p4096 ABBA above stands for B at 2906 as it is: 4104 is above both thresholds, so every pass takes the same
  route, and the two settings differ in nothing else.
- Decode-path check, as for the first ABBA: A's arms part from the reference at 50, 8, 7 and 8 tokens, B's at 8,
  29, 29 and 8; every divergence is at a near tie in the teacher-forced log-probabilities (position 7: 0.04 nats,
  8: 0.12, 29: 0.07) or at the shared miss at 50. The tier was 5520 slots after its growth in every B arm, 5456-5472
  in A's; MemAvailable stayed 18-35 MiB above the floor; no watchdog.

#### U1: the graph half

The review was right that the U1 run above exercised the sink only. `y9-rev2.sh` step u1g ran the graph half:
`~/ssdl/ab/runs/1021-193602-y9-rev-u1g-stress`, the bare launch (no SETTINGS: the tier from one block, the 4096 MiB
margin, 8 checkpoints, context 131072, ubatch 8192, depth 2) with one override, `GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0`,
so that every pass of 32 tokens or more takes the graph. The client (`y9-u1g2-client.sh`): the 4096 prompt, then the
three follow-ups of the framework's `work-4turn.json` (1.4-2.5K tokens of text each), 256 tokens a turn, prompt
cache on; then the 32K prompt (`work-long.json`'s second entry) and the 4096 prompt, each as a new conversation on
the same server. Watchdog 3 GiB.

| turn | new tokens (cached) | route | TTFT s | decode 1-128 / 129-256 | lowest MemAvailable MiB | ledger floor MiB |
|---|---|---|---|---|---|---|
| 1 | 4104 (0) | graph | 31.28 | 4.186 / 6.169 | 4757 | 4729-4730 |
| 2 | 1784 (4100) | graph | 23.96 | 5.751 / 6.184 | 4747 | 4730 |
| 3 | 3421 (6139) | graph | 34.90 | 5.411 / 6.219 | 4605 | 4740-4741 |
| 4 | 2937 (9815) | graph | 32.52 | 5.338 / 5.593 | 4519 (at the next request's start) | 4749 |
| 32K, new | 32036 (0) | graph, 4 passes (8192 x 3, 7460) | 328.12 | 4.668 / 5.365 | 4430 | 5652-5975 |
| 4096, new | 4104 (0) | graph | 32.09 | 5.377 / 6.250 | 6340 | 5975-6318 |

- **No kill, and MemAvailable stayed above 3 GiB** (lowest 4430 MiB, 1.3 GiB above the watchdog). No ledger refusal,
  no allocation failure. Every follow-up took the graph after the tier had grown: each graph prefill took its
  workspace back from the tier (189-236 slots at 1.8-3.4K tokens, 470 and three reactive blocks at 8192), and decode
  grew it back to 5536-5632 slots. 10 checkpoints of 5 MiB. The ledger learned no margin (`margin_updates=0`), so
  there is no learned margin to run at.
- **MemAvailable fell below the floor the ledger published.** Briefly when a graph prefill took its workspace
  from a full tier: 128-135 MiB under for a second at turn 3's start, 230 MiB at the 32K request's start. Then for
  much of the 32K prefill: the ledger's file reserve grew from 0.68 GB before the request to 1.63 GB at pass 1 and
  1.97 GB at pass 4 as the prefill refaulted file pages (318 thousand refaults by its last pass), so the floor rose
  from 4750 to 5652 and then 5971 MiB, while MemAvailable was 4430-5881 MiB through the first three passes (by the
  sampler's steps). It was below the floor's pass-1 value for 182 of the prefill's 328 seconds, by up to 1221 MiB.
  In this log the tier gave slots back for the floor (`host_reclaim reason=floor`) only in decode; during the
  prefill only for the graph's buffers (`reason=host_reserve`) and reactively, so nothing restored the floor until
  decode. The 32K prefill took 328 s, where the prefill model gives the four graph passes 167 s. File pages (dense
  weights, Engram) falling out of the page cache are the likely cause: the README's Limits record the same slowdown
  for DeepSeek once the RAM cache has filled the memory (a 32k prompt 35-75 % slower, with 1.5 million major
  faults).
- **So U1's gate is not met on DeepSeek**: no kill and above 3 GiB, but not above the published floor, on the graph
  route with a long prompt. The launcher at 6333f49 already starts unified memory without the guard (design U1's
  commit); that removal stands on this run's no-kill result only. Keep the harness's 3 GiB watchdog on unified
  memory until the ledger holds its floor during a multi-pass graph prefill.
- Decode: the cold protocol turn (turn 1) decoded 4.186 / 6.169 tok/s at depth 2, against 4.176 / 6.453 in the sink
  half and 4.077 / 6.079 in spark-b9a5's depth-2 ABBA on the graph; the 4096 prompt after everything else 5.377 /
  6.250 on a warm tier. These are single arms, not a comparison; nothing in them suggests a loss.

#### The other findings

- **The route check flagged only a route's total** (71994ff). Now it flags each prefill more than 15 % off and sums
  per route and tier state. On the validation it flags all 6 warm prefills of the two static conversations. Its
  summary also gives each conversation's prefill seconds and mean decode rate, so that placements can be compared
  on decode when their follow-ups re-prefilled differently (The validation, above).
- **The plan chose a configuration priced below zero without complaint** (f12f9f8). It now sets aside any
  configuration whose total, prefill or decode rate is at or below zero, and every configuration of a placement
  whose layer table prices a class at or below zero, with the reasons in `plan.json`; with nothing left it stops.
  On the first decision's inputs it sets aside every `all` configuration (the singular fit priced an all-miss layer
  at -6.2 ms) and chooses `split` with 5624 slots at 358.0 s, as the fixed run did.
- **The ubatch probe compared two different quantities** (59cf072). b029cdd compared the most the tier held over 4
  probe tokens with the most it held over the bootstrap's 512, which includes 14-24 slots of watermark growth. And a
  discrete GPU grows its tier before it reports the allotment, so the maximum over 4 tokens could exceed the old
  reading. Now both sides read the tier after its first decode growth, and each placement's tier at a larger ubatch
  loses the difference: `all@16384` 5598 and `split@16384` 5576 slots (both 5585 before), 5358 and 5336 at 32768
  (5350). The plan's choice does not change. **M6 stays open:** on unified memory that reading sees only the buffers
  that stay allocated in decode, the host's compute buffer (586, 1684 and 5416 MiB at 8192, 16384 and 32768 tokens:
  48 and 288 slots). The GPU compute buffer (4640, 10456 and 29633 MiB) is taken back from the tier by every graph
  prefill and refilled by decode (design F_g); nothing prices it yet. conv-p414 shows the mechanism at ubatch 6400:
  146-155 slots a graph re-prefill.
- **Teacher-forced agreement cannot tell A from B, and the claim was limited** (Correctness on ysy9, above, with the
  decode-path check from the ABBA's replies).
- **C0's quota wording** (Warm-up C0, above): the 5482 and 5532 slots are the C0 runs' grown tier, a stand-in for B's
  tier at load; with fewer seeds the net is larger.

### What remains

- **Not run on ysy9:** Muse (not copied, on purpose); U1's run at a learned margin, since no run learned one;
  prompts near n* on both routes ([ask]).
- **U1's floor during a graph prefill** (addressed by b704fad, see the warm-up section below). On the graph route
  MemAvailable fell below the ledger's published floor,
  by up to 1.2 GiB during a 32K prompt (After the review on ysy9). The ledger reclaimed for its floor only at decode
  token boundaries, and the floor rose with the file pages the prefill refaulted. The launcher already starts unified
  memory without the guard; the harness's 3 GiB watchdog stays until the ledger holds its floor through a multi-pass
  graph prefill, and U1 runs again.
- **DeepSeek's derived values stay pending.** The Spark ABBA accepts them on the Spark. The deployment's change waits
  for D1-D4 on 125 (formerly 176).
- **Teacher-forced 61/64 on eng-dev**, one miss more than eng-route's 62/64 on spark-b9a5, for the manual best and
  the framework alike. The source changes between them touch no forward pass, so the build's flags or the machine
  are left. The same check of both builds on one machine would tell which. A teacher-forced check after a request
  that grows the tier would cover the decode path token by token; the ABBAs' greedy replies cover it only up to
  their first divergence. (`probe-tf-decode.py` does that since the warm-up's review, below.)
- **The warm sink at a few hundred tokens is predicted 35-75 % slow**, and the cold sink at 523 tokens 4 % slow.
  The model takes the tier's share of a pass's experts from the warm prefill of 4616 tokens (41 %); the experts of a
  short pass are mostly the hot ones the tier holds. A share that follows n, from the decode trace as U(n) does,
  would fix it. It biases the warm break-even low; for this workload the threshold sits in 539-4103, where every
  value gives the served workload the same time, so the error does not move a route here.
- **A reply that renders back in other tokens** (the server keeps the reply's tokens since 0d4a174 and 403fa3e, see
  the warm-up section below). 7 of the validation's 12 follow-ups and 2 of U1's 3 prefilled the
  previous reply again (279-539 tokens); with B's settings, 1 of 6. The threshold now prices both forms (5c2fc47).
  The plan's time still counts each follow-up as its own tokens (README, Limits), and the framework does not learn
  how often a model's replies do this.
- **The graph's workspace taken from the tier is not priced** (design F_g, M6): 146-155 slots a graph re-prefill at
  ubatch 6400, 189-236 at 1.8-3.4K tokens and 470 at 8192 on the bare launch, regrown in the next decode.
- **C1-C6 are not built** (C1-C5 since are, see the warm-up section below). C0 says seeding pays on DeepSeek. Its exposed copy is three roles a layer, so C5's net
  gate should be priced that way. A fence at the next use of the copy's source, instead of before the split's
  compute, would let the copy overlap the GPU work that follows.

## Warm-up, the floor in a prefill and reply tokens on ysy9

The evening of 10-21 (20:20-23:40 CST) ysy9 ran the warm-up design (`wf-warm/design.md`: A, B, C), a fix for U1's
floor and the server's reuse of reply tokens, each against the build before it. Everything ran under
`flock ~/ssdl/spark.lock`, one server at a time, with `~/ssdl/STOP` checked before each step and the 3 GiB watchdog;
it never fired. Each arm is a cold server (`w2-launch.sh`: `dsv-launch.sh` plus a sampler of MemAvailable against the
floor the ledger last published, 4 Hz, `memfloor.txt`, and the run's vmstat counters). B is the framework's derived
settings with the threshold 2906 (`y9-fw-dsv-ysy9-5c2fc47.settings`) unless a row says otherwise.

A review of this work followed (After the review of the warm-up, below): it changed A's first price, the reply
reuse's matching and the seeds' statistics, and found that the teacher-forced check below runs no decode step. The
numbers in this part are those of the builds in the table; the ones after the review are in their own part.

| build | commits | what it adds |
|---|---|---|
| `eng-base` | 8251825 | dev: route per ubatch, batch 2, M8, the ysy9 framework fixes |
| `eng-warm` (until the review) | 40f0585, 0c77461, 36220a2 | A: admissions priced from the first decode token's second layer; B: the tier's storage at load |
| `eng-seed2` | + 589ea94, 7569720 | C: seeds from the graph prefill's expert copies (built from an earlier C with the prefetch marks' call removed; it behaves as 7569720) |
| `eng-floor` | + b704fad | the ledger's floor held during a prefill (`w2-u1g-floor` and `w2-ab6` ran its first version) |
| `eng-reuse` | + 0d4a174 | a follow-up keeps the cached tokens of the reply it repeats |

### A and B: the tier from the first decode token

- **A.** The tier measures an upload when it gets its storage (one slot's roles from pageable memory, 337 us for
  18.8 MB here) and prices admissions from the layer after the first that reports a miss. The first
  `[kimi_vram_admission]` line now comes in decode token 1 (`cost_hits` 0.06-0.07, priced with that layer's cold miss
  cost); before, in token 3 or 4. MIN_COUNT decides only the first decode layer of a tier's life. (The review moved
  the first price to the second token's boundary, 7d7a347.)
- **B.** At load, after the worst-case graph reserve, the tier grows into what the ledger leaves:
  `decode_grow ... decode_token=0`, 16 to 5232-5248 slots in 2.93-3.00 s. The first request's graph takes ~300 slots
  back, and decode token 2 regrows them in 0.17-0.20 s. Before, token 2 grew 16 to 5504-5536 slots in 3.09-4.06 s.
  The server's load is about 3 s longer (load_s 40-42 against 36-40).

Tier hits per decode step (traced p4096 arms `w2-tr2-p4096-eng-base`, `-eng-warm`, `w2-tr4-p4096-eng-seed`; 240
routed experts a step):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | tier share 1-127 | read GB 1-127 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 15 | 102 | 68 | 107 | 105 | 144 | 157 | 0.780 | 57.2 |
| A+B | 0 | 77 | 78 | 120 | 128 | 131 | 136 | 121 | 160 | 160 | 0.808 | 51.1 |
| A+B+C | 122 | 192 | 170 | 212 | 195 | 196 | 202 | 189 | 209 | 207 | 0.897 | 43.4 |

The design's replay for A+B gave 74, 70 and 118 hits at steps 2-4 (at depth 2; these runs are at depth 6).

**The tier's threads.** The first A+B build decoded tokens 129-256 9 % slower than base (ABBA `w2-ab1`, 2 cycles:
-7.4 % and -10.5 %). The tier's worker and upload threads take the affinity of the thread that creates them. A sink
call pins its caller to the first compute CPU, so they used to run there; created at load by the unpinned context
constructor they ran anywhere. Profiled arms (`w2-pf-p4096-*`, `GGML_KIMI_CPU_SINK_VRAM_TIMING=1`) showed it: the
launch-to-worker handoff took 1.59 s against 0.51 s, the sink's wait for the GPU batch 11.8 s against 5.3 s, uploads
563 against 396 us. The load call is now pinned as the sink's calls are (36220a2).

ABBA, base against A+B, 2 cycles (`w2-ab3-*`, `w2-ab3.json`):

| prompt | base: TTFT, decode 1-128 / 129-256 | A+B | cycle 1 | cycle 2 | all |
|---|---|---|---|---|---|
| 4096 (graph) | 31.21 s, 4.527 / 7.143 | 31.30 s, 5.281 / 7.181 | +0.66 %, +17.9 % / +0.2 % | -0.06 %, +15.5 % / +0.9 % | +0.30 %, +16.7 % / +0.5 % |
| 512 (sink) | 16.46 s, 4.654 / 7.059 | 16.49 s, 5.458 / 7.108 | +0.30 %, +17.8 % / +1.0 % | +0.09 %, +16.8 % / +0.4 % | +0.20 %, +17.3 % / +0.7 % |

- Decode 1-128: 4.0 s less at 4096 (28.3 to 24.2 s) and 3.7 s at 512; the design expected about 3.7 s.
- TTFT moved by +0.09 s and +0.03 s, inside the 1.5 s noise of the protocol (one A+B arm read 31.65 s).
- MemAvailable against the published floor (4 Hz): in this ABBA every A+B arm stayed at least 15 MiB above it; one
  base arm touched -125 MiB for one sample. Over all twelve A+B arms of `w2-ab3`, `w2-ab4` and `w2-ab5`, one sample of
  one arm was 114 MiB under (`w2-ab4-p4096-a-4`, early decode); over the base arms of `w2-ab1` and `w2-ab3`, two
  (-207 and -125 MiB). The floor gate is therefore not met in the strict sense, by single samples, in base as in A+B.
  Two of these three arms are among the three with direct reclaim (below); the floor watcher below is for them.
- Teacher-forced with the prompt on the graph (the threshold 414, `w2-tf-eng-*-414b`): base, A+B and A+B+C read
  61/64 with the same rows (misses at 5, 8 and 50, the same log-probabilities). At 2906 the 523-token prompt takes
  the sink, whose CPU/GPU split follows measured times, and the misses move between near ties from run to run (base
  60/64 and 59/64 for A+B, positions 12, 17 and 21 within 0.1-0.4 nats). This check asks for one token per position,
  so every prediction comes from prompt processing and no decode step runs: it cannot see A, B or C (the seeded
  build whose decode failed with NaN read 61/64 there too). The check through decode is in the review's part.
- Every A+B arm's greedy reply at 4096 equals base's for all 256 tokens.
- **Three slow growths at token 2.** Three arms grew 4.06 s (base, `w2-ab3-p4096-a-1`), 1.92 s and 2.30 s (A+B,
  `w2-ab4-p4096-a-4`, `w2-ab5-p4096-a-1`) instead of 3.1 s and 0.16-0.20 s. They are exactly the three p4096 arms of
  `w2-ab1` to `w2-ab7` in which the kernel reclaimed pages directly (`pgscan_direct` +101589 to +157306 against 0, or
  14624 in one normal arm; `pgscan_kswapd` +0.5 to 1.5 M). MemAvailable counts the page cache, and a build or a
  debug run of another checkout had just left some: the growth paid for its reclaim. B does not cause it, and base
  shows it too.

### C: seeds from the graph prefill

At p4096 the prompt takes the graph. The scheduler offers each layer's gate, up and down expert copies with the
pass's routing counts; the tier plans a layer's share of its empty slots (free / layers left) with the most-routed
experts and copies their roles device to device. Every free slot was seeded: 5216-5232 seeds, 15648-15696 role
copies, 98 GB, 116-124 ms of host time in the callbacks.

**The fence.** The first build fenced the backend's stream at the next split, so that the copies overlapped the
split's compute. Its decode failed at step 1 (`invalid CPU sink expert id or weight` at layer 3: NaN). A debug build
compared seeded roles with the file: right after each copy they matched, but at publish 210 of 3924 checked roles
held other bytes. A node after the expert matmul can reuse the input's memory within the same split. The fence is
now before the split computes, so the copies are exposed (C0 priced that at 0.94 s), and every arm decoded normally.

**The prefetch.** A build that also marked the seeded experts as tier hits for the next-layer prefetch kept the
prefetch from reading them after decode had evicted them unused: its reads issued fell from 13695 to 6954 and its
hits from 4139 to 1327, and tokens 129-256 were 3.8 % slower than A+B (`w2-ab4`). The committed C leaves the prefetch
alone.

ABBA, A+B against A+B+C at 4096, 2 cycles (`w2-ab5-*`, `w2-ab5.json`):

| | TTFT | decode 1-128 | decode 129-256 |
|---|---|---|---|
| A+B | 31.27 s | 5.111 | 7.173 |
| A+B+C | 32.27 s | 6.135 | 7.102 |
| cycle 1 / 2 | +2.9 % / +3.5 % | +25.1 % / +15.3 % | -1.1 % / -1.0 % |

- The first A+B arm (4.526 tok/s) is one of the slow growths above. Without it: decode 1-128 +15.6 % (5.307 to
  6.135 tok/s, 3.2 s less), 129-256 0.2 s more, TTFT 1.0 s more (the exposed copy): net 2.0 s for the first request
  after load, against the replay's 1.8-2.2 s. With it the cycles read +25.1 % and +15.3 %, net 3.0 s.
- Three of the four seeded arms' replies equal base's for all 256 tokens; one parts at character 1155, and one A+B
  arm of the same ABBA parts at character 552. No log-probabilities were recorded there; the check through decode
  after the review covers it.
- The 512 prompt takes the cold sink at 2906, which the graph producer does not touch.

### The ledger's floor during a prefill

U1's graph half failed the floor (After the review on ysy9): the floor rose with refaulted file pages during a 32K
prompt's four graph passes, and the tier restored it only at decode token boundaries or when an allocation asked
through the ledger. Now (b704fad) a higher floor is published only after the tier gave back what it adds, and a
watcher in the sink measures the ledger ten times a second and has the tier give back blocks when MemAvailable is
under the floor, the RAM cache's demand and 64 MiB. It takes the sink's run lock without waiting, so in decode it acts
between layers.

The gate run repeats U1's graph half on A+B+C and the watcher (`w2-u1g-floor2`): the launcher's bare unified-memory
defaults (the tier from one block and grown at load, the 4096 MiB margin, 8 checkpoints, context 131072, ubatch 8192,
depth 2) with `GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0`, the client `y9-u1g2-client.sh` (the 4096 prompt and work-4turn's
three long follow-ups, then the 32K prompt and the 4096 prompt as new conversations, 256 tokens a turn).

| turn | new tokens (cached) | TTFT s | decode 1-128 / 129-256 |
|---|---|---|---|
| 1 | 4104 (0) | 32.22 | 5.826 / 6.355 |
| 2 | 1524 (4359) | 21.94 | 5.660 / 6.148 |
| 3 | 3421 (6138) | 33.52 | 5.489 / 6.100 |
| 4 | 2937 (9814) | 32.48 | 5.546 / 5.616 |
| 32K, new | 32036 (0), 4 graph passes | 263.78 | 4.719 / 5.352 |
| 4096, new | 4104 (0) | 31.27 | 5.398 / 6.108 |

- No kill. The gate, MemAvailable never under the published floor, is nearly met: of 2759 samples at 4 Hz, one was
  under it, by 56 MiB, at 650.8 s, the start of the last request's prefill (an allocation burst, back above it at the
  next sample, while the floor rose from 5044 to 5596 MiB with refaults). The floor went from 4728 to 5626 MiB; the
  lowest MemAvailable was 4738 MiB. The tier gave back blocks for the floor 13 times.
- Before: 182 s of the 32K prefill under the floor, by up to 1221 MiB (`1021-193602-y9-rev-u1g-stress`). That run
  is `eng-dev` 6333f49, not this branch without the watcher, and a single run each; the control on this branch is
  in the review's part. A first version of the watcher (four times a second, the floor published before the tier
  made room, `w2-u1g-floor`) left 5 single samples under it, down to -448 MiB at a floor jump.
- The 32K prefill took 263.8 s against 328.1 s in that run of another engine; the control decides what the watcher
  changes.

ABBA, A+B+C against A+B+C and the watcher at 4096, 2 cycles (`w2-ab7-*`, `w2-ab7.json`): TTFT 32.28 against
32.16 s (-0.4 %), decode 1-128 6.157 against 6.156 (0.0 %), 129-256 7.100 against 7.041 (-0.8 %: -0.5 % and -1.2 %),
inside the protocol's noise (3 % for decode). The watcher at four times a second and outside decode (`w2-ab6`) read
-0.2 %, +0.5 % and +0.3 %.

### The server keeps the reply's tokens

A follow-up's history renders the previous reply as text, and a greedy reply's text does not always tokenize back to
the tokens it was generated as. The server then parted from its cache inside the reply, restored the checkpoint at
the end of the previous prompt and prefilled the reply again. Now (0d4a174), when a completion is assigned to a slot,
the prompt's tokens that spell exactly the text of the slot's cached tokens, up to the furthest offset where both end
a token, are replaced by the cached ids (`server_tokens_reuse_text`, `tests/test-server-text-reuse.cpp`: 7 cases,
pass on ysy9). After the review (403fa3e) only normal and byte tokens match by text; a control or user-defined
token, or one without text, matches only its own id where both texts end a token, so a control token and its
literal spelling never trade places and no cached token the prompt lacks is taken. The same conversation can now
tokenize differently depending on what the slot holds: the KV then holds the ids the model produced, for the same
text.

The protocol's 4-turn conversation (`y9-conv-client.sh`: 512 tokens a turn, EOS ignored, the replies as generated)
with the launcher's bare defaults (every prefill on the sink, depth 2, 8 checkpoints), before and after:

| follow-up | before (`w2-conv-bare-eng-floor`): prefilled, cached, TTFT | after (`w2-conv-bare-eng-reuse`) |
|---|---|---|
| 2 | 539, 4100, 7.52 s | 23, 4615, 1.71 s |
| 3 | 534, 4634, 7.06 s | 17, 5149, 1.51 s |
| 4 | 534, 5164, 6.54 s | 18, 5677, 1.48 s |

- Before, every follow-up restored the checkpoint at the end of the previous prompt and prefilled the 511-token reply
  again. After, the cache grows by each reply (4104 + 511 = 4615, + 23 + 511 = 5149, ...) and only the new tokens are
  prefilled: 16.4 s of the three follow-ups' 21.1 s of TTFT are gone. The server's line reads, for example,
  `shared 4489 -> 4615 of 4615 cached tokens`: the ids parted 385 tokens into the first reply, every turn.
- The replies after reuse stay on topic and well formed (three headed answers about the document, `mt.json`).
- With the derived settings (threshold 2906) neither conversation re-prefilled (`w2-conv-2906-*`: 23, 17 and 18 new
  tokens each); whether a reply tokenizes back differently depends on the reply.

### After the review of the warm-up

Two reviews (10-22) found two real defects, gates that did not test what they claimed, and claims wider than the runs.
Six commits on top of the branch above (dcc0773), then every gate they can affect again on ysy9 (night of 10-21 to
10-22, 00:50-04:00 CST, the same rules: one lock, `~/ssdl/STOP`, the 3 GiB watchdog; each ABBA now holds the lock
across its eight arms, so no other run comes between them).

| commit | what |
|---|---|
| 7d7a347 | A prices at token boundaries only: the first price comes at the second decode token from the first token's misses, not at the first token's second layer from one layer's |
| 403fa3e | reply reuse: a control or user-defined token, or one without text, matches only its own id |
| eccdaa3 | a seed decode selects becomes an ordinary entry admitted at that selection's count (design 3.3) |
| aec730c | one identity for a role's payload (`kimi_vram_source_identity`), used by the sink's key and the seed's |
| f593262 | test: the seed quota, its roll-over and residency |
| a485f71 | the tier logs the CPUs its threads inherit |

`eng-warm` is now this branch (a485f71; its binaries are 965ca26's, the same code with two comments different).
`eng-base` is dev 8251825 as before.

**Scope.** A, B and C act on the first request after the server loads. B moves only the first request's tier growth
to the load. C seeds only empty slots, and after the first reply the tier is full: the U1 run seeded once in six
requests, and the later cold conversation got no seed. A prices free slots, which exist only before the tier first
fills. Every ABBA arm is a fresh server, so "a cold request" above means the first request after load. The cost is
paid at load: about 3 s more of it (load_s below).

#### A: the first price

At the first token's second layer A priced every admission of the first token with that one layer's miss cost
(4093-5917 us in every arm, `hit_us` 0). On the Spark it did not matter (min_count 0 instead of 0.25); on a discrete
GPU with an upload of two settled misses it would have admitted every count-1 miss of the first token. Now the first
token is under MIN_COUNT, and the first price comes at the second token's boundary from the miss cost over the first
token, as every later price comes (7d7a347).

What the runs show: the first token's average is no lower than its first layer. `updates=1` reads `miss_us` 4993.6
(`w2r-vfy-p4096`) and 3866-4960 us (`w2r-n2` B arms); base's first price, at about the fourth token, reads 1284-1313
us. The first token's misses are first touches, all read from the drive; from about the fourth token recurring experts
come from the RAM cache. So the second token is still priced with an expensive miss. That is what a miss costs then,
and dev already prices the second token of every later request the same way: 233's stored 4-turn conversation reads
`miss_us` 2.3-6.6 ms from token to token, `cost_hits` 0.57-1.77. On 233 the first request now behaves like dev's later
ones; whether its early count-1 admissions pay there is part of the pending 233 ABBA (with the admissions of tokens
1-3). On the Spark the price changes nothing: min_count is 0 from the second token either way.

#### Correctness through decode

`probe-tf.py` asks for one token per position (`n_predict` 1): every prediction comes from prompt processing, and no
decode step runs. The gate's runs show it: the seeded build whose decode failed with NaN read 61/64 there, the same
as base. `probe-tf-decode.py` sends request k with the prompt and the reference tokens up to 2k+1, forces token
2k+1 as the first generated token (`logit_bias`), and reads from the raw logits (`post_sampling_probs` false) the
prompt pass's prediction of position 2k+1 and the decode step's of position 2k+2. The cache then holds the next
prompt but its last token: each request prefills one token and decodes one, with no rollback. 64 requests cover
positions 1-128, the even ones from a decode step. The derived settings with the threshold 414 put the 523-token
prompt on the graph, so request 0 seeds.

| build | decode positions | prefill positions | tier hits / uploads | seeds published / freed unused |
|---|---|---|---|---|
| base (`w2r-tfd-eng-base`) | 62/64 | 64/64 | 840 / 51 | - |
| branch (`w2r-tfd-eng-warm`) | 62/64 | 63/64 | 22170 / 4311 | 5216 / 5018 |

- Both decode misses are base's own (positions 50 and 82, the same two tokens on top in both builds: base 0.98 and
  0.70 nats apart, the branch 0.81 and 0.25).
- The branch's prefill miss is a near tie of the same two tokens: position 29, ' operates' against ' ' (base -0.43
  and -1.05, the branch -0.85 and -0.56).
- Both runs decoded 64 times (`decode_active=1` 64 times). The branch hit the tier 26 times as often, the seeds
  included: 198 of 5216 were hit by the first decode step, the rest freed at the next prefill.

The ABBAs' greedy replies (below) add the free-running view.

#### Seeds byte for byte

A debug tier made from the committed one (`mk-verify.py`, not committed) compared every published seed's three roles
with the file (O_DIRECT reads through the resolver the seed keys come from), at publish and again at the second
decode token's begin for every seed slot that still held its key.

| run | seeds | at publish | after decode token 1 |
|---|---|---|---|
| p4096 at 2906 (`w2r-vfy-p4096`, one graph pass) | 5216 | 5214 equal, 0 differ | 5214 equal, 0 differ |
| 32K at the bare defaults (`w2r-vfy-32k`, 8 graph passes of 6400 and 1792 tokens) | 5152 (16 cancelled: a role not offered) | 5149 equal, 0 differ | 5149 equal, 0 differ |

Two and three seeds could not be read back by the check (an O_DIRECT read past the end of its file); none differs.
The check's reads put 26 s into each TTFT, so these runs time nothing.

#### ABBA, base against the branch

Arms alternate A B B A A B B A, base (A) against the branch (B), the derived settings (2906), each arm a cold server.

**n2: the first request after load and a new conversation after it** (`w2r-n2-*`, `w2r-n2.json`): the p4096
protocol turn, then on the same server a new conversation with another text (`work-4turn.json` entry 2, the README's
first part, 3420 tokens, the graph), 256 tokens each, EOS ignored (`y9-conv2.py --turns 1 --new`).

| request | base: TTFT, decode 1-128 / 129-256 | branch | cycle 1 | cycle 2 |
|---|---|---|---|---|
| 1, p4096, first after load | 31.33 s, 4.556 / 7.150 | 32.26 s, 6.146 / 7.079 | +2.8 %, +34.1 % / -1.3 % | +3.1 %, +35.7 % / -0.7 % |
| 2, new text, same server | 28.83 s, 6.723 / 7.177 | 28.87 s, 6.723 / 7.082 | +0.2 %, +0.1 % / -1.6 % | +0.1 %, -0.1 % / -1.0 % |

- Request 1: decode 1-128 takes 7.2 s less (27.9 to 20.7 s), 129-256 0.2 s more, TTFT 0.9 s more (the exposed seed
  copy): 6.1 s less of a 77 s request (7.9 %). The server's load takes 42 s against 38-40 s: B's growth at load.
- Request 2 gains nothing, as the scope says: the tier is full after request 1, its prefill seeds nothing (one seed
  publish per server, `freed_unused=0`), and the branch prices from the first measured costs as dev does by then.
- Decode 129-256 is 0.7-1.6 % slower in all four comparisons, inside the protocol's 3 % noise but of one sign. The
  pre-review C ABBA read -1.0 % and -1.1 % there. The likely cause is the prefetch's stale flag (What remains). The
  long-reply ABBA below measures whether it lasts.
- The first price (`updates=1`) comes at the branch's second token with `miss_us` 3866-4960; base's, at about the
  fourth token, with 1284-1313. The branch's miss estimate is down to 784 us by token 12 (`w2r-n2-p4096-b-2`).
- Every token-2 regrowth took 0.17-0.18 s on the branch (3.07-3.13 s on base), and no arm reclaimed directly.
- Floor: the branch's arms stayed 7-19 MiB above the published floor at their lowest; one base arm was under it in
  76 samples, by up to 41 MiB.
- Greedy replies. Request 1: six of the eight arms, three of each build, give the same 256 tokens; one base and one
  branch arm part at the same character, 973 ("buffer and flush readings into" against "buffer and write"). Request
  2: base's four arms agree; three branch arms part at character 123 ("16 GiB RAM, 16 threads (...)" against the same
  two items in the other order), the fourth agrees. Both are choices between equal wordings, where the GPU and CPU
  expert paths differ in the last bits; the check through decode above found only such near ties.

**cv: the protocol's 4-turn conversation** (`w2r-cv-*`, `w2r-cv.json`; `y9-conv-client.sh`: 512 tokens a turn,
EOS ignored, prompt cache on):

| turn | base: new tokens, TTFT, decode 1-128 / 129-256 / 129-512 | branch | all arms, B against A |
|---|---|---|---|
| 1 | 4104, 31.31 s, 4.557 / 7.110 / 7.023 | 4104, 32.31 s, 6.159 / 7.098 / 7.033 | +3.2 %, +35.2 % / -0.2 % / +0.1 % |
| 2 | 23, 1.76 s, 6.356 / 6.598 / 6.694 | 23, 1.71 s, 6.329 / 6.649 / 6.747 | -2.9 %, -0.4 % / +0.8 % / +0.8 % |
| 3 | 17 or 534, 2.94 s, 6.532 / 6.883 / 6.900 | 17, 1.56 s, 6.441 / 6.804 / 6.882 | -47 %, -1.4 % / -1.2 % / -0.3 % |
| 4 | 534 in 3 of 4 arms, 5.40 s, 6.795 / 7.194 / 7.114 | 18, 1.51 s, 6.780 / 7.176 / 7.079 | -72 %, -0.2 % / -0.3 % / -0.5 % |

- Turn 1 is the first request after load: +35 % in decode 1-128 as in n2.
- Turns 2-4: base prefilled the previous reply again in 4 of its 12 follow-ups (534 tokens, 6.6-7.1 s); the branch in
  none, its reply reuse fired in the 3 follow-ups where the reply tokenized back differently (`shared 5364 -> 5677`,
  `4999 -> 5149`, `4999 -> 5677`), with the stricter match on the real vocabulary.
- Decode in turns 2-4 is within 1.4 % over all arms (within 2.8 % per cycle), inside the 3 % noise: the warm turns
  neither gain nor lose from A, B or C.
- Floor: every arm's lowest sample was 2-17 MiB above the published floor; none fell under it.

**p5: the p512 turn** (`w2r-p5-*`, `w2r-p5.json`; the 523-token prompt takes the cold sink at 2906, so no seeds):
TTFT 16.47 against 16.49 s (+0.5 % and -0.2 %), decode 1-128 4.715 against 5.421 (+15.6 % and +14.4 %: 3.5 s less),
129-256 7.077 against 7.073 (-0.9 % and +0.8 %). Before the review A+B read +17.3 % here; the first token no longer
admits, which is about the difference. The branch's first price reads `miss_us` 3456-3635 at its second token.
Every arm stayed above the floor (lowest margin 15-42 MiB).

**lg: a 1024-token reply** (`w2r-lg-*`, `w2r-lg.json`; `w2r-long.py`: the p4096 protocol turn, EOS ignored), to see
whether the 129-256 loss lasts:

| window | base (tok/s) | branch | cycle 1 | cycle 2 |
|---|---|---|---|---|
| TTFT | 31.24 s | 32.17 s | +3.1 % | +2.8 % |
| 1-128 | 4.584 | 6.180 | +33.9 % | +35.7 % |
| 129-256 | 7.177 | 7.111 | -1.1 % | -0.8 % |
| 257-512 | 7.040 | 7.019 | -0.1 % | -0.5 % |
| 513-1024 | 6.778 | 6.921 | -7.1 % | +11.5 % |

The loss is about 1 % in 129-256 and 0.3 % in 257-512; in 513-1024 the arms of either build spread from 6.2 to 7.9
tok/s, so that window shows nothing either way. Up to token 512 the branch takes 5.9 s less than base (113.1
against 107.2 s, TTFT included) for the first request after load.

#### The floor, with a control

U1's graph half as before (bare defaults, every prefill on the graph, `y9-u1g2-client.sh`), once on `eng-seed2` (the
branch before b704fad: A, B and C without the floor watcher) and once on the branch (`w2r-u1g-*`). Single runs.

| | control (`eng-seed2`) | branch |
|---|---|---|
| MemAvailable under the published floor (4 Hz samples) | 322 of 2766, in 3 episodes: 20.5 s from 80 s (to -146 MiB), 59.4 s from 354 s (the 32K prompt's start, to -114 MiB), one sample at 647 s (-55 MiB) | 1 of 2757, at 303 s (-51 MiB) |
| floor (MiB), lowest MemAvailable | 4746-5731, 4600 | 4728-5738, 4677 |
| 32K prefill (4 graph passes) | 263.3 s | 261.9 s |
| turns 1-4 TTFT | 32.40, 21.94, 33.56, 32.45 s | 32.50, 21.96, 33.61, 32.47 s |
| watchdog | not fired | not fired |

- The watcher holds the floor through the multi-pass prefill: 80 s under it, by up to 146 MiB, becomes one sample.
  The floor gate (never under) is still not met by that one sample, as in `w2-u1g-floor2` (-56 MiB): an allocation
  at a request's start can take what the tier left until the next tick.
- The 32K prefill takes the same time with and without the watcher. The 328 s of the review's run were `eng-dev`
  6333f49 (the tier from one block, no B), not this branch: the watcher does not make the prefill faster, and the
  earlier sentence that it does is withdrawn.

#### Reply reuse, stricter

The bare-defaults conversation again on the branch (`w2r-conv-bare-eng-warm`): follow-ups prefilled 23, 17 and 18
tokens with 4615, 5149 and 5677 cached, TTFT 1.81, 1.59 and 1.45 s, as `eng-reuse` did. This time every reply
tokenized back as generated, so the reuse did not fire; in the conversation ABBA above it fired three times.

On the real vocabulary, deterministically (`w2r-reuse-test.py`, one cold server each, bare defaults):

| case | 0d4a174 (`w2r-reuse-eng-reuse`) | branch (`w2r-reuse-eng-warm`) |
|---|---|---|
| " acknowledgement" cached as 5 tokens, the next prompt's text has it as 1 | `shared 20 -> 25`: 7 new tokens prefilled | the same |
| the end-of-sentence control token (id 1) cached, the next prompt has its spelling as 12 literal tokens | `shared 17 -> 18`, prompt 34 -> 23 tokens: the control token took the literal text's place, 5 prefilled | no match: the 34 tokens prefilled as the prompt has them |

The second row is the review's defect, reproduced on the device and gone on the branch.

`tests/test-server-text-reuse.cpp` (7 groups: a reply tokenized back differently, partial agreement, no common
boundary, a different text, an extension and a short prompt, tokens without text, control tokens) passes on ysy9.

### Discrete GPUs (125, 233): from the code

Neither machine ran (both busy); their ABBA against dev is pending. From the code, after the review:

- **A** prices from the second decode token with the probed upload and the first token's misses; the first token is
  under MIN_COUNT, as in dev. Dev prices from the boundary after the first measured upload (about the fourth token).
  So on a discrete GPU the second and third tokens are now priced, with the first tokens' expensive misses: on 233 an
  upload of 4.6 ms against a miss of up to about 5-6 ms in those tokens can admit count-1 misses into free slots,
  where the settled price (miss 2.2-2.6 ms) refuses them. Dev does the same at the second token of every later
  request (233's stored conversation, above). Whether it pays there is measured by the pending ABBA. The probe is one
  slot copied twice at allocation (about 10 ms over PCIe).
- **B** returns at once on a discrete GPU. The only change there: the sink is created, pinned like its calls, at the
  end of the context's construction instead of in the warm-up pass; the tier and its threads are still created by
  the first prefill or decode call, which pins its caller (the threads' CPUs are now logged).
- **C** seeds empty slots of any graph prefill that finds a table. For MXFP4 (DeepSeek) the first prefill has no
  table (the arena comes at the first decode), so the first request is not seeded. A later request's prefill seeds
  whatever slots are still empty, which on a discrete GPU, where the cost rule fills slots only with experts worth
  their upload, can be many after a short reply: the device-to-device copy is then exposed before each split
  computes, and the seeds' first decode hits pay for it. For Q2 experts (IQ2_XXS/Q2_K) the cold prefill allocates the
  table, so the first graph prefill is seeded as on the Spark; and Q2's own cold admissions now carry the seed bit:
  out of the earned statistics until decode selects them, freed at the next prefill if it never does. No Q2 model ran.
- **The floor watcher** starts only on unified memory.
- **The reply reuse** is the server's and helps every device.

### What remains of the warm-up

- **125 and 233:** the ABBA against dev of the p512 and p4096 protocol turns with a second request on the same
  server (as `w2r-n2`), the 4-turn conversation, and an A-only arm (A is the only part that acts on DeepSeek there);
  report the first price (`updates=1`) and the admissions of decode tokens 1-3. The elastic test's discrete branch
  (B returns at once) has run only where the device is unified. Both machines were busy.
- **Muse** (not on ysy9): A and B apply as they are; C's net there was -0.4 to +1.1 s in the design's replay.
- **Q2 experts:** the first graph prefill seeds on a discrete GPU too, and Q2's cold admissions now carry the seed
  bit; no Q2 model ran.
- **The miss cost of the first tokens.** A token's misses are mostly drive reads until the RAM cache holds the
  recurring experts, so the first prices of every request follow an expensive miss (above). A price from the miss an
  admission saves (the next one of the same expert, usually a RAM-cache hit) would not; that is a change of the cost
  model for every request and device, to be measured on a discrete GPU.
- **C's exposed copy (1.0 s):** a fence at the first node that can reuse a split's input, rather than before its
  compute, would let the copies overlap the expert matmul.
- **The prefetch and evicted entries:** the next-layer prefetch skips an expert whose last selection was a tier hit
  even after the tier evicted it; a live residency query would serve both seeds and ordinary entries. It is the
  likely cause of the 0.7-1.6 % in decode 129-256.
- **The floor:** single 4 Hz samples can still fall under the published floor at a request's start, by tens of MiB,
  in base and branch alike: an allocation outside the ledger takes what the tier left until the next watcher tick.
  The floor watcher has no unit test of its own (start, stop, the run lock it does not wait for); U1 runs it.
- **The seed check's knob:** the design's `SEED_VERIFY` was not added; `mk-verify.py` makes the debug tier from the
  committed one when needed.

### The merge of edge and warm-up on ysy9 (M1)

Early on 10-22 (04:05-04:50 CST), ysy9 built and checked the merge 2826e92. It joins edge (e0962e2: weights homed on
the drive through `DENSE_HOMES`, the graph route's memory line, the tier's start-up safety from the elastic margin
on discrete GPUs) and warm-up (27fe8f2). The one conflict (`ggml-backend.cpp`, the resolver block) keeps both sets of
resolvers behind one macro. The question was only whether the merge breaks anything on the Spark with DeepSeek-V4.1-Flash-1017
Q8_0. The same rules held as before: one lock (`~/ssdl/spark.lock`), `~/ssdl/STOP` checked before each step and each
arm, one server at a time, and the 3 GiB watchdog, which never fired. The reference is `eng-warm` (a485f71, binaries
965ca26). The merge is `eng-m1`, built from `git archive 2826e92`. Scripts: `~/ssdl/ab/m1-*.sh`, `m1-a3.py`,
`m1-tfcmp.py`.

**Build.** `LIBURING_PREFIX=~/ssdl/deps BUILD_TESTS=1 tools/adaptive/build.sh`: `BUILD_OK` in 3 min 10 s, liburing
static, architectures 121a, the same CMake flags as `eng-warm`. No errors. The warnings are the usual kinds
(missing field initializers, unused variables). The test targets built too (`TESTS_BUILD_RC=0`).

**Unit tests** (each in a user scope of 8 GiB with no swap, under the lock):

| test | result |
|---|---|
| test-kimi-vram-elastic | PASS, 127 assertions, unified=1 (as `eng-warm`) |
| test-kimi-vram-pool | PASS, 36 assertions |
| test-kimi-vram-reclaim | PASS, 52 assertions |
| test-kimi-vram-deferred | PASS, 31 assertions, uploads 5, published 3 |
| test-kimi-vram-q2 | PASS, 140 checks, worst rel L2 0.0157312 (as `eng-warm`) |
| test-kimi-dense-stream (edge) | PASS, stage 327680 |
| test-kimi-runtime-routing | PASS |
| test-kv-cache-iswa-state-at | PASS |
| test-server-text-reuse | PASS |
| test-kimi-cpu-sink, and with `--prefetch-tests` | PASS, PASS |
| test-kimi-vram-mxfp4 | rc=1: `up_gate_c512/scales` row 0, cpu 0x1.242ab8p+12, gpu 0x1.242ab6p+12. The output is identical to `eng-warm`'s and to base's (`eng-base`) apart from the cgroup scope's name. A known failure of this host, not the merge. |

On the VM (no build): the framework's Python tests (68) pass, and so do `test_launcher_settings.sh` (100 checks) and
`test_launcher_replica.sh` (6).

**Edge A3 on the Spark, DeepSeek in place of Muse** (derived settings at 2906, no `DENSE_HOMES`, p4096, one AB
pair, `m1-a3-p4096-*`):
- `graph splits` are the same in both builds: 126 at bs 6400 and at bs 4104, and 86 at bs 1.
- The launcher's 63 receipts are equal once the build directory is replaced by a placeholder.
- `eng-m1` logs `[dense_homes] gpu_bytes=8944517568 ram_bytes=0 drive_bytes=0 drive_tensors=0` and no `[dense_stream]`
  line. That holds in every `eng-m1` server of this check, 8 of them.
- TTFT 32.246 against 32.170 s (-0.24 %), decode 1-128 6.179 against 6.156 (-0.37 %), 129-256 7.143 against 7.055
  (-1.23 %). Each is within 3 %, so the rule asks for no ABBA. The n2 ABBA below is one anyway.

**Edge A5 on unified memory** (`m1-a5/`): dry runs of both launchers with the derived settings (414), with 2906 and
with the bare defaults. Once the build directory is replaced, every ENV and ARG line is byte-identical.
`GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024 source=profile` in all six. Every difference, explained:
- The SCOPE unit name (a timestamp and a pid).
- `memory.max` and SCOPE `MemoryMax`. They are computed from the memory in use at that moment, and they differ by
  15-167 MB from run to run of either build.
- `eng-warm` warns that `libkimi-mxfp4-parity.so` is older than `libggml-base.so` (20:24 against 00:57). This comes
  from that checkout's history, not from the merge.
- The first `eng-m1` launch rebuilt the expert index once (~4 s), because edge's index adds a `dense` key. The
  new `experts.tsv` is byte-identical to the one `eng-warm`'s `expert_source.py` writes (checked in /dev/shm). The
  new `experts.json` differs from the old one only by `dense`. `eng-warm` reads the new index without rebuilding it.

**Regression: n2 at p4096**: `m1-n2-*`, `m1-n2.json`. The derived settings at 2906, request 1 and then a new
conversation on the same server. Two ABBA cycles ran with the lock held over each cycle's four arms; the second was
added after the first read -2.3 % in 129-256. A is `eng-warm`, B is `eng-m1`.

| request | eng-warm: TTFT, decode 1-128 / 129-256 | eng-m1 | cycle 1 | cycle 2 | all |
|---|---|---|---|---|---|
| 1, p4096, first after load | 32.19 s, 6.156 / 7.088 | 32.34 s, 6.128 / 6.997 | +0.85 %, -0.84 % / -2.33 % | +0.05 %, -0.08 % / -0.24 % | +0.45 %, -0.46 % / -1.28 % |
| 2, new text, same server | 28.87 s, 6.762 / 6.980 | 28.91 s, 6.742 / 6.930 | +0.05 %, -0.62 % / -0.10 % | +0.20 %, +0.02 % / -1.33 % | +0.12 %, -0.30 % / -0.71 % |

- `eng-m1`'s request 1 by arm: TTFT 32.44, 32.46, 32.25 and 32.21 s, and decode 1-128 6.149, 6.078, 6.143 and 6.142.
  The warm-up's spread was 32.2-32.4 s and 6.08-6.23. Two cycle-1 arms are 0.04-0.06 s above that TTFT range.
  Cycle 2 is inside it.
- Cycle 1's -2.3 % in 129-256 follows the text. Both of its `eng-m1` replies part from the reference at characters
  552 and 591, so tokens 129-256 decode another text. In cycle 2 both `eng-m1` replies equal the reference for all
  1247 characters, and the difference is -0.24 %.
- Greedy request-1 replies: `eng-warm` 4 of 4 equal the reference. `eng-m1` 2 of 4, plus the A3 arm, which parts at
  706. Each of these points is an earlier near tie: 552 (pre-review ABBA), 706 (`w2-ab7`), 591 (`eng-dev`, U1 stress);
  973, 1155 and 757 also occur on the warm-up builds. Request 2 parts at character 123 in arms of both builds, as in the warm-up.
- Seeds published per arm: 5216 (`eng-warm`); 5232, 5200, 5216 and 5216 (`eng-m1`). In request 1 the token-2
  regrowth took 0.166-0.178 s in every arm (request 2: 0.130-0.132 s, and 0.324 s in `eng-warm` a-8); load 39-43 s.
- Floor (4 Hz): each build has one arm with one sample under the published floor, at the same moment, the second
  request's prefill-to-decode boundary (~148 s after the start): `eng-m1` b-7 by 409 MiB, `eng-warm` a-8 by 139
  MiB. The other six arms stayed 0-22 MiB above the floor. The lowest 4 Hz MemAvailable was 4141 MiB, about 1.07 GiB
  above the 3 GiB watchdog. This is the open item
  of the warm-up (an allocation that takes what the tier left until the next tick), larger than the tens of MiB seen
  before, in both builds.

**Correctness through decode** (`probe-tf-decode.py`, threshold 414, 64 requests, positions 1-128):

| run | decode | prefill | misses |
|---|---|---|---|
| `m1-tfd-eng-m1` | 62/64 | 62/64 | 21 (' operates' -0.808 vs ' runs' -0.819), 29, d50, d82 |
| `m1-tfd2-eng-warm-1` | 62/64 | 64/64 | d50, d82 |
| `m1-tfd2-eng-m1-2` | 62/64 | 64/64 | d50, d82 |
| `m1-tfd2-eng-m1-3` | 62/64 | 64/64 | d50, d82 |
| `m1-tfd2-eng-warm-4` | 62/64 | 63/64 | 29, d50, d82 |
| `w2r-tfd-eng-warm` (earlier) | 62/64 | 63/64 | 29, d50, d82 |

- The decode misses are base's own two: the same tokens on top in every run (' from', ' retains').
- The first `eng-m1` run read 62/64 in prefill, under the 63/64 bar. Position 21 was a tie of 0.011 nats, decided in
  a one-token sink pass, whose CPU/GPU split follows measured times. The ABBA rerun read 64/64 twice on `eng-m1`, and
  64/64 and 63/64 on `eng-warm`.
- The spread does not depend on the build. The mean absolute difference of the reference token's log-probability
  between two runs is 0.009-0.016 nats within `eng-warm`, 0.014 within `eng-m1` and 0.007-0.017 across builds. The
  largest single difference is 0.15-0.40, 0.23-0.48 and 0.13-0.50.
- Every run decoded: `decode_active=1` 64 times. Every run hit the tier: on `eng-m1`, 21966-22224 hits and 4282-4297
  uploads (on `eng-warm`, 22170-22184 and 4311-4317). Seeds: 5232 published, 5034 freed unused.

**Verdict.** The merge breaks nothing that these checks can see on the Spark. The unit tests pass, and the one
failing test fails with base's values. Edge's rules do not fire without `DENSE_HOMES`. The launcher passes the same
environment and arguments on unified memory. Time, correctness and the floor are within the noise of the warm-up
build. What is not covered here: anything with drive homes (A4, on 233), discrete GPUs (A5's derived safety, 125 and
233), and Muse.
