# Adaptive framework

Decides, from nothing, how to run an SSD-streamed MoE model on the machine it is on, for prefill and decode, and
keeps deciding while it serves. A configuration is

- the size of the GPU expert tier (the share of GPU memory the engine may fill with experts),
- the CPU placement of the expert sink (`all` CPUs unpinned; `split`: per-phase pools, decode on the fastest CPU
  group and prefill on all; `pin`: the fastest group only),
- the prefill ubatch (a longer prompt in fewer passes, each reading the experts its tokens touch, for GPU memory
  the tier then lacks).

Nothing is tuned by hand: the machine is measured, the model of it predicts every configuration, and the loop
corrects that model from the requests it serves. The workload is not known in advance either.

## Run

```sh
tools/adaptive/build.sh                                     # this checkout, built for this machine in build-cuda130
# DeepSeek-V4.1-Flash: the GGUF shards under ROOT/models (the conf's MODEL_GLOB, up to two directory levels down;
# MODEL_DIR or MODEL say otherwise; they and EXPERT_PACK replace the old DSV41_DIR and DSV41_PACK)
ROOT=/data tools/adaptive/adaptive.sh serve                 # an OpenAI-compatible server on 127.0.0.1:8080
ROOT=/data tools/adaptive/adaptive.sh validate              # the four-conversation check below
# Muse Spark Open UD-Q4_K_XL on the Spark: one server with its best measured settings (a 4K benchmark turn).  On a
# unified-memory GPU the engine's host-memory ledger arbitrates the one DRAM, from a margin of at least 4 GiB (the
# launcher's start there); a run under the framework keeps a 3 GiB watchdog floor there (WATCHDOG_MIN_AVAIL_MB=3072)
# until a stress run passes a margin the ledger learned
SETTINGS=tools/adaptive/models/examples/muse-spark-gb10.settings MODEL_CONF=tools/adaptive/models/muse-spark.conf \
  MODEL_DIR=/data/models ./run-ssd-llama.sh

# Qwen3.8-Flash-Next on its port's engine (branch deepseek2/qwen38, built in its own build-cuda130)
ROOT=/data MODEL_CONF=tools/adaptive/models/qwen38.conf QWEN_SRC=/src/qwen38 tools/adaptive/adaptive.sh serve
```

`adaptive.sh serve|validate [OUT] [-- llama-server args]` writes everything to `OUT` (default
`$ROOT/fwruns/<date>`); a directory that has steps done keeps them, so a second `serve` on it starts at once.
Needs python3 (the standard library only), a C compiler (the drive probe), nvcc (the memory probe), cgroup v2
with a systemd user scope (the launchers bound the server's memory), and the GPU to itself. `build.sh` builds the
engine for the machine it runs on (CUDA architecture, Arm extensions and liburing derived from the machine).

## Steps

1. **Device probe** (`probe.sh`): CPU groups from the kernel's labels (`coresets.py`), memory, the GPU's kind and
   host link (`probe/memprobe.cu`), the drive at the queue depths a decode layer issues and beyond, read the
   engine's way (`reads.sh`, `probe/ioreplay.c`: O_DIRECT reads of whole widened expert roles through raw io_uring
   syscalls); `device_profile.py` records the knee QD*, the smallest depth as fast as the fastest within the
   repeat spread.
2. **Bootstrap** per CPU placement (`arm.sh`, `boot_client.py`): one real-text turn with 512 decode tokens and the
   engine's per-token trace and profile on, then the same prompt and reply prefilled again on the warm tier; the
   client counts what the drives of the expert files read in decode (`/proc/diskstats`). Then the GPU graph's
   prefill on a fresh server at the threshold 0 (`boot_client.py --graph`): a cut of the text that the decode trace
   covers and the whole text, each from scratch. For each larger prefill ubatch the model allows, a start-up probe
   of the tier it leaves (`ubprobe.py`): the tier after its first decode growth, compared with the same reading of
   the reference bootstrap; each placement's tier loses the difference. On unified memory this sees only the
   buffers that stay allocated in decode (the host's compute buffer: 48 and 288 slots at 16384 and 32768 tokens,
   DeepSeek on ysy9); the GPU compute buffer a graph prefill takes back from the tier (4.5, 10.2 and 28.9 GiB at
   8192, 16384 and 32768 tokens), which decode refills afterwards, is not priced (design F_g, M6).
3. **In-situ costs** (`insitu.py`): per placement, the per-layer decode time by class (tier hits, ready, read
   from the SSD) with its CPU compute part, the time outside the expert sink against the context position, the
   prefill's CPU compute and bytes streamed, the drive's service time per MB while it decoded, the depth the
   next-layer prefetch ran at with the engine's statistics of the hint per rank and the drive's idle window, and
   what other readers took from the drives; the graph's prefills (the bytes and read time of its expert copy, its
   prompt time, the GPU memory with its compute buffers); the model's routing sample and its reuse curve.
4. **Plan** (`plan.py`): every configuration of the machine model priced over the expected conversation; the
   least total time wins. `settings.py` writes the choice as `settings.env`, a `SETTINGS` file for
   `run-ssd-llama.sh` in which every value carries its receipt (`KEY=VALUE  # source=...; input=...`); it is
   also the writer of the knobs of every server the framework starts after the plan. Besides the configuration
   it derives the prefill threshold and ubatch (below), the next-layer prefetch depth (below), the context a
   known workload fills (`WORKLOAD=known`, rounded up to 256 tokens), for a workload of more than one turn one checkpoint where the memory
   saves an earlier state after each prompt batch (the server says so at start; none for one turn), and on unified
   memory a host-memory margin where the engine's
   ledger learned one in the framework's runs (above the launcher's start there, 4 GiB); `validate` writes it again
   with the validation's runs. On unified memory the launcher itself starts the GPU tier from one block (it grows
   after the first decode token into what the ledger leaves) and the margin at 4 GiB, for every server.

Then `serve` (`serve.py`) starts the plan's configuration behind an HTTP front end that runs the closed loop, or
`validate` serves `prompts/work-4turn.json` four times (`converse.py`): from the plan's choice and from the best
configuration with another placement, each static and under the loop, and sets them side by side
(`compare_online.py`; it also sets each static prefill against the prefill model by the route the server took, and
flags a route predicted more than 15 % off). It also reports each server's decode plan: how many experts a token the
GPU computed from host memory, the plan's median error on its predicted per-layer time (flagged above 15 %), the RAM
cache slots the engine page-locked and the copy rate it measured (flagged below half the probe's page-locked rate).

## The model

`machine.py` prices a configuration; `plan.py` asks it before anything is served, `online.py` at every request.

- **Decode.** A token is the MoE sink's time plus the time outside the sink. The sink's time is the routing
  replayed through the tier (`sim.py`: the engine's count-decay admission over one pool of whole experts) and
  priced with the placement's measured per-layer table; it splits into CPU compute and the rest (waiting on the
  drive, the GPU). Outside the sink: a level per placement and a slope against the position.
- **Prefill** (`prefill.py`). Pass by pass on the route the engine takes for it: up to the prefill threshold, and
  below the GPU's offload minimum of 32 tokens, the CPU sink, the larger of its reads (less what the tier holds) at
  the drive's rate and its CPU compute (the routes a prompt token takes times the compute per route), plus its
  time a token besides; above it the ggml graph on the GPU, its reads at the rate its expert copy read in the
  bootstrap's graph prefills, plus their time beyond the reads (a part per pass, a part per token: the cut and the
  whole text give both). A pass reads the experts its tokens touch, U(n) per layer: the bootstrap's prefill
  passes (the sink's distinct experts, the graph's copied bytes), and below the smallest of them the decode
  trace's n consecutive tokens, scaled to it (decode tokens touch fewer experts than prompt tokens: about 7 % fewer
  at 669 tokens, Muse on the Spark).
  The threshold gives the workload its least prefill time as the plan prices it: one conversation on a freshly
  started server, its first turn on a cold tier and the others on the warm one. One threshold serves every
  request, and a sink that computes the tier's experts on the GPU (MXFP4) reads less on a warm tier, while the
  graph copies every expert it touches. DeepSeek on the Spark at 4104 tokens, by the model: the cold sink 40.0 s
  and the graph 31.1 s (measured 40.1 and 31.3 s), the warm sink 30.1 s (never measured). Where the workload's
  time leaves the threshold free, between its pass sizes, the threshold is the break-even of the two routes
  (found by bisection) over the tier states the workload's passes of at least 32 tokens meet. A follow-up counts in
  two forms, 1/2 each: its own new tokens, and the reply before it prefilled again with them. When a reply's text
  renders back in other tokens, the server's cache parts inside the reply, and the server prefills again from its
  latest checkpoint, the end of the previous prompt (DeepSeek on the Spark ysy9: 9 of 15 follow-ups, 279-539
  tokens instead of 17-23; Muse on spark-b9a5: none of 3). Which form a follow-up takes is not known before its
  reply exists. So a long first prompt with short follow-ups weighs 1 cold pass against 1.5 warm ones (DeepSeek on
  ysy9: 2906, against 414 for the cold tier alone, which sends a 534-token re-prefill to the graph: 17 s there
  against 7 s on the warm sink), and four turns of one size weigh the states 1 : 3. Without a workload it is the
  warm tier's. `settings.py` writes it with the ubatch when a graph pass larger than it
  fits the GPU's memory, and otherwise keeps every pass on the sink. Without graph prefills the graph is priced
  from the probe's rate and the sink's time a token, and the receipt says it is provisional.
- **Prefetch depth** (`settings.py`). The graph hands the sink the next layer's router output on this layer's
  input, best first; at depth k the sink reads the first k novel experts (neither held by the GPU tier now nor in
  the RAM cache) while the drive would idle, from the end of the layer's own reads to the start of the next layer.
  A rank's read pays when the share of it the next layer uses (`p_j`, the share of novel hints it selects) times
  the demand time a used read saves (the bootstrap's layer fit: a read expert less a ready one) exceeds what it
  costs: its spill into the next layer's demand reads and, when it starts after the window, its own transfer. The
  engine measures the spill (`spill_us`, `spill_us_r`, `started_r`): the readers' time past a window's close on
  reads the next run does not take, shared out over the ranks by each rank's own time past the close, over the
  rank's reads; a rank the bootstrap did not read costs the most a read rank does, plus the transfer past the
  window. A report from an older engine gives only the readers' busy time beyond the windows' total, over the
  reads, a lower bound. The window is the idle window
  less what other readers take of it, `k_win` ranks start inside it (the novel reads before them, `sum nu_j` at an
  expert's read time at the probe's rate for the engine's readers, do not fill it). `k* = min(k_pay, k_mem)`, with
  `k_pay` the leading ranks that pay (0, prefetch off, where the drive has no idle window or the spill outweighs
  what a read saves) and `k_mem` the largest depth whose every rank saves more reads (`nu_j p_j`) than the slot it
  takes would serve a token. On unified memory depth k stages max(1, k) slots a layer in
  the DRAM the GPU tier would hold: rank 1 takes none, rank j the tier's slots between depths j - 1 and j (the
  bootstrap's routing replayed through the tier at both sizes). On a discrete GPU the prefetch lands in the RAM
  cache's slots in place of its coldest experts: rank j takes its j-th slot from the end (the cache behind the
  replayed tier, by the engine's count-decay policy), which serves more the smaller the cache. The decode model
  keeps the bootstrap's depth: it prices the plan with the per-layer table the bootstrap measured.
- **The drive** is one state that every placement shares: bootstraps that ran at different drive states are put
  on the reference's (the drive's service time per MB in decode scales the part that waits on it), and the loop
  carries the latest measured state to every configuration.

**The loop** (`controller.py`, `online.py`). When a request arrives its new prompt tokens are known; the rest of
the conversation is expected from the requests served so far (`HORIZON` turns ahead). Every configuration's
rest is predicted from the conversation's own routing and from what the served turns measured, part by part:
CPU compute and the outside level per placement, the drive's state shared, the prefill bytes. Another
configuration costs a restart (measured) and a cold tier; the engine's disk prompt cache brings the history
back (`SWITCH_KEEPS_HISTORY=1`), without one it is prefilled again. The configuration changes before the request
when the best beats staying by 5 % (at least 15 s).

## Files

| file | role |
|---|---|
| `adaptive.sh` | entry: steps 1–4, then `serve` or `validate` |
| `build.sh` | the engine built for this machine |
| `gguf_header.py`, `expert_source.py` | GGUF headers (standard library only); where the experts are, what one costs to read, read plans |
| `probe.sh`, `reads.sh`, `probe/ioreplay.c`, `coresets.py`, `device_profile.py`, `probe/memprobe.cu` | step 1 |
| `arm.sh`, `boot_client.py`, `client.py`, `placement.py`, `ubprobe.py` | step 2 |
| `insitu.py`, `logparse.py` | step 3 |
| `machine.py`, `sim.py`, `prefill.py`, `plan.py`, `settings.py` | the model, step 4; the settings it decides |
| `online.py`, `controller.py`, `serve.py`, `converse.py`, `compare_online.py` | the loop, its two front ends, the check |
| `../../run-ssd-llama.sh` | the launcher of every model on this engine (`run-deepseek-v41-flash-mxfp4.sh` wraps it) |
| `models/*.conf`, `models/examples/` | one data-only file per model; SETTINGS files of measured configurations |
| `models/qwen38.conf`, `models/launch-qwen38.sh` | the Qwen port: its own engine and launcher, for any machine |
| `tests/test_launcher_replica.sh`, `tests/test_launcher_settings.sh`, `tests/test_settings.py` | the wrapped DeepSeek launcher starts the server exactly as the old one (1f174b4, from git) did; the launcher's own rules; settings.py's |
| `prompts/` | real texts: the bootstrap turn and the validation conversations (`source` says where each is from) |
| `baselines/` | llama.cpp (PR #28696) and colibri on the Spark, a 255H + RTX 5090, a 16 GB M1 and a 100U + RTX 5060 Ti, with ssd-llama for reference: settings, the changes that made them run, results, scripts |

A model's `.conf` is data (`KEY=VALUE` lines, never executed): the model's name (`NAME`), its first shard
(`MODEL_GLOB`), an expert pack if the experts are not read from the GGUF itself (`EXPERT_PACK`), a chat template
the GGUF lacks (`TEMPLATE`, `TEMPLATE_KWARGS`), the model's sampling where its GGUF does not carry it (`SAMPLING`,
arguments, which override the GGUF's `general.sampling.*`), the model's
own engine switches (`ENGINE_ENV`), and for the framework the placements and ubatches to try (`MODEL_PLACEMENTS`,
`MODEL_UBATCHES`), and the settings the framework derives but must not pass for this model yet (`PENDING`: the
DeepSeek deployment keeps its own until an ABBA on its machine accepts the derived value; `settings.py` writes them
as comments). Everything else follows from the GGUF header (the experts' types and ranges, its sampling) and
from the machine. `run-ssd-llama.sh` reads it, together with an optional `SETTINGS` file (the launcher's settings
and engine variables, in the same format: `models/examples/muse-spark-gb10.settings` is Muse's best measured
configuration on the Spark) and the caller's environment, which wins; `LAUNCH_DRY_RUN=1` prints the environment
and arguments the server would get, and every engine variable is logged with its source (`[ssd_setting]`). What
none of them gives comes from the launcher's legacy profile: the DeepSeek launcher's values for an RTX 5090 with
16 GiB of RAM (131072 tokens of context, 8 checkpoints, the CPU sink's grouped prefill, ...), the same for every
model and machine until each value moves to the source that derives or decides it.
`qwen38.conf` is a script with its own launcher (`LAUNCH`, `KNOBS_VIA=extra_env`, `SWITCH_KEEPS_HISTORY=0`):
`adaptive.sh` sources a conf only when it names a `LAUNCH`.

The engine reports what the framework reads in default-off log lines: `GGML_KIMI_CPU_SINK_TRACE=1` and
`GGML_KIMI_CPU_SINK_PROFILE=1` (every decode token's routing and the sink's costs), `GGML_KIMI_CPU_SINK_MRC=1`
(online miss-ratio curves). The per-phase pools are `GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS`; the launcher takes
`UBATCH` and any engine variable from its caller's environment.

## Results

From nothing, the workload not given (the plan assumes four turns the size of the bootstrap text), each
conversation four turns of real text with 512-token replies. RTX 5090 on both machines.

| machine | model | plan | the other placement | plan predicted / measured | loop from the other placement |
|---|---|---|---|---|---|
| Core Ultra 7 255H, 15 GB, PCIe Gen4 x4 | DeepSeek-V4.1-Flash | `all` | `split`, 12.6 % slower | 989 / 997 s | switched after turn 0 |
| Core Ultra 5 230F, 30 GB, PCIe Gen5 x16 | DeepSeek-V4.1-Flash | `all` | `split`, 14 % slower | 408 / 415 s | switched after turn 0 |
| same, second turn 32k tokens | DeepSeek-V4.1-Flash | `all` | `split`, 18 % slower | (the plan does not know the 32k turn) | both loops took `all@16384` for it: 596 s against 610 static, 624 against 722 |
| Core Ultra 5 230F | Qwen3.8-Flash-Next | `all` | `pin`, 5–24 % slower | 210 / 178 s | stayed: re-prefilling the history costs more than it saves |
| Core Ultra 7 255H | Qwen3.8-Flash-Next | `pin` | `all`, 13.7 % slower | 354 / 346 s | stayed, as above |

Time from nothing to the decision: 3–12 minutes (two placements; the slower drive and machine take longer). The
last row is this directory's code on a machine and model pair it had not seen (the plan predicted the other
placement 13.8 % slower), followed by `serve` with the same conversation through the front end.  On the 230F
machine `serve` with the 32k conversation switched to `all@16384` when the 32k request arrived (7.7 s; the history
came back from the disk prompt cache) and stayed there. The rows above
used the same model before it was organized into this directory: `plan.py` reproduces those plans to 0.02 s, and
the loop those decisions.

On the Spark (unified memory) the framework from zero reaches the best manual settings of Muse Spark Open, within
noise at both protocol prompts, and predicts its 4-turn conversation; on a second Spark, from zero on
DeepSeek-V4.1-Flash-1017, it ties the manual best's TTFT and decodes 3-11 % faster (another CPU placement):
[results-spark-20261021.md](results-spark-20261021.md).

## Limits

- One conversation at a time: `serve` takes generation requests one after another (the engine has one slot),
  and a request's new tokens are its prompt beyond the context the previous one left.
- A request that is not streamed adds its prefill to the feedback but not its decode rate.
- The loop reads the engine's per-token trace from its log: about 50 MB gzipped for a four-turn conversation
  with 2k decode tokens (DeepSeek), several times that while the server runs.
- The drive is modelled in its normal state; a drive that slows down over a hot region is not.
- The RAM expert cache is sized by the engine, not chosen: once it has filled the memory, the weights the engine
  maps from files (dense, Engram) no longer stay in the page cache, and a 32k prompt then prefills 35-75 % slower
  (DeepSeek, 30 GB: 232-294 s against 167-171 s on a fresh server, with 1.5 million major page faults against 2.6
  thousand).  The prefill model's rates are the bootstrap's (one prompt of 3-4K tokens), so such prompts are
  predicted too fast.
- The Qwen port's engine has no per-phase pools, no disk prompt cache and a fixed ubatch, so its configurations
  are the tier and `all`/`pin`.
- The graph's time beyond its reads is fitted on two prefills of the bootstrap text (the cut and the whole), and
  U(n) below the cut is the decode trace's shape scaled to it, so prompts far from those sizes (16K tokens and
  more) are extrapolated. The slots the graph's workspace takes from the tier, which decode refills afterwards
  (design F_g), are not priced. Muse on the Spark: n* is 411 with two graph prefills of the protocol prompts (669
  and 4221 tokens: 2.74 s a pass and 2.84 ms a token besides reads at 11.93 GB/s), against 295-302 when the graph
  was priced like the sink.
- The model keeps a pass below 32 tokens on the sink, the engine's default offload minimum; it does not read
  `GGML_OP_OFFLOAD_MIN_BATCH`. One graph prefill cannot split the graph's time beyond its reads into a part per
  pass and a part per token; all of it then goes per token, and the graph looks faster from the first token
  (DeepSeek on the Spark with its 4104-token graph prefill alone: n* = 0). The cut and the whole text give both
  parts. DeepSeek on the Spark, offline, with a 523-token graph prefill rebuilt from the numbers kept of a lost
  run: n* = 313 for the protocol's conversation, one turn or four.
- A follow-up that prefills its reply again is priced in the threshold only (two forms, 1/2 each). The plan's time
  counts each follow-up as its own new tokens, so a served conversation whose replies do not render back in their
  own tokens takes longer than planned (DeepSeek on ysy9: 7 s a re-prefill on the warm sink). The context is not
  affected: a re-prefill computes the same positions again.
- The workload is one conversation on a freshly started server, for the plan and for the threshold alike. The
  learned workload of `serve` (four turns the size of the bootstrap text) therefore weighs a cold tier once in
  four turns, while a serving server meets it once per start. The warm state is the bootstrap's second prefill of
  the same text, after its decode: its tier holds more of that prompt's experts than a new prompt would find, so
  the warm sink's reads are a lower bound.
- The context follows the workload only where it is known (`WORKLOAD=known`: `validate`, a benchmark). The plan's
  learned workload (four turns the size of the bootstrap text) is a forecast, so `serve` keeps the launcher's
  context (131072 tokens) and the GPU memory its KV cache takes; growing the context by a restart when a request
  would not fit (design M11) is not built.
- The prefetch depth comes from the bootstrap, which runs at the launcher's depth (2): the drive's idle window and
  the hint's statistics move a little with the depth (the window's median 2.9-3.4 ms a hinted layer at depths 0-8,
  Muse on the Spark), and the plan prices decode at the bootstrap's depth, not at the derived one.
