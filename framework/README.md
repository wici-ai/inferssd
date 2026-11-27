# InferSSD framework tools

The package supplies the `inferssd` command, typed settings, source metadata and
an owned runtime that supervises one local server. The runtime uses only the
Python standard library.

For download-and-run usage, `inferssd cli -hf REPO` opens terminal chat and
`inferssd serve -hf REPO` starts the API. These commands prepare model files and
a default resource policy, then use the owned runtime. Install the optional
Hub client with `python3 -m pip install './framework[hf]'`, or use the
[one-command installer](../docs/quickstart.md#one-command-installation), which also
builds the native engine. `-m FILE` uses local weights without the Hub dependency.

## Commands

| Command | Input | Output |
| --- | --- | --- |
| `inferssd cli` | Tested Hugging Face repository or local GGUF | Streaming terminal chat; owned server stops on exit |
| `inferssd serve` | Tested Hugging Face repository or local GGUF | OpenAI-compatible API with automatic resource policy |
| `inferssd profile` | This machine, optional model paths and the native probes | Machine characteristics file and a summary per section; see [machine profile](../docs/machine-profile.md) |

## Settings and launch choices

`serve` and `cli` take `--settings FILE` for the versioned registry-defined
configuration and repeated `--set key=literal` for explicit overrides. CLI values take precedence over file
values, then the launch's own choices for this machine, then registry defaults. Comments following a file value are preserved as
source receipts. The generated native file contains file/generated values;
CLI-origin values become the matching registry-generated native `--ssd-*` option
so native effective telemetry preserves that priority. There is no environment
translation. Retired settings/launcher environment names fail by name without
printing their values.

CPU threads are server arguments, not settings. The SSD executor computes in the
server's ggml threadpools and adds only its I/O, prefetch-reader and watch
threads, which `cpu.io_cpus` places. The launch chooses the decode `-t` and
prefill `-tb` counts and, on Linux, the CPUs they run on.

An executed launch measures the machine before its first launch and reuses the
result afterwards (`--reprofile` measures again; `--machine PATH` uses a given
characteristics file and measures nothing); see [machine profile](../docs/machine-profile.md).
From that file, the engine's facts of the model and the CPUs, GPUs and memory it finds
at that moment, the launch chooses thread counts, CPU placement, `place.costs`
and, for an engine with CUDA but no usable CUDA GPU, `vram.budget = 0` (on macOS also
`host.budget_mib` from the runtime policy). It prefills each prompt in one pass
(`-b` and `-ub` as large as the context) on every platform, so that a pass reads
each expert it routes to once; the engine reserves each prompt's own graph. On a
CUDA GPU that shares host memory it keeps `host.headroom_mib = 4096`. It fills only settings left at their registry
defaults and thread options and sizes not given, so explicit values always win.
The choices are written to the generated `settings.ssd` with `source=strategy`
receipts and to `strategy.json`, with the values your settings kept and the
choices left out because they contradicted them. The server keeps one context
checkpoint (`--ctx-checkpoints 1`) unless you set `--ctx-checkpoints` or
`LLAMA_ARG_CTX_CHECKPOINTS`.

The model profile (chat template and model settings) follows the GGUF's
`general.architecture`. The `dsv` profile also sets
DeepSeek-V4.1's own sampling defaults (`--temp 1.0 --top-p 0.95 --top-k 0
--min-p 0`), which a request's own values or your extra server arguments
replace.

The prompt cache lives on disk, one directory per model under
`inferssd/prompts` in your user cache directory, and survives restarts
(`--cache-disk`, 4 GiB per model unless `--prompt-cache-mib` names another
limit). A chat that starts with a system message keeps one state there: a
newer state replaces the earlier ones of the same conversation, the ones
with the same system messages, tools and user messages up to the first
reply. With the `dsv` profile the DeepSeek-V4.1
template keeps earlier turns' thinking (`drop_thinking` false), so a
conversation's history renders as the server cached it.

```text
inferssd-settings 1
prefetch.depth = 4 # measured predictor depth
host.headroom_mib = 3072 # operator resource reserve
vram.budget = auto # use admitted capacity
```

A model must be a local GGUF or the first shard of a split set. `--ple FILE`
(or `--shard`, repeatable, comma separated) adds its extra shards, such as a
per-layer embedding table kept apart from it ([DeepSeek-V4.1
Engram](../docs/deepseek-v41-engram.md)). An omitted `expert.source` names the
files the engine loads the model from, and the engine indexes the routed experts from the
GGUF headers; explicit sources must be nonempty existing paths (GGUF files, expert
packs or alias indexes). Native source validation remains a separate boundary. `serve` and `cli` download a tested model with `-hf`, or take
a local one with `-m`. Other compute processes on the GPU do not stop a launch:
it leaves them VRAM (`vram.reserve_mib` in the [settings](../docs/configuration.md)).

The framework reads no GGUF file itself. Before it starts a server, a launch
asks that server for the model's facts (`inferssd-server --ssd-model-facts
MODEL`, with the model's `--shard` values): the engine reads the headers as its load does, without reading a weight
or starting a backend, and prints its files, metadata, tensors and the layout of
the routed experts as JSON (`inferssd.engine.model`). The strategy and the RAM
admission take their model facts from it, and a recorded dry run takes them from
its host facts.

## Compatibility and failure behavior

The package was extracted from earlier adaptive scripts. Their recorded
reference revision is `8b12192920b339a4ef31c9006d1ebe20304ce45a`; verbatim
reference sources and synthetic observations are under `tests/fixtures`. Extraction preserves arithmetic, sum and mapping order, stable
ties, rounding, borrowed references, callback order and exception timing.
Toy logs are synthetic inputs, not claimed engine-run evidence. Historical
archives establish only their actual recorded input coverage.

The installation requires Python 3.9 or later. Normal-wheel regression and
exact reference transcripts are verified on Python 3.9 and 3.12. Their original
results can differ because of float summation, resident-set iteration or native
exception wording. Tests retain separate runtime transcripts and compare exactly
within each runtime, without a numeric tolerance or production version branch.
Recorded transcript checks run only on those two versions; direct comparisons
with the reference sources that do not require a recorded transcript still run.
Future versions are not assumed identical to either verified runtime.

Legacy artifact mappings are typing views, not new schemas. They are not copied,
normalized, validated or filled with defaults on ingestion. Missing, null, empty
and zero values retain distinct behavior, including failures when consumed.

## Python interfaces

| Area | Modules and boundary |
| --- | --- |
| Machine | `machine.*`: identity, probes, characteristics and their store, behind `inferssd profile` |
| Launch | `launch.*`, `strategy.*`, `configuration.settings`: typed settings, the launch strategy and the typed launch behind `serve` and `cli` |
| Owned runtime | `engine.*`, `platform.*`: the guardian, worker and owned server, and the engine's facts of a model (`engine.model`) |
| Native observations | `telemetry.jsonl`, `telemetry.observations`, `telemetry.files`: versioned native events and local file access |

Import and adapter construction perform no file or process work. These interfaces preserve existing semantics;
they are not a claim that every legacy artifact now has a versioned, validated
contract.

## How serve and cli run the server

`inferssd serve` and `cli` resolve the launch for Linux, macOS or Windows from the live
machine, then start one supervised server on a loopback port, or for `serve`
on the IPv4 address `--host` gives. Each launch writes
a resource policy for the whole machine, `memory_budget_bytes` set to its
physical memory, and claims a new run directory under
`~/.local/share/inferssd/state/runs`.

On Linux the server runs in a systemd scope with cgroup v2 whose memory limit is
`memory_budget_bytes`, with zero swap and readable ancestor OOM counters and
memory peak; see the [Linux requirements](../docs/quickstart.md#linux-requirements).
The worker checks the actual envelope and the server's inherited scope and
memlock limit. It never raises system limits.

On Windows the server runs in its guardian's job object, whose commit limit is
`memory_budget_bytes` plus the dedicated memory of the discrete GPUs, since WDDM
charges their allocations to commit; the worker checks that limit and that the
server stays in the job. Paging-file use and the memory compression store are system-wide, so
their growth is sampled, as on macOS.

On macOS the budget is a **sampled soft server RSS threshold**, not a cgroup or a
hard unified-memory cap. Every platform is monitored while the
server runs, and fail the execution if a threshold or the worker fails. Host
utility calls have a ten-second timeout. These checks do not replace per-host
validation or prove a model will fit.

Execution claims `.inferssd-owner` atomically before opening logs. A directory
may already exist, but an existing execution claim or execution artifact is
never overwritten, including after successful completion. A cooperating device
lock is acquired before writing the launch specification. The guardian, worker
and server receive the same locked file description; only owned groups and the
fresh exact Linux scope may be cleaned up. A disconnected controller, STOP or
signal requests cleanup. If cleanup is unknown, the guardian retains the lock
and writes quarantine evidence; a timeout never authorizes killing that guardian.
An unrelated listener on the health port is never killed.

`settings.ssd` records the file/generated settings and their receipts;
`native-events.jsonl` is exclusively created by the context-owned native writer.
`runtime.json` records server argv and scope, `memory-samples.jsonl` records
resource samples (every one of the first minute, then one every 30 seconds and
any whose memory moved by 2%), and `ownership.json` records final status and cleanup.
Keep these artifacts private if arguments or paths contain private information.
Server loader variables are carried in the process environment and restored
only for the server, not serialized into the launch specification or applied to
the Python lock owner. Guardian and scoped worker use the same installed interpreter
with Python isolated mode; a package in the caller's directory or PYTHONPATH
cannot replace their implementation. A successful cleanup does not turn a failed server or
resource check into a successful exit.

Model, listener, parallel, thread and batch arguments are owned by the launch, so
settings cannot replace them. macOS cannot pin threads to CPUs. A positive
`cpu.decode_participants` cannot exceed the decode thread count. The launch
receipt and server argv record the thread counts and CPUs.

`inferssd.telemetry.observations.NativeObservationSession` reads a run's owned
`native-events.jsonl` stream; the benchmark CI uses it to record what each run
did. It requires the native observation contract (the `ssd.controller.begin`
event, schema 1) and versioned placement/prefill records, checks event sequence
and field types, and rejects failed layer observations. Human-readable server
logs remain diagnostic artifacts and are never parsed. Static launch does not
require debug channels. See [native observations](../docs/architecture/native-observations.md).

## Remaining integration boundaries

The native context integration must be validated with this installed package on
each target host; Python substitutes for OS effects or synthetic model headers do
not establish native execution correctness. No new performance conclusion
follows from these package tests. Importing these modules starts no process,
host probe or network call.

## Development checks

```sh
python3 -m pytest framework/tests
mypy --config-file framework/pyproject.toml framework/src
ruff check framework
ruff format --check framework
```

Use a mypy release supporting the configured Python 3.9 target (verified with
1.18.2). CI installs the normal package before running its tests and these checks.
The preserved reference sources under `tests/fixtures` are intentionally
excluded from formatting/lint rewrites.

## Installed engine resources

`serve` and `cli` use the engine the installer set up. `--resources FILE` points
them at the descriptor of a source build instead. The descriptor identifies the
executable, per-profile templates, optional native probes, and supported
configuration/telemetry contract. Paths resolve relative to the descriptor, never
the current directory. See the [deployment guide](../docs/engine-resources.md).

### Decode estimates before inference

`inferssd profile --model <GGUF-or-directory> --estimate-compute --json` adds
synthetic CUDA matrix measurements to the hardware profile. It does not load or
run a language model. Rebuild `inferssd-cpuprobe` first; the probe rotates 256 MiB
of synthetic weights and measures one-token products, including both DeepSeek
expert matrix orientations. Run profiling on an idle GPU. Saved measurements are
reused; `--reprofile` replaces them.

Estimation uses the decode thread pool, each expert quantization type, and measured
GPU matvec rates rather than charging GPU work to the CPU. `--memory-gib 32` sets
the **estimated host budget** in place of the memory available now (it does not
set a process limit). Optional
`--host-work-mib` and `--gpu-work-mib` reserve known KV/graph working memory. Without
these, cache capacity is an upper estimate; these options do not alter launches.

Routed experts follow one routing assumption for every model, since routing cannot be
observed before the model runs: warmed caches hold the most popular experts, and the
share of accesses at rank `r` is `min(1/top_k, a/r)` (`strategy/routing.py`). JSON
gives that reference in `decode_tokens_per_second`, its name in `reference_scenario`
(`rank_serial` for routed experts), component times in `components` and the
assumptions taken. `scenarios` and `routing_scenarios` bound how much the RAM and
VRAM caches overlap (`rank_inclusive`, `rank_disjoint`), and
`scenario_range_tokens_per_second` adds I/O overlap and the measured CPU, GPU and
drive quartiles. The range carries **no probabilities**: it is not a confidence
interval or a guaranteed bound, and microbenchmark quartiles describe only the
profiling window. No historical tokens/s multiplier is used.

`--estimate-gguf FILE` and `--estimate-facts JSON` estimate a specific checkpoint.
A dense decoder can run without a named model profile. Routed experts need a profile
and a layout the SSD executor supports. An architecture or expert format that the
engine cannot run yields a refusal reason in place of a decode rate. Placement uses the
selected engine's GPU backends, including when replaying a saved machine profile.
Checks the engine makes when it starts are left to the launch. These include the
devices chosen through launch settings and values inside the model.

For a routed model, the estimator prices dense GPU work at the measured Q8_0 matvec
rate for both catalog entries and explicit checkpoints. It checks each routed layer's
up, gate and down formats against the executors used by the launch. An executable
expert format with no timing measurement can borrow the rate of the measured format
nearest it in bits per weight. The estimate identifies these proxies in its assumptions.
When another compute process is using the GPU, the estimate leaves the launcher's
VRAM reserve free before counting how many routed experts fit. On unified memory, the
GPU's working memory and that reserve come out of host memory.

For a dense model such as Gemma 4 E4B, the catalog records the bytes in each layer
weight format. The estimator prices that mix using matvec measurements near
4096 × 4096. An explicit checkpoint supplies individual matrix shapes and types.
On a single profiled CUDA or ROCm GPU, non-streamable weights must fit in available
VRAM. Other layers can stream from pinned host memory or the model drive. The
`dense_pipeline` scenario overlaps GPU compute, host-to-device transfer and drive
reads. The host scenario uses the measured CPU or RAM rates. Attention, lookup I/O
and cache warm-up remain unpriced (issue #395).
