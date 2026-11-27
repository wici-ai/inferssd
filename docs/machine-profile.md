<!-- SPDX-License-Identifier: Apache-2.0 -->
# Machine profile

InferSSD measures each machine once and keeps the result in a characteristics
file. The file describes the machine only: its CPUs, memory, GPUs and
the drives that hold models. It holds no model data, no placement and no cache
sizes; each launch works those out from the engine's facts of the model and this file.

You do not need to run anything for this: the first `inferssd serve` or `cli` on
a machine measures it, and later launches reuse the file (see
[launches](#launches)). `inferssd profile` measures or shows it on its own.

Linux, macOS and Windows are supported. On Windows nothing measures the GPUs
yet: the file records the NVIDIA GPUs `nvidia-smi` lists, with a partial `gpus`
section, and the other display adapters a Vulkan build can use.

## Usage

```sh
# Measure the machine and the drive that holds a model.
inferssd profile --model /models/model-00001-of-00004.gguf --build build-linux-cuda

# Print the saved profile without measuring anything.
inferssd profile --show

# Measure the drive of a models directory, then print the profile and the
# supported models' speed estimates as one JSON object.
inferssd profile --model ~/models --json

# Measure every section again.
inferssd profile --reprofile --model /models/model-00001-of-00004.gguf
```

| Option | Meaning |
| --- | --- |
| `--model PATH` | A model file or directory whose drive is measured. Repeat it for models on other drives. A file named like `name-00001-of-00004.gguf` brings all shards of its set; a directory brings every file below it, and one without a file of 1 GiB is measured with a test file (see [drive measurement](#drive-measurement)). The [speed estimates](#speed-estimates) take the models to be on the drive that holds most of the given files, or with none, on the first directory's. |
| `--resources FILE` | An [engine resource descriptor](engine-resources.md) whose `probes` map names the probe executables. |
| `--build DIR` | An engine build directory; the probes are `DIR/bin/inferssd-*probe` (`inferssd-*probe.exe` on Windows). |
| `--estimate-compute` | Measure single-token matrix throughput on idle CUDA/ROCm GPUs with synthetic weights. |
| `--memory-gib`, `--host-work-mib`, `--gpu-work-mib` | Set the host memory the estimates plan with, in place of the memory available now, and additional known working-memory reservations for estimation. |
| `--estimate-gguf FILE` | Estimate a specific GGUF. The engine reads its headers and discovers standard shards. Pass extra tables or nonstandard shards with repeatable `--shard` or `--ple`. |
| `--estimate-facts JSON` | Estimate exported native model facts against the current machine profile. Use `--model PATH` to select the target drive when the weights will spill. |
| `--reprofile` | Ignore the saved file and measure every section. |
| `--output PATH` | Use this file instead of the default location. |
| `--show` | Print the saved file's summary and exit; with `--model`, also the estimates for that drive. Measures nothing. |
| `--json` | Print the summary and the estimates as one JSON object on stdout instead of the text lines. |

Without `--resources` or `--build`, the command uses the installed engine, as
`cli` and `serve` do: `INFERSSD_RESOURCES` when nonempty (set by the one-command
installer's wrapper), otherwise `current/engine-resources.json` in the data
directory. Without an installed engine it exits with status 2. `--show` only
reads the saved profile. Adding `--estimate-gguf` also needs the engine header reader.
Adding `--estimate-facts` works offline. For saved catalog and explicit-target estimates,
the selected or installed engine descriptor supplies its GPU backends. If an older descriptor omits
them, the estimator checks the build cache beside its binary. With no engine or build
information, the saved machine GPU inventory and live availability determine placement.
The command prints one progress line per step on stderr and one summary line per
section on stdout, with `--model` the host memory the estimates plan with and
one estimate line per supported model (with `--estimate-gguf` or
`--estimate-facts`, the memory line and one line for that target), then
names the sections of this run that are partial or failed (on stderr with
`--json`). It exits with status 1 when a section measured in this run failed
(the file is still saved, with the reason recorded), and with status 2 on a
usage error, a `--model` path that does not exist, or when another
`inferssd profile` is running.

## Launches

Before a launch starts the server, it checks that the
characteristics file exists and is current for this machine and for the drive
that holds the model. When it is not, the launch measures what is missing
first, printing its progress on stderr. On a new machine this takes one to
three minutes; later launches find the file current and measure nothing.

The launch finds the probes next to the server executable, then in the
[engine resource descriptor](engine-resources.md)'s `probes` map, then in the
checkout's build directories. A probe that is missing or fails is reported as
a warning: its section is recorded as failed and the launch continues without
what it would have measured.

| Option | Meaning |
| --- | --- |
| `--reprofile` | Measure every section again before launching. |
| `--machine PATH` | Use this characteristics file as it is and measure nothing, for example one file measured once and shared by several runs. A dry-run reads it too. |

Each launch then combines the file with the engine's facts of the model (its
server reads the GGUF headers, `inferssd-server --ssd-model-facts`) and with what
it reads at that moment (the CPUs it may use, the usable GPUs and their free
memory, the available memory) to choose:

- the decode thread count: the fewest threads that stream the model's weight
  format at one token as fast as the best count measured, judged by the
  calls' middle half (more threads add no bandwidth, only waits; the engine
  runs decode on fewer while the machine's other work needs their CPUs); where
  the CPUs allow it, which CPUs the decode threads use: the performance cluster
  alone, its first CPUs, one thread pinned to each, when one token streams the
  format faster there than on those decode threads unpinned;
- the prefill thread count: the measured count that computes the format fastest at
  the longest prompt measured, of the counts that leave one CPU to the sink's I/O
  threads (every CPU but one is measured too; a profile without that count takes it
  when the most threads it measured below it are the fastest there and more threads
  measured faster still); the same rule on every platform;
- the CPU the sink's I/O threads keep, where threads can be pinned (`cpu.io_cpus`):
  outside the pinned decode CPUs, on the slower cluster where there is one, the
  highest such. Unpinned beside a prefill pool on every other CPU, the I/O thread and
  its drive queue and completion interrupts move between CPUs that sleep in between,
  and the drive reads slower;
- the starting estimates the engine uses to place experts between the CPU and
  the GPU (it refines them as it runs);
- `vram.budget = 0` when the engine has CUDA and no CUDA GPU is usable, and on
  macOS and Windows the host memory budget from the runtime policy;
- with an engine that has CUDA and one usable CUDA GPU, one prefill pass per prompt; on
  a GPU sharing host memory, a host margin of 4096 MiB (`host.headroom_mib`);
- for a model published with an MTP draft head (an `mtp-*.gguf` file beside the model's
  files), whether to load it. The server measures the launch's decode steps with the head
  once (`--ssd-decode-probe`, about as long as loading the model): the dense time of a step
  of one to four rows and of a draft. Drafts save only the dense work the rows of a step
  share: each accepted token still pays its own routed experts, which the
  [speed estimate](#speed-estimates) prices with and without the head, its tensors and its
  routed layer taking expert cache. The head is loaded when drafting some length, every
  draft accepted, costs less a token than decoding without it; the server then drafts the
  length that pays as it runs, down to none. A `--spec-*` option of yours leaves speculative
  decoding to you, and nothing is measured.

Without a measurement of the model's weight format, both pools take every
allowed CPU but one.

Only the pass size changes the model's output, the way
any batch shape does; the rest keeps it bit for bit. The launch fills only the settings
you left at their defaults: a value from `--set` or from the settings file always
wins, and a choice that contradicts one of your values is left out. Every choice and its reason is written to `strategy.json` in the run
directory and to the settings file the launch generates.

## What is measured

| Section | Source | Content |
| --- | --- | --- |
| `cpu` | sysfs on Linux, `sysctl` on macOS, CPU Sets on Windows; ISA from `inferssd-cpuprobe` | Logical CPUs with core, package, SMT siblings, `cpu_capacity` and maximum frequency; performance and efficiency clusters; whether threads can be pinned; the instruction-set features ggml reports (build options such as OpenMP or repacking are left out) |
| `memory` | `/proc/meminfo`, `hw.memsize` or `GlobalMemoryStatusEx`; `inferssd-cpuprobe` | Total memory, page size, whether the GPU shares system memory, DRAM read bandwidth per thread count |
| `compute` | `inferssd-cpuprobe` | ggml CPU matrix-multiply speed per weight format at 1, 2, 4, ... threads up to every CPU, at the performance cluster's size and at every CPU but one: weight bytes per second for one token, FLOP/s for more tokens, with each point's median call and quartiles; on Linux machines with several CPU clusters, also one token with a thread pinned to each CPU of each cluster and of every CPU, and on the performance cluster's first 1, 2, 4, ... CPUs (rows with `cpus`) |
| `gpus` | `inferssd-memprobe` (Linux with NVIDIA GPUs, or AMD GPUs on ROCm) | VRAM size, host-to-device (pinned and pageable), device-to-host and device-to-device copy rates, and device read bandwidth, for every GPU the engine's CUDA or ROCm runtime finds |
| `storage:<drive>` | `inferssd-ssdprobe`, one per drive | Random-read bandwidth and latency by request size and queue depth, read with the engine's own reader; the knee queue depth per size |

The identity lists the GPUs, each with the backend the engine runs it on: the
NVIDIA GPUs `nvidia-smi` reports, on CUDA; on Linux with ROCm installed (its
`rocminfo` on PATH), the AMD GPUs the kernel's compute driver amdkfd lists
(`/sys/class/kfd/kfd/topology`), on ROCm, by ROCm's UUIDs (`GPU-` and the
device's unique ID in hex, as `rocminfo` and HIP spell them); on macOS the Apple GPU
with its core count as `ioreg` reports it, on Metal (when `ioreg` cannot say, the
list is empty and profiling continues); and the other AMD and the Intel display
controllers, on Vulkan, found on the PCI bus on Linux and among the display
adapters in the registry on Windows. Other display controllers, such as a server's
management display or a virtual machine's, are left out. CUDA and ROCm carry the
expert VRAM tier. The installers build the engine for the first of CUDA, ROCm,
Metal and Vulkan among these GPUs, and for the CPU alone when there is none. The
GPU probe measures every GPU on Linux, regardless of `CUDA_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`; on macOS the GPU shares the
machine's memory and is not measured. A GPU the probe cannot measure (too
little free VRAM, or a runtime error) makes the `gpus` section partial, with the
GPU's UUID and the reason. When `nvidia-smi` is
installed but fails, for example after a driver upgrade without a reboot, the
`gpus` section fails and the saved GPU identity is kept, so the machine is not
mistaken for new hardware.

The CPU probe runs with one thread, the size of the performance cluster and
all logical CPUs. Clusters follow the engine's own rule: on Linux each CPU's
`cpu_capacity` (or, without it, `cpuinfo_max_freq`) is sorted in descending
order and split once at the largest relative drop above 10%; a CPU with
neither value makes the whole machine one cluster. On macOS the
`hw.perflevel0.logicalcpu` CPUs form the performance cluster and the rest the
efficiency cluster. macOS has no thread affinity, so its CPU numbers only count
CPUs. On Windows, a CPU number is 64 times its processor group plus its index in
that group. This also keeps core and SMT sibling records from different groups
distinct. The highest CPU Set efficiency class forms the performance cluster.
A machine without a CPU Set for every CPU has one cluster. `capacity` records
the efficiency class. Package and maximum frequency are unknown. On machines
with multiple processor groups, profiling and inference leave thread placement
to Windows. The probe sizes its thread counts to the CPUs available to the
process. Explicit CPU pinning requires a single-group machine.

The knee of a request size is the smallest queue depth whose best run reaches
the worst run of the fastest queue depth at that size. Each point's bandwidth is
the median of its runs.

Planning takes the measured size nearest an expert-role read on a log scale,
then the best positive measured rate from its knee through the engine's or
reader's queue-depth cap. If the knee exceeds that cap, it uses the best
measured point within the cap and records that limitation; it does not
extrapolate to an unmeasured depth.

Every number names its unit: `_bytes`, `_Bps` (bytes per second), `_us`, `_s`,
`_khz`, and `flops` (FLOP per second).

Facts that change while the machine runs are never stored: available memory,
free VRAM, running GPU processes and cgroup or affinity limits are read at
launch. Machine IDs and drive serial numbers are stored only as SHA-256 digests.

## Drive measurement

The SSD probe never writes. It reads random, aligned extents inside existing
files on the drive being measured, using direct I/O and the same reader the
engine uses (io_uring on Linux, IoRing or overlapped reads on Windows, a pool of
reader threads on macOS). Model shards
are read as raw bytes, so no model format is involved. For each drive the
probe reads the eight largest files of at least 1 GiB among the given paths.

A given directory with no such file below it, such as an empty models directory
before the first download, has its drive measured in a test file instead:
`inferssd profile` writes 4 GiB of random data into it, past the page cache
(`F_NOCACHE` on macOS), has the probe read it, and removes it, also when the
probe fails. It needs 5 GiB free there. The file's name
(`.inferssd-drive-test-<pid>-*`) carries the run's process ID: test files are never
read as model files, and the next run removes those whose process has ended.
When no such directory is given, or there is not enough room, the drive's section
is recorded as failed with the reason.

The test file is also used when every given file is already in the page cache,
which is how a model looks right after its download on a Mac with memory to
spare: the probe skips cached pages on macOS, since reading them would measure
memory, and it finds nothing left to read. The file is written beside the model
and needs the same 5 GiB free there; without that room the section is recorded
as failed, and a later run measures the drive once the pages have been evicted.

A drive is identified by its model and a digest of its serial number, or by
its kernel name when it reports no serial (common for virtual disks). On Linux
the file's device is followed through partitions and device-mapper or md
layers to the physical disks; overlay and network filesystems have no disk and
are identified by filesystem type and mount point. On macOS the key uses the
model and protocol reported by `diskutil` and a digest of the APFS volume UUID;
a filesystem without a disk (network shares, FUSE) is identified by its type
and mount point. On Windows the file's volume is followed to its physical disks,
and each disk is identified by the vendor, product, bus and serial number the
storage driver reports; network shares and volumes without a disk (RAM disks,
substituted drives) are identified by filesystem type and mount point.

## Speed estimates

Estimate single-stream generation after prefill, before the model is downloaded
or run. With `--model`, the report estimates each supported model from the
engine's facts of its tested quantization, which the framework keeps.
`--estimate-gguf` and `--estimate-facts` estimate another checkpoint or
quantization of a supported architecture from its own header: the engine
supplies tensor roles, sizes, formats, layer counts and routed expert layouts,
and the same formulas apply to every checkpoint.

```sh
inferssd profile --estimate-gguf /models/model.gguf --estimate-compute --json
inferssd profile --show --estimate-gguf /models/model.gguf --ple /models/table.gguf --json

# The GGUF can be on another machine. Select the target drive in the saved profile.
inferssd-server --ssd-model-facts /models/model.gguf > model-facts.json
inferssd profile --show --estimate-facts model-facts.json --model /models --json
```

The JSON row gives the reference speed in `decode_tokens_per_second` and its
method in `reference_scenario`. `components` breaks down per-token service
times, `assumptions` records proxies, and `scenarios` compares the bounds on
cache coverage. The reported range adds I/O overlap and the measured timing
spread, without assigning probabilities. `--memory-gib`, `--host-work-mib` and
`--gpu-work-mib` set available host memory and known working allocations.
An estimate plans with the memory available when it is made. `--memory-gib`
replaces that reading with the amount given, up to the installed memory and, on
Linux, the tightest cgroup `memory.max` above the process, since a launch never
gets more. The `estimate memory` line names the amount used. GPU memory is
still the free memory read when the estimate is made.
Where a launch measured a routed model's decode steps on this machine, as it is profiled
now, with the selected engine, the measured dense time of a one-row step replaces the
priced one, and `assumptions` says so.
An explicit target is estimated where a launch runs it as its header describes it: a
decoder needs no model profile; separated routed experts need one, a layout the SSD
path runs and expert formats its CPU executor computes. GPU placement follows the GPU
backends of the selected engine. When its CUDA or ROCm expert tier is active, each
routed layer's up, gate and down combination must also be executable there. The JSON
row gives a null speed and the reason when an executor rejects the format.

**Routed experts.** Routing is not uniform: a few experts take many of the
routes, so a cache with space for 10% of the expert store serves far more than
10% of them. Without a run to observe routing, the estimate assumes one routing
distribution for every model, `rank_serial`: warmed cache slots hold the most
popular experts, and the share of accesses at rank `r` is `min(1/top_k, a/r)`,
with `a` chosen so that the shares sum to one. A token can select an expert at
most once. The estimate converts host and device bytes to
whole aligned slots. Their distinct coverage lies between the larger cache's
slot count and their sum capped at the full store. The reference uses the
midpoint and exposes both endpoints. Device hits run on the GPU. For ready
host-sourced experts, the estimate compares moving zero or more whole experts
in each layer and chooses the fewest moves that minimize completion time. A
fractional mean number of resident hits interpolates between its neighboring
integer counts. `gpu_execution_fraction` includes experts moved to the GPU,
and `h2d_ms` prices their payload at the measured pinned link rate.
Without a measured link or space for a device slot, host-sourced work stays
on the CPU in the estimate. A miss in both caches also incurs a drive read. Prefetch and
changing working sets can move the actual miss rate substantially.

Small expert products also have fixed dispatch cost. When the target is smaller
than all measured shapes of its weight format, two different measured sizes
can supply a fixed time and a per-byte time. The measured shape remains the
fallback. Nearest-format proxies and the dense Q8_0 matvec proxy for MoE are
identified in the report.

**Other expert layouts.** The native SSD store describes separate gate, up and
down tensors. Fused gate/up and nongated experts can execute in the ordinary
graph. For a recognized packed expert tensor, the full tensor consumes
placement capacity, while one token uses only `top_k / expert_count` of its
bytes. The GGUF count, tensor shape and role must agree. Otherwise the estimator
charges a full scan and reports how many expert bytes lacked validated routing
metadata. A model can contain both kinds of experts. A shared tensor is counted
only once. These layout cases have regression coverage with synthetic headers.

**Dense models.** The estimate counts full matrix scans, keeps a tied embedding
when it serves as the output projection and excludes known row lookups and
encoder-only blocks. It prices the quantization mix and allocates weights to
VRAM, pinned RAM and the drive. Disk-sourced bytes also cross the host-to-device
link. The steady pipeline costs the slowest of compute, transfer and storage.
The serial scenario adds them. With no applicable single-GPU measurements the
host reference uses measured host rates and prices spill when the target drive
is known. Multi-GPU sharding, KV reads at long contexts, kernel fusion and
whole-layer rounding can change the observed rate. Missing required measurements
produce a reason alongside a null speed.

**Measured accuracy.** The target was a predicted/actual ratio between 0.5 and
2. Under the [benchmark decode window](benchmark.md#measurement-definitions), five
4K single-GPU MoE comparisons gave the following rates in tok/s. The previous
estimate assumed uniform routing.

| Model | Measured | Previous | Estimate |
| --- | ---: | ---: | ---: |
| DeepSeek-V4.1-Flash IQ2_XS | 10.61 | 3.49 | 11.16 |
| Qwen3.8-Flash-Next Q2_0 | 32.26 | 137.08 | 50.74 |
| Qwen3.8-Flash-Next IQ2_XS | 31.30 | 127.70 | 50.72 |
| Qwen3.8-Flash-Next IQ3_XXS | 27.52 | 96.77 | 50.45 |
| Qwen3.8-Flash-Next IQ3_S | 28.78 | 77.48 | 48.67 |

The Qwen3.8 estimates are 1.6 to 1.8 times the measured rates: attention,
non-matrix graph work and the per-layer dispatch of routed experts are not
priced. On the 4K DeepSeek run the estimate charged 20.85% of routes to disk,
while the engine observed 6.14% missing both caches. The close aggregate rate
therefore masks countervailing errors in individual service times. The routing
assumption put 70.7% of DeepSeek's routes in VRAM, against 73.6% observed, and
86% to 97% of Qwen3.8's, against 97% to 99%.

The dense formula, checked on Qwen2.5-7B-Instruct Q8_0, estimated 123.09 tok/s
against 152.06 measured.
On a mechanical model drive, four identical 19-token prompts in one server
increased from 0.78 to 6.86 tok/s as the cache warmed, while the estimate gave
0.25 tok/s each time: it does not model a cache warmed by earlier requests.
Raw rates, provenance and failed prefill attempts are in the
[validation receipt](benchmarks/decode-estimate-rtx5090.json).

## Where the file lives

| System | Default path |
| --- | --- |
| Linux | `$XDG_STATE_HOME/inferssd/machine.json`, by default `~/.local/state/inferssd/machine.json` |
| macOS | `~/Library/Application Support/inferssd/machine.json` |
| Windows | `%LOCALAPPDATA%\inferssd\machine.json` |

The file is JSON with schema `inferssd-machine/1`. It is replaced atomically,
and a `machine.json.lock` file next to it keeps two profiles from running at
once. Each section records when it was measured, how long it took, the method,
its status (`ok`, `partial` or `failed`) and, when relevant, why.

The decode steps a launch measured with a draft head are kept next to it in
`decode-steps.json` (schema `inferssd-decode-steps/1`), each with what it holds for:
the server build, the characteristics, the model's and the head's files and sizes, and
the launch's arguments other than its port and run files. A launch with all of them the
same reuses the measurement.

## When sections are measured again

The saved file is reused section by section. A run measures:

- everything when there is no saved file, on `--reprofile`, or when the
  hardware identity differs: platform, architecture, machine ID, machine model,
  CPU brand, logical CPU count, GPUs, or total memory by more than 1%;
- the drives given with `--model` when the kernel or OS release changed (saved
  measurements of other drives are dropped, since they predate the change);
- `gpus` when the GPU driver changed;
- the sections of a probe whose output schema changed;
- a drive given with `--model` that the file does not describe yet;
- any section whose last attempt failed or was partial (for example a GPU
  that was busy), except the partial `gpus` section on Windows, which no probe
  measures.

`cpu`, `memory` and `compute` come from one CPU-probe run and are measured
together. When nothing needs measuring, the file is left untouched.

## How long it takes

Reading identity, topology and memory size takes well under a second. The
probes dominate, with these bounds from their default settings:

- The CPU probe times each format, thread count and token count for at least
  0.5 s; with seven formats, three thread counts and three token counts that is
  at least about 30 s, plus the DRAM reads.
- The SSD probe stops each point after 2 GiB or 5 s. With three request sizes,
  six queue depths and two repeats that is at most 36 points, about three
  minutes per drive on a slow drive and much less on a fast one.
- The GPU probe times copies of five sizes, from 256 KiB to 64 MiB, on each device.

Keep other heavy work off the machine while it is being profiled.
