# Typed SSD settings v1

`engine/ggml/src/ggml-ssd/settings/registry.def` is the authoritative settings registry.
The C++ schema/help and packaged Python schema are generated from that table;
the cross-language contract compares the actual C++ output with the installed
Python resource. There are no environment mirrors or legacy-name aliases.

| Key | Kind / unit | Default literal | Producer | Scope | Backend |
|---|---|---|---|---|---|
| `expert.source` | paths / path | `[]` | plan | all |  |
| `cpu.io_cpus` | cpus / CPU ID | `empty` | plan | all |  |
| `cpu.decode_participants` | integer / compute participants | `0` | plan | all |  |
| `prefetch.depth` | integer / predicted experts | `64` | plan | all |  |
| `place.costs` | costs / named us and GB/s terms | `empty` | plan | all |  |
| `prefill.lookahead` | integer / boolean | `1` | plan | all |  |
| `vram.budget` | budget / bytes | `auto` | plan | all |  |
| `vram.max_slots` | integer / slots | `0` | plan | all | CUDA, ROCm |
| `vram.reserve_mib` | integer / MiB | `0` | launch | all | CUDA, ROCm |
| `host.headroom_mib` | integer / MiB | `512` | plan | all |  |
| `host.budget_mib` | integer / MiB | `0` | plan | darwin, windows |  |
| `kv.spill_directory` | paths / path | `[]` | launch | all |  |
| `model.engram_prefetch` | integer / boolean | `0` | model | all |  |
| `model.hc_fused` | integer / boolean | `1` | model | all |  |

These are conservative parser defaults and retained policy priors, not claimed
per-device performance optima. The framework must provide its measured CPU,
cache and placement configuration. The platforms are `linux`, `darwin` and
`windows`; a scope is `all` or the platforms it lists. A backend lists the GPU
backends whose expert tier a key configures: the key parses on every platform, and
an engine built or running without that tier refuses it when it is set. Expert compute runs on
ggml's threads (`-t`, `-tb`, `-C`, `-Cb`, `--cpu-strict`; see
[how it works](architecture.md#threads)); `cpu.io_cpus` places the executor's
own I/O, prefetch and watch threads, and macOS, which cannot pin threads, only
validates it. Zero decode participants computes each decode phase on every decode thread; a
positive value counts the leading ggml threads of each decode phase, and a
decode call with fewer threads fails. Non-applicable platform keys cannot be
explicitly set. The `vram.*` keys size the expert VRAM tier, which CUDA builds have and ROCm
builds for AMD GPUs too (ggml's HIP backend builds the same sources). Its slots each hold
one expert and are sized for the model's largest, so a model that quantizes its
layers differently keeps experts of every layer there. `vram.budget = auto`
gives the tier all the room
the GPU admits; on a GPU that shares the host's memory, such as GB10, that is
the host memory envelope below less the slots the next-layer prefetch lands in,
which are all the RAM cache keeps beside a tier there (with `vram.budget = 0`
the RAM cache takes the envelope). Auto gives no tier where no CUDA or ROCm
backend runs. On Vulkan and Metal, a GPU that shares the host's memory takes the heavy
experts of a prompt pass through the [GPU lane](architecture.md#the-gpu-lane),
which keeps nothing across layers, and `vram.budget = 0` leaves them on the CPU.
On a CPU-only build, or with another GPU backend, the GPU runs the dense layers
while the CPU sink computes every routed expert, and an explicit nonzero budget
is an error. The lane holds no expert across layers, so for it a nonzero budget,
like auto, only turns it on.

`kv.spill_directory` accepts one existing local SSD directory. With a growable attention
cache and one sequence, context fitting admits a larger logical window when the SSD
has room for the cache and the original devices can stage the active layers. The
cache starts resident. Crossing the fitted resident boundary backs it with private
temporary storage and stages attention inputs to their existing devices. Existing
backends that cannot grow their cache keep their normal fitted context.

```sh
inferssd serve -m /path/to/model.gguf --set 'kv.spill_directory=["/path/to/kv-ssd"]'
```

Dense attention can read the entire active KV on each decode token, so generation
throughput after spill depends on how much of the mapping stays cached in memory.
The temporary file is released when the engine exits or the remaining cache fits
back in resident memory.

`vram.reserve_mib` is VRAM that prefill and decode leave free for other processes.
On a discrete GPU it is kept against what this process holds, not against what the
card has free: another process using the reserve takes nothing from the tier, and
one going beyond it fails its own allocation. When other processes release memory
beyond the reserve, the tier grows into it. The tier also leaves 64 MiB above the
reserve for the driver's own memory. A reserve too large for the model fails at
launch. The framework chooses 1024 MiB when other compute processes are on the GPU
at launch, and 0 otherwise; a setting of yours replaces that choice. On a GPU that
shares the host's memory, the reserve adds to the host memory floor below, and the
tier gives memory back under pressure from any process.

In decode, each layer's graph also predicts the next layer's experts, best first, and
the experts of its first `prefetch.depth` ranks that neither the GPU tier nor the
RAM cache holds are read into that layer's RAM cache slots while the drive would
otherwise wait for the next router. It reads them likeliest first, each while its
read pays: a read the drive finishes in the gap saves a read if the expert is used;
one the gap ends during saves the part it has read if the expert is used and takes
the rest of its drive time from the next layer's reads if not, so it pays when the
expert's chance of use is above the share of its read left at the gap's end; one
the gap ends before it starts is dropped. The chance of use is how often the next
layer selected that rank's predictions so far, the gaps and an expert's read time
are measured as decode runs. A read takes the slot of a less likely entry, never
one of another predicted expert. `0` turns the prefetch off.

`prefill.lookahead = 1` keeps the drive reading through the gap between two layers
of a prompt pass, while the GPU computes the next layer's attention and routing.
Each layer's graph also applies the next layer's router to its input, which
predicts the next layer's experts for every token; an expert's chance of use comes
from how often each rank of the prediction was right so far. The experts the
prediction names that neither the GPU tier nor the RAM cache holds are read into
that layer's RAM cache slots, likeliest first, weighed by the decode prefetch's
rule over the gaps the pass has had so far. What a read gains if it lands is the
expert's chance of use, less the chance the pass would have used what its slot
held; a gap that ends during the read costs the share of it left, and one that ends
before the read is issued costs nothing. A plan keeps the run of reads worth the
most, since the gaps differ from layer to layer. The next layer takes what landed
as RAM hits instead of reading them. A read takes a free slot of the layer first, else the layer's coldest
entry of an expert the prediction does not name. When the layer's own slots are all
taken, it borrows slots of the other layers, except the one the pass is computing:
an empty slot first, then the RAM copy of an expert the GPU tier also holds, then
the coldest entry of a layer the pass has computed, and last the coldest entry of a
layer it has yet to reach, which the pass would have used as often as its calls so
far used the entries they held. A borrowed slot serves that one layer's read and
comes back empty, so a pass cycles a few slots instead of emptying other layers. The
read-ahead holds no memory of its own: what it read into the layer's own slots stays
cached. It reads until the next layer's call starts, and a slot's entry goes only
once a read into it is issued. It needs a RAM cache that a prompt pass reads. `0`
turns this off, for A/B timing.

At load, once the server's first pass is done and before any request, the expert
caches take their experts from the drive. No request has named an expert yet, so
each layer's experts are taken in their order: the GPU tier's empty slots, shared
alike among the layers, take each layer's first experts, and the RAM cache's empty
slots the next ones. Nothing from earlier runs is kept or used. An expert loaded
this way and not used since is as good as an empty slot to whatever allocates
later: a prompt's seeds and decode's admissions take its slot as they take an
empty one, and a prompt pass's read-ahead takes it unless the pass is likely to
use it. The experts a prompt puts into the GPU tier, over all the passes the
server runs it in, become the same once the next prompt starts if the decode
between never used them: they stay cached until something takes their slot. On a
discrete GPU the tier is made at load and first grows into the card, once the
load's pass has shown that the device memory other work holds has
settled. The load then leaves the graph reserved for decode's single token, as a
request's prefill does, so the tier grows into the memory a prompt's graph does
not need; each prompt's prefill reserves its own graph and takes back what that
needs from the tier. Every device allocation is admitted against the card's free
memory before it is made, and the tier gives back what it needs: on Windows an
allocation past the free memory would not fail (WDDM backs the excess with host
memory, read over the link and charged to commit). Whenever the tier gives memory back, it keeps its most
valuable experts: the ones in the block it frees move into the slots of the
least used ones first.

The host memory ledger has two terms on every platform, and they size the expert
caches. The envelope is a finite limit on this process's memory less a 512 MiB
reserve: on Linux the tightest cgroup v2 `memory.max`; on Windows a memory limit
of the process's job, when the job sets one, against commit; and on macOS and
Windows `host.budget_mib` against the process's host memory (macOS phys_footprint;
Windows PrivateUsage less the video memory the process holds on discrete GPUs,
which WDDM charges to commit without taking host memory, so the framework's job
limit is the memory budget plus the discrete GPUs' dedicated memory). With `host.budget_mib = 0` and no job limit, the limit is
the machine's physical memory, the budget the framework passes by default. The
system term is the memory available now less the floor, which is at least
`host.headroom_mib` and the reserve: Linux MemAvailable, Windows the smaller of available physical
memory and the commit still available (including what the paging files may grow
by, less the video memory the process may still allocate on the discrete GPUs it
uses, which WDDM will charge to commit; a job limit counts that memory as used
too), macOS what a new allocation takes before the kernel
compresses anything, by its own live thresholds: the free pages above its free
target, the file-backed pages above its file cache minimum
(`vm.vm_page_filecache_min`), and the purgeable pages. Reusable pages are not
counted: they stay resident while the kernel compresses other memory. So a slot
the RAM cache gives back is freed at once on every OS (macOS: each slot is its
own mapping, replaced by a fresh one). The RAM
expert cache admits only under a readable envelope. On a GPU
that shares the host's memory (unified memory, such as GB10), a CUDA allocation
is not charged to the cgroup, so the Linux envelope counts the CUDA expert
tier's storage as used beside the cgroup's own usage: the tier and the RAM
cache share one envelope, and the tier also stays within MemAvailable less the
floor. Every other allocation, such as a compute or work buffer, a scratch
buffer or a server checkpoint, is one the run needs. When it does not fit both
terms it first takes room back from the expert caches: on unified memory from
the GPU tier, whose blocks also answer any CUDA allocation that would fail while
no layer holds them, then from the RAM cache's coldest slots. Then it goes
ahead unless it would exceed one of the limits above itself. The reserve, the
floor and the memory available now only decide how much the caches hold, so a
machine short of memory runs with smaller caches instead of failing.

## File and CLI contract

```text
inferssd-settings 1
cpu.io_cpus = "3"  # source=derived; input=measured CPU placement
cpu.decode_participants = 2
expert.source = ["/path/to/index # 1.tsv", "/path/to/other.tsv"]
place.costs = "cpu_fixed_us=204,link_gbps=12.3"
vram.budget = auto
```

Integers are unsigned decimal literals, not quoted strings. Source lists are
JSON arrays of nonempty UTF-8 strings; `#` inside a quoted path is preserved.
CPU lists accept ordered comma-separated IDs and inclusive ranges. Cost tables
accept the eleven schema-listed `name=value` terms separated by commas or
whitespace; values must be finite, nonnegative and at most 1e6, with positive
`link_gbps`. Omitted cost terms retain the executor priors until measured.

The precedence is command line > file > registry defaults. A launch
fills keys still at their registry default with choices for the machine (see
[machine profile](machine-profile.md#launches)) before it writes the native
file. A file must have
exactly the v1 header. Duplicate file keys, unknown keys, wrong types, bad ranges
and dependency errors fail with key/source context; later CLI occurrences win.
Trailing file comments are retained exactly after the `#` byte. CLI overrides
receive `origin=cli`; file and default values retain their own location/receipt.
Each context owns a copy of the immutable snapshot; no mutable global or ambient
environment is consulted during parsing.

The C++ `ssd::settings` API returns `[[nodiscard]] result<T>` and status/error
values. No parser, loader or telemetry error is thrown across its public boundary.
Typed accessors return const pointers, including null for an absent/wrong kind,
and never perform an unchecked `std::get`. `load` reads a file; `parse` accepts
explicit content, source name, overrides and platform. `schema_json`, `help`,
`cli_option` and `cli_override` use the same registry. The startup retired-prefix
guard accepts an explicit list of environment names and rejects the declared old
prefixes plus `TASK[0-9]+_`; it accepts `TASKRC` and `TASKDATA`.

`inferssd serve` and `cli` take a file with `--settings settings.ssd` and single
values with `--set key=value`:

```sh
inferssd serve -hf REPO --settings settings.ssd --set prefetch.depth=4
```

The launch checks both before it starts the server, and an invalid file or value
stops it with the key and its source. The table above lists every setting; the
machine-readable schema is `framework/src/inferssd/configuration/schema.json`.
Settings are never translated to environment variables. The launch
passes the settings file to the server with `--ssd-settings`, and each setting
with a command-line origin as the matching `--ssd-*` option. CPU thread counts and
placement are server threadpool arguments, not settings; the launch chooses them
for the machine (see the [framework guide](../framework/README.md)).
