# Shared provider prototype

This is an isolated, default-off C0 experiment. It has not passed a CUDA engine
build or the required original-provider versus extracted-provider golden test.
The supported migration scope is the common v1/alias source used by DeepSeek and
Muse. The source-layout migration and release gates remain incomplete.

## Boundary

With `LLAMA_C0_LINKED_STORE_PROTOTYPE=ON`, `llama` and `ggml-cuda` link one shared
`ggml-ssd-c0-store` library. A static archive linked separately into both would
create two owners, so this experiment deliberately requires a shared library.
The option requires Linux and the bundled ggml; CUDA additionally requires
`GGML_CUDA_MOE_STREAM_BATCH=ON`.

The shared library owns the v1/alias entry vector, source vector, buffered FILE
handles, direct descriptors, mapping cleanup, once state and first-shape state.
CUDA's existing state borrows references to those exact vectors. Entries and
vector addresses stay fixed after publication; CUDA's existing lazy mmap fields
inside each source remain mutable. No independent CUDA copy of the index is
published. The entry pointer subtraction used by the RAM tier consequently
continues to refer to the same array.

The CUDA provider registers before any lookup and supplies its existing
initializer and miss-handling lookup. Pack/alias/list/overlay parsing order,
page-cache mode and existing initialization side effects stay in that callback.
CPU-only builds use the portable parser instead. The oracle fragments were
frozen from the original source before changing the CUDA adapter; their source
blob and SHA-256 are in `oracle/extraction.json`. The extractor only copies this
frozen material and never derives its reference from the implementation under
test.

V2 indexes, Colibri object metadata and lossless sidecar ownership stay in their
original transport state. In particular, the lossless source descriptor is an
alias of the sidecar descriptor, not a new portable-owned handle. Retaining
those old initialization branches is not evidence of full compatibility: this
prototype has not executed their CUDA paths. CPU-only initialization explicitly
rejects active Colibri/lossless sources. The direct CPU entry point rejects the
non-common K3 shape; it does not create a second parser for an unsupported model.

## Publication and shutdown

Registration is final once initialization starts. Duplicate and late
registration are rejected. CUDA's static registration treats rejection as a
fatal configuration error. A CUDA-enabled store without its registered provider
fails initialization instead of silently selecting the CPU-only loader.

The initializer may populate a private-to-initialization half catalog, but the
release/acquire publication flag is set only after it returns successfully.
Lookups wait for `call_once`. Diagnostics only query publication and do not
trigger initialization or inspect an unpublished vector. Failed or disabled
initialization clears entries, unmaps sources and closes both kinds of handle;
the failure remains final. The runtime-disable environment variable is still
checked at every raw lookup, after initialization.

Consumers borrow descriptors and pointers until all reads stop. Provider
unregistration retires callbacks and prevents a later fallback or replacement;
it does not wait for active callbacks or outstanding I/O. **Concurrent backend
unload is unsupported.** The caller must stop all consumers before unloading a
backend. A load-once, process-fixed provider is the intended experiment mode.
The existing ggml backend registry itself deliberately keeps dynamic backend
handles loaded at shutdown because backend resources can include live threads;
the mock-module test does not change that engine limitation.

Normal construction puts the owner before CUDA's borrowing state, and the
registration object after that state. Runtime atexit handlers stop readers and
report before retirement; the owner is destroyed after borrowing state. Its
actual close/munmap at teardown is new relative to legacy process-exit cleanup.
The mock module test verifies the ownership contract, not the real CUDA worker
shutdown order. That remains a required engine/device gate. No singleton is
deliberately leaked to avoid this check.

## Lightweight tests

Configure `experiments/c0-store` as a standalone project. `C0_SANITIZERS=ON`
enables ASan/UBSan. The tests link an executable and a dynamically loaded mock
provider against the same shared store and cover:

- CPU-only initialization and CUDA-required missing-provider rejection;
- identical catalog, entry and source addresses across two consumers;
- concurrent first initialization, stable descriptors and path replacement;
- dynamic runtime disable after initialization and one first-shape decision;
- duplicate/late registration, failed and disabled half catalogs;
- a paused half catalog that observers cannot see before publication;
- retirement before module unload, no fallback, normal final descriptor closure;
- page-cache rejection by the Linux direct-range API and logical offset/length.

The Linux tests also exercise real O_DIRECT files and the existing portable
parser oracle. No GPU, large model or performance inference is involved. The
remaining C0 acceptance test is an original CUDA provider versus shared provider
comparison through the same CPU sink, with finite RAM and full numerical
evidence on the frozen F/F-prime pair.
