# Recorded M5 Mac raw-output reference

These are executed reference outputs from the original reference engine on an
Apple M5 Max with macOS 26.4, Apple Clang 21 and SDK 26.5. Three synthetic GGUF
models cover DeepSeek4.1 MXFP4 and Muse Q4_K/Q5_K. Each model ran in two
independent processes; each process made two KV-cleared requests. All four
full-vocabulary float32 rows matched across those repetitions. The files here
retain the first request of the first original recording, never regenerated
expected outputs.

The profile uses the actual Mac GGUF-alias reader, a 4 MiB LFU expert cache,
read-ahead depth 2, 4096-byte alignment and expert prefill. Experts execute on
CPU; dense operations execute on Metal. The execution gate checked positive
reader/prefill/decode activity, read-ahead submitted/used, bounded cache
occupancy, second-request hits and CPU/Metal compute-node assignment. It also
bound actual source files and mapped libraries and required normal shutdown.
It did not claim LFU eviction or physical SSD bandwidth.

`manifest.json` binds exact input and library hashes, generator recipe,
compiler/native features, settings, phase calls and output rows. There are 259
finite little-endian IEEE float32 logits per row. Selection is raw argmax with
the lowest token ID on ties. Four prompt tokens `[3,4,5,6]`, followed by three
single-token calls, produce four output decisions at positions 3 through 6.
Requested context 128 becomes 256; batch and ubatch are 16. No sampler, EOS
shortening or tolerance comparison is used. The exact C API runner is included.
Its clock boundaries include phase calls and raw-file writes and are not pure
inference timing measurements.

The engine uses native Release arithmetic with Metal, embedded Metal shaders,
Accelerate/Apple BLAS and LLAMAFILE. It does not set `-ffp-contract=off`; that flag
belongs only to the small runner. Linux/aarch64 and CUDA references are separate
profiles and cannot be substituted for these bytes.

## Inputs and offline comparison

`make-inputs.py` wraps the unchanged `c0-store/make-tiny.py` generator using
NumPy 2.0.2, seed 20261022 and the recorded `gguf-py` source. Only the expert
count changes: DeepSeek=384 and Muse=256, so the actual production Mac reader
guard is entered. The embedding width remains 256, with two layers and two
selected experts. The original 16-expert Linux inputs do not cover this reader.
The recipe was recorded at revision `27a69e724e36d8c054e42f64913d24c02a89eb4d`
of this repository, which contains the generator and the
exact recorded GGUF Python sources. The reference engine revision is earlier and
does not itself contain the input generator. Verify all generated GGUF hashes
against this manifest. The generator also writes auxiliary packs, but the Mac
gate executes only GGUF aliases.

```sh
python3 make-inputs.py --repo path/to/pinned/source --out path/to/new/inputs
python3 verify.py --model deepseek41-mxfp4 --actual-dir path/to/completed/arm
```

Alias paths depend on the output directory. Their normalized hash replaces only
the source-path column with `<MODEL_DIR>/` plus the GGUF basename; all other
columns, row order, tabs and line endings remain fixed. Regenerate valid actual
paths for the selected directory and bind the source bytes before execution.
The recipe only generates inputs, never output expectations.

The verifier reads a completed arm's `raw-ready.json`, `result.json`,
`runner-events.stdout`, two request JSON files and exactly eight raw float rows.
It checks READY, saved begin/end/DONE ordering, complete requests, geometry,
finite raw bytes and exact token selection against the original reference. It
never launches a model or changes the stored expectations. A matching artifact
cannot establish actual process/library/source identity, cache/Metal coverage,
resource limits or cleanup; those remain separate execution-gate requirements.

## Resource evidence and limits

The original numerical recording had a sampling gap: most sub-second processes
exited between periodic loaded-RSS samples. A separately admitted six-process
supplement added forced samples after READY and after both requests while the
same PID was alive. All 48 supplement rows matched the immutable originals;
all 12 loaded samples were positive, approximately 45-55 MiB. The supplement
used a 2 GiB sampled child-RSS budget, 3 GiB free+inactive reserve and sampled
compressor/swap-growth thresholds. These are soft observations, not a Darwin
cgroup or proof of an instantaneous unified-memory upper bound. The original
gap remains documented and is not rewritten as a passing resource measurement.

This reference is specific to the recorded M5/native/compiler/OS profile. On a
different platform, first record the reference engine on that same host/profile,
then keep the byte-exact comparison. These tiny inputs omit DeepSeek compressed
attention and do not revalidate production long contexts, large cache pressure,
failure recovery, full-model quality or throughput. Run the included runner only
under a reviewed controller with the device lock, STOP, deadlines,
source/library identities and cleanup checks; this directory is an offline
reference artifact.
