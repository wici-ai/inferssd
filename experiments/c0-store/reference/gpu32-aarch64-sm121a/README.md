# Recorded GPU32 raw-output reference

These outputs were recorded from the original F engine in six tiny model/source
cases, then compared with the extracted shared-provider candidate. Every case
matched all four complete float32 rows and all four independently selected
argmax tokens exactly. The files here retain the original F rows. They are
executed reference outputs, not generated expectations.

The profile uses 32 input tokens, GPU graph prefill with BATCH1 io_uring expert
transport, then three single-token CPU-sink decode calls. Those four calls yield
four output decisions at positions 31, 32, 33 and 34. Selection uses finite raw
logits, the lowest token ID for ties, and no EOS early exit. The exact runner is
included unchanged. Each process makes one request. This differs from the
CPU-only reference's four-token prompt and two requests; do not compare their
rows as if they represented the same protocol.

The recorded platform is Linux aarch64 with NVIDIA GB10, CUDA 13.0.88, sm_121a,
GCC 13.3.0 and driver 580.82.09. `manifest.json` includes the build flags, actual
library hashes, synthetic generator/seed, input hashes, normalized alias
identity, environment, phase transitions, and per-row geometry/hash/selection.
CUDA uses `-use_fast_math`; host C/C++ uses `-ffp-contract=off`. Do not interpret
the host flag as disabling CUDA fast math. Requested context size 128 became
256 in the engine; the saved READY artifact must contain the actual value.
The generator uses NumPy 2.0.2 and the pinned `gguf-py` source.

The alias normalization replaces only the `source_path` column with its
basename. It retains the header, row order, other column text, tab separators
and an LF after each row. This normalization is valid only for these one-GGUF
fixtures. Generate the inputs using the pinned recipe and bind their hashes;
do not use an alias pointing at unrelated payloads. The reference does not
contain real model weights or private machine paths.

## Offline comparison

Run the verifier against an existing completed GPU32 arm directory:

```sh
python3 verify.py --actual-dir "$ACTUAL_DIR" --model deepseek41-mxfp4 --mode pack
```

The directory must contain `raw-ready.json`, the original worker `result.json`,
`request-1.json`, and exactly four `request-1-step-*.f32` files. The verifier
checks one complete request, READY, worker-attested completion, row positions,
batch shape, full-vocabulary bytes, finite values, and lowest-ID argmax. It
rejects extra requests or raw rows. It never launches a model, writes expected
rows, or accepts a numerical tolerance.

The original worker checked the runner's DONE event live before writing
`requests_complete=true`. Raw DONE event lines were **not retained**. Therefore
this verifier checks that completion receipt; it does not independently prove
event ordering. It also does not attest actual process/library/source identity,
transport coverage, finite resources, or cleanup. Those checks remain in the
separate execution gate. A successful artifact comparison cannot replace them.

## Scope

The executed gate used 4 GiB effective RAM, zero swap, a machine reserve, the
shared device lock and STOP protocol, actual mapped-library/source identities,
positive BATCH1 I/O and H2D counters, complete CQE histories, zero final in-flight
reads, and normal process/scope exit. DeepSeek exercised six transport batches
per process; Muse exercised three. This does not require every prefill layer to
run on GPU: the engine can prune the final layer to its requested output.

This is a reference for this exact architecture, compiler/ISA profile, inputs
and request protocol. It is not a cross-platform checksum, GPU-decode or VRAM-tier
golden, throughput benchmark, proof of explicit main-ring queue destruction,
failed-drain recovery, or hot backend unloading. On another architecture or
compiler profile, first record the frozen baseline on that same host/profile;
keep the byte-exact comparison requirement.
