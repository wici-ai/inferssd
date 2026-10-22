# Recorded CPU raw-output reference

These are executed reference outputs for the Linux aarch64 CPU sink, recorded
after the shared source-store extraction. They are not generated expectations.
Three synthetic model types each ran with pack and GGUF-alias sources. For every
case, the original F provider and two independent CPU-only candidate processes
produced exactly the same finite full-vocabulary float32 bytes and selected IDs.
Each process repeated the request after clearing KV, with identical results.
The matrix also required grouped prefill, cache admission/reuse, actual source
and library identities, bounded memory, and normal process/scope cleanup.

`manifest.json` binds the recording source, compiler/profile, input GGUF and pack
hashes, settings, phase transitions, runner, token IDs, and each raw row. The
files retain the first candidate request; the second request and independent
recording matched it byte for byte. There are 259 logits per row, encoded as
little-endian IEEE binary32. Selection is argmax with the lowest-ID tie break.

The synthetic generator is `experiments/c0-store/make-tiny.py`, using the pinned
repository `gguf-py`, NumPy 2.0.2, and seed 20261022. Verify the resulting GGUF
and pack hashes against this manifest before using them. Alias TSV absolute
paths depend on the fixture directory, so their whole-file hashes are not
portable: regenerate their ranges from the same GGUF and verify those ranges.
The manifest records each source-selection environment key and a normalized
alias-table hash, replacing only the source-path column with its basename.
The generator's candidate labels are retained to preserve the exact input
bytes; generating a file does not create or approve golden outputs.

`raw-logits-runner.cpp` is the exact runner used for the recording. It directly
links the pinned engine's existing phase lifecycle export, which is not part
of the public llama header. It accepts explicit model/output/backend paths and
the commands `RUN 1`, `RUN 2`, then `EXIT` on stdin. A controller must wait for
each `ready`/`done` acknowledgement. The requested context is 128; the actual
engine context is 256. All effective counts are recorded in the ready event.

Run this runner only through a reviewed device controller that retains the
existing device lock, STOP handling, deadlines, finite memory/swap limits,
library/source identity checks, path coverage, and normal cleanup evidence.
This directory does not replace those resource and execution checks. Its
standalone comparison command consumes already completed output:

```sh
python3 experiments/c0-store/reference/cpu-aarch64/verify.py \
  --model deepseek41-mxfp4 --mode alias --actual-dir path/to/completed-arm
```

Both requests must contain all four output rows. Any byte, ID, shape, position,
or request mismatch fails; the command never rewrites expected data. Matching
artifacts alone do not prove the intended provider, cache, source mode or device
was used. Check those facts separately before accepting a refactoring gate.

The bitwise reference applies to the recorded aarch64 compiler/build profile.
It is not a universal x86, CUDA or Metal golden, a large-model quality result,
or a performance benchmark. DeepSeek compressed attention is absent from these
tiny inputs. GPU transport and final real-model validation are separate gates.
