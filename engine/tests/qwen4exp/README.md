# Qwen4Exp CPU integration fixture

The fixture uses deterministic small weights, Q8_0 routed experts (12 experts,
top-10), BF16 QSA indexer projections, four HC streams, three GDN layers and one
QSA layer. PLE has 16 hash heads, a quantized mmap table and nonzero dilated
convolution weights. This checks implementation behavior; it is not a model
quality or speed benchmark.

Generate it with Python and NumPy:

```sh
python3 tests/qwen4exp/make_fixture.py /tmp/qwen4exp-fixture.gguf
```

Build the fork CPU library, then compile the probe against that build:

```sh
c++ -std=c++17 -O2 tests/qwen4exp/probe.cpp -Iinclude -Iggml/include \
  -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama -o /tmp/qwen4exp-probe
/tmp/qwen4exp-probe /tmp/qwen4exp-fixture.gguf /path/to/real-text.txt /tmp/logits.bin 8 0
```

The probe uses the first 48 bytes of the supplied text as IDs in its fixture byte
vocabulary. It emits all 256 logits at every position and checks whole-context
restore, sequence restore with other occupied cells, and sequence copy. Chunk
sizes 48, 8 and 1 cover prefill and decode. The final argument selects flash
attention. `QWEN_UNIFIED=0` exercises separate KV streams. `QWEN_PROBE_ONLY=1`
omits the state checks for syscall tracing. Compile with `QWEN_INTERNAL` and
`-Isrc -lggml-base` to print the mapped PLE address for page-advice auditing.

The same probe compiles against Unsloth eaf9376 with `-DQWEN_UPSTREAM` to adapt
its load-mode API. Compare the binary logits using identical fixture files and
text. The upstream revision has a known separate-stream sequence-copy failure:
its indexer update context omits the pending stream copy. The fork's Qwen memory
wrapper applies the indexer update alongside attention.

To exercise the optional boolean recurrent-layer metadata array, generate a
second fixture with `--recurrent-array` after the output path. Its weights and
layer pattern are identical.

On shared host 166, wrap compilation and CPU-heavy work in `quiet-build.sh`, use
at most `-j8`, and run probes under the session's `gpu-lock.sh` verification label.

The streamed fixture matrix uses the same GGUF and real text for its ordinary
CPU/mmap baseline and every expert-pack path. It emits 96 byte-vocabulary IDs,
24,576 logits per case, and JSONL route IDs/normalized weights through the graph
evaluation callback. Compile the probe against the CUDA-enabled build with
`-DQWEN_INTERNAL -Isrc -lggml-base -Wl,--no-as-needed -lggml-cuda`, and build the
`kimi-vram-tier` target. Create the fixture pack with
`create-moe-expert-pack.py`, using its default expert-major v1 layout.

```sh
python3 tests/qwen4exp/streamed.py --probe /path/to/probe \
  --fixture /path/to/fixture.gguf --text /path/to/real-text.txt \
  --pack /path/to/fixture.pack --tier build-cuda130/bin/libkimi-vram-tier.so \
  --output /path/to/evidence
```

Run this under the GPU lock and quiet-build wrapper. RAM admission requires a
finite memory cgroup (the validation uses a private 8 GiB systemd scope). The
matrix checks CPU decode, generic/two/four-column grouped prefill, chunked and
mixed execution, resident VRAM, and RAM/SSD transient uploads. Scheduler costs
are deliberately forced to select GPU work and are not performance estimates.
`--cases` selects diagnostic cases, and `--ring` changes the bounded I/O ring
(default 2). Logs and counters must confirm the selected path actually ran.

The probe invokes the existing request-phase hook for streamed runs: chunk size
alone does not switch between prefill and decode. `QWEN_PREFILL_TOKENS` selects
the initial prefill length; subsequent tokens decode singly. `QWEN_WARMUP=1`
warms 24 decode tokens before clearing model memory, with tier retention enabled
by the matrix. `QWEN_GPU_LAYERS=99 QWEN_STREAMED=1 QWEN_LOAD_ONLY=1` with an
internal probe verifies that deferred expert buffers remain on the host while
dense layers offload. `QWEN_UBATCH` configures the probe batch bound, and
`QWEN_ROUTE_AUDIT` records route tensors. The fixture remains a correctness test,
not a real-model quality or speed measurement.

For Phase 2c real-model GPU dense controls, run `build-gpu-dense-probes.sh`
under a verify GPU lock and quiet-build, then `run-gpu-dense-matrix.sh` under
that lock. The matrix checks preserved code64/code544 IDs and batch schedules.
`QWEN_EXPERTS_CPU=1` supplies the CLI `exps=CPU` weight override;
`QWEN_OP_OFFLOAD=0` also prevents CPU expert operations from being offloaded
back to CUDA during prefill. Dense GPU weight operations and attention remain
on CUDA. `QWEN_CAPTURE_OFF=1` disables intermediate observation for quality
measurements: capture callbacks can split graphs and change CUDA fusion and
quantized arithmetic. Captured diagnostic results must therefore be labeled
separately from the unobserved quality controls.

`QWEN_CAPTURE_REGEX` selects named diagnostic nodes.
`QWEN_CAPTURE_SOURCE_REGEX` additionally records the underlying matmul's real
operands and integer operation parameters (with a strict 64 MiB operand cap).
HC gate captures also preserve their down-projection operands. Compile
`replay-matmul.cpp` against each backend to replay these exact operands,
independent of the full model, sink, and tier. Its positional arguments are
`cpu|gpu operand0-stem operand1-stem output-stem`; each stem has native `.meta`
and `.bin` files. The tool requires contiguous operands and fails explicitly
otherwise. `PROBE_BIN` selects an alternate real probe in the private launcher's
probe mode without changing server mode.

The preliminary real-server driver uses `pack-gate.py` to gate the active pack.
A failed gate probes the specified read-only cold shard, permits one direct-I/O
copy-and-cmp refresh only if that cold gate passes, and stops on a subsequent
failure. `summarize-measure.py` retains full/first128/tail128 audit throughput,
prefill and TTFT, own-scope read counters and their source, and disjoint expert
VRAM/RAM/SSD routes. These single-run results are preliminary observations.

For normalization replay, set `QWEN_REPLAY_RMS_EPS` to the verified model
epsilon. Only operand0 is loaded in that mode; the required operand1 argument
is ignored. Preserve the input tensor native shape so the normalization group
width matches the model.

`QWEN_CAPTURE_SKIP_DUMP=1` preserves the original callback selection and
graph split points but omits tensor file writes. Use it only for an explicitly
labeled observation A/B diagnostic; `QWEN_CAPTURE_OFF=1` remains the setting
for quality measurements.
