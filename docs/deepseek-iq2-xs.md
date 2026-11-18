<!-- SPDX-License-Identifier: Apache-2.0 -->
# DeepSeek-V4.1 GSQ-RCO IQ2_XS

The SSD executor supports text generation with the six-shard
DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XS checkpoint from
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF).
The validation below covers short requests on one RTX 5090. Long contexts,
multimodal input, MTP and full-model CUDA numerical parity are not established.

## Layout and execution

The checkpoint uses `deepseek41.ple.*` metadata, a shared IQ4_NL
`per_layer_token_embd.weight` table and per-layer fused `engram_gate` scales.
It uses the existing shared-table loader, with checked global bucket offsets,
mapped padding tokens and history barriers for token-map entries of `-1`.
The fused gate uses the positive branch for a zero dot product, matching the
upstream reference. The legacy `engram.*` layout keeps its existing behavior.

With mmap enabled, the shared table remains an ordinary CPU mmap; selected rows
page in on demand. Explicit `--no-mmap`, `--mlock` and whole-table tensor validation
retain their eager-loading behavior and can require substantially more RAM.
Nonzero offsets, gaps, overlapping buckets and trailing table padding are
accepted when all bucket indices fit I32 and the table covers them.

Pass a shared table explicitly with `--ple /path/to/table.gguf` (alias
`--shard`). The table can be in a different directory and have any filename.
The native loader and Python source index identify it by its contents: an
extra GGUF must contain only `per_layer_token_embd.weight`. It replaces an
embedded PLE tensor or supplies a separate PLE tensor. Two explicit tables,
mixed-content extras and duplicate physical files are rejected.

Keep the numbered backbone shards together. One missing numbered shard is
accepted only when PLE metadata and the declared tensor count identify a
single missing PLE table; all other split and tensor checks still apply.
A complete numbered checkpoint continues to work without `--ple`. Creating
a numbered link for the external table is unnecessary.

The complete expert index contains 46,080 entries in 120 tensors. Experts mix
IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS; up and gate match in every layer.
These formats use existing quantization kernels. No weights are rewritten.

## Validation

The model files total 265,821,189,184 bytes. Every physical file passed full
SHA-256 verification against the publisher's manifest. The source checkpoint
revision is `65d53bc8be65123e255ff02f3ded958b6dbf54fd` on Hugging Face.

Synthetic tests use real GGUF files to cover external tables, embedded-table
replacement, missing PLE-only shards, malformed splits, duplicate files and
tensors, mixed-content extras, table bounds and explicit loading modes.
They also cover metadata overflow, I32 boundary indices, padding/history
hashes and fused gate values. Legacy Engram and MXFP8 decoder tests remain
in the same suite. Mixed-IQ CPU experts agree bitwise with ordinary CPU MoE
execution. CUDA matrix and cache tests exercise the four expert types.

The integration configuration uses one RTX 5090, an AMD EPYC 7B13 CPU,
CUDA 12.8, context 2048, batch/microbatch 64, eight CPU threads, greedy sampling
and at most 32 generated tokens per request. A Linux cgroup enforces 32 GiB
RAM with swap disabled. Prefetch and prefill lookahead are disabled.
The configured 8 GiB expert tier is a base budget; elastic allocation uses
most of the card. This is not an 8 GiB VRAM result.

Three requests exercise arithmetic, an ice-density explanation and an MoE
definition. Arithmetic returns `5`; the ice explanation stops normally; the
MoE definition reaches the configured output limit. Acceptance requires
successful prefill/decode placement, actual expert reads and VRAM hits,
zero read errors, zero fallbacks, no OOM and a clean server exit.

Implementation commit `083c53a` passed these checks on 2026-11-18 from fresh
CPU and CUDA builds. The model directory contained only numbered shards 1–5;
the PLE table was a regular file in another directory with an arbitrary name.
The index contained 46,080 expert entries. All 1,920 placement events succeeded,
with 16,521 expert reads, 53,362,057,216 expert read bytes and 8,376 VRAM hits.
Peak cgroup memory was 34,041,057,280 bytes (31.70 GiB); whole-device VRAM peaked
at 32,088 MiB including a 308 MiB baseline. Read errors, fallbacks and OOM events
were zero; the server exited successfully and released the GPU. The complete
native CI suite passed all 173 tests under an 8 GiB cgroup without skips.

CPU attention uses the same implementation as `main`; this change does not
alter its precision or KV representation. The stable softplus and
DeepSeek-V4.1 sum-plus-epsilon router fixes are already on `main`.

The remote filesystem exposes a virtual block device whose physical backing
is unknown. This validates disk-offload integration, not SSD throughput.

## Reproduction

Run from a CUDA source build with the framework package available. Use a
fresh run directory and telemetry file:

```sh
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf"
PLE=/path/to/shared-tables/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
RUN_DIR=$(mktemp -d "$PWD/deepseek-run.XXXXXX")
PYTHONPATH=framework/src python -m inferssd.source.gguf_alias \
  --model-glob "$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-*-of-00006.gguf" \
  --ple "$PLE" --out "$RUN_DIR/experts.tsv" --summary-json "$RUN_DIR/experts.json"
cat > "$RUN_DIR/settings.ssd" <<EOF
inferssd-settings 1
expert.source = ["$RUN_DIR/experts.tsv"]
vram.budget = 8589934592
prefetch.depth = 0
prefill.lookahead = 0
EOF
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/inferssd-server \
  -m "$MODEL" --ple "$PLE" -ngl 99 --fit off -c 2048 -b 64 -ub 64 -t 8 -tb 8 \
  --no-warmup --defer-experts --ssd-settings "$RUN_DIR/settings.ssd" \
  --ssd-telemetry "$RUN_DIR/events.jsonl" --jinja \
  --chat-template-file engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --parallel 1 --host 127.0.0.1 --port 8080
```

Use the same `--ple` / `--shard` inputs when indexing and loading. The native
CLI also accepts comma-separated or repeated options and `LLAMA_ARG_SHARD`.
The framework `serve` and `cli` commands forward these inputs to automatic
source indexing and native model loading.

The wrapper requires a Linux user session with memory-controller delegation.
After the server is ready, send a request from another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2 + 3? Answer with just the number."}],"max_tokens":32,"temperature":0,"seed":42,"cache_prompt":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

Build the relevant test targets before running the focused checks:

```sh
ctest --test-dir build-linux-cpu -R '^(test-dsv41-(ple|engram|engram-layouts)|test-moe-router-norm|test-cpu-softplus|test-ple-args)$' --output-on-failure
build-linux-cpu/bin/test-ssd-cpu-sink --dsv41-iq-test --threads 9 --ring-experts 8
CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/test-ssd-vram-mm --dsv41-iq
CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/test-ssd-vram-generic --dsv41-iq
```
