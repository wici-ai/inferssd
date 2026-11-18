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

Keep all six numbered shards together. The shared
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` file may be linked under shard 6's
numbered name. Source indexing validates the numbered names before resolving
physical files, and still checks split metadata and duplicate sources.
The complete index contains 46,080 entries in 120 expert tensors. Experts mix
IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS; up and gate match in every layer.
These formats use existing quantization kernels. No weights are rewritten.

## Validation

The model files total 265,821,189,184 bytes. Every physical file passed full
SHA-256 verification against the publisher's manifest. The source checkpoint
revision is `65d53bc8be65123e255ff02f3ded958b6dbf54fd` on Hugging Face.

Synthetic tests cover malformed metadata, overflow, I32 boundary indices,
padding/history hashes and fused gate values. Existing legacy Engram and
MXFP8 decoder tests remain in the same suite. Mixed-IQ CPU experts agree
bitwise with ordinary CPU MoE execution. CUDA matrix tests exercise real
expert dimensions and prefill/decode shapes (64 comparisons, NMSE below
`5e-4`); cache tests exercise hits and misses across the four types
(100 comparisons, relative L2 below `0.02`).

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
The final source snapshot, based on main `746253a` with the accompanying
numerical fixes, passed these checks on 2026-11-17. Peak cgroup memory was
34,039,660,544 bytes (31.70 GiB); whole-device VRAM peaked at 32,098 MiB,
including a 308 MiB baseline. All 1,920 placement events succeeded, with
18,381 expert read requests, 59,428,999,168 expert read bytes and 8,421 VRAM
hits. The executor reported zero read errors and zero fallbacks, the cgroup
reported no OOM events, and the server exited successfully and released the GPU.
The remote filesystem exposes a virtual block device whose physical backing
is unknown. This establishes disk-offload integration, not SSD throughput.

The accompanying numerical fixes honor explicit `GGML_PREC_F32` in CPU
attention, retain softplus's small positive negative-tail values and preserve
DeepSeek-V4.1's sum-plus-epsilon router denominator. Default attention
precision and other architectures' router guards remain unchanged.
Earlier focused CPU comparisons against llama.cpp
`9581154a2250cc9e6948e4c2a2fc4b4e19d1a57e` matched all generated token IDs
and returned top-five log probabilities for two prompts, after aligning
prefill partitions and disabling upstream tiled matmul and fusion.
That evidence is specific to those CPU samples and execution settings.

## Reproduction

Run from a CUDA source build with the framework package available. Use a
fresh run directory and telemetry file:

```sh
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf"
RUN_DIR=$(mktemp -d "$PWD/deepseek-run.XXXXXX")
PYTHONPATH=framework/src python -m inferssd.source.gguf_alias \
  --model-glob "$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-*-of-00006.gguf" \
  --out "$RUN_DIR/experts.tsv" --summary-json "$RUN_DIR/experts.json"
cat > "$RUN_DIR/settings.ssd" <<EOF
inferssd-settings 1
expert.source = ["$RUN_DIR/experts.tsv"]
vram.budget = 8589934592
prefetch.depth = 0
prefill.lookahead = 0
EOF
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/inferssd-server \
  -m "$MODEL" -ngl 99 --fit off -c 2048 -b 64 -ub 64 -t 8 -tb 8 \
  --no-warmup --defer-experts --ssd-settings "$RUN_DIR/settings.ssd" \
  --ssd-telemetry "$RUN_DIR/events.jsonl" --jinja \
  --chat-template-file engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --parallel 1 --host 127.0.0.1 --port 8080
```

The wrapper requires a Linux user session with memory-controller delegation.
After the server is ready, send a request from another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2 + 3? Answer with just the number."}],"max_tokens":32,"temperature":0,"seed":42,"cache_prompt":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

Build the relevant test targets before running the focused checks:

```sh
ctest --test-dir build-linux-cpu -R '^(test-dsv41-(ple|engram|engram-layouts)|test-moe-router-norm|test-cpu-attention-precision|test-cpu-softplus)$' --output-on-failure
build-linux-cpu/bin/test-ssd-cpu-sink --dsv41-iq-test --threads 9 --ring-experts 8
CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/test-ssd-vram-mm --dsv41-iq
CUDA_VISIBLE_DEVICES=0 build-linux-cuda/bin/test-ssd-vram-generic --dsv41-iq
```
