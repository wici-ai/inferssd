<!-- SPDX-License-Identifier: Apache-2.0 -->
# Inkling 2 Mini

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) from Thinking Machines is a 26B-parameter mixture-of-experts model with about 3B active parameters per token. Its 24 decoder layers hold 128 experts each, of which a token uses 8. Layers 6 to 17 form a loop core that runs three times, so a token makes 48 layer visits. Each visit keeps its own KV cache and short-convolution state. The GGUF architecture is `inkling2_loop`, with the same key and tensor names as upstream llama.cpp.

The checkpoint also ships a DSpark drafter (`mtp.safetensors`, 318M parameters): a three-layer transformer trained with eight positions per block. This implementation can extend a draft and verify multiple candidate paths in one target pass. Its GGUF architecture is `inkling2_dspark`.

This branch is a runnable reference for contributors evaluating speculative decoding on other InferSSD models. On a 48 GB M4 Pro, DSpark took the built-in greedy test from 33.59 to 62.09 generated tokens/s. Both runs produced the same 256 token IDs. With all 24 physical layers streaming experts from SSD, the target used 14.59 GiB less Metal weight memory. The short HTTP test below compares resident and streamed experts with drafting on and off.

## Weights

The converter writes the checkpoint's values without re-quantizing them.

| Model tensors | GGUF type |
| --- | --- |
| Experts of layers 0 to 19, NVFP4 | `NVFP4` blocks with an F32 scale per expert, repacked bit for bit |
| Experts of layers 20 to 23, MXFP8 | `MXFP8` |
| Attention projections, embeddings, LM head (BF16) | `BF16` |
| Norms, router, convolution kernels, clip scales, loop controllers | `F32`, widened from BF16 as upstream does |

| Drafter tensors | GGUF type |
| --- | --- |
| Attention and feed-forward projections, Markov head | `BF16` |
| Norm weights | `F32` |

The model file is 18.97 GB and the drafter file is 648 MB. Dequantizing the experts with ggml's CPU math reproduces the checkpoint's reference dequantization exactly.

The loop controllers and the router hold BF16 values in F32. At load the engine packs them into BF16 copies, which hold the same values and halve their reads.

## Convert

```sh
python3 -I engine/convert_hf_to_gguf.py /path/to/Inkling-2-Mini --outfile Inkling-2-Mini-native.gguf
python3 -I engine/convert_hf_to_gguf.py /path/to/Inkling-2-Mini --mtp --outfile Inkling-2-Mini-dspark-native.gguf
```

Install the converter dependencies from `engine/requirements/requirements-convert_hf_to_gguf.txt` in a compatible Python environment. Use the full checkpoint, including its tokenizer files and `mtp.safetensors`. The first command reads the language model and skips the vision tower. The second converts the drafter. Two environment variables change the expert format:

- `INKLING2_MXFP8_BF16=1` writes the MXFP8 experts as exact BF16, for backends without MXFP8 matrix multiplication (the file grows to about 22.9 GB).
- `INKLING2_NVFP4_NATIVE=0` dequantizes the NVFP4 experts instead of repacking them.

## Build and Run on Apple Silicon

From the root of this checkout, use a Python virtual environment and build the Metal server, probes and tests.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./framework
cmake -S . -B build-mac -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-mac -j 8 --target inferssd-server inferssd-ssdprobe inferssd-cpuprobe test-inkling-dspark test-flash-attn-indexed-bias test-inkling-sconv-rollback test-ssd-source-index
python scripts/engine-resources.py --build build-mac --output build-mac/engine-resources.json
```

Start the server or run a single prompt with the converted files.

```sh
inferssd serve -m /path/to/Inkling-2-Mini-native.gguf --resources build-mac/engine-resources.json --port 8080
inferssd cli   -m /path/to/Inkling-2-Mini-native.gguf --resources build-mac/engine-resources.json -p "Implement an O(1) LRUCache in Python."
```

Both commands load the drafter when it sits next to the model as `<stem>-dspark-native.gguf` (for a model named `<stem>-native.gguf`), `<stem>-dspark.gguf` or `<stem>-mtp.gguf`. `--draft-model PATH` names another drafter and `--draft-model off` disables drafting. `--draft-max N` caps the draft per step. The launcher passes 16 by default. Tree verification takes up to 15 candidates plus the anchor token.

The launcher defaults to resident experts and an 8,192-token context on the tested machine, keeping 4 context checkpoints (`--ctx-checkpoints 4`) when at least 8 GiB of unified memory headroom remains and `--inkling-ssd-layers` is not set. Multi-turn conversations persist both the target prompt state and a `.prompt.dspark` drafter KV sidecar in the disk prompt cache so branching or interleaved sessions restore both caches without re-prefilling. Drafting adds the drafter's 648 MB and about 258 MiB for its 48,000-slot draft vocabulary. The chat template maps OpenAI reasoning efforts `none`, `minimal`, `low`, `medium`, `high` and `xhigh` (or `max`) to 0, 0.1, 0.2, 0.7, 0.9 and 0.99. The default is `high`.

## Stream Experts From SSD

On macOS Metal, `--inkling-ssd-layers N` streams experts for the first N **physical** target layers. Choose 0 for resident weights, 1 for the first layer, or 24 for all layers of Inkling 2 Mini. For example, start the server with the target GGUF on a local SSD. Leave the sibling drafter in place to use DSpark, or append `--draft-model off` to run the target alone.

```sh
inferssd serve -m /path/to/Inkling-2-Mini-native.gguf \
  --resources build-mac/engine-resources.json --port 8080 \
  --context 2048 --inkling-ssd-layers 24
```

For each visit to an offloaded layer, the router selects experts for the microbatch. The engine reads each distinct expert's fused gate/up and down weights once from the GGUF with macOS file caching disabled on the read descriptor. Metal runs the existing quantized matmuls on the staged weights. Dense target weights, expert scales and the optional drafter stay resident. A later visit reads its selected experts again.

| Physical layers on SSD | DSpark | Target Metal weight buffer (MiB) | Decode (tokens/s) |
| ---: | --- | ---: | ---: |
| 0 | Off | 17,200.77 | 35.41 |
| 1 | Off | 16,654.02 | 26.18 |
| 24 | Off | 2,256.27 | 2.41 |
| 0 | On | 17,200.77 | 74.60 |
| 24 | On | 2,256.27 | 3.57 |

Each row is one warmed HTTP request on the same 48 GB M4 Pro. All five returned the same 24 greedy token IDs. The weight column excludes compute buffers, CPU weights, KV cache and drafter allocations. With drafting off, each one-token decode step read 35,831,808 expert bytes for one layer and 1,839,366,144 bytes for all 24. In DSpark mode the byte count depends on the experts selected by the verification batch. Inspect `inkling_ssd: batch_tokens=… direct_read_bytes=…` in the server log for GGUF payload bytes read at each step.

Restart the server for each configuration. Run this warmup and measurement against each server, then compare the `tokens` arrays. The rate uses the server's decode timing and excludes prefill.

```sh
for n in 1 24; do
  jq -n --argjson n "$n" '{
    prompt: "def merge_sorted(a, b):\n    ", n_predict: $n, temperature: 0,
    seed: 42, ignore_eos: true, cache_prompt: false, return_tokens: true
  }' | curl -fsS http://127.0.0.1:8080/completion \
       -H 'Content-Type: application/json' --data-binary @- | \
       jq -e --argjson n "$n" 'select((.tokens | length) == $n and .timings.predicted_n == $n)
         | {tokens, tokens_per_second: .timings.predicted_per_second}'
done
```

## Reproduce the Decode Comparison

```sh
build-mac/bin/test-inkling-dspark /path/to/Inkling-2-Mini-native.gguf /path/to/Inkling-2-Mini-dspark-native.gguf
build-mac/bin/test-flash-attn-indexed-bias
build-mac/bin/test-inkling-sconv-rollback
```

The full-model test formats a 53-token Python `merge_sorted` prompt, sets reasoning effort to `low` and uses greedy decoding with a 2,048-token context. It warms the kernels, generates 256 tokens with each configuration and compares their IDs. The timer covers 255 tokens after the first token. Passing both file paths selects the full-model test.

| Workload | Prompt tokens | Target only (tokens/s) | With DSpark (tokens/s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Greedy parity test | 53 | 33.59 | 62.09 | 1.85× |
| Server, Python LRU cache | 27 | 33.63 | 49.37 | 1.47× |
| Server, rail telemetry summary | 4,458 | 30.76 | 44.25 | 1.44× |

These are single paired runs on an Apple M4 Pro with 48 GB unified memory and macOS 27.2.1. The two server workloads used resident experts, the same build, native weights and 8,192-token context. Start the server once with the sibling drafter and once with `--draft-model off`. In each process send the following user messages in order:

1. `Implement an O(1) LRU cache in Python.`
2. The full contents of [the rail telemetry prompt](../bench/prompts/prompt-4096.txt), including its final newline.

Use `/v1/chat/completions` with `temperature=0`, `seed=42`, `reasoning_effort="low"`, `max_tokens=256`, `ignore_eos=true`, `cache_prompt=false` and `timings_per_token=true`. The server rates use `timings.predicted_per_second` and exclude prefill. Both arms returned 256 tokens for each request. Their returned text differed on both prompts. The token-for-token match in the table applies only to the built-in parity test.

The tested [checkpoint revision](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a) produced these local GGUF files.

| File | SHA-256 |
| --- | --- |
| `Inkling-2-Mini-native.gguf` | `c537e19573ec121bb4ff6b4ab6fb5d967df3f84f7cc3e3b031f9f424ee2a03ef` |
| `Inkling-2-Mini-dspark-native.gguf` | `ce4f577781b4244443bac6b4584afbf3b0ec7653f55a1433878137ff5705494b` |

## What to Try on Another Model

| Idea | Starting point in this branch | First experiment |
| --- | --- | --- |
| Stage routed weights once per microbatch | [Inkling expert reader](../engine/src/ssd/inkling-experts.cpp) maps repeated routes to compact slots and reads each selected expert once per layer visit. | On another model with packed expert banks, compare returned token IDs, direct-read bytes and decode rate at the same offload depth. |
| Choose how many draft candidates to verify | [DSpark candidate selection](../engine/common/speculative.cpp) scores each tree size by expected accepted tokens per millisecond. The timing tables cover short and 32K contexts on the M4 Pro. | For a model with a compatible drafter, measure verification time by batch size and context length. Fit its own table, then compare accepted tokens per second with a fixed draft cap. |
| Reuse execution plans for recurring verify shapes | [Context graph cache](../engine/src/llama-context.cpp) checks graph compatibility, scheduler validity and the memory generation before reusing a plan. | Compare host scheduling time with reuse enabled and disabled. Exercise prompt growth and cache changes before comparing end-to-end decode rates. |
| Share KV reads across queries | [Metal attention dispatch](../engine/ggml/src/ggml-metal/ggml-metal-ops.cpp) lets several query rows and heads in a verify batch share a KV tile (with `FA_MQ_RESCALE` skipping unchanged row tiles), alongside coalesced Split-K/`BK=32` BF16 kernels and sorted MoE dispatch through `B = 16`. | On a model with grouped-query attention and compatible head widths, compare both kernels at a fixed prompt length and batch size. Check logits and decode speed. |
| Persist drafter KV with hybrid prompt caches | [Server prompt cache](../engine/tools/server/server-context.cpp) checks hybrid/SWA rollback feasibility before preferring slot state over disk cache, saves large dropped branches (`>= 2,048` tokens), recycles checkpoint buffers, and writes a `.prompt.dspark` sidecar for target-conditioned drafters. | On a multi-turn agentic or tool-calling loop with branching requests, compare prefill reuse and post-restore draft acceptance with and without sidecar persistence. |

Inkling's tapped layers, Markov head and per-visit short convolution belong to this checkpoint. A verifier for another architecture must restore all of that model's state after rejecting a branch. For an SSD-streamed model, record expert bytes read and verify latency alongside output agreement and generated tokens/s.

<details>
<summary>Earlier linear-drafting measurements and bandwidth analysis</summary>

The measurements below predate the tree verifier and the current Metal fusions. They describe the earlier implementation and its working set.

### Earlier Performance

These figures come from a MacBook Pro with an M4 Pro (20-core GPU, 48 GB) with no other application using the GPU.

`llama-bench -ngl 99 -fa 1 -p 512 -n 128`, without drafting:

| Test | token/s |
| --- | ---: |
| pp512 | 465 |
| tg128 | 31.2 |

`inferssd serve` with drafting, 640 generated tokens per prompt at `reasoning_effort` `low`:

| Prompt | Drafts accepted | token/s |
| --- | ---: | ---: |
| Python LRU cache, temperature 0 | 497 of 621 | 70.5 |
| TypeScript rate limiter, temperature 0 | 483 of 635 | 65.7 |
| PostgreSQL schema and window query, temperature 0 | 421 of 687 | 50.1 |
| C++20 SPSC ring buffer, temperature 0 | 463 of 647 | 60.6 |
| Python Dijkstra, temperature 0.7 | 470 of 635 | 62.7 |

The mean is 61.9 token/s. The contexts stay under 710 tokens (prompts of 35 to 66 tokens and 640 generated ones). `test-inkling-dspark` generates its 256-token answer at 58.1 token/s with drafting and 31.6 token/s without. Earlier idle runs, which accepted the same drafts, measured 63.8 token/s for the mean, 61.0 and 32.4 token/s for `test-inkling-dspark`, and 471 and 32.2 token/s for `llama-bench`, so idle runs differ by 1 to 5%. Other GPU load lowers these figures further: with another application holding the GPU at 55 to 60% utilization, the same prompts measured 38 to 51 token/s.

### Earlier Long-Context Measurements

The figures above use short contexts. The 8 full-attention layer visits read the whole cache for every token, and the cache grows by 32 KiB per token for the model and 4 KiB for the drafter; the 40 sliding-window visits keep a fixed 1,536-token cache of 240 MiB. A 128k-token context therefore needs about 4.5 GB of cache, and the full 1M tokens about 36 GB, which does not fit beside the model in 48 GB.

`llama-bench -ngl 99 -fa 1 -p 0 -n 128 -d N`, without drafting:

| Cached tokens | 0 | 8,192 | 32,768 |
| --- | ---: | ---: | ---: |
| token/s | 31.2 | 29.6 | 26.3 |

Another process started on the GPU during the 32,768-token run; the serve figure without drafting at 32,083 tokens below, measured on an idle GPU, is within 2% of it. An earlier run under 70 to 99% GPU load from other work measured 22.9 token/s at 0 cached tokens and 15.7 at 65,536.

A verify pass reads the full-attention caches once for all its tokens. For batches under 20 tokens, Metal's flash-attention vec kernel runs a threadgroup per query and head, so each drafted token reads the whole cache again, about 4 ms at 32k cached tokens. For 2 to 8 tokens over at least 2,048 cells (16,384 for one token), the backend runs `kernel_flash_attn_ext_mq` instead: one threadgroup holds all the query rows of the 3 heads that share a KV head, and reads each K and V row once for them. The 1,536-cell sliding-window caches stay with the per-query kernel; with a full-size sliding-window cache, the new kernel skips the chunks that the window masks. `GGML_METAL_FA_MQ_MIN_KV=N` moves the threshold to N cells for every batch size, which is how the kernels were compared. One pass at 32,069 cached tokens, alternating the kernels while another application held the GPU at 84%:

| Tokens in the batch | 1 | 2 | 4 | 5 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A threadgroup per KV head, ms | 57.6 | 69.0 | 101.9 | 108.5 | 110.8 | 131.1 |
| A threadgroup per query and head, ms | 57.8 | 72.9 | 109.9 | 122.1 | 122.0 | 150.4 |

The next-token logits of the two kernels differed by at most 0.004, and their top token matched.

`inferssd serve`, 256 generated tokens after a prompt of C++ source and a coding question:

| Prompt tokens | 4,082 | 16,083 | 32,083 |
| --- | ---: | ---: | ---: |
| Prefill, token/s | 393 to 402 | 384 | 353 |
| With drafting, token/s | 48.1 | 41.1 | 40.9 |
| Drafts accepted | 159 of 216 | 140 of 200 | 147 of 195 |
| Without drafting, token/s | 30.1 | 28.7 | 26.8 |

Under 91 to 98% GPU load from other work, two drafting runs one after the other measured 27.9, 28.2 and 28.4 token/s with `kernel_flash_attn_ext_mq` and 29.8, 28.2 and 26.0 with the per-query kernel. Prefill, which the kernel choice does not touch, ran about 10% slower in the first run, so the two runs did not see the same load.

### Earlier Bandwidth Estimate

Decoding is bound by how fast the GPU reads weights. A dense F32 multiply over a 1.85 GB matrix reads 213 to 223 GB/s at best on this machine and 170 to 200 GB/s in the median run, against the 273 GB/s the memory supports on paper.

A single token reads about 5.6 GB: 3.8 GB of dense weights (attention projections, loop controllers, LM head) and 1.8 GB for the 384 experts it routes to. A verify batch reads the dense weights once. Its tokens share experts, and the GPU cache already serves a shared expert to all of them:

| Tokens in the batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distinct experts per visit | 8.0 | 12.4 | 16.3 | 19.1 | 21.8 | 24.0 | 26.2 | 28.4 |
| Verify pass, ms | 31 | 38 | 43 | 48 | 51 | 54 | 59 | 64 |

A full block of 8 tokens therefore reads about 10.3 GB to verify, and the draft that precedes it reads about 1.2 GB (the drafter's layers, and the draft vocabulary's rows of the LM head and of the Markov head, which is read seven times). At 215 GB/s that is 53 ms for at most 8 tokens, before any other cost. The five prompts above accept 3.5 to 4.7 tokens per step, which bounds them at roughly 65 to 90 token/s even if the engine read at the full rate and did nothing else. In `test-inkling-dspark` the engine spends 58 ms per step at a mean batch of 4 tokens: 49.6 ms to verify, 7.0 ms to draft and 0.7 ms to pass the step's states to the drafter.

The dense BF16 multiplies reach 205 to 225 GB/s at every batch size from 1 to 8, so they sit at the limit. The expert multiplies are bound by arithmetic as well as by reads: a kernel that reads each distinct expert once for all its tokens measured no faster than the per-token kernel, and the matrix kernel that prefill uses is slower at batches of 2 to 8 tokens because most of its columns stay empty. Most of what remains above the bound is the roughly 4,000 small kernels of a pass: a single-token pass takes 31 ms where its 5.6 GB of reads account for about 26 ms.

Smaller dense weights do not pay either. A narrower float type would lose information: the BF16 tensors use every mantissa bit, with about 4,400 distinct values per tensor. Lossless packing is possible. Their entropy is 10.5 bits per value, and a 12.5-bit layout (a sign-and-mantissa byte per value, a 4-bit exponent code against the largest exponent of each 32 values, and up to 3 escaped exponents per 256 values) holds every block of this checkpoint exactly, which cuts the BF16 matrices a token reads from 3.51 to 2.74 GB. A kernel for that layout, timed against the BF16 kernel on one layer's projections while other apps loaded the GPU, took 0.82 to 0.95 of the BF16 time with 1 or 2 inputs, 0.93 to 1.01 with 4 to 6, and 1.09 with 8. With its exponent decode cut to one shift it was still slower at 8 inputs, so the extra on-chip work per byte, not the decode, absorbs what the smaller reads save: at most about 1 ms of a 52 ms verify pass at the batches drafting produces.

</details>

## Coverage

- MXFP8 matrix multiplication on the CPU backend. Use `INKLING2_MXFP8_BF16=1` for CPU-only runs.
- The vision tower.
- Earlier experiments with the n-gram draft sources of `--spec-type` alongside linear DSpark drafting showed mixed results. With DSpark the engine caps their drafts at 24 tokens, the most the short-convolution history can roll back; longer drafts left the cache inconsistent and failed the request. Prompt lookup (`ngram-simple` with a 6-token key) then lowered the mean of the five prompts above from 63.6 to 56.8 token/s, and raised a request that rewrites a given file by 4%.
