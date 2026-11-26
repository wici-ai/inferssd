<!-- SPDX-License-Identifier: Apache-2.0 -->
# Experimental phase-specific expert routing

This branch combines two approximate DeepSeek-V4.1 routing policies to reduce
expert reads. **The measured throughput improves, but the combined version fails
manual task-quality review. It is an experiment, not a quality-preserving
optimization or a recommendation for production.**

Both policies activate automatically for the DeepSeek41 SSD sink path in this
branch. There is no separate user opt-in switch yet. Other architectures retain
their original expert selection. This activation behavior must be considered
before any merge that would change the default inference path.

## Sources

| Policy | Upstream design | Pinned upstream implementation |
| --- | --- | --- |
| Miss-aware prefill union budget | [Colibri PR #254](https://github.com/JustVugg/colibri/pull/254), by woolcoxm | [`dd0d60692d338c01ee03e3854284a3eca8601da6`](https://github.com/JustVugg/colibri/commit/dd0d60692d338c01ee03e3854284a3eca8601da6) |
| Bounded-rank resident decode selection | [Colibri PR #199](https://github.com/JustVugg/colibri/pull/199); performance lead from [issue #811](https://github.com/JustVugg/colibri/issues/811) | [`47b5f075e266384e6a6445089b638babb5f945de`](https://github.com/JustVugg/colibri/commit/47b5f075e266384e6a6445089b638babb5f945de) |

The first policy adapts the miss-aware union selection. The second independently
implements the top-J / resident top-M selection mechanism. Other Colibri kernels,
MTP features and tuning are not included. Attribution is retained in [NOTICE](../../NOTICE).
Upstream performance and routing-agreement figures are not evidence of this
DeepSeek adaptation's quality.

## Behavior and numerical consequences

During **multi-token prefill**, collect the selected expert union for one layer.
Keep every ready GPU/RAM hit. Rank the misses by their aggregate gate weight over
the batch and retain at most `max(0, 4 - ready_hits)` of them. Remove the remaining
routes before expert reads and computation. The budget is not four experts per
token, and ready hits may exceed four. Rescale surviving weights to preserve each
token's original routed weight sum; if no route survives, its routed-expert
contribution is zero. The residual and other model branches still execute.

This can discard an expert important to an individual token but uncommon in the
batch. Renormalization preserves a scalar sum, not the missing expert's output
vector. The changed prompt hidden states affect downstream layers and KV state.

During **single-token decode**, keep the true top-J expert (`J=1`), prefer ready
resident experts inside the original top-M (`M=16`), and fill any remaining slots
in original rank order. Keep K unchanged (six for this model). The original
unbiased gate scores still pass through the existing normalization. As many as
five of six expert identities can change: residency is not a guarantee of
semantic suitability, and the result depends on cache state.

The policies have separate phase guards. GPU residency ABI v5 exposes separate
snapshots: grouped prefill can see all published keyed slots, while decode only
accepts slots in the layer's current allocation. Both exclude unfinished uploads;
normal execution still acquires payload leases. Host-cache eligibility remains
phase-specific. Only worker zero executes the decode selector; other CPU workers
return before touching its state, and the executor barrier publishes the IDs.

## Measured performance and limits

These results use frozen main `8b1b6500e6b69bc37605ab1b0f1f9eff17c45ef4` as A and
combined commit `5e5bcc242552b4f45a57dd95be07ab85e5895e30` as B. A has five samples;
each candidate has two. The PR subsequently integrates main
`910afae91c3c2578f890df7ffd9a1e55b6237680`; **performance was not rerun after that
integration**, so these numbers are not measurements of the final PR head.

NVIDIA DGX Spark / GB10, Linux aarch64, 20 CPU cores, whole-machine memory budget
130,663,170,048 bytes, swap disabled. DeepSeek-V4.1-Flash-1017 GGUF named Q8_0;
metadata reports 120 MXFP4 expert tensors alongside F32, Q8_0 and BF16 tensors.
A and all candidates use the same formats and hidden quantizer; the combination
introduces no additional weight-bit-width reduction.

Workload: [the retained 4K prompt](../../bench/prompts/prompt-4096.txt), 4104 actual
input tokens, 256 output tokens, context 8192, batch=ubatch=4104, decode threads=10,
batch threads=19. Requests use greedy top-k=1, temperature=0, seed=0, no prompt
cache and forced 256-token generation. The CUDA expert tier is enabled and its
actual use is verified. Primary decode is tokens 129-256.

| Variant | Prefill tok/s | vs A | Last-128 decode tok/s | vs A |
| --- | ---: | ---: | ---: | ---: |
| Frozen main, A5 | 152.534 | baseline | 7.269 | baseline |
| Prefill union budget, B2 | 390.024 | +155.70% | 6.954 | -4.33% |
| Bounded-rank decode, B2 | 151.901 | -0.42% | 8.083 | +11.19% |
| Combined policies, B2 | 393.604 | +158.04% | 8.052 | +10.76% |

Combined samples are 390.262 / 396.946 prefill tok/s and 8.006 / 8.098 last-128
decode tok/s. Overall 256-token decode averages 7.325 tok/s. The baseline is
historical, not interleaved with these two combined runs. A single prompt and two
candidate samples cannot establish general performance or quality.

The [machine-readable record](../benchmarks/experimental-resident-routing.json)
contains exact source revisions, sample values, prompt hashes, build settings,
routing counters and both combined outputs. It is a curated summary of retained
native evidence, not a replacement for all raw traces.

## Observed quality regression

The measured prompt has 984,960 baseline token-to-expert contributions over 40
routed layers. The combination retains 188,119 / 188,587: a mean **80.88% omission**.
Decode replaces 12,771/61,200 (20.87%) and 8,705/61,200 (14.22%) selected slots.
These are routing-change fractions, not accuracy-loss percentages.

The same prompt asks for a few-paragraph summary of a telemetry storage design:

- Baseline outputs are on-topic summaries with concrete system details.
- Prefill-only outputs are an outline-like response or a reproduction of the
  introduction, rather than the requested summary.
- Decode-only outputs remain on-topic in the two samples inspected. This does
  not establish that the decode policy is lossless.
- Combined sample 1 largely reproduces the introduction; sample 2 produces an
  off-topic, repetitive chain about time periods and universe variants.

The existing basic output screen accepts nonempty valid text with 256 token IDs;
it does not grade instruction adherence or factual accuracy. It accepted these
combined samples even though manual review failed. Both policies were positively
identified as active in both measured runs. The observed quality issue therefore
must not be hidden by the throughput table or described merely as hypothetical.

No perplexity or standardized accuracy score was measured, and no numerical
accuracy-drop percentage is claimed. The prefill-only failure implicates the
aggressive omission policy; a fixed validation set and separate ablations are
still needed to quantify each policy's independent and combined quality cost.

## Integration and validation

Current main already uses CPU executor ABI v9 for `vram_reserve_pass`. The PR
preserves that callback, appends `route_ready`, and advances the ABI to v10. Tests
reject a missing callback or a v9 table. The measured source's original CPU ABI
was v9 with `route_ready`; it has not been overwritten or relabelled.

The measured combination passed CPU policy/reference tests, 1/2/10-worker decode
graph-reuse checks, unchanged expert-computation regressions, and a CUDA server
build followed by two full inference samples. These establish implementation
behavior and throughput, not model accuracy. PR-head checks are reported in the
PR description and CI; measured-source checks are not represented as PR-head
performance validation.

Before considering production use, make the approximation explicitly opt-in,
retain important experts per token rather than only by batch popularity, bound
the omitted gate mass, and evaluate perplexity and task accuracy on held-out data.
Those safeguards are follow-up work and are not implemented by this experiment.
