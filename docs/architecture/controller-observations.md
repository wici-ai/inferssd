# Native controller observations

The online Linux controller reads `native-events.jsonl` through
`inferssd.telemetry.controller.NativeObservationSession`. Human-readable server
logs and `INFERSSD_DEBUG` are optional diagnostics. Historical calibration and
record import still use the explicitly named legacy adapters; Darwin online
adaptation remains unsupported.

The existing JSONL envelope is `v: 1`, a contiguous per-writer `seq`, and `ev`.
After `ssd.settings`, `ssd.controller.begin` declares `schema: 1`. Each writer
belongs to one native runtime and is closed after its users have drained.

| Event | Controller fields |
| --- | --- |
| `ssd.place.layer` | epoch, token, has_token, layer, decode, success, selected_experts, vram_hit, cache_hit, wall_us, wait_io_us, compute_us |
| `ssd.prefill.graph` | requested_tokens, configured_max, selected_ubatch |
| `ssd.prefill.layer` | layer, n_tokens, read_bytes |
| `ssd.prefill.copy` | bytes, read_us, passes, tokens |

Route arrays are captured from the completed native layer, not reconstructed
from text. Wall-time counters are already collected by the executor; enabling
this contract does not enable per-thread profiling or logits tracing. The
controller preserves the prior epoch/token/layer reset rule, including repeated
positions in successive requests. Prefill fallback retains the historical
unreported configured-max value of zero; selected_ubatch records the real size.

The consumer validates the envelope, required fields and successful layers,
rejects missing contracts or truncated files, and waits on incomplete final
lines. Unknown event names remain forward-compatible. The live boundary exposes
only groups, routes, layer costs and prefill summaries actually used by the
controller. It does not claim that all historical log fields are now a public API.

Tests compare the live observer with recorded legacy records through served-turn
aggregation, feedback and the real decision function, including reused token
positions. A native unit verifies production prefill events with debug channels
off. Device readiness additionally consumes a real short Linux model request's
events; that receipt is separate from the offline parity tests.
