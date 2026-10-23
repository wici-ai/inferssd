# Expert VRAM tier

This optional CUDA component caches routed expert weights and runs explicit
expert formats for DeepSeek and Muse. Build `kimi-vram-tier` and
`kimi-mxfp4-parity` together with the same CUDA toolkit and architecture as the
backend. The MXFP4 component retains strict floating-point compilation.

The owner supplies tensor types, geometry, expert count and layer IDs through
`kimi_vram_format_api`; it supplies source file identities and ranges through
`kimi_vram_request`. The tier shares admission policy, asynchronous and deferred
uploads, residency leases, elastic blocks and failure cleanup across formats.
A borrowed upload keeps its source pinned until collect/finish or failure drain.
Reclaim refuses live compute/residency leases and drains queued copies before
freeing their blocks. The CPU sink owns the request and phase boundaries.

MXFP4 uses its dedicated parity or clamped hidden-chain component. Other
supported CUDA expert formats use the generic format adapter; grouped prefill
uses the retained batch adapters. The host MXFP4 bridge checks its parity against
the CPU implementation. The optional GPU hidden chain has its existing
numerical tolerance and does not promise byte equality with that host bridge.

The former implicit K3/Q3_K transport path and test2 component have been removed.
The private transport ABI remains version 2 with its existing field order;
implicit `prepare`/`launch` slots return an explicit unsupported-format error and
drain their context. Use the explicit format extension. Generic upstream Q3_K
quantizers and kernels remain part of ggml.

GPU lifecycle and numerical tests require an owned device and the matching
component DSOs. They are built as explicit targets and are not automatically
run by CTest. The historical component's original provenance and notices remain
in Git history; see [THIRD_PARTY.md](THIRD_PARTY.md).
