# Expert VRAM tier

This optional CUDA component caches routed expert weights and runs explicit
expert formats for DeepSeek and Muse. The `kimi-vram-tier` and
`kimi-mxfp4-parity` object targets link normally into `ggml-cuda`. The strict
MXFP4 target keeps `--fmad=false --ftz=false`, the explicit architecture, and
the original kernel bodies. No component DSO or preload is used.

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
the CPU implementation through an immutable L1 kernel table copied into each
tier. The GPU backend does not link the CPU backend. The optional GPU hidden chain has its existing
numerical tolerance and does not promise byte equality with that host bridge.

The former implicit Q3_K transport path has been removed. The internal
transport ABI remains version 2 with its existing field order; implicit
`prepare`/`launch` slots return an explicit unsupported-format error and drain
their context. Use the explicit format extension. Generic upstream Q3_K
quantizers and kernels remain part of ggml.

GPU lifecycle and numerical tests require an owned device and the matching
normal backend. They are built as explicit targets and are not automatically
run by CTest. License and origin notes for this directory are in
[THIRD_PARTY.md](THIRD_PARTY.md).
