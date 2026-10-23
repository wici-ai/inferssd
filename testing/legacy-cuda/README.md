# Retired CUDA streaming experiments

These sources preserve the former shape dispatch, wrapper/stub exports, role
caches, object pipeline, and experimental dynamic tier for historical comparison.
They are outside the production backend source glob and are not compiled into
`ggml-cuda`. Ordinary header-only oracle tests use the explicit paths here.

Production expert I/O uses `ggml-ssd/io` and the context-owned CUDA transport in
`ggml-cuda/ssd`. The active grouped matmul is `ssd/group-matmul.cu`; its normally
linked MMVF/MMVQ/MMQ kernel exports retain the backend's numerical implementation.
