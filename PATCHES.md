# Upstream integration

Commit `8884f8314ce8ea96d62cd602b6bbf7fc25758a09` is the bundled llama.cpp snapshot described in
[PROVENANCE.md](PROVENANCE.md). Owned SSD implementations are mapped in
[ARCHITECTURE.md](ARCHITECTURE.md). Relocations preserve source attribution and
retained license files; no upstream history is rewritten by these changes.

| Integration group | Reason and boundary |
| --- | --- |
| ggml scheduler and allocator | Borrowed SSD copy hooks; graph buffer measurement/admission; first touch; backend scratch reservation. No core implementation is linked into ggml-base. |
| CPU/CUDA registries | Normal private provider discovery and context-service attachment. CPU and CUDA do not link each other. |
| Backend operation files | New model operations and narrow native kernel entry points stay with their upstream operation/architecture files. Strict MXFP4 math is compiled separately with contraction and FTZ disabled. |
| llama context | Runtime assembly, request phases, resource-aware graph reservation and failure recovery. Source, cache and telemetry owners reside in `src/ssd`. |
| Model loading/mapping | Deferred expert ranges and compact Muse Metal allocations; fixed complete-page advice. Source/range helpers reside in `src/ssd`. |
| Model/architecture support | DeepSeek V4.1, Engram and Muse architecture, metadata, graph and KV operations follow upstream model/op organization. They are intentional model exceptions. |
| Common/server | CLI dispatch, rendering/parser hooks, prompt-cache/checkpoint and lifecycle calls. Owned implementations reside in `common/inferssd` and `tools/server/inferssd`. |
| State serialization | Virtual/C API entry points call positional/range codecs under `src/inferssd`. |

The original 3,000-line hook target is a planning budget, not evidence that every
changed upstream line is a hook. The inventory below includes model/op additions
and inherited downstream edits. Further reduction of those edits, formatting and
comment cleanup are deferred; production dependency directions are checked now.
Private candidate validation must finish before claiming operational acceptance.

Regenerate the path inventory with
`python3 scripts/inferssd/patch-surface.py`. It compares the working tree with
the local `upstream` alias set up in [CONTRIBUTING.md](CONTRIBUTING.md) and excludes
newly owned paths. Review model/op exceptions separately from integration hooks
when assessing future patch growth.

<!-- BEGIN UPSTREAM INVENTORY -->
| Upstream path | Added | Removed |
| --- | ---: | ---: |
| `engine/common/CMakeLists.txt` | 6 | 1 |
| `engine/common/arg.cpp` | 60 | 2 |
| `engine/common/chat-auto-parser-generator.cpp` | 1 | 1 |
| `engine/common/chat-peg-parser.cpp` | 4 | 2 |
| `engine/common/chat-peg-parser.h` | 3 | 0 |
| `engine/common/chat.cpp` | 175 | 199 |
| `engine/common/chat.h` | 34 | 3 |
| `engine/common/common.cpp` | 2 | 0 |
| `engine/common/common.h` | 13 | 2 |
| `engine/common/debug.cpp` | 16 | 13 |
| `engine/common/jinja/caps.cpp` | 1 | 1 |
| `engine/common/jinja/parser.cpp` | 10 | 2 |
| `engine/common/jinja/runtime.cpp` | 6 | 1 |
| `engine/common/jinja/string.cpp` | 16 | 5 |
| `engine/common/jinja/string.h` | 5 | 6 |
| `engine/common/jinja/value.cpp` | 170 | 133 |
| `engine/common/jinja/value.h` | 17 | 18 |
| `engine/common/peg-parser.cpp` | 16 | 0 |
| `engine/common/peg-parser.h` | 4 | 0 |
| `engine/common/reasoning-budget.h` | 1 | 53 |
| `engine/common/sampling.cpp` | 78 | 12 |
| `engine/common/sampling.h` | 11 | 2 |
| `engine/common/speculative.cpp` | 23 | 0 |
| `engine/common/speculative.h` | 6 | 0 |
| `engine/ggml/CMakeLists.txt` | 1 | 0 |
| `engine/ggml/include/ggml-alloc.h` | 4 | 0 |
| `engine/ggml/include/ggml-backend.h` | 58 | 0 |
| `engine/ggml/include/ggml-cpu.h` | 3 | 0 |
| `engine/ggml/include/ggml-cuda.h` | 23 | 0 |
| `engine/ggml/include/ggml-rpc.h` | 2 | 2 |
| `engine/ggml/include/ggml.h` | 106 | 1 |
| `engine/ggml/src/CMakeLists.txt` | 2 | 0 |
| `engine/ggml/src/ggml-alloc.c` | 38 | 2 |
| `engine/ggml/src/ggml-backend-impl.h` | 5 | 0 |
| `engine/ggml/src/ggml-backend-meta.cpp` | 14 | 1 |
| `engine/ggml/src/ggml-backend.cpp` | 386 | 64 |
| `engine/ggml/src/ggml-blas/ggml-blas.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-cann/ggml-cann.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-common.h` | 1 | 0 |
| `engine/ggml/src/ggml-cpu/CMakeLists.txt` | 17 | 0 |
| `engine/ggml/src/ggml-cpu/arch/arm/quants.c` | 0 | 1 |
| `engine/ggml/src/ggml-cpu/arch/x86/quants.c` | 138 | 7 |
| `engine/ggml/src/ggml-cpu/ggml-cpu-impl.h` | 3 | 0 |
| `engine/ggml/src/ggml-cpu/ggml-cpu.c` | 3445 | 3650 |
| `engine/ggml/src/ggml-cpu/ggml-cpu.cpp` | 187 | 7 |
| `engine/ggml/src/ggml-cpu/ops.cpp` | 539 | 12 |
| `engine/ggml/src/ggml-cpu/ops.h` | 4 | 0 |
| `engine/ggml/src/ggml-cpu/quants.c` | 1 | 0 |
| `engine/ggml/src/ggml-cpu/quants.h` | 7 | 0 |
| `engine/ggml/src/ggml-cpu/unary-ops.cpp` | 271 | 1 |
| `engine/ggml/src/ggml-cpu/unary-ops.h` | 3 | 0 |
| `engine/ggml/src/ggml-cuda/CMakeLists.txt` | 9 | 0 |
| `engine/ggml/src/ggml-cuda/argsort.cu` | 92 | 71 |
| `engine/ggml/src/ggml-cuda/argsort.cuh` | 6 | 0 |
| `engine/ggml/src/ggml-cuda/binbcast.cu` | 12 | 2 |
| `engine/ggml/src/ggml-cuda/common.cuh` | 162 | 8 |
| `engine/ggml/src/ggml-cuda/concat.cu` | 69 | 46 |
| `engine/ggml/src/ggml-cuda/cpy.cu` | 1 | 1 |
| `engine/ggml/src/ggml-cuda/fattn-common.cuh` | 35 | 7 |
| `engine/ggml/src/ggml-cuda/fattn.cu` | 10 | 0 |
| `engine/ggml/src/ggml-cuda/fattn.cuh` | 3 | 0 |
| `engine/ggml/src/ggml-cuda/ggml-cuda.cu` | 1433 | 125 |
| `engine/ggml/src/ggml-cuda/mmq.cu` | 54 | 3 |
| `engine/ggml/src/ggml-cuda/mmq.cuh` | 28 | 10 |
| `engine/ggml/src/ggml-cuda/mmvf.cu` | 22 | 0 |
| `engine/ggml/src/ggml-cuda/mmvq.cu` | 258 | 40 |
| `engine/ggml/src/ggml-cuda/mmvq.cuh` | 12 | 0 |
| `engine/ggml/src/ggml-cuda/norm.cu` | 15 | 0 |
| `engine/ggml/src/ggml-cuda/quantize.cu` | 238 | 21 |
| `engine/ggml/src/ggml-cuda/quantize.cuh` | 18 | 0 |
| `engine/ggml/src/ggml-cuda/rope.cu` | 93 | 59 |
| `engine/ggml/src/ggml-cuda/set-rows.cu` | 65 | 4 |
| `engine/ggml/src/ggml-cuda/ssm-conv.cu` | 11 | 3 |
| `engine/ggml/src/ggml-cuda/top-k.cu` | 248 | 0 |
| `engine/ggml/src/ggml-cuda/top-k.cuh` | 9 | 0 |
| `engine/ggml/src/ggml-cuda/unary.cu` | 377 | 0 |
| `engine/ggml/src/ggml-cuda/unary.cuh` | 16 | 0 |
| `engine/ggml/src/ggml-cuda/vecdotq.cuh` | 2 | 1 |
| `engine/ggml/src/ggml-hexagon/ggml-hexagon.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-metal/ggml-metal-device.m` | 2 | 1 |
| `engine/ggml/src/ggml-metal/ggml-metal.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-opencl/ggml-opencl.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-openvino/ggml-openvino.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-quants.c` | 5 | 0 |
| `engine/ggml/src/ggml-rpc/ggml-rpc.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-sycl/ggml-sycl.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-virtgpu/ggml-backend.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-webgpu/ggml-webgpu.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-zdnn/ggml-zdnn.cpp` | 1 | 0 |
| `engine/ggml/src/ggml-zendnn/ggml-zendnn.cpp` | 1 | 0 |
| `engine/ggml/src/ggml.c` | 272 | 8 |
| `engine/ggml/src/gguf.cpp` | 7 | 0 |
| `engine/include/llama.h` | 88 | 1 |
| `engine/src/CMakeLists.txt` | 19 | 0 |
| `engine/src/llama-arch.cpp` | 114 | 2 |
| `engine/src/llama-arch.h` | 77 | 2 |
| `engine/src/llama-context.cpp` | 1352 | 46 |
| `engine/src/llama-context.h` | 150 | 2 |
| `engine/src/llama-cparams.h` | 9 | 0 |
| `engine/src/llama-ext.h` | 16 | 0 |
| `engine/src/llama-graph.cpp` | 758 | 49 |
| `engine/src/llama-graph.h` | 158 | 5 |
| `engine/src/llama-hparams.cpp` | 4 | 0 |
| `engine/src/llama-hparams.h` | 45 | 1 |
| `engine/src/llama-impl.h` | 35 | 0 |
| `engine/src/llama-kv-cache-iswa.cpp` | 41 | 3 |
| `engine/src/llama-kv-cache-iswa.h` | 28 | 0 |
| `engine/src/llama-kv-cache.cpp` | 250 | 123 |
| `engine/src/llama-kv-cache.h` | 57 | 0 |
| `engine/src/llama-kv-cells.h` | 132 | 25 |
| `engine/src/llama-memory-recurrent.cpp` | 6 | 0 |
| `engine/src/llama-memory.h` | 70 | 0 |
| `engine/src/llama-mmap.cpp` | 132 | 3 |
| `engine/src/llama-mmap.h` | 7 | 0 |
| `engine/src/llama-model-loader.cpp` | 127 | 2 |
| `engine/src/llama-model-loader.h` | 27 | 0 |
| `engine/src/llama-model-saver.cpp` | 6 | 2 |
| `engine/src/llama-model.cpp` | 784 | 38 |
| `engine/src/llama-model.h` | 81 | 0 |
| `engine/src/llama-quant.cpp` | 22 | 1 |
| `engine/src/llama-sampler.cpp` | 1 | 0 |
| `engine/src/llama-vocab.cpp` | 67 | 8 |
| `engine/src/llama-vocab.h` | 5 | 0 |
| `engine/src/llama.cpp` | 8 | 1 |
| `engine/src/models/deepseek2.cpp` | 10 | 4 |
| `engine/src/models/models.h` | 56 | 0 |
| `engine/tools/server/CMakeLists.txt` | 8 | 0 |
| `engine/tools/server/README.md` | 2 | 1 |
| `engine/tools/server/server-chat.cpp` | 66 | 16 |
| `engine/tools/server/server-common.cpp` | 288 | 5 |
| `engine/tools/server/server-common.h` | 22 | 1 |
| `engine/tools/server/server-context.cpp` | 685 | 127 |
| `engine/tools/server/server-task.cpp` | 195 | 183 |
| `engine/tools/server/server-task.h` | 23 | 80 |

Total: 15564 added, 5345 removed across 135 upstream paths.
<!-- END UPSTREAM INVENTORY -->
