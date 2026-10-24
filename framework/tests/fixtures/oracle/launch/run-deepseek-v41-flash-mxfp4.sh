#!/usr/bin/env bash
set -Eeuo pipefail

# Start llama-server for DeepSeek-V4.1-Flash (Q8_0 dense + MXFP4 experts) on an RTX 5090 behind a real PCIe Gen4 x4
# link and 16 GiB of RAM, through the generic launcher run-ssd-llama.sh with tools/adaptive/models/dsv41.conf; it
# gives the same server environment and arguments as this launcher always did (tools/adaptive/tests/
# test_launcher_replica.sh checks it).  The context and the decode VRAM headroom are set for the target machine:
# 131072 tokens, VRAM_ELASTIC_SAFETY_MIB 64 MiB, and the 16/64 MiB watermarks keep free VRAM in that range during
# decode.  VRAM_RESERVE_MIB leaves GPU memory to another process that keeps running on the card (a deployment's
# monitoring listener, for example): the CUDA allocator reclaims expert blocks rather than go below it, the
# watermarks move up by the same amount, and the launcher then accepts that process instead of requiring an idle GPU.
#
# Required:  MODEL, and EXPERT_ALIAS or EXPERT_PACK: where the experts are read from. EXPERT_ALIAS indexes their
#            ranges in the GGUF itself (tools/adaptive/expert_source.py index MODEL DIR writes DIR/experts.tsv);
#            EXPERT_PACK is a separate expert pack built from GGUF shards 1-7.
# Optional:  CONTEXT, SERVER_BIN, PRELOAD_LIB, LLAMA_SERVER_HOST, LLAMA_SERVER_PORT, THREADS,
#            MEMORY_HEADROOM_MIB, MEMORY_RESERVE_MIB, MEMORY_MAX, VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB, PROMPT_CACHE_DIR,
#            PROMPT_CACHE_MIB, UBATCH (the prefill ubatch, 8192), GGML_KIMI_CPU_SINK_VRAM_BYTES (21000000000), and any
#            other engine variable (GGML_*, LLAMA_*), which replaces the launcher's value (run-ssd-llama.sh)
# Extra llama-server arguments can be appended after the script name.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL:?Set MODEL to the first DeepSeek-V4.1-Flash Q8_0 GGUF shard (...-00001-of-00010.gguf)}"
if [[ -z "${EXPERT_ALIAS:-}" && -z "${EXPERT_PACK:-}" ]]; then
    echo "Set EXPERT_ALIAS (the GGUF's expert ranges) or EXPERT_PACK (an expert pack)" >&2
    exit 1
fi
MODEL_CONF=$SCRIPT_DIR/tools/adaptive/models/dsv41.conf
PROMPT_CACHE_DIR=${PROMPT_CACHE_DIR-${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/deepseek-v41-flash-mxfp4}
# sourced, not run: these settings reach the launcher without entering the server's environment
source "$SCRIPT_DIR/run-ssd-llama.sh" "$@"
