#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_file="${SSD_LLAMA_MODEL:?Set SSD_LLAMA_MODEL to the first GGUF shard}"
export SSD_LLAMA_MAC_EXPERT_PACK="${SSD_LLAMA_MAC_EXPERT_PACK:?Set SSD_LLAMA_MAC_EXPERT_PACK to the matching expert pack}"
build_dir="${SSD_LLAMA_BUILD_DIR:-$repo_dir/build-mac}"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This expert-pack profile requires an Apple Silicon Mac." >&2
    exit 1
fi
default_cache_mib=4096
if (( $(sysctl -n hw.memsize) >= 48 * 1024 * 1024 * 1024 )); then
    default_cache_mib=10240
fi
export SSD_LLAMA_MAC_EXPERT_CACHE_MIB="${SSD_LLAMA_MAC_EXPERT_CACHE_MIB:-$default_cache_mib}"
for file in "$model_file" "$SSD_LLAMA_MAC_EXPERT_PACK" "$build_dir/bin/llama-server"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing required file: $file" >&2
        exit 1
    fi
done
if ! [[ "$SSD_LLAMA_MAC_EXPERT_CACHE_MIB" =~ ^[0-9]+$ ]] ||
        (( SSD_LLAMA_MAC_EXPERT_CACHE_MIB > 16384 )); then
    echo "SSD_LLAMA_MAC_EXPERT_CACHE_MIB must be 0..16384" >&2
    exit 2
fi

export LLAMA_MMAP_PREFETCH=0
export LLAMA_DSV41_TOPK_REUSE=0

exec "$build_dir/bin/llama-server" \
    -m "$model_file" --mmap --no-repack --fit off --no-warmup \
    -ngl 99 --cpu-moe --no-op-offload \
    -ot 'blk\.[0-9]+\.engram_embd\.weight=CPU' \
    -c "${SSD_LLAMA_CTX:-8192}" -b "${SSD_LLAMA_BATCH:-512}" \
    -ub "${SSD_LLAMA_UBATCH:-512}" -t "${SSD_LLAMA_THREADS:-8}" \
    -tb "${SSD_LLAMA_PREFILL_THREADS:-12}" \
    --chat-template-file "$repo_dir/models/templates/deepseek-ai-DeepSeek-V4.1.jinja" \
    --reasoning off --cache-ram 0 --host 127.0.0.1 \
    --port "${SSD_LLAMA_PORT:-18089}" -np 1 "$@"
