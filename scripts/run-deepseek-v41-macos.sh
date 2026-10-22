#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_file="${SSD_LLAMA_MODEL:?Set SSD_LLAMA_MODEL to the first GGUF shard}"
build_dir="${SSD_LLAMA_BUILD_DIR:-$repo_dir/build-mac}"
mode="${1:-chat}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This profile requires an Apple Silicon Mac." >&2
    exit 1
fi
if [[ ! -f "$model_file" ]]; then
    echo "Model not found: $model_file" >&2
    exit 1
fi

# Avoid prefetching the entire model, which is larger than unified memory.
export LLAMA_MMAP_PREFETCH=0
# Reader-layer top-k reuse changed the answer in the 130-token Metal
# regression case. Keep the reference attention path for this profile.
export LLAMA_DSV41_TOPK_REUSE=0
args=(
    -m "$model_file" --mmap --no-repack --fit off --no-warmup
    -ngl 99 --cpu-moe --no-op-offload
    -ot 'blk\.[0-9]+\.engram_embd\.weight=CPU'
    -c "${SSD_LLAMA_CTX:-512}" -b 32 -ub 32 -t "${SSD_LLAMA_THREADS:-8}"
    --chat-template-file "$repo_dir/models/templates/deepseek-ai-DeepSeek-V4.1.jinja"
    --reasoning off --cache-ram 0
)

case "$mode" in
    chat)
        exec "$build_dir/bin/llama-cli" "${args[@]}" -n 128 "$@"
        ;;
    server)
        exec "$build_dir/bin/llama-server" "${args[@]}" \
            --host 127.0.0.1 --port "${SSD_LLAMA_PORT:-8080}" -np 1 "$@"
        ;;
    *)
        echo "Usage: $0 [chat|server] [llama arguments...]" >&2
        exit 2
        ;;
esac
