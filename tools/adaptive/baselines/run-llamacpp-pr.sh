#!/usr/bin/env bash
set -Eeuo pipefail

# llama.cpp baseline for DeepSeek-V4.1-Flash: the runtime of the GGUF's author as proposed upstream, PR #28696
# (vcruz305 feat/deepseek-v41-convert @ 2076412), built unmodified. Nothing of ssd-llama is loaded. Experts are
# ordinary GGUF mmaps computed by the CPU backend (-ot exps=CPU); GPU_EXPERT_LAYERS > 0 keeps the experts of the
# first layers on the GPU (the usual "fill the VRAM" tuning). The engram tables (208 GB) stay with the CPU too:
# the PR places nothing itself, and -ngl 99 alone asks the GPU for them. LOAD_MODE=mmap is needed on an
# integrated GPU (GB10), where llama.cpp's automatic mode reads the whole model into memory instead of mapping it. Same scope as the system under test: memory.max =
# MemAvailable - 512 MiB unless MEMORY_MAX is given, no swap, the same thread count.
#
# Required: MODEL, SERVER_BIN, THREADS. Optional: SRC_DIR (the PR tree), TEMPLATE, LOAD_MODE, CONTEXT (131072),
# GPU_EXPERT_LAYERS (0), BATCH (2048), UBATCH (512), MEMORY_MAX, LLAMA_SERVER_HOST/PORT.

: "${MODEL:?}"; : "${SERVER_BIN:?}"; : "${THREADS:?}"
BIN_DIR=$(cd -- "$(dirname -- "$SERVER_BIN")" && pwd)
SRC_DIR=${SRC_DIR:-$(cd -- "$BIN_DIR/../.." && pwd)}
TEMPLATE=${TEMPLATE:-$SRC_DIR/models/templates/deepseek-ai-DeepSeek-V4.1-Flash.jinja}
HOST=${LLAMA_SERVER_HOST:-127.0.0.1}
PORT=${LLAMA_SERVER_PORT:-18181}
CONTEXT=${CONTEXT:-131072}
K=${GPU_EXPERT_LAYERS:-0}
if (( K == 0 )); then
    OT='ffn_(up|gate|down)_exps=CPU'
else
    layers=$(seq -s '|' "$K" 39)
    OT="blk\\.(${layers})\\.ffn_(up|gate|down)_exps=CPU"
fi
read -r MEM_AVAILABLE_KIB < <(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
MEMORY_MAX=${MEMORY_MAX:-$(( (MEM_AVAILABLE_KIB - 512 * 1024) * 1024 ))}
if (( EUID == 0 )); then SCOPE=(systemd-run --scope); else SCOPE=(systemd-run --user --scope); fi

echo "llama.cpp baseline (PR #28696) on http://$HOST:$PORT (threads=$THREADS, context=$CONTEXT, gpu_expert_layers=$K)"
echo "memory.max=$MEMORY_MAX -ot $OT,engram_embd=CPU load_mode=${LOAD_MODE:-auto} template=$TEMPLATE"
exec "${SCOPE[@]}" --quiet --unit="dsv41-pr28696-$(date +%Y%m%d-%H%M%S)-$$" \
    -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
    env LD_LIBRARY_PATH="$BIN_DIR" "$SERVER_BIN" \
    -m "$MODEL" \
    -c "$CONTEXT" -np 1 -t "$THREADS" -tb "$THREADS" -b "${BATCH:-2048}" -ub "${UBATCH:-512}" \
    --ctx-checkpoints 8 --fit off -ngl 99 -fa on -ot "$OT,engram_embd=CPU" ${LOAD_MODE:+--load-mode "$LOAD_MODE"} \
    --chat-template-file "$TEMPLATE" \
    --chat-template-kwargs '{"drop_thinking": false}' \
    --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
    --no-repack --cache-ram 0 --no-warmup \
    --host "$HOST" --port "$PORT" \
    "$@"
