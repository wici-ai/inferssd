#!/usr/bin/env bash
set -Eeuo pipefail
# Qwen3.8-Flash-Next Q8_0 on the Qwen port's engine (branch deepseek2/qwen38), on any machine: the port's
# run-qwen38-q8.sh with the machine's own CPUs, memory and link where that one has a test server's.  Threads follow
# the DeepSeek launcher (the P and E cores of an Intel hybrid CPU, else every CPU); memory.max is MemAvailable at
# launch less 512 MiB, of which the RAM expert cache gets what the port's 16 GB envelope leaves it (13.98 GB with a
# 6.4 GB cache: 7.5 GB for the rest), at most 24 GB; the prefill cost tuple is the port's for the link (LINK_GBPS,
# measured by the device probe).
# Env: MODEL PACK SERVER_BIN (required), LINK_GBPS (6.74), PORT, CONTEXT, THREADS, EXTRA_ENV; args go to llama-server.
: "${SERVER_BIN:?SERVER_BIN, the llama-server of the port, is required}"
BIN_DIR=$(cd -- "$(dirname -- "$SERVER_BIN")" && pwd)
: "${MODEL:?MODEL (first GGUF shard) is required}" "${PACK:?PACK (expert pack) is required}"
CONTEXT=${CONTEXT:-131072}
PORT=${PORT:-${LLAMA_SERVER_PORT:-18382}}
PRELOAD="$BIN_DIR/libkimi-no-mmap-prefetch.so:$BIN_DIR/libkimi-vram-tier.so"
THREADS=${THREADS:-$(
    if [[ -r /sys/devices/cpu_core/cpus && -r /sys/devices/cpu_atom/cpus ]]; then
        awk -F, '
            { for (i = 1; i <= NF; ++i) {
                n = split($i, r, "-")
                for (cpu = r[1] + 0; cpu <= (n == 2 ? r[2] + 0 : r[1] + 0); ++cpu) seen[cpu] = 1
            } }
            END { for (cpu in seen) ++count; if (count) print count; else exit 1 }
        ' /sys/devices/cpu_core/cpus /sys/devices/cpu_atom/cpus || nproc
    else
        nproc
    fi
)}
MEMORY_MAX=$(awk '/MemAvailable:/ {printf "%.0f\n", ($2 - 512 * 1024) * 1024}' /proc/meminfo)
CACHE_BYTES=$(( MEMORY_MAX - 7534063616 ))
(( CACHE_BYTES > 24000000000 )) && CACHE_BYTES=24000000000
(( CACHE_BYTES < 1073741824 )) && { echo "only $(( MEMORY_MAX >> 20 )) MiB available" >&2; exit 1; }
CACHE_MAX=128; VRAM_BYTES=${VRAM_BYTES:-18000000000}
LINK=${LINK_GBPS:-6.74}
# the port's cost tuples, measured at a Gen5 x16 link (53.5 GB/s) and an emulated Gen4 x4 one (6.74 GB/s)
if awk -v l="$LINK" 'BEGIN { exit !(l < 20) }'; then
    Q8_COST="81.3465,14.7154,20.9598,2.0006,$LINK,10,50"
else
    Q8_COST="107.7209,12.5199,19.5017,1.9089,$LINK,4,50"
fi
for f in "$MODEL" "$PACK" "$BIN_DIR/llama-server" "$BIN_DIR/libkimi-vram-tier.so" "$BIN_DIR/libkimi-no-mmap-prefetch.so"; do
    [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done
ENVIRONMENT=(
    LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    LD_PRELOAD="$PRELOAD"
    GGML_KIMI_CPU_SINK=1
    GGML_KIMI_CPU_PREFILL=1
    GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE=1
    GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY=1
    GGML_MOE_EXPERT_PACK="$PACK"
    GGML_MOE_IO_BACKEND=iouring
    GGML_MOE_IO_DEPTH=64
    GGML_KIMI_CPU_SINK_THREADS="$THREADS"
    GGML_KIMI_CPU_SINK_RING_EXPERTS=8
    GGML_KIMI_CPU_SINK_READY_FIRST=1
    GGML_KIMI_CPU_SINK_SPLIT_CV=1
    GGML_KIMI_CPU_SINK_DECODE_RESERVE=1
    GGML_KIMI_CPU_SINK_CACHE_BYTES="$CACHE_BYTES"
    GGML_KIMI_CPU_SINK_CACHE_MAX_PER_LAYER="$CACHE_MAX"
    GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES=536870912
    GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM=1
    GGML_KIMI_CPU_SINK_CACHE_POLICY=count_decay2
    GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform
    GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS=1
    GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE=1
    GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ=1
    GGML_KIMI_CPU_SINK_VRAM_BYTES="$VRAM_BYTES"
    GGML_KIMI_CPU_SINK_VRAM_POLICY=count_decay2
    GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT=2.5
    GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE=16
    GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL=1
    GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE=1
    GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD=1
    GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN=0
    GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS=1
    GGML_KIMI_CPU_SINK_VRAM_PREFILL=1
    GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC=1
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_DEMAND_RECLAIM=1
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS=0
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB="64"
    GGML_KIMI_CPU_SINK_VRAM_WATERMARK_LOW_MIB=16
    GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB=64
    GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD=1
    GGML_KIMI_CPU_SINK_EARLY_UP_GATE=1
    GGML_KIMI_CPU_SINK_ROLE_MAJOR=1
    GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED=1
    GGML_KIMI_CPU_SINK_PREFETCH=2
    GGML_KIMI_CPU_SINK_DYNAMIC_ROWS=1
    GGML_KIMI_CPU_SINK_DYNAMIC_CHUNK_ROWS=64
    GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE=1
    GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROMPT_ONLY=1
    GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_BATCH=1
    GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_CACHE_BYPASS=1
    LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT=1
    LLAMA_DROP_MMAP_EVERY_PROMPT_BATCH=1
    LLAMA_DROP_MMAP_FADVISE_DONTNEED=1
    LLAMA_DROP_DENSE_MMAP_CACHE=1
    MALLOC_ARENA_MAX=1
    GGML_KIMI_CPU_PREFILL_Q8_COST="$Q8_COST"
)
if [[ -n ${EXTRA_ENV:-} ]]; then read -ra EXTRA <<< "$EXTRA_ENV"; ENVIRONMENT+=("${EXTRA[@]}"); fi
echo "Qwen3.8 memory.max=$MEMORY_MAX cache=$CACHE_BYTES threads=$THREADS link=$LINK context=$CONTEXT"
exec systemd-run --user --scope --quiet --unit="${UNIT:-qwen38-fw-$$}" \
 -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
 env "${ENVIRONMENT[@]}" "$BIN_DIR/llama-server" \
 -m "$MODEL" -c "$CONTEXT" -np 1 -t "$THREADS" -tb "$THREADS" -b 4096 -ub 4096 \
 --ctx-checkpoints 4 --fit off -ngl 99 -fa on --defer-experts --no-repack \
 --cache-ram 0 --no-warmup --host 127.0.0.1 --port "$PORT" \
 --jinja --chat-template-kwargs '{"enable_thinking":false}' --reasoning off --reasoning-budget 0 --temp 0 --top-p 1 --top-k 1 --metrics --slots "$@"
