#!/usr/bin/env bash
set -Eeuo pipefail

# Start llama-server for DeepSeek-V4.1-Flash (Q8_0 dense + MXFP4 expert pack) with the
# hybrid_prefill_adaptive_u2048 profile of configs/deepseek-v41-flash-mxfp4/local-5090.json
# on the target machine: RTX 5090 behind a real PCIe Gen4 x4 link and 16 GiB of RAM.
# The context and the decode VRAM headroom are set for the target machine and differ from
# the profile: 131072 tokens instead of 2048, VRAM_ELASTIC_SAFETY_MIB defaults to 64 MiB
# instead of 512, and the 16/64 MiB watermarks keep free VRAM in that range during decode.
# VRAM_RESERVE_MIB leaves GPU memory to another process that keeps running on the card
# (a deployment's monitoring listener, for example): the CUDA allocator reclaims expert
# blocks rather than go below it, the watermarks move up by the same amount, and the
# launcher then accepts that process instead of requiring an idle GPU.
# The test server's envelope emulation (libh2dgov at 7 GB/s, CPU pinning, a 16 GiB cgroup
# on a larger host) is not applied; the real link and memory are the envelope here.
#
# Required:  MODEL, and EXPERT_ALIAS or EXPERT_PACK: where the experts are read from. EXPERT_ALIAS indexes their
#            ranges in the GGUF itself (tools/adaptive/expert_source.py index MODEL DIR writes DIR/experts.tsv);
#            EXPERT_PACK is a separate expert pack built from GGUF shards 1-7.
# Optional:  CONTEXT, SERVER_BIN, PRELOAD_LIB, LLAMA_SERVER_HOST, LLAMA_SERVER_PORT, THREADS,
#            MEMORY_HEADROOM_MIB, MEMORY_RESERVE_MIB, MEMORY_MAX, VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB, PROMPT_CACHE_DIR,
#            PROMPT_CACHE_MIB, UBATCH (the prefill ubatch, 8192), GGML_KIMI_CPU_SINK_VRAM_BYTES (21000000000)
# Extra llama-server arguments can be appended after the script name.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:?Set MODEL to the first DeepSeek-V4.1-Flash Q8_0 GGUF shard (...-00001-of-00010.gguf)}"
: "${EXPERT_ALIAS:=}" "${EXPERT_PACK:=}"
if [[ -z "$EXPERT_ALIAS" && -z "$EXPERT_PACK" ]]; then
    echo "Set EXPERT_ALIAS (the GGUF's expert ranges) or EXPERT_PACK (an expert pack)" >&2
    exit 1
fi
[[ -n "$EXPERT_ALIAS" ]] && EXPERT_PACK=

SERVER_BIN=${SERVER_BIN:-"$SCRIPT_DIR/build-cuda130/bin/llama-server"}
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "Missing llama-server executable: $SERVER_BIN (set SERVER_BIN)" >&2
    exit 1
fi
BIN_DIR=$(cd -- "$(dirname -- "$SERVER_BIN")" && pwd)
PRELOAD_LIB=${PRELOAD_LIB:-"$BIN_DIR/libkimi-no-mmap-prefetch.so"}
HOST=${LLAMA_SERVER_HOST:-127.0.0.1}
PORT=${LLAMA_SERVER_PORT:-18181}
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
CONTEXT=${CONTEXT:-131072}
# Saved prompt states go to disk, not to the RAM the expert cache needs: the checkpoints a conversation falls back
# to, the shared system prompt and tools that a new conversation starts from, and conversations that another
# request displaced. They outlive the server. PROMPT_CACHE_DIR= keeps none.
PROMPT_CACHE_DIR=${PROMPT_CACHE_DIR-${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/deepseek-v41-flash-mxfp4}
PROMPT_CACHE_MIB=${PROMPT_CACHE_MIB:-16384}
if [[ -n "$PROMPT_CACHE_DIR" ]]; then
    PROMPT_CACHE_ARGS=(--cache-disk "$PROMPT_CACHE_DIR" --cache-disk-mib "$PROMPT_CACHE_MIB")
else
    PROMPT_CACHE_ARGS=(--cache-ram 0)
fi
VRAM_RESERVE_MIB=${VRAM_RESERVE_MIB:-0}
if ! [[ "$VRAM_RESERVE_MIB" =~ ^[0-9]+$ ]] || (( VRAM_RESERVE_MIB > 8192 )); then
    echo "VRAM_RESERVE_MIB must be in 0..8192" >&2
    exit 2
fi
# the decode watermarks keep 16..64 MiB free above the reserve, e.g. 576/640 MiB for a reserve of 576 MiB
VRAM_WATERMARK_LOW_MIB=$(( VRAM_RESERVE_MIB > 16 ? VRAM_RESERVE_MIB : 16 ))
VRAM_WATERMARK_HIGH_MIB=$(( VRAM_RESERVE_MIB + 64 ))
VRAM_ELASTIC_SAFETY_MIB=${VRAM_ELASTIC_SAFETY_MIB:-$VRAM_WATERMARK_HIGH_MIB}

for asset in "$MODEL" "${EXPERT_ALIAS:-$EXPERT_PACK}" "$PRELOAD_LIB" \
        "$BIN_DIR/libkimi-mxfp4-parity.so" "$BIN_DIR/libkimi-vram-tier.so"; do
    if [[ ! -f "$asset" ]]; then
        echo "Missing required asset: $asset" >&2
        exit 1
    fi
done

if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || (( CONTEXT > 1048576 )); then
    echo "CONTEXT must be a positive number of tokens up to 1048576" >&2
    exit 2
fi
if ! [[ "$VRAM_ELASTIC_SAFETY_MIB" =~ ^[1-9][0-9]*$ ]] || \
        (( VRAM_ELASTIC_SAFETY_MIB < VRAM_WATERMARK_HIGH_MIB || VRAM_ELASTIC_SAFETY_MIB > 8192 )); then
    echo "VRAM_ELASTIC_SAFETY_MIB must be in $VRAM_WATERMARK_HIGH_MIB..8192 (the decode watermark above VRAM_RESERVE_MIB)" >&2
    exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "LLAMA_SERVER_PORT must be in 1..65535" >&2
    exit 2
fi
if command -v ss >/dev/null 2>&1 && [[ -n "$(ss -Hltn "sport = :$PORT")" ]]; then
    echo "Port $PORT is already in use" >&2
    exit 1
fi

# The elastic VRAM tier takes all GPU memory that the graphs leave free, so a second CUDA
# process on the card can make either one fail, unless VRAM_RESERVE_MIB leaves it memory.
if command -v nvidia-smi >/dev/null 2>&1; then
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    if [[ -n "$busy" ]] && (( VRAM_RESERVE_MIB == 0 )); then
        echo "The GPU already runs a compute process; stop it first or set VRAM_RESERVE_MIB:" >&2
        echo "$busy" >&2
        exit 1
    elif [[ -n "$busy" ]]; then
        echo "Sharing the GPU with (VRAM_RESERVE_MIB=$VRAM_RESERVE_MIB):" >&2
        echo "$busy" >&2
    fi
else
    echo "warning: nvidia-smi not found; skipping the idle-GPU check" >&2
fi

# The CPU sink's RAM expert cache admits slots only while some ancestor cgroup has a
# finite cgroup-v2 memory.max (check_room in ggml/src/ggml-cpu/kimi-cpu-sink.cpp); on an
# unbounded cgroup it caches nothing. This scope therefore only publishes how much memory
# the server can have: MemAvailable at launch (memory the running system does not need,
# counting reclaimable page cache as free) minus MEMORY_HEADROOM_MIB for later growth of
# the system and for driver allocations outside the cgroup, and minus MEMORY_RESERVE_MIB
# that other processes keeping running on the machine will still take. The cache's guard keeps a
# 512 MiB reserve inside the scope and gives back slots under pressure; the cache sizes itself
# at the first decode token from the room the scope still has (with a unified-memory GPU, only
# the slots the next-layer prefetch lands in: the GPU tier holds experts in the same DRAM).
# The guard requires that its pages cannot be swapped, hence MemorySwapMax=0.
if [[ ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
    echo "cgroup v2 is required (the RAM expert cache reads memory.max)" >&2
    exit 1
fi
MEMORY_HEADROOM_MIB=${MEMORY_HEADROOM_MIB:-512}
MEMORY_RESERVE_MIB=${MEMORY_RESERVE_MIB:-0}
if ! [[ "$MEMORY_HEADROOM_MIB" =~ ^[0-9]+$ && "$MEMORY_RESERVE_MIB" =~ ^[0-9]+$ ]]; then
    echo "MEMORY_HEADROOM_MIB and MEMORY_RESERVE_MIB must be nonnegative integers" >&2
    exit 2
fi
read -r MEM_TOTAL_KIB MEM_AVAILABLE_KIB < <(awk '
    /^MemTotal:/ { total = $2 } /^MemAvailable:/ { available = $2 }
    END { print total, available }' /proc/meminfo)
if [[ -n "${MEMORY_MAX:-}" ]]; then
    MEMORY_SOURCE="MEMORY_MAX"
else
    MEMORY_MAX=$(( (MEM_AVAILABLE_KIB - (MEMORY_HEADROOM_MIB + MEMORY_RESERVE_MIB) * 1024) * 1024 ))
    MEMORY_SOURCE=$(awk -v t="$MEM_TOTAL_KIB" -v a="$MEM_AVAILABLE_KIB" -v h="$MEMORY_HEADROOM_MIB" -v r="$MEMORY_RESERVE_MIB" 'BEGIN {
        printf "total %.2f GiB, in use by the system %.2f GiB, headroom %.2f GiB, reserved for others %.2f GiB",
            t / 1048576, (t - a) / 1048576, h / 1024, r / 1024 }')
    if (( MEMORY_MAX < 4 * 1073741824 )); then
        echo "Only $(( MEMORY_MAX >> 20 )) MiB would be left for llama-server ($MEMORY_SOURCE)" >&2
        exit 1
    fi
fi
if (( EUID == 0 )); then
    SCOPE=(systemd-run --scope)
else
    SCOPE=(systemd-run --user --scope)
    controllers=/sys/fs/cgroup/user.slice/user-$EUID.slice/user@$EUID.service/cgroup.controllers
    if ! systemctl --user show-environment >/dev/null 2>&1; then
        echo "No systemd user manager; run from a login session (ssh) or 'loginctl enable-linger $USER'" >&2
        exit 1
    fi
    if ! grep -qw memory "$controllers" 2>/dev/null; then
        echo "The memory controller is not delegated to $controllers" >&2
        exit 1
    fi
fi

ENVIRONMENT=(
    LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    LD_PRELOAD="$PRELOAD_LIB:$BIN_DIR/libkimi-mxfp4-parity.so:$BIN_DIR/libkimi-vram-tier.so"
    GGML_KIMI_CPU_SINK=1
    # Cover prefill graph buffers and VMM workspaces, not just decode growth.
    GGML_CUDA_MEMORY_RESERVE_MIB="$VRAM_RESERVE_MIB"
    GGML_KIMI_CPU_PREFILL=1
    GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE=1
    GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY=1
    GGML_MOE_EXPERT_PACK="$EXPERT_PACK"
    GGML_MOE_EXPERT_GGUF_ALIAS_TSV="$EXPERT_ALIAS"
    GGML_MOE_IO_REQUIRE_DIRECT=1
    GGML_MOE_IO_BACKEND=iouring
    GGML_MOE_IO_DEPTH=64
    GGML_KIMI_CPU_SINK_THREADS="$THREADS"
    GGML_KIMI_CPU_SINK_RING_EXPERTS=8
    GGML_KIMI_CPU_SINK_READY_FIRST=1
    GGML_KIMI_CPU_SINK_SPLIT_CV=1
    GGML_KIMI_CPU_SINK_DECODE_RESERVE=1
    GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES=536870912
    GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM=1
    GGML_KIMI_CPU_SINK_CACHE_POLICY=count_decay2
    GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform
    GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS=1
    GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE=1
    GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ=1
    GGML_KIMI_CPU_SINK_VRAM_BYTES="${GGML_KIMI_CPU_SINK_VRAM_BYTES:-21000000000}"
    GGML_KIMI_CPU_SINK_VRAM_POLICY=count_decay2
    GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT=2.5
    GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE=16
    GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL=1
    GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE=1
    GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD=1
    GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN=0
    GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS=1
    GGML_KIMI_CPU_SINK_VRAM_PREFILL=1
    GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB="$MEMORY_HEADROOM_MIB"
    GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC=1
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_DEMAND_RECLAIM=1
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS=0
    GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB="$VRAM_ELASTIC_SAFETY_MIB"
    GGML_KIMI_CPU_SINK_VRAM_WATERMARK_LOW_MIB="$VRAM_WATERMARK_LOW_MIB"
    GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB="$VRAM_ELASTIC_SAFETY_MIB"
    GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1
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
    LLAMA_DSV41_ENGRAM_PREFETCH=1
    LLAMA_DSV41_HC_FUSED=1
)

echo "DeepSeek-V4.1-Flash MXFP4 on http://$HOST:$PORT (threads=$THREADS, context=$CONTEXT)"
echo "memory.max=$MEMORY_MAX ($MEMORY_SOURCE)"

# LD_PRELOAD goes through env so that it reaches llama-server, not systemd-run.
exec "${SCOPE[@]}" --quiet --unit="dsv41-mxfp4-$(date +%Y%m%d-%H%M%S)-$$" \
    -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
    env "${ENVIRONMENT[@]}" "$SERVER_BIN" \
    -m "$MODEL" \
    -c "$CONTEXT" -np 1 -t "$THREADS" -tb "$THREADS" -b "${UBATCH:-8192}" -ub "${UBATCH:-8192}" \
    --ctx-checkpoints 8 --fit off -ngl 99 -fa on \
    --chat-template-file "$SCRIPT_DIR/models/templates/deepseek-ai-DeepSeek-V4.1.jinja" \
    --chat-template-kwargs '{"drop_thinking": false}' \
    --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
    --defer-experts --no-repack "${PROMPT_CACHE_ARGS[@]}" --no-warmup \
    --host "$HOST" --port "$PORT" \
    "$@"
