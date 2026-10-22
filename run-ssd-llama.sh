#!/usr/bin/env bash
set -Eeuo pipefail

# Start llama-server on the SSD-streaming engine for any model with a conf in tools/adaptive/models/: dense weights
# on the GPU, routed experts read from NVMe by the CPU sink, a GPU expert tier and a RAM expert cache.  The model's
# facts come from MODEL_CONF and from the GGUF itself, the machine's from what it reports, and the engine's settings
# from the caller's environment, SETTINGS and the legacy profile below.  The legacy profile is temporary: it holds
# the values of the DeepSeek launcher of 2026-10-20, defaults of its RTX 5090 deployment included, the same for
# every model and machine, and each value leaves it in a commit of its own for the engine's profile or for the
# source that derives, decides or tunes it (tools/adaptive design, section 9).  Outside that block this file sets
# no model, machine or tuning value.
#
# Inputs (environment):
#   LAUNCH_SCOPE optional unique systemd scope name supplied by the benchmark owner; not a tuning setting.
#   MODEL_CONF   required: a model's data-only conf (tools/adaptive/models/*.conf): NAME, MODEL_GLOB, EXPERT_PACK,
#                TEMPLATE, TEMPLATE_KWARGS, SAMPLING (the model's sampling, passed whenever the conf names it: it
#                overrides the GGUF's general.sampling.*, and a conf whose GGUF carries the model's leaves it out),
#                ENGINE_ENV (model-intrinsic engine switches)
#   MODEL        the first GGUF shard; default: MODEL_GLOB searched for in MODEL_DIR and two levels below it
#   EXPERT_ALIAS a GGUF alias index (experts.tsv) or EXPERT_PACK an expert pack; default: the GGUF's own expert
#                ranges, indexed once by tools/adaptive/expert_source.py into EXPERT_INDEX
#                (~/.cache/ssd-llama/experts/<model>)
#   DENSE_HOMES  a file of tensor<TAB>ram|drive lines (tools/adaptive settings.py writes it): the dense weights that
#                leave the GPU, to RAM (pinned host memory, computed by the CPU in decode) or to the drive (the GGUF
#                itself, read block by block through the expert transport, which needs liburing, and computed on the
#                GPU); their alias rows go to DENSE_HOMES.alias.tsv; default: every dense weight on the GPU
#   SETTINGS     a KEY=VALUE file (the format of MODEL_CONF): launcher settings (the names below), engine variables
#                (GGML_*, LLAMA_*) and ARGS (more llama-server arguments, split at blanks)
#   settings     the launcher's own: PROMPT_CACHE_DIR (~/.cache/ssd-llama/NAME; empty keeps none),
#                VRAM_RESERVE_MIB (0; GPU memory left to another process on the card), MEMORY_RESERVE_MIB (0; host
#                memory left to other processes), MEMORY_MAX (default: what the machine has available less the
#                headroom and the reserve), CPUSET (a taskset CPU list), SERVER_BIN, PRELOAD_LIB, LLAMA_SERVER_HOST
#                (127.0.0.1), LLAMA_SERVER_PORT (18181); the legacy profile's, with their defaults there: THREADS
#                (the CPU sink's threads and -t/-tb), CONTEXT, UBATCH, CTX_CHECKPOINTS, PROMPT_CACHE_MIB,
#                MEMORY_HEADROOM_MIB, VRAM_ELASTIC_SAFETY_MIB
#   the prefill route is one input, a threshold the engine applies to each prefill ubatch: the CPU sink's grouped
#                prefill for ubatches of up to GGML_KIMI_CPU_PREFILL_MAX_TOKENS tokens (and for any the GPU would not
#                offload), the ggml graph on the GPU above, which copies each layer's experts from the expert source
#                with batched io_uring reads.  Without MAX_TOKENS, GGML_KIMI_CPU_PREFILL spells it: 1 (the legacy
#                profile) every ubatch on the sink, 0 none.  The two are one input: over SETTINGS that give MAX_TOKENS,
#                a caller sets MAX_TOKENS too (4294967295 is every ubatch on the sink, as for the engine).  What the
#                graph route needs follows when it is reachable (a threshold below UBATCH).
#   precedence   the caller's environment, then SETTINGS, then the conf, then the legacy profile.  An empty value is
#                no value (the next source decides), except for PROMPT_CACHE_DIR, MEMORY_MAX, EXPERT_ALIAS,
#                EXPERT_PACK and CPUSET, where empty means none.  A value the launcher derives (the expert source,
#                the sink's threads, the GPU memory reserve) is not replaced: an engine variable that contradicts it
#                stops the launch.  Every engine variable the server gets, the ones it inherits too, is printed as an
#                [ssd_setting] line with its source.
#   unified memory  on a GPU without memory of its own (nvidia-smi reports no memory.total) the engine's
#                host-memory ledger arbitrates the one DRAM.  The GPU tier then starts from one elastic block, and the
#                ledger's margin at MEMORY_HEADROOM_MIB but at least 4096 MiB, the smallest a stress run has passed
#                (both derived, see below); the framework's SETTINGS carry a larger margin where its runs learned one
#   LAUNCH_DRY_RUN=1  print the scope, the environment (ENV lines) and the argv (ARG lines) and stop
#   LAUNCH_RESOLVE=1  print the model and the conf as shell assignments and stop
# Extra llama-server arguments can be appended after the script name; they come last.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fw=$SCRIPT_DIR/tools/adaptive
dry_run=${LAUNCH_DRY_RUN:-0}
die() { echo "$*" >&2; exit "${2:-1}"; }
# a fact about the machine that would stop the server: a dry run reports it and goes on
refuse() { if [[ $dry_run == 1 ]]; then echo "dry run: the launch would stop here: $*" >&2; else die "$@"; fi; }

# KEY=VALUE files are data: VALUE runs to the end of the line or to a ' #' comment, optionally in single quotes;
# other lines are blank or comments
declare -A conf=() settings=()
read_kv() {   # FILE ARRAY
    local line k v n=0
    local -n into=$2
    while IFS= read -r line || [[ -n $line ]]; do
        n=$(( n + 1 ))
        [[ $line =~ ^[[:space:]]*(#.*)?$ ]] && continue
        [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] ||
            die "$1:$n: not a KEY=VALUE line (no 'export', no blanks around '='): $line" 2
        k=${BASH_REMATCH[1]} v=${BASH_REMATCH[2]}
        v=${v%%[[:space:]]#*}
        v=${v%"${v##*[![:space:]]}"}
        [[ $v =~ ^\'(.*)\'$ ]] && v=${BASH_REMATCH[1]}
        into[$k]=$v
    done < "$1"
}
: "${MODEL_CONF:?Set MODEL_CONF to a model conf (tools/adaptive/models/*.conf)}"
[[ -f $MODEL_CONF ]] || die "Missing model conf: $MODEL_CONF"
read_kv "$MODEL_CONF" conf
[[ -z ${SETTINGS:-} ]] || { [[ -f $SETTINGS ]] || die "Missing SETTINGS file: $SETTINGS"; read_kv "$SETTINGS" settings; }
launcher_settings=" CONTEXT UBATCH CTX_CHECKPOINTS THREADS PROMPT_CACHE_DIR PROMPT_CACHE_MIB MEMORY_HEADROOM_MIB
    MEMORY_RESERVE_MIB MEMORY_MAX VRAM_RESERVE_MIB VRAM_ELASTIC_SAFETY_MIB CPUSET SERVER_BIN PRELOAD_LIB
    LLAMA_SERVER_HOST LLAMA_SERVER_PORT EXPERT_ALIAS EXPERT_PACK EXPERT_INDEX DENSE_HOMES ARGS "
for name in "${!settings[@]}"; do
    [[ $name =~ ^(GGML|LLAMA)_ || $launcher_settings == *[[:space:]]${name}[[:space:]]* ]] ||
        die "$SETTINGS: $name is neither a launcher setting nor an engine variable (GGML_*, LLAMA_*)" 2
done
model_name=${conf[NAME]:-$(basename "$MODEL_CONF" .conf)}

# where NAME's value comes from: environment (the caller's), settings (SETTINGS) or default
empty_is_none=" PROMPT_CACHE_DIR MEMORY_MAX EXPERT_ALIAS EXPERT_PACK DENSE_HOMES CPUSET "
given_by() {   # NAME
    local none=0
    [[ $empty_is_none == *" $1 "* ]] && none=1
    if [[ -n ${!1:-} || ( $none == 1 && -n ${!1+x} ) ]]; then echo environment
    elif [[ -n ${settings[$1]:-} || ( $none == 1 && -n ${settings[$1]+x} ) ]]; then echo settings
    else echo default; fi
}
# a launcher setting: the given value, else DEFAULT
setting() {   # VAR DEFAULT
    case $(given_by "$1") in
        environment) ;;
        settings) printf -v "$1" %s "${settings[$1]}" ;;
        *) printf -v "$1" %s "$2" ;;
    esac
}
# an engine variable's value: the given value, else DEFAULT
engine_value() {   # NAME DEFAULT
    case $(given_by "$1") in
        environment) printf '%s\n' "${!1}" ;;
        settings) printf '%s\n' "${settings[$1]}" ;;
        *) printf '%s\n' "$2" ;;
    esac
}

# the model: MODEL, or the conf's glob under MODEL_DIR
if [[ -z ${MODEL:-} ]]; then
    : "${MODEL_DIR:?Set MODEL (the first GGUF shard) or MODEL_DIR (where to look for ${conf[MODEL_GLOB]:-MODEL_GLOB})}"
    [[ -n ${conf[MODEL_GLOB]:-} ]] || die "$MODEL_CONF has no MODEL_GLOB; set MODEL"
    shopt -s nullglob
    found=("$MODEL_DIR"/${conf[MODEL_GLOB]} "$MODEL_DIR"/*/${conf[MODEL_GLOB]} "$MODEL_DIR"/*/*/${conf[MODEL_GLOB]})
    shopt -u nullglob
    (( ${#found[@]} == 1 )) || die "MODEL_GLOB ${conf[MODEL_GLOB]} matches ${#found[@]} files under $MODEL_DIR: ${found[*]:-none}; set MODEL"
    MODEL=${found[0]}
fi
# LAUNCH_RESOLVE=1: the model and the conf as shell assignments (MODEL, CONF_<key>), for adaptive.sh
if [[ ${LAUNCH_RESOLVE:-0} == 1 ]]; then
    printf 'MODEL=%q\n' "$MODEL"
    for k in "${!conf[@]}"; do printf 'CONF_%s=%q\n' "$k" "${conf[$k]}"; done
    exit 0
fi

# the launcher's own settings
setting SERVER_BIN "$SCRIPT_DIR/build-cuda130/bin/llama-server"
[[ -x $SERVER_BIN ]] || die "Missing llama-server executable: $SERVER_BIN (set SERVER_BIN)"
BIN_DIR=$(cd -- "$(dirname -- "$SERVER_BIN")" && pwd)
setting PRELOAD_LIB "$BIN_DIR/libkimi-no-mmap-prefetch.so"
setting LLAMA_SERVER_HOST 127.0.0.1
setting LLAMA_SERVER_PORT 18181
HOST=$LLAMA_SERVER_HOST PORT=$LLAMA_SERVER_PORT
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    die "LLAMA_SERVER_PORT must be in 1..65535" 2
fi
setting CPUSET ""
# Saved prompt states go to disk, not to the RAM the expert cache needs: the checkpoints a conversation falls back
# to, the shared system prompt and tools that a new conversation starts from, and conversations that another
# request displaced. They outlive the server. PROMPT_CACHE_DIR= keeps none.
setting PROMPT_CACHE_DIR "${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/$model_name"
if [[ -n "$PROMPT_CACHE_DIR" ]]; then
    fs=$(stat -f -c %T "$(d=$PROMPT_CACHE_DIR; while [[ ! -e $d ]]; do d=$(dirname "$d"); done; echo "$d")")
    [[ $fs == overlayfs || $fs == tmpfs ]] &&
        echo "warning: PROMPT_CACHE_DIR $PROMPT_CACHE_DIR is on $fs; put it on the drive that holds the model" >&2
fi
setting VRAM_RESERVE_MIB 0
if ! [[ "$VRAM_RESERVE_MIB" =~ ^[0-9]+$ ]] || (( VRAM_RESERVE_MIB > 8192 )); then
    die "VRAM_RESERVE_MIB must be in 0..8192" 2
fi
setting MEMORY_RESERVE_MIB 0
setting MEMORY_MAX ""

# where the experts are read from
setting EXPERT_ALIAS ""
setting EXPERT_PACK "${conf[EXPERT_PACK]:-}"
[[ -n "$EXPERT_ALIAS" ]] && EXPERT_PACK=
if [[ -z "$EXPERT_ALIAS" && -z "$EXPERT_PACK" ]]; then
    setting EXPERT_INDEX "${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/experts/$(basename "$MODEL" .gguf)"
    # indexed once: expert_source.py keeps an index that is the model's and newer than its files
    python3 -B "$fw/expert_source.py" index "$MODEL" "$EXPERT_INDEX" > /dev/null ||
        die "Cannot index the experts of $MODEL into $EXPERT_INDEX"
    EXPERT_ALIAS=$EXPERT_INDEX/experts.tsv
fi

for asset in "$MODEL" "${EXPERT_ALIAS:-$EXPERT_PACK}" "$PRELOAD_LIB"; do
    [[ -f "$asset" ]] || die "Missing required asset: $asset"
done
# the model's facts the settings follow from: the experts' types, how many alias entries start off a 4 KiB boundary
facts=$(python3 -B "$fw/expert_source.py" facts "$MODEL" ${EXPERT_ALIAS:+"$EXPERT_ALIAS"}) ||
    die "Cannot read the GGUF header of $MODEL"
eval "$facts"
# the dense weights homed off the GPU: the drive's alias rows in a file of their own (a row of a matrix without a
# split into equal blocks stops the launch there), and one buffer override each way
setting DENSE_HOMES ""
dense_drive=0 dense_drive_bytes=0 dense_ram=0 dense_ram_bytes=0 dense_stage_bytes=0 dense_alias=
dense_args=()
if [[ -n $DENSE_HOMES ]]; then
    [[ -f $DENSE_HOMES ]] || die "Missing DENSE_HOMES file: $DENSE_HOMES"
    homes=$(python3 -B "$fw/expert_source.py" dense "$MODEL" "$DENSE_HOMES" "$DENSE_HOMES.alias.tsv") ||
        die "DENSE_HOMES $DENSE_HOMES cannot be served (above)" 2
    eval "$homes"
    (( dense_drive > 0 )) && dense_alias=$DENSE_HOMES.alias.tsv
    declare -A dense_re=()
    while IFS=$'\t' read -r name home _; do
        [[ -n $name && -n $home ]] && dense_re[$home]+="${dense_re[$home]:+|}${name//./\\.}"
    done < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$DENSE_HOMES")
    [[ -n ${dense_re[drive]:-} ]] && dense_args+=(-ot "^(${dense_re[drive]})\$=CPU")
    [[ -n ${dense_re[ram]:-} ]] && dense_args+=(-ot "^(${dense_re[ram]})\$=CUDA_Host")
fi
mxfp4=0
[[ " $expert_types " == *" MXFP4 "* ]] && mxfp4=1
# the MXFP4 expert kernels are a library of their own, loaded only for MXFP4 experts
libs=("$PRELOAD_LIB")
(( mxfp4 )) && libs+=("$BIN_DIR/libkimi-mxfp4-parity.so")
libs+=("$BIN_DIR/libkimi-vram-tier.so")
preload=$(IFS=:; echo "${libs[*]}")
for lib in "${libs[@]:1}"; do
    [[ -f $lib ]] || die "Missing required asset: $lib"
    if [[ -f $BIN_DIR/libggml-base.so && $lib -ot $BIN_DIR/libggml-base.so ]]; then
        echo "warning: $lib is older than the libggml-base.so it was linked against: rebuild it (tools/adaptive/build.sh)" >&2
    fi
done

# The server's environment.  An engine variable (GGML_*, LLAMA_*) given by the caller or in SETTINGS replaces the
# value below; every one is printed with its source.
ENVIRONMENT=()
declare -A env_set=()
receipt() { echo "[ssd_setting] component=launcher name=$1 value=$2 source=$3${4:+ input=$4}"; }
add_env() {   # NAME VALUE
    [[ -n ${env_set[$1]+x} ]] && die "$1 is set twice" 3
    env_set[$1]=1
    ENVIRONMENT+=("$1=$2")
}
put() {   # NAME VALUE SOURCE [INPUT]: an engine variable, VALUE unless given
    local name=$1 value=$2 source=$3 input=${4:-}
    case $(given_by "$name") in
        environment) value=${!name} source=override input=environment ;;
        settings) value=${settings[$name]} source=settings input=$SETTINGS ;;
    esac
    add_env "$name" "$value"
    receipt "$name" "$value" "$source" "$input"
}
derived() {   # NAME VALUE INPUT: follows from the launcher's input INPUT; an engine variable given another value stops
    local given
    if [[ $1 =~ ^(GGML|LLAMA)_ && $(given_by "$1") != default ]]; then
        given=$(engine_value "$1" "")
        [[ $given == "$2" ]] || die "$1=$given contradicts $1=$2, which the launcher derives from $3: set $3 instead" 2
    fi
    add_env "$1" "$2"
    receipt "$1" "$2" derived "$3"
}

# A GPU without memory of its own (nvidia-smi reports no memory.total) takes the GPU tier, the staging slots and the
# KV cache from the same DRAM as the RAM expert cache and the checkpoints, which the engine's host-memory ledger
# arbitrates: the tier's start and the ledger's margin follow from it (below)
unified=0 gpu_mib=
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || gpu_mib=
    [[ -n $gpu_mib && ! $gpu_mib =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && unified=1
fi

# ---- The legacy profile (temporary) ------------------------------------------------------------------------------
# The validated SSD-streaming engine path as the DeepSeek launcher set it on 2026-10-20 for an RTX 5090 behind PCIe
# Gen4 x4 with 16 GiB of RAM, the same for every model and machine.  Each value leaves this block in a commit of its
# own: into the engine's profile where it stays, or to the source that derives, decides or tunes it (tools/adaptive
# design, sections 5 and 9); a change of value needs a 176 ABBA.  Pending value changes noted elsewhere in this file:
# GGML_MOE_IO_ALIGNED_ALIAS_BATCH when every prefill ubatch takes the sink.
# the CPU sink's threads (and -t/-tb): the CPUs of CPUSET, else the P- and E-cores of a hybrid Intel CPU (its LP-E
# cores left out), else every CPU
cpu_count() { awk -F, '
    { for (i = 1; i <= NF; ++i) {
        n = split($i, r, "-")
        for (cpu = r[1] + 0; cpu <= (n == 2 ? r[2] + 0 : r[1] + 0); ++cpu) seen[cpu] = 1
    } }
    END { for (cpu in seen) ++count; if (count) print count; else exit 1 }' "$@"; }
if [[ -n $CPUSET ]]; then
    setting THREADS "$(cpu_count <<< "$CPUSET")"
elif [[ -r /sys/devices/cpu_core/cpus && -r /sys/devices/cpu_atom/cpus ]]; then
    setting THREADS "$(cpu_count /sys/devices/cpu_core/cpus /sys/devices/cpu_atom/cpus || nproc)"
else
    setting THREADS "$(nproc)"
fi
setting CONTEXT 131072
setting UBATCH 8192
setting CTX_CHECKPOINTS 8
setting PROMPT_CACHE_MIB 16384
setting MEMORY_HEADROOM_MIB 512
# the decode watermarks keep 16..64 MiB free above the reserve, e.g. 576/640 MiB for a reserve of 576 MiB
VRAM_WATERMARK_LOW_MIB=$(( VRAM_RESERVE_MIB > 16 ? VRAM_RESERVE_MIB : 16 ))
VRAM_WATERMARK_HIGH_MIB=$(( VRAM_RESERVE_MIB + 64 ))
setting VRAM_ELASTIC_SAFETY_MIB "$VRAM_WATERMARK_HIGH_MIB"
put GGML_KIMI_CPU_SINK 1 profile
# the prefill threshold: GGML_KIMI_CPU_PREFILL_MAX_TOKENS when given, else the legacy spelling (1: every ubatch)
max_sink_tokens=$(engine_value GGML_KIMI_CPU_PREFILL_MAX_TOKENS "")
if [[ -n $max_sink_tokens ]]; then
    [[ $(given_by GGML_KIMI_CPU_PREFILL) == default ]] ||
        die "GGML_KIMI_CPU_PREFILL and GGML_KIMI_CPU_PREFILL_MAX_TOKENS are both given: the threshold is one input, set MAX_TOKENS" 2
    # a number of tokens in base 10 (as the engine reads it), up to its "every ubatch", 2^32 - 1
    [[ $max_sink_tokens =~ ^[0-9]{1,10}$ ]] && (( 10#$max_sink_tokens <= 4294967295 )) ||
        die "GGML_KIMI_CPU_PREFILL_MAX_TOKENS must be a number of tokens (at most 4294967295, every ubatch)" 2
    put GGML_KIMI_CPU_PREFILL_MAX_TOKENS "$max_sink_tokens" settings
    max_sink_tokens=$(( 10#$max_sink_tokens ))
else
    cpu_prefill=$(engine_value GGML_KIMI_CPU_PREFILL 1)
    [[ $cpu_prefill == 0 || $cpu_prefill == 1 ]] || die "GGML_KIMI_CPU_PREFILL must be 0 (the GPU graph) or 1 (the CPU sink)" 2
    put GGML_KIMI_CPU_PREFILL 1 profile
    max_sink_tokens=$(( cpu_prefill ? 4294967295 : 0 ))
fi
put GGML_MOE_IO_REQUIRE_DIRECT 1 profile
put GGML_MOE_IO_BACKEND iouring profile
put GGML_MOE_IO_DEPTH 64 profile
put GGML_KIMI_CPU_SINK_RING_EXPERTS 8 profile
put GGML_KIMI_CPU_SINK_READY_FIRST 1 profile
put GGML_KIMI_CPU_SINK_SPLIT_CV 1 profile
put GGML_KIMI_CPU_SINK_DECODE_RESERVE 1 profile
put GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES 536870912 profile
put GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM 1 profile
put GGML_KIMI_CPU_SINK_CACHE_POLICY count_decay2 profile
put GGML_KIMI_CPU_SINK_CACHE_ALLOCATION uniform profile
put GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS 1 profile
put GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE 1 profile
put GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ 1 profile
(( unified )) || put GGML_KIMI_CPU_SINK_VRAM_BYTES 21000000000 profile
put GGML_KIMI_CPU_SINK_VRAM_POLICY count_decay2 profile
put GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT 2.5 profile
put GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE 16 profile
put GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL 1 profile
put GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE 1 profile
put GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD 1 profile
put GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN 0 profile
put GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS 1 profile
put GGML_KIMI_CPU_SINK_VRAM_PREFILL 1 profile
(( unified )) || put GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB "$MEMORY_HEADROOM_MIB" profile MEMORY_HEADROOM_MIB
# the GPU tier's start-up safety, what its decode arena must leave free when the first decode token allocates it: a
# tier that grows (the engine's condition: elastic, one global pool kept across requests) on a discrete GPU grows down
# to its elastic margin right after that allocation, so the safety is that margin (the engine's, when it is MiB alone),
# and a start the margin admits does not stop on more; otherwise (unified memory, a tier that does not grow) 1 GiB
elastic_margin=$(engine_value GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB "$VRAM_ELASTIC_SAFETY_MIB")
if (( !unified )) && [[ $(engine_value GGML_KIMI_CPU_SINK_VRAM_ELASTIC 1) == 1 &&
        $(engine_value GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL 1) == 1 &&
        $(engine_value GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS 1) == 1 &&
        $(engine_value GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS 0) == 0 ]]; then
    put GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB "$elastic_margin" derived \
        "the elastic margin (GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB) of a tier that grows on a discrete GPU"
else
    put GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB 1024 profile
fi
put GGML_KIMI_CPU_SINK_VRAM_ELASTIC 1 profile
put GGML_KIMI_CPU_SINK_VRAM_ELASTIC_DEMAND_RECLAIM 1 profile
put GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS 0 profile
put GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB "$VRAM_ELASTIC_SAFETY_MIB" profile "VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB"
put GGML_KIMI_CPU_SINK_VRAM_WATERMARK_LOW_MIB "$VRAM_WATERMARK_LOW_MIB" profile VRAM_RESERVE_MIB
put GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB "$VRAM_ELASTIC_SAFETY_MIB" profile "VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB"
# the mxfp4 adapter's GPU path (not bitwise, accepted 09-24); only the mxfp4 adapter reads it
(( mxfp4 )) && put GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN 1 profile
put GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD 1 profile
put GGML_KIMI_CPU_SINK_EARLY_UP_GATE 1 profile
put GGML_KIMI_CPU_SINK_ROLE_MAJOR 1 profile
put GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED 1 profile
put GGML_KIMI_CPU_SINK_PREFETCH 2 profile
put GGML_KIMI_CPU_SINK_DYNAMIC_ROWS 1 profile
put GGML_KIMI_CPU_SINK_DYNAMIC_CHUNK_ROWS 64 profile
put GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE 1 profile
put GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROMPT_ONLY 1 profile
put GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_BATCH 1 profile
put GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_CACHE_BYPASS 1 profile
put LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT 1 profile
put LLAMA_DROP_MMAP_EVERY_PROMPT_BATCH 1 profile
put LLAMA_DROP_MMAP_FADVISE_DONTNEED 1 profile
put LLAMA_DROP_DENSE_MMAP_CACHE 1 profile
# ---- end of the legacy profile -----------------------------------------------------------------------------------

if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    die "THREADS must be a positive integer" 2
fi
if ! [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] || (( CONTEXT > 1048576 )); then
    die "CONTEXT must be a positive number of tokens up to 1048576" 2
fi
if ! [[ "$UBATCH" =~ ^[1-9][0-9]*$ && "$CTX_CHECKPOINTS" =~ ^[0-9]+$ && "$PROMPT_CACHE_MIB" =~ ^[1-9][0-9]*$ ]]; then
    die "UBATCH and PROMPT_CACHE_MIB must be positive integers, CTX_CHECKPOINTS a nonnegative one" 2
fi
if ! [[ "$VRAM_ELASTIC_SAFETY_MIB" =~ ^[1-9][0-9]*$ ]] || \
        (( VRAM_ELASTIC_SAFETY_MIB < VRAM_WATERMARK_HIGH_MIB || VRAM_ELASTIC_SAFETY_MIB > 8192 )); then
    die "VRAM_ELASTIC_SAFETY_MIB must be in $VRAM_WATERMARK_HIGH_MIB..8192 (the decode watermark above VRAM_RESERVE_MIB)" 2
fi
if ! [[ "$MEMORY_HEADROOM_MIB" =~ ^[0-9]+$ && "$MEMORY_RESERVE_MIB" =~ ^[0-9]+$ ]]; then
    die "MEMORY_HEADROOM_MIB and MEMORY_RESERVE_MIB must be nonnegative integers" 2
fi
# the ggml graph takes a prefill ubatch above the threshold: reachable when a ubatch can be larger
graph_route=0
(( max_sink_tokens < UBATCH )) && graph_route=1

# what follows from the launcher's inputs
derived LD_LIBRARY_PATH "$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" SERVER_BIN
derived LD_PRELOAD "$preload" "SERVER_BIN, PRELOAD_LIB, expert types $expert_types"
# Cover prefill graph buffers and VMM workspaces, not just decode growth.
derived GGML_CUDA_MEMORY_RESERVE_MIB "$VRAM_RESERVE_MIB" VRAM_RESERVE_MIB
derived GGML_MOE_EXPERT_PACK "$EXPERT_PACK" EXPERT_PACK
derived GGML_MOE_EXPERT_GGUF_ALIAS_TSV "$EXPERT_ALIAS${dense_alias:+${EXPERT_ALIAS:+:}$dense_alias}" \
    "EXPERT_ALIAS${dense_alias:+, DENSE_HOMES}"
derived GGML_KIMI_CPU_SINK_THREADS "$THREADS" THREADS
# On unified memory the GPU tier starts from one elastic block and grows after the first decode token into what the
# host-memory ledger leaves, instead of taking a base budget of the one DRAM at the first request.  The ledger's
# margin starts at MEMORY_HEADROOM_MIB, and at least at 4096 MiB, the smallest a stress run of the ledger has passed
# there (2026-10-21): the ledger learns only the growth it does not track in decode.  A margin given as the engine
# variable replaces it.  Muse on the Spark: decode 1.763/2.194 tok/s with both, against 1.726/2.024 with the 21 GB
# base and a 16 GiB margin (2026-10-20).
if (( unified )); then
    put GGML_KIMI_CPU_SINK_VRAM_BYTES auto derived "unified memory (memory.total $gpu_mib)"
    put GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB $(( MEMORY_HEADROOM_MIB > 4096 ? MEMORY_HEADROOM_MIB : 4096 )) derived \
        "unified memory (memory.total $gpu_mib), MEMORY_HEADROOM_MIB"
fi
# graphs sized to each request on either route, at the engine's granularity (one token): a request reserves its
# prefill graph at its own size and leaves decode only the one-token graph, in the sink's phase transitions
actual_shape=1
[[ $(engine_value GGML_KIMI_CPU_SINK 1) == 0 ]] && actual_shape=0
put GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE "$actual_shape" derived GGML_KIMI_CPU_SINK
# The CUDA expert copy reads an alias entry that does not start on a 4 KiB boundary with a synchronous fallback
# unless GGML_MOE_IO_ALIGNED_ALIAS_BATCH widens it to 4 KiB boundaries.  It is derived when the graph route is
# reachable; with every ubatch on the sink it is a legacy hold that keeps DeepSeek's environment (every entry
# unaligned) as it was until a 176 ABBA takes the switch for the sink's reads too.
if (( alias_unaligned > 0 && graph_route )); then
    put GGML_MOE_IO_ALIGNED_ALIAS_BATCH 1 derived "$alias_unaligned unaligned entries of EXPERT_ALIAS"
fi
read -ra model_env <<< "${conf[ENGINE_ENV]:-}"
for kv in "${model_env[@]}"; do
    put "${kv%%=*}" "${kv#*=}" model "$MODEL_CONF"
done
# engine variables in SETTINGS that nothing above sets, then those the server inherits from the caller
for name in $(printf '%s\n' "${!settings[@]}" | sort); do
    [[ $name =~ ^(GGML|LLAMA)_ && -z ${env_set[$name]+x} && $launcher_settings != *[[:space:]]${name}[[:space:]]* ]] ||
        continue
    put "$name" "${settings[$name]}" settings "$SETTINGS"
done
for name in $(compgen -e | grep -E '^(GGML|LLAMA)_' | sort); do
    [[ -z ${env_set[$name]+x} && $launcher_settings != *[[:space:]]${name}[[:space:]]* ]] || continue
    put "$name" "${!name}" override environment
done

if command -v ss >/dev/null 2>&1 && [[ -n "$(ss -Hltn "sport = :$PORT")" ]]; then
    refuse "Port $PORT is already in use"
fi
# the GPU route's expert copy reads through liburing when the CUDA backend calls it (as tools/adaptive/build.sh checks);
# the weights homed on the drive are read only through it
uring=1
LC_ALL=C grep -qs io_uring_queue_init "$BIN_DIR/libggml-cuda.so" "$SERVER_BIN" || uring=0
if (( graph_route && !uring )); then
    echo "warning: the CUDA backend does not call liburing: the GPU prefill route reads every expert with synchronous preads" >&2
fi
if (( dense_drive > 0 && !uring )); then
    refuse "DENSE_HOMES puts $dense_drive weights on the drive, but the CUDA backend does not call liburing: build it with liburing (tools/adaptive/build.sh)"
fi
if [[ -n $DENSE_HOMES ]]; then
    echo "[ssd_setting] component=launcher name=DENSE_HOMES value=$DENSE_HOMES source=$(given_by DENSE_HOMES)" \
        "input=drive=$dense_drive drive_bytes=$dense_drive_bytes ram=$dense_ram ram_bytes=$dense_ram_bytes" \
        "stage_bytes=$dense_stage_bytes"
fi

# The elastic VRAM tier takes all GPU memory that the graphs leave free, so a second CUDA
# process on the card can make either one fail, unless VRAM_RESERVE_MIB leaves it memory.
if command -v nvidia-smi >/dev/null 2>&1; then
    busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    if [[ -n "$busy" ]] && (( VRAM_RESERVE_MIB == 0 )); then
        refuse "The GPU already runs a compute process; stop it first or set VRAM_RESERVE_MIB:"$'\n'"$busy"
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
    refuse "cgroup v2 is required (the RAM expert cache reads memory.max)"
fi
read -r MEM_TOTAL_KIB MEM_AVAILABLE_KIB < <(awk '
    /^MemTotal:/ { total = $2 } /^MemAvailable:/ { available = $2 }
    END { print total, available }' /proc/meminfo)
if [[ -n "$MEMORY_MAX" ]]; then
    MEMORY_SOURCE="MEMORY_MAX"
else
    MEMORY_MAX=$(( (MEM_AVAILABLE_KIB - (MEMORY_HEADROOM_MIB + MEMORY_RESERVE_MIB) * 1024) * 1024 ))
    MEMORY_SOURCE=$(awk -v t="$MEM_TOTAL_KIB" -v a="$MEM_AVAILABLE_KIB" -v h="$MEMORY_HEADROOM_MIB" -v r="$MEMORY_RESERVE_MIB" 'BEGIN {
        printf "total %.2f GiB, in use by the system %.2f GiB, headroom %.2f GiB, reserved for others %.2f GiB",
            t / 1048576, (t - a) / 1048576, h / 1024, r / 1024 }')
    if (( MEMORY_MAX < 4 * 1073741824 )); then
        refuse "Only $(( MEMORY_MAX >> 20 )) MiB would be left for llama-server ($MEMORY_SOURCE)"
    fi
fi
if (( EUID == 0 )); then
    SCOPE=(systemd-run --scope)
else
    SCOPE=(systemd-run --user --scope)
    controllers=/sys/fs/cgroup/user.slice/user-$EUID.slice/user@$EUID.service/cgroup.controllers
    if ! systemctl --user show-environment >/dev/null 2>&1; then
        refuse "No systemd user manager; run from a login session (ssh) or 'loginctl enable-linger $USER'"
    fi
    if ! grep -qw memory "$controllers" 2>/dev/null; then
        refuse "The memory controller is not delegated to $controllers"
    fi
fi

if [[ -n "$PROMPT_CACHE_DIR" ]]; then
    PROMPT_CACHE_ARGS=(--cache-disk "$PROMPT_CACHE_DIR" --cache-disk-mib "$PROMPT_CACHE_MIB")
else
    PROMPT_CACHE_ARGS=(--cache-ram 0)
fi
args=(-m "$MODEL"
    -c "$CONTEXT" -np 1 -t "$THREADS" -tb "$THREADS" -b "$UBATCH" -ub "$UBATCH"
    --ctx-checkpoints "$CTX_CHECKPOINTS" --fit off -ngl 99 -fa on)
if [[ -n ${conf[TEMPLATE]:-} ]]; then
    template=${conf[TEMPLATE]}
    [[ $template == /* ]] || template=$SCRIPT_DIR/$template
    args+=(--chat-template-file "$template")
fi
[[ -n ${conf[TEMPLATE_KWARGS]:-} ]] && args+=(--chat-template-kwargs "${conf[TEMPLATE_KWARGS]}")
if [[ -n ${conf[SAMPLING]:-} ]]; then
    read -ra sampling <<< "${conf[SAMPLING]}"
    args+=("${sampling[@]}")
fi
args+=(--defer-experts --no-repack "${dense_args[@]}" "${PROMPT_CACHE_ARGS[@]}" --no-warmup --host "$HOST" --port "$PORT")
if [[ -n ${settings[ARGS]:-} ]]; then
    read -ra extra <<< "${settings[ARGS]}"
    args+=("${extra[@]}")
fi
args+=("$@")

cpumask=()
[[ -n $CPUSET ]] && cpumask=(taskset -c "$CPUSET")
scope_name=${LAUNCH_SCOPE:-$model_name-$(date +%Y%m%d-%H%M%S)-$$}
[[ $scope_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && ${#scope_name} -le 200 ]] || die "Invalid LAUNCH_SCOPE"
SCOPE+=(--quiet --unit="$scope_name" -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0)

route="prefill on the CPU sink"
(( graph_route )) && route="prefill on the CPU sink up to $max_sink_tokens tokens and where the GPU does not offload, on the GPU graph above"
echo "$model_name on http://$HOST:$PORT (threads=$THREADS, context=$CONTEXT, $route)"
echo "memory.max=$MEMORY_MAX ($MEMORY_SOURCE)"
if [[ $dry_run == 1 ]]; then
    printf 'SCOPE %s\n' "${SCOPE[@]}" "${cpumask[@]}"
    printf 'ENV %s\n' "${ENVIRONMENT[@]}"
    printf 'ARG %s\n' "$SERVER_BIN" "${args[@]}"
    exit 0
fi
# LD_PRELOAD goes through env so that it reaches llama-server, not systemd-run.
exec "${SCOPE[@]}" "${cpumask[@]}" env "${ENVIRONMENT[@]}" "$SERVER_BIN" "${args[@]}"
