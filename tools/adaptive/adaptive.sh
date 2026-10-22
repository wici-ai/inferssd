#!/usr/bin/env bash
# The adaptive framework on one machine, from nothing.  Steps 1-4 measure the machine and decide:
#   1  device probe (probe.sh): CPU groups, memory, the GPU link, the drive at a decode layer's queue depths
#   2  a bootstrap per CPU placement (arm.sh + boot_client.py): a real-text turn with a cold and a warm prefill and
#      512 decode tokens, traced; the GPU graph's prefill of a cut of that text and of the whole; then, for each
#      larger prefill ubatch the model allows, a start-up probe of the GPU tier it leaves
#   3  in-situ costs and the model's routing profile (insitu.py)
#   4  the plan (plan.py): the configuration with the least time for the expected conversation, written as
#      settings.env (settings.py), a SETTINGS file for run-ssd-llama.sh
# then
#   serve      serve.py: an HTTP front end on SERVE_PORT, the closed loop re-planning at every request
#   validate   the conversation WORK four times (converse.py), against what the plan and the loop predicted:
#                loop-plan      closed loop starting from the plan's choice
#                static-plan    the plan's choice for the whole conversation
#                static-other   the best configuration with another CPU placement, for the whole conversation
#                loop-other     closed loop starting from that other configuration (a wrong start on purpose)
#              every conversation on a fresh server with an empty disk prompt cache and the model's DROP_FILES out
#              of the page cache
# A run directory that has steps done keeps them (a second `serve` starts at once).
# usage: ROOT=<dir with models/> [MODEL_CONF=models/<model>.conf] adaptive.sh serve|validate [OUT] [-- server args]
#        (the model: MODEL, the first GGUF shard, else the conf's MODEL_GLOB under MODEL_DIR, default ROOT/models)
# env:   WORKLOAD learned (the plan assumes PRIOR_TURNS turns the size of the bootstrap text and the loop learns the
#        rest from the requests; default) or known (validate: the conversation's prompt lengths); HORIZON (3),
#        PLACEMENTS (the model's), SERVE_PORT (8080), SERVE_HOST (127.0.0.1)
#        ADAPTIVE_VRAM_RESERVE_MIB, ADAPTIVE_MEMORY_RESERVE_MIB: GPU and host memory left to other processes that keep
#        running (default 0: the framework assumes the machine to itself and does not infer what others hold)
set -uo pipefail
mode=${1:-}
[[ $mode == serve || $mode == validate ]] || { sed -n '2,26p' "$0"; exit 2; }
shift
out=
[[ $# -gt 0 && $1 != -- ]] && { out=$1; shift; }
[[ ${1:-} == -- ]] && shift
export ROOT=${ROOT:-$PWD}
[[ -x $ROOT/fio/bin/fio ]] && export PATH=$ROOT/fio/bin:$PATH
export FW=${FW:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
# the engine checkout the framework lives in: the model configurations take the launcher and the build from it
export SRC=${SRC:-$(cd "$FW/../.." && pwd)}
MODEL_CONF=${MODEL_CONF:-$FW/models/dsv41.conf}
[[ -z ${DSV41_DIR:-}${DSV41_PACK:-} ]] ||
  { echo "DSV41_DIR and DSV41_PACK are gone: set MODEL_DIR (or MODEL, the first shard) and EXPERT_PACK" >&2; exit 2; }
if grep -q '^LAUNCH=' "$MODEL_CONF"; then
  source "$MODEL_CONF"   # a conf that names its own launcher and engine (qwen38.conf), a script
else
  # a data-only conf of this checkout's engine, never executed: run-ssd-llama.sh reads it and finds the model
  # (MODEL, else the conf's MODEL_GLOB under MODEL_DIR, default ROOT/models)
  export MODEL_CONF LAUNCH=$SRC/run-ssd-llama.sh
  resolved=$(MODEL_DIR=${MODEL_DIR:-$ROOT/models} LAUNCH_RESOLVE=1 bash "$LAUNCH") || exit 1
  eval "$resolved"
  EXPERT_PACK=${EXPERT_PACK:-${CONF_EXPERT_PACK:-}}
  # settings the framework derives but does not pass for this model (settings.py)
  PENDING=${CONF_PENDING:-}
  MODEL_PLACEMENTS=${CONF_MODEL_PLACEMENTS:-all} MODEL_UBATCHES=${CONF_MODEL_UBATCHES:-}
  SERVER_BIN=${SERVER_BIN:-$SRC/build-cuda130/bin/llama-server}
  # cold: no page of any shard of the model cached
  DROP_FILES=$MODEL
  [[ $MODEL =~ ^(.*)-[0-9]{5}-of-([0-9]{5})\.gguf$ ]] &&
    DROP_FILES=$(ls "${BASH_REMATCH[1]}"-*-of-"${BASH_REMATCH[2]}".gguf | tr '\n' ' ')
  # llama-server arguments of the bootstraps and the validation (deterministic, no thinking); the launcher takes
  # the knobs from its caller's environment; a restarted server gets the conversation back from its disk prompt cache
  SERVER_ARGS=${SERVER_ARGS---reasoning off --temp 0 --top-k 1} KNOBS_VIA=env SWITCH_KEEPS_HISTORY=1
fi
# where the experts are read from: the GGUF's own ranges, indexed once per model (expert_source.py keeps an index
# that is up to date, the launcher's too: EXPERT_INDEX), or the conf's EXPERT_PACK
EXPERT_PACK=${EXPERT_PACK:-}
export EXPERT_SOURCE=${EXPERT_SOURCE:-$ROOT/experts/$(basename "${EXPERT_PACK:-$MODEL}")}
python3 $FW/expert_source.py index "$MODEL" ${EXPERT_PACK:+"$EXPERT_PACK"} "$EXPERT_SOURCE" > /dev/null ||
  { echo "no expert index for $MODEL" >&2; exit 1; }
[[ -n $EXPERT_PACK ]] || export EXPERT_INDEX=$EXPERT_SOURCE
EXPERT_ALIAS=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['alias'] or '')" "$EXPERT_SOURCE/experts.json")
export MODEL EXPERT_PACK EXPERT_ALIAS LAUNCH SERVER_BIN DROP_FILES SERVER_ARGS KNOBS_VIA SWITCH_KEEPS_HISTORY
export PORT=${PORT:-18250} WATCHDOG_MIN_AVAIL_MB=${WATCHDOG_MIN_AVAIL_MB:-256}
# what other processes keep, for every server the framework starts (bootstraps, validation, serve)
export VRAM_RESERVE_MIB=${ADAPTIVE_VRAM_RESERVE_MIB:-0} MEMORY_RESERVE_MIB=${ADAPTIVE_MEMORY_RESERVE_MIB:-0}
out=${out:-$ROOT/fwruns/$(date +%Y%m%d-%H%M%S)}
export RUNS=$out
mkdir -p "$out"
T0=$(date +%s)
log() { echo "$(date +%T) +$(( $(date +%s) - T0 ))s $*" | tee -a "$out/pipeline.log"; }
TR="GGML_KIMI_CPU_SINK_TRACE=1 GGML_KIMI_CPU_SINK_PROFILE=1"
BOOT_PROMPT=${BOOT_PROMPT:-$FW/prompts/boot-en-doc.json}
WORK=${WORK:-$FW/prompts/work-4turn.json}
DECODE=512 WINDOW=129-512

# 1: the device (the drive is read by ioreplay, built by reads.sh on first use)
export IOREPLAY=${IOREPLAY:-$ROOT/probe/ioreplay}
MEMPROBE=${MEMPROBE:-$ROOT/probe/memprobe}
if [[ ! -x $MEMPROBE ]] && command -v nvcc > /dev/null; then
  mkdir -p "$(dirname "$MEMPROBE")" && nvcc -O2 -o "$MEMPROBE" $FW/probe/memprobe.cu >> "$out/pipeline.log" 2>&1
fi
# the grouped-prefill lanes are timed by the build's test-kimi-vram-group (BUILD_TESTS=1), when the engine has one
export GROUP_TIMER=${GROUP_TIMER:-$(dirname "$SERVER_BIN")/test-kimi-vram-group}
if [[ ! -s $out/probe/device.json ]]; then
  log "step 1: device probe"
  bash $FW/probe.sh "$out/probe" "$EXPERT_SOURCE" "$MEMPROBE" >> "$out/pipeline.log" 2>&1 ||
    { log "device probe failed (the reason is above in pipeline.log): stop"; exit 1; }
fi
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for p in d['slow_files']:
    print(f'warning: {p} reads at {d[\"expert_files\"][p]} GiB/s, under half the other expert files: its drive blocks'
          ' are degraded; rewrite it (copy, then move over the original)')" "$out/probe/device.json" | tee -a "$out/pipeline.log"
# the host-to-GPU link the probe measured, for a launcher that sets its prefill costs from it
LINK_GBPS=${LINK_GBPS:-$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['gpu'].get('h2d_pinned_gbs') or '')" "$out/probe/device.json" 2>/dev/null)}
export LINK_GBPS

# 2: a bootstrap per CPU placement (placement.py); the first is the engine default and the reference, and also
# counts the tokens of WORK's turns; each client counts what the drives of the expert files read in decode
bootok() { grep -q "^health_ok=1" "$(head -1 "$out/$1.out")/meta.txt" 2>/dev/null; }
PLACEMENTS=${PLACEMENTS:-${MODEL_PLACEMENTS:-all}}
runs=""
first=1
for p in $PLACEMENTS; do
  if (( first )); then
    penv="GGML_KIMI_CPU_SINK_MRC=1"
    extra=(CLIENT_ARGS="--tokenize $WORK --tokenize-out $out/work-tokens.json --drive $EXPERT_SOURCE/experts.json")
  else
    penv=$(python3 $FW/placement.py "$out/probe/device.json" $p); extra=(CLIENT_ARGS="--drive $EXPERT_SOURCE/experts.json")
    [[ -n $penv ]] || { log "placement $p does not exist on this machine, skipped"; continue; }
  fi
  if [[ ! -s $out/boot-$p.out ]]; then
    log "step 2: bootstrap, placement $p (${penv:-engine default}), $(basename $BOOT_PROMPT), 512 decode tokens"
    env PROMPT_CACHE_DIR=$out/pcache CLIENT=boot_client.py "${extra[@]}" \
      bash $FW/arm.sh boot-$p $BOOT_PROMPT 512 "$TR $penv" > "$out/boot-$p.out" 2>&1
    rm -rf "$out/pcache"
  fi
  bootok boot-$p || { log "bootstrap $p: server did not come up, stop"; exit 1; }
  (( first )) && { [[ -s $out/work-tokens.json ]] || { log "no workload tokens, stop"; exit 1; }; }
  runs="$runs $p=$(head -1 "$out/boot-$p.out")"
  first=0
done
# the GPU graph's prefill (prefill.py prices the graph route from it): a fresh server of the engine default at the
# threshold 0 prefills, each from scratch on the ggml graph, a cut of the bootstrap text that the decode trace covers
# and the whole text.  A server that does not start, or prefills that fail, leave the graph route unmeasured.  A conf
# with a launcher of its own (qwen38.conf) has no graph route.
GRAPH=""
if [[ $LAUNCH == "$SRC/run-ssd-llama.sh" ]]; then
  if [[ ! -s $out/boot-graph.out ]]; then
    log "step 2g: the GPU graph's prefill, $(basename $BOOT_PROMPT) cut and whole"
    env PROMPT_CACHE_DIR= CLIENT=boot_client.py CLIENT_ARGS=--graph \
      bash $FW/arm.sh boot-graph $BOOT_PROMPT 512 "GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0" > "$out/boot-graph.out" 2>&1
  fi
  g=$(head -1 "$out/boot-graph.out")
  [[ -d $g ]] && GRAPH="--graph $g"
fi
# larger prefill ubatches (MODEL_UBATCHES, the launcher's UBATCH): a start-up of the engine default with each and a
# short request gives the tier the GPU memory leaves, the prefill buffers growing with the ubatch; one that does not
# start is out.  The request decodes four tokens: a discrete GPU allots the tier at the first request, unified memory
# grows it from one block after the first decode token (ubprobe.py reads the most it held)
UBP=""
if [[ -n ${MODEL_UBATCHES:-} ]]; then
  if [[ ! -s $out/ubatch.json ]]; then
    python3 -c "
import json, sys
t = json.load(open(sys.argv[1], encoding='utf-8'))[0]['text']
cut = t.find('. ', 800)
json.dump([{'text': t[:cut + 1 if cut > 0 else 1000]}], open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)" \
      "$BOOT_PROMPT" "$out/probe-prompt.json"
    for ub in $MODEL_UBATCHES; do
      log "step 2u: start-up probe, prefill ubatch $ub"
      env PROMPT_CACHE_DIR= CLIENT=client.py bash $FW/arm.sh ubprobe-$ub $out/probe-prompt.json 4 "UBATCH=$ub" > "$out/ubprobe-$ub.out" 2>&1
    done
    python3 $FW/ubprobe.py "$out" $MODEL_UBATCHES > "$out/ubatch.json"
    log "ubatch probe: $(cat $out/ubatch.json)"
  fi
  UBP="--ubatch-probe $out/ubatch.json"
fi

# 3: in-situ costs
if [[ ! -s $out/insitu.json ]]; then
  log "step 3: in-situ costs + model profile ($runs)"
  python3 $FW/insitu.py $(for r in $runs; do echo --run $r; done) --skip 128 -o "$out/insitu.json" \
    --profile "$out/model.json" $UBP $GRAPH > "$out/insitu.txt" 2>&1 || { log "insitu failed"; exit 1; }
fi

# 4: the plan
WORKLOAD=${WORKLOAD:-learned} HORIZON=${HORIZON:-3} PRIOR_TURNS=${PRIOR_TURNS:-4}
export HORIZON
if [[ $WORKLOAD == known ]]; then
  TOK=$(python3 -c "import json, sys; print(','.join(map(str, json.load(open(sys.argv[1])))))" "$out/work-tokens.json")
else
  TOK=$(python3 -c "
import json, sys
c = [json.loads(x) for x in open(sys.argv[1])]
n = next(r['timings']['prompt_n'] for r in c if r.get('kind') == 'chat'); print(','.join([str(n)] * int(sys.argv[2])))" \
    "$(head -1 "$out/boot-${PLACEMENTS%% *}.out")/client.jsonl" $PRIOR_TURNS)
fi
log "step 4: plan (workload $WORKLOAD: prompt tokens $TOK, $DECODE decode tokens a turn)"
python3 $FW/plan.py "$out" --prompts $TOK --decode $DECODE --window $WINDOW --pending "${PENDING:-}" \
  $([[ $WORKLOAD == known ]] && echo --known) -o "$out/plan.json" > "$out/plan.txt" 2>&1 || { log "plan failed"; exit 1; }
read -r CHOICE OTHER < <(python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))
print(*(f\"{r['tier']},{r['cpus']}\" for r in (p['plan'], p['other']) if r))" "$out/plan.json")
log "DECISION_READY plan $CHOICE${OTHER:+, other $OTHER}"
# the plan's choice as a SETTINGS file for run-ssd-llama.sh, every value with its source
python3 $FW/settings.py "$out" -o "$out/settings.env" || { log "settings failed"; exit 1; }
log "settings: $out/settings.env"

if [[ $mode == serve ]]; then
  log "serving from $CHOICE on ${SERVE_HOST:-127.0.0.1}:${SERVE_PORT:-8080}"
  exec python3 $FW/serve.py "$out" "$@"
fi

# validate
conv() {   # label mode tier,cpus
  local d=$out/conv-$1
  mkdir -p "$d"
  python3 $FW/expert_source.py iolog "$EXPERT_SOURCE" 100 $RANDOM "$d/gate.iolog" &&
    bash $FW/reads.sh "$d/gate.iolog" 18 "$d/drive.json"
  rm -f "$d/gate.iolog"
  python3 -c "
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" $DROP_FILES
  log "conversation $1: $2 from $3 (reserved for others: $VRAM_RESERVE_MIB MiB GPU, $MEMORY_RESERVE_MIB MiB host)"
  PROMPT_CACHE_DIR=$d/pcache python3 $FW/converse.py --boot "$out" --script "$WORK" \
    --tokens "$out/work-tokens.json" --decode $DECODE --window $WINDOW --mode $2 --start "$3" --out "$d" \
    --workload $WORKLOAD --horizon $HORIZON > "$d/converse.out" 2>&1
  log "conversation $1 exit $? $(tail -1 "$d/converse.out")"
  rm -rf "$d/pcache"
}
conv loop-plan loop "$CHOICE"
conv static-plan static "$CHOICE"
if [[ -n $OTHER ]]; then
  conv static-other static "$OTHER"
  conv loop-other loop "$OTHER"
fi
python3 $FW/compare_online.py "$out" > "$out/compare-online.txt" 2>&1
# settings.env again, with what the host-memory ledger learned in the validation's servers too
python3 $FW/settings.py "$out" --ledgers "$out"/conv-*/inst* -o "$out/settings.env" || log "settings failed"
log "VALIDATE_DONE"
