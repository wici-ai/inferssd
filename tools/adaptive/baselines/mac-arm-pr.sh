#!/usr/bin/env bash
# One cold llama.cpp baseline arm on macOS (Apple Silicon, Metal build; BIN/TEMPLATE pick the llama.cpp tree), the macOS
# counterpart of arm.sh + run-llamacpp-baseline.sh: GGUF mmap, ssd-llama mechanisms off (they need Linux),
# -b 2048 -ub 512, ctx 131072, llama.cpp's own thread default, two real-text turns of 256 tokens, 2700 s cap.
# The page cache is purged first (cold). macOS has no cgroups: the server gets the whole machine.
# usage: mac-arm.sh <label> <replay.json> <experts: cpu|gpu>
#   cpu: dense layers on Metal, routed experts computed by the CPU backend from the mmap (-ot exps=CPU)
#   gpu: everything on Metal (llama.cpp's default on a Mac), experts read from the same mmap
set -uo pipefail
label=$1; script=$2; experts=$3
ROOT=$HOME/dsv41
M=$ROOT/models/DeepSeek-V4.1-Flash-Q8_0
BIN=${BIN:-$ROOT/src/build/bin/llama-server}
PORT=18231
run=$ROOT/runs/$(date +%Y%m%d-%H%M%S)-$label
mkdir -p "$run"; echo "run=$run"
OT=(); [[ $experts == cpu ]] && OT=(-ot 'ffn_(up|gate|down)_exps=CPU,engram_embd=CPU')
{ echo "label=$label"; echo "script=$script"; echo "experts=$experts"; echo "host=$(hostname)"; echo "date=$(date)";
  sysctl -n machdep.cpu.brand_string hw.memsize; } > "$run/meta.txt"
sync; sudo -n purge
"$BIN" -m "$M/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf" -c ${CTX:-131072} -np 1 -b 2048 -ub 512 \
    --ctx-checkpoints 8 --fit off -ngl 99 -fa on ${OT[@]+"${OT[@]}"} \
    --chat-template-file "${TEMPLATE:-$ROOT/src/models/templates/deepseek-ai-DeepSeek-V4.1.jinja}" \
    --chat-template-kwargs '{"drop_thinking": false}' --no-repack --cache-ram 0 --no-warmup \
    --reasoning off --temp 0 --top-k 1 --host 127.0.0.1 --port $PORT > "$run/server.log" 2>&1 &
spid=$!
start=$(date +%s); ok=0
while (( $(date +%s) - start < 2700 )); do
  curl -fsS -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
  kill -0 $spid 2>/dev/null || break
  sleep 2
done
echo "health_ok=$ok load_s=$(( $(date +%s) - start ))" >> "$run/meta.txt"
( while kill -0 $spid 2>/dev/null; do
    printf '%s ' "$(date +%s)"; vm_stat | awk '/Pages free|Pages active|Pages inactive|Pages wired|File-backed|Pageins|Swapouts/{gsub(/\./,"",$NF); printf "%s=%s ", $0 ~ /free/?"free":$0 ~ /File-backed/?"file":$0 ~ /Pageins/?"pageins":$0 ~ /Swapouts/?"swapouts":$0 ~ /wired/?"wired":$0 ~ /inactive/?"inactive":"active", $NF}'
    iostat -d -c 1 disk0 | tail -1 | awk '{printf "disk_mb_s=%s", $3}'; echo; sleep 1
  done ) > "$run/monitor.txt" 2>/dev/null &
mon=$!
if (( ok )); then
  left=$(( 2700 - ($(date +%s) - start) ))
  python3 - "$left" "$ROOT/bench/replay-stream.py" --port $PORT --script "$script" --out "$run/client.jsonl" > "$run/client.stdout" 2>&1 <<'PY'
import subprocess, sys
left = int(sys.argv[1])
try:
    rc = subprocess.run([sys.executable] + sys.argv[2:], timeout=max(left, 1)).returncode
except subprocess.TimeoutExpired:
    rc = 124
sys.exit(rc)
PY
  echo "client_rc=$?" >> "$run/meta.txt"
fi
kill $spid 2>/dev/null; sleep 5; kill -9 $spid 2>/dev/null; kill $mon 2>/dev/null
grep -E "turn=|ERROR" "$run/client.stdout" | head -4
