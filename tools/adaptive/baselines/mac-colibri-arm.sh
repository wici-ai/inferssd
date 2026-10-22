#!/usr/bin/env bash
# One colibri baseline arm on macOS: pinned ce370e87d, CPU engine deepseek_v41 (OpenMP via libomp), official
# safetensors, the paper's environment, two real-text turns of 256 tokens, 2700 s cap with startup. The page
# cache is purged first. macOS has no cgroups. Threads: colibri's own policy (physical cores).
# usage: mac-colibri-arm.sh <label> <replay.json> <cap> <ctx>
set -uo pipefail
label=$1; script=$2; cap=$3; ctx=$4
R=$HOME/dsv41; SNAP=$R/models/DeepSeek-V4.1-Flash-official
threads=$(sysctl -n hw.physicalcpu)
run=$R/runs/$(date +%Y%m%d-%H%M%S)-colibri-$label
mkdir -p "$run"; echo "run=$run"
{ echo "label=$label"; echo "script=$script"; echo "cap=$cap"; echo "ctx=$ctx"; echo "threads=$threads"; echo "date=$(date)";
  echo "binary_sha256=$(shasum -a 256 $R/colibri/c/deepseek_v41 | cut -d' ' -f1)"; } > "$run/meta.txt"
sync; sudo -n purge
( while :; do printf '%s ' "$(date +%s)"; vm_stat | awk '/Pages free/{gsub(/\./,"",$NF); printf "free=%s ", $NF}'
    iostat -d -c 1 disk0 | tail -1 | awk '{printf "disk_mb_s=%s", $3}'; echo; sleep 1; done ) > "$run/monitor.txt" 2>/dev/null &
mon=$!
env SNAP="$SNAP" SERVE=1 CTX="$ctx" OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE MALLOC_ARENA_MAX=1 \
    V41_DIRECT=1 V41_READ_DEPTH=8 V41_ENGRAM_ROWS=65536 V41_STATS=1 COLI_KV_PREFIX=0 V41_DSPARK=1 \
    V41_DSPARK_MINACC=60 SEED=61004 \
    python3 - "$R/colibri-venv/bin/python" "$R/bench/colibri-replay.py" --binary "$R/colibri/c/deepseek_v41" \
        --cap "$cap" --script "$script" --out "$run" > "$run/replay.stdout" 2> "$run/replay.stderr" <<'PY'
import subprocess, sys
try:
    rc = subprocess.run(sys.argv[1:], timeout=2700).returncode
except subprocess.TimeoutExpired:
    rc = 124
sys.exit(rc)
PY
echo "rc=$?" >> "$run/meta.txt"
pkill -f "colibri/c/deepseek_v4[1]" 2>/dev/null
kill $mon 2>/dev/null
python3 - "$run" <<'PY'
import json, os, sys
run = sys.argv[1]; p = os.path.join(run, "turns.json")
rows = json.load(open(p)) if os.path.exists(p) else []
f = lambda v: "%.2f" % v if isinstance(v, (int, float)) else "na"
for r in rows:
    print("turn=%d prompt_n=%s ttft_s=%s decode_129_256=%s rss_GB=%s" % (r["turn"], r.get("prompt_n"), f(r.get("ttft_s")),
          f(r.get("decode_tps_129_256")), f(r.get("engine_rss_GB"))))
if len(rows) < 2: print("incomplete: %d of 2 turns" % len(rows))
PY
