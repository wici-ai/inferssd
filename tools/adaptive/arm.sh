#!/usr/bin/env bash
# One cold server with the given environment, one replay script, then stop.
# usage: arm.sh LABEL SCRIPT MAX_TOKENS "K=V K=V ..."
# env:   RUNS LAUNCH SERVER_BIN MODEL EXPERT_SOURCE EXPERT_PACK EXPERT_ALIAS DROP_FILES SERVER_ARGS KNOBS_VIA PORT
#        WATCHDOG_MIN_AVAIL_MB
#        [CLIENT CLIENT_ARGS VRAM_RESERVE_MIB MEMORY_RESERVE_MIB]  (adaptive.sh sets them from models/<model>.conf
#        and the caller's ADAPTIVE_*_RESERVE_MIB)
set -uo pipefail
label=$1; script=$2; maxtok=$3; armenv=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
run=$RUNS/$(date +%Y%m%d-%H%M%S)-$label
mkdir -p "$run"
echo "$run"
printf 'label=%s\nscript=%s\nmax_tokens=%s\narm_env=%s\nserver_bin=%s\ndate=%s\n' \
  "$label" "$script" "$maxtok" "$armenv" "$SERVER_BIN" "$(date -Is)" > "$run/meta.txt"
# drive state (diagnostic only): the engine's reads of 100 random experts at QD 18
python3 "$here/expert_source.py" iolog "$EXPERT_SOURCE" 100 $RANDOM "$run/gate.iolog" &&
  bash "$here/reads.sh" "$run/gate.iolog" 18 "$run/drive.json"
rm -f "$run/gate.iolog"
# every arm starts with the model's DROP_FILES (DeepSeek: the engram shards) out of the page cache
python3 - ${DROP_FILES-$ENGRAM_SHARDS} <<'PY'
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
# the framework has the machine to itself; what other processes keep is only what the caller says (default 0)
echo "vram_reserve_mib=${VRAM_RESERVE_MIB:-0} memory_reserve_mib=${MEMORY_RESERVE_MIB:-0}" >> "$run/meta.txt"
extra=()
[[ ${KNOBS_VIA:-env} == extra_env ]] && extra=(EXTRA_ENV="$armenv")
t0=$(date +%s.%N)
env LLAMA_SERVER_PORT=$PORT SERVER_BIN=$SERVER_BIN $armenv "${extra[@]}" bash "$LAUNCH" \
  ${SERVER_ARGS---reasoning off --temp 0 --top-k 1} \
  > "$run/server.log" 2>&1 < /dev/null &
spid=$!
ok=0
for _ in $(seq 1 900); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 $spid 2>/dev/null || break
  sleep 1
done
# load_s: from the launch to the health check passing (the model, the context, the caches' storage at load)
echo "health_ok=$ok load_s=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')" >> "$run/meta.txt"
if [[ $ok == 1 ]]; then
  ( i=0
    while kill -0 $spid 2>/dev/null; do
      a=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
      if (( a < WATCHDOG_MIN_AVAIL_MB )); then echo "$(date -Is) watchdog $a MiB" >> "$run/watchdog.txt"; kill -9 $spid; break; fi
      # every second: the server's page faults (minor, major) and resident set, the machine's available memory and
      # page cache (MiB), to tell what a prefill waits on besides the expert reads
      (( i++ % 2 )) || echo "$(date +%s.%N) $(awk '{print $10, $12, int($24 * 4 / 1024)}' /proc/$spid/stat 2>/dev/null) $a $(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)" >> "$run/mem.txt"
      sleep 0.5
    done ) &
  wd=$!
  python3 "$here/${CLIENT:-client.py}" --port "$PORT" --script "$script" --max-tokens "$maxtok" --out "$run/client.jsonl" ${CLIENT_ARGS:-} \
    > "$run/client.out" 2>&1
  echo "client_rc=$?" >> "$run/meta.txt"
  kill $wd 2>/dev/null
fi
kill $spid 2>/dev/null
for _ in $(seq 1 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
kill -9 $spid 2>/dev/null
left=$(ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
[[ -n "$left" ]] && { kill -9 "$left" 2>/dev/null; echo "killed_leftover=$left" >> "$run/meta.txt"; }
gzip -1 -f "$run/server.log"
echo "done=$(date -Is)" >> "$run/meta.txt"
