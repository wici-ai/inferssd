#!/usr/bin/env bash
# One cold server with the given environment, one replay script, then stop.
# usage: arm.sh LABEL SCRIPT MAX_TOKENS "K=V K=V ..."
# env:   RUNS LAUNCH SERVER_BIN MODEL EXPERT_SOURCE EXPERT_PACK EXPERT_ALIAS DROP_FILES SERVER_ARGS KNOBS_VIA PORT
#        WATCHDOG_MIN_AVAIL_MB
#        [CLIENT CLIENT_ARGS VRAM_RESERVE_MIB MEMORY_RESERVE_MIB]  (adaptive.sh sets them from models/<model>.conf
#        and the caller's ADAPTIVE_*_RESERVE_MIB)
set -uo pipefail
# Never attach to an existing listener or infer process ownership from its port.
if ! command -v ss >/dev/null || ! command -v systemctl >/dev/null; then
  echo "arm requires ss and systemctl" >&2; exit 2
fi
[[ ${PORT:-} =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "invalid PORT" >&2; exit 2; }
listeners=$(ss -Hltn "sport = :$PORT") || exit 2
[[ -z $listeners ]] || { echo "PORT $PORT is already in use" >&2; exit 2; }
# The standard launcher uses this exact unit. Custom launchers must honor LAUNCH_SCOPE.
scope=ssdash-arm-$(date +%s)-$$-$RANDOM.scope
manager=(systemctl)
(( EUID == 0 )) || manager+=(--user)
spid= wd= scope_started=0
stop_scope() {
  [[ $scope_started == 1 ]] || return 0
  "${manager[@]}" kill --kill-whom=all --signal=TERM "$scope" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    "${manager[@]}" is-active --quiet "$scope" || break
    sleep 0.2
  done
  "${manager[@]}" kill --kill-whom=all --signal=KILL "$scope" >/dev/null 2>&1 || true
}
cleanup() {
  [[ -z $wd ]] || { kill "$wd" 2>/dev/null || true; wait "$wd" 2>/dev/null || true; wd=; }
  stop_scope
  # Only the direct child wrapper; a listener PID is never a kill target.
  [[ -z $spid ]] || { kill "$spid" 2>/dev/null || true; kill -9 "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; spid=; }
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
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
scope_started=1
env LLAMA_SERVER_PORT=$PORT SERVER_BIN=$SERVER_BIN $armenv "${extra[@]}" LAUNCH_SCOPE="$scope" bash "$LAUNCH" \
  ${SERVER_ARGS---reasoning off --temp 0 --top-k 1} \
  > "$run/server.log" 2>&1 < /dev/null &
spid=$!
ok=0
for _ in $(seq 1 900); do
  if "${manager[@]}" is-active --quiet "$scope" && curl -fsS -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 $spid 2>/dev/null || break
  sleep 1
done
# load_s: from the launch to the health check passing (the model, the context, the caches' storage at load)
echo "health_ok=$ok load_s=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')" >> "$run/meta.txt"
client_rc=1
if [[ $ok == 1 ]]; then
  ( i=0
    while kill -0 $spid 2>/dev/null; do
      a=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
      if (( a < WATCHDOG_MIN_AVAIL_MB )); then echo "$(date -Is) watchdog $a MiB" >> "$run/watchdog.txt"; "${manager[@]}" kill --kill-whom=all --signal=KILL "$scope"; break; fi
      # every second: the server's page faults (minor, major) and resident set, the machine's available memory and
      # page cache (MiB), to tell what a prefill waits on besides the expert reads
      (( i++ % 2 )) || echo "$(date +%s.%N) $(awk '{print $10, $12, int($24 * 4 / 1024)}' /proc/$spid/stat 2>/dev/null) $a $(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)" >> "$run/mem.txt"
      sleep 0.5
    done ) &
  wd=$!
  python3 "$here/${CLIENT:-client.py}" --port "$PORT" --script "$script" --max-tokens "$maxtok" --out "$run/client.jsonl" ${CLIENT_ARGS:-} \
    > "$run/client.out" 2>&1
  client_rc=$?
  echo "client_rc=$client_rc" >> "$run/meta.txt"
fi
cleanup
trap - EXIT
gzip -1 -f "$run/server.log"
echo "done=$(date -Is)" >> "$run/meta.txt"
exit "$client_rc"
