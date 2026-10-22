#!/usr/bin/env bash
# One cold server, one replay script, then stop.  Portable across the three devices.
#
# usage: arm.sh <label> <replay.json> [extra llama-server args...]
# env:   ROOT (work dir with launcher/src/bin), MODEL, EXPERT_PACK, SERVER_BIN, ENGRAM_SHARDS ("a b"),
#        LAUNCH (launcher path, default $ROOT/src/run-deepseek-v41-flash-mxfp4.sh),
#        ARM_ENV (extra "K=V K=V" passed to the launcher), MAX_TOKENS (256), PORT (18231),
#        FIO_MIN_GIB (skip arm if the pack reads slower; 0 = no gate)
set -euo pipefail
label=${1:?label}; script=${2:?replay json}; shift 2
ROOT=${ROOT:?}
LAUNCH=${LAUNCH:-$ROOT/src/run-deepseek-v41-flash-mxfp4.sh}
PORT=${PORT:-18231}
MAX_TOKENS=${MAX_TOKENS:-256}
FIO_MIN_GIB=${FIO_MIN_GIB:-0}
run=$ROOT/runs/$(date +%Y%m%d-%H%M%S)-$label
mkdir -p "$run"
echo "run=$run"
{
  echo "label=$label"; echo "script=$script"; echo "extra=$*"; echo "arm_env=${ARM_ENV:-}"
  echo "host=$(hostname)"; echo "date=$(date -Is)"
  echo "server_bin=${SERVER_BIN:-}"; echo "model=$MODEL"; echo "pack=$EXPERT_PACK"
} > "$run/meta.txt"

# drive state probe (read-only, O_DIRECT, 10 s)
fio --name=pack-read --filename="$EXPERT_PACK" --readonly --direct=1 --rw=randread --bs=6266880 \
    --iodepth=18 --ioengine=libaio --size=250G --norandommap=1 --randrepeat=0 --time_based=1 --runtime=10 \
    --output-format=json --output="$run/fio.json" >/dev/null 2>&1 || true
fio_gib=$(python3 -c "import json;print('%.3f'%(json.load(open('$run/fio.json'))['jobs'][0]['read']['bw_bytes']/2**30))" 2>/dev/null || echo 0)
echo "fio_gib_s=$fio_gib" | tee -a "$run/meta.txt"
if python3 -c "import sys; sys.exit(0 if float('$fio_gib') >= float('$FIO_MIN_GIB') else 1)"; then :; else
  echo "drive below gate ($fio_gib < $FIO_MIN_GIB GiB/s)"; exit 42
fi
# NVMe temperature (best effort)
(cat /sys/class/nvme/nvme0/hwmon*/temp*_input 2>/dev/null | tr '\n' ' '; echo) > "$run/nvme-temp-before.txt" || true

# cold engram page cache for the two engram shards
if [[ -n "${ENGRAM_SHARDS:-}" ]]; then
  python3 - $ENGRAM_SHARDS <<'PY' > "$run/pagecache.txt"
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print("dropped", p)
PY
fi

cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "${spid:-}" ]]; then
    kill "$spid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
    kill -9 "$spid" 2>/dev/null || true
  fi
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  [[ -n "${mon:-}" ]] && kill "$mon" 2>/dev/null || true
  [[ -n "${wd:-}" ]] && kill "$wd" 2>/dev/null || true
  (cat /sys/class/nvme/nvme0/hwmon*/temp*_input 2>/dev/null | tr '\n' ' '; echo) > "$run/nvme-temp-after.txt" || true
  echo "arm_exit=$rc" >> "$run/meta.txt"
  gzip -1 -f "$run/server.log" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT

env LLAMA_SERVER_PORT=$PORT ${ARM_ENV:-} bash "$LAUNCH" --reasoning off --temp 0 --top-k 1 "$@" > "$run/server.log" 2>&1 &
spid=$!
ok=0
for _ in $(seq 1 900); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$spid" 2>/dev/null || break
  sleep 1
done
echo "health_ok=$ok" >> "$run/meta.txt"
[[ $ok == 1 ]] || { tail -30 "$run/server.log"; exit 1; }

# Optional memory watchdog: on a unified-memory machine the GPU allocations are invisible to the
# cgroup, so a host OOM is possible; kill the server before MemAvailable falls below the floor.
if [[ -n "${WATCHDOG_MIN_AVAIL_MB:-}" ]]; then
  (
    while kill -0 "$spid" 2>/dev/null; do
      a=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
      if (( a < WATCHDOG_MIN_AVAIL_MB )); then
        echo "$(date -Is) watchdog: MemAvailable ${a} MiB < ${WATCHDOG_MIN_AVAIL_MB}; killing server" >> "$run/watchdog.txt"
        kill -9 "$spid" 2>/dev/null; pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null
        break
      fi
      sleep 0.5
    done
  ) &
  wd=$!
fi

# 1 Hz system monitor: disk read bytes, memory, GPU memory
(
  while :; do
    printf '%s ' "$(date +%s.%N)"
    awk '$3=="nvme0n1"{printf "rd_sect=%s ", $6}' /proc/diskstats
    awk '/^MemAvailable:/{printf "memavail_kb=%s ", $2} /^Cached:/{printf "cached_kb=%s ", $2}' /proc/meminfo
    (nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed 's/^/gpu=/') || true
    echo
    sleep 1
  done
) > "$run/monitor.txt" 2>/dev/null &
mon=$!

${ARM_DEADLINE_S:+timeout --signal=TERM --kill-after=30s ${ARM_DEADLINE_S}s} python3 "$ROOT/bench/replay-stream.py" --port "$PORT" --script "$script" --max-tokens "$MAX_TOKENS" \
  --ignore-eos --out "$run/client.jsonl" > "$run/client.stdout" 2>&1 || echo "client_rc=$?" >> "$run/meta.txt"
python3 - "$run/client.jsonl" <<'PY' | tee "$run/summary.txt"
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "error" in r:
        print("turn", r["turn"], "ERROR", r["error"]); continue
    t = r["timings"]
    f = lambda v: "%.2f" % v if isinstance(v, (int, float)) else "na"
    print("turn=%d prompt_n=%s prefill_tps=%s ttft_s=%s decode_1_128=%s decode_129_256=%s decode_all=%s predicted_n=%s" % (
        r["turn"], t.get("prompt_n"), f(t.get("prompt_per_second")), f(r.get("ttft_s")),
        f(r.get("decode_first128_tps")), f(r.get("decode_last128_tps")), f(r.get("decode_all_tps")), t.get("predicted_n")))
PY
