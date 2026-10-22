#!/usr/bin/env bash
# One colibri baseline arm, as in the paper (pinned ce370e87d, CPU engine, official FP4/FP8 safetensors):
# two real-text turns through colibri-replay.py, 256 tokens each, greedy, KV-prefix reuse off, 2700 s cap
# including startup. Cold: page cache of all 48 shards dropped first.
# usage: colibri-arm.sh <label> <replay.json> <cap> <threads> [memory_max_bytes]
# env: ROOT (work dir with colibri/, bench/, runs/), SNAP (official model dir with dsv41_engram.json),
#      COLIBRI_CTX (131072 as in the paper; smaller only when the engine cannot start in this memory)
set -uo pipefail
label=$1; script=$2; cap=$3; threads=$4
ctx=${COLIBRI_CTX:-131072}
ROOT=${ROOT:?}; SNAP=${SNAP:?}
avail_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
memory=${5:-$(( (avail_kib - 512 * 1024) * 1024 ))}
run=$ROOT/runs/$(date +%Y%m%d-%H%M%S)-colibri-$label
mkdir -p "$run"
echo "run=$run"
python3 - "$SNAP" > "$run/pagecache.txt" <<'PY'
import json, os, sys
snap = sys.argv[1]
names = sorted(set(json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"].values()))
for n in names:
    fd = os.open(os.path.join(snap, n), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
print("dropped", len(names))
PY
fio --name=gate --filename="$SNAP/model-00048-of-00048.safetensors" --readonly --direct=1 --rw=randread --bs=6291456 \
    --iodepth=18 --ioengine=libaio --size=4G --norandommap=1 --randrepeat=0 --time_based=1 --runtime=5 \
    --output-format=json --output="$run/fio.json" >/dev/null 2>&1 || true
{
  echo "label=$label"; echo "script=$script"; echo "cap=$cap"; echo "threads=$threads"; echo "ctx=$ctx"; echo "memory_max=$memory"
  echo "binary_sha256=$(sha256sum "$ROOT/colibri/c/deepseek_v41" | cut -d' ' -f1)"; echo "host=$(hostname)"; echo "date=$(date -Is)"
  python3 -c "import json;print('fio_gib_s=%.3f'%(json.load(open('$run/fio.json'))['jobs'][0]['read']['bw_bytes']/2**30))" 2>/dev/null
} > "$run/meta.txt"
env_args=(SNAP="$SNAP" SERVE=1 CTX="$ctx" OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE MALLOC_ARENA_MAX=1
          V41_DIRECT=1 V41_READ_DEPTH=8 V41_ENGRAM_ROWS=65536 V41_STATS=1 COLI_KV_PREFIX=0 V41_DSPARK=1
          V41_DSPARK_MINACC=60 SEED=61004)
(
  while :; do
    printf '%s ' "$(date +%s.%N)"
    awk '$3=="nvme0n1"{printf "rd_sect=%s ", $6}' /proc/diskstats
    awk '/^MemAvailable:/{printf "memavail_kb=%s ", $2}' /proc/meminfo
    echo; sleep 1
  done
) > "$run/monitor.txt" 2>/dev/null &
mon=$!
timeout --signal=TERM --kill-after=30s 2700s systemd-run --user --scope --quiet --unit="colibri-$label-$$" \
    -p MemoryMax="$memory" -p MemorySwapMax=0 env "${env_args[@]}" \
    "$ROOT/colibri-venv/bin/python" "$ROOT/colibri-replay.py" --binary "$ROOT/colibri/c/deepseek_v41" \
    --cap "$cap" --script "$script" --out "$run" > "$run/replay.stdout" 2> "$run/replay.stderr"
rc=$?
kill $mon 2>/dev/null
echo "rc=$rc" >> "$run/meta.txt"
journalctl --user -n 50 --no-pager 2>/dev/null | grep -i -E "oom|colibri-$label" | tail -5 > "$run/journal-tail.txt"
python3 - "$run" <<'PY'
import json, os, sys
run = sys.argv[1]
p = os.path.join(run, "turns.json")
rows = json.load(open(p)) if os.path.exists(p) else []
f = lambda v: "%.2f" % v if isinstance(v, (int, float)) else "na"
for r in rows:
    print("turn=%d prompt_n=%s ttft_s=%s decode_129_256=%s engine_tps_incl_prefill=%s rss_GB=%s" % (
        r["turn"], r.get("prompt_n"), f(r.get("ttft_s")), f(r.get("decode_tps_129_256")),
        f(r.get("engine_total_including_prefill_tps")), f(r.get("engine_rss_GB"))))
if len(rows) < 2:
    print("incomplete: %d of 2 turns" % len(rows))
PY
