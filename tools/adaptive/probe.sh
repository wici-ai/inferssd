#!/usr/bin/env bash
# Device probe (adaptive.sh step 1): this machine from nothing: CPU groups (the candidate placements), memory, the
# GPU's kind and host link (memprobe), the drive at the queue depths a decode layer issues, every expert file alone.
# The drive is read the engine's way (reads.sh: ioreplay, or fio with PROBE_READER=fio).
# usage: probe.sh OUT EXPERT_SOURCE MEMPROBE   (EXPERT_SOURCE: the directory expert_source.py index wrote)
#        GROUP_TIMER: the build's test-kimi-vram-group, for the grouped-prefill lanes (none: not timed)
# env:   PROBE_QDS (the queue-depth grid, in role reads), PROBE_REPEATS (2), PROBE_MIN_EXPERTS (60 per run),
#        PROBE_FILE_EXPERTS (100 per expert file)
# exit:  nonzero, with the reason, when the drive cannot be read the way the engine reads it
set -uo pipefail
out=$1; src=$2; memprobe=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
t0=$(date +%s)
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  n=${c##*/cpu}
  echo "$n $(cat $c/cpu_capacity 2>/dev/null || echo 0) $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)" \
       "$(cat $c/topology/thread_siblings_list 2>/dev/null || echo -)"
done | sort -n > "$out/cpus.txt"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo > "$out/meminfo.txt"
"$memprobe" 4 > "$out/memprobe.txt" 2>&1 < /dev/null || true
echo "probe: host done $(( $(date +%s) - t0 )) s"

# drive: the engine's reads of random experts (every role an O_DIRECT range of the expert source); a layer that
# misses n experts issues 3n of them at once.  The engine reads experts only with O_DIRECT through io_uring, so a
# file that refuses O_DIRECT, or a kernel that refuses io_uring (the first replay), stops the probe here with the
# reason instead of the server later.
python3 - "$src/experts.json" << 'PY' || exit 1
import json, os, sys
bad = []
for p in json.load(open(sys.argv[1]))['files']:
    try:
        os.close(os.open(p, os.O_RDONLY | os.O_DIRECT))
    except OSError as e:
        bad.append(f'{p}: {e.strerror}')
if bad:
    sys.exit('probe: the engine reads experts only with O_DIRECT; these files cannot be opened so:\n  ' + '\n  '.join(bad))
PY
replay() {   # IOLOG QD OUT: one replay; a failed one stops the probe (the engine would read the same way)
  bash "$here/reads.sh" "$@" || { echo "probe: replay of $1 at QD $2 failed (above): stop" >&2; exit 1; }
}
plan() {   # N SEED [FILE]: a read plan of N random experts into reads.iolog
  python3 "$here/expert_source.py" iolog "$src" "$1" "$2" "$out/reads.iolog" ${3:+"$3"} || exit 1
}
# The grid runs past where the drive saturates, so that device_profile.py finds the knee QD* (the smallest depth as
# fast as the fastest, within the repeat spread).  Each repeat is a sweep of its own, in the direction opposite to
# the one before, so that a drive that drifts during the probe (heat) widens the repeat spread instead of bending
# the curve.  A run is at least 12 depths long (4 QD experts, 60 at least): its drain, the last depth's reads
# completing with ever fewer in flight, then costs about 1/24 of its time, the same share at every depth.
QDS=${PROBE_QDS:-1 3 6 9 12 18 24 36 48 72 96} REPEATS=${PROBE_REPEATS:-2} MIN=${PROBE_MIN_EXPERTS:-60}
reversed=$(printf '%s\n' $QDS | tac | tr '\n' ' ')
for r in $(seq 1 "$REPEATS"); do
  sweep=$QDS
  (( r % 2 )) || sweep=$reversed
  for qd in $sweep; do
    plan $(( 4 * qd > MIN ? 4 * qd : MIN )) $(( qd * 100 + r ))
    replay "$out/reads.iolog" $qd "$out/reads-qd$qd-$r.json"
  done
done
# every file that holds experts, alone: one on degraded blocks reads far slower than the others
: > "$out/files.txt"; i=0
while IFS= read -r f; do
  plan ${PROBE_FILE_EXPERTS:-100} $i "$f"
  replay "$out/reads.iolog" 6 "$out/reads-file$i.json"
  printf '%s %s\n' $i "$f" >> "$out/files.txt"; i=$(( i + 1 ))
done < <(python3 -c "import json, sys; print(*json.load(open(sys.argv[1]))['files'], sep='\n')" "$src/experts.json")
rm -f "$out/reads.iolog"
gb=$(python3 -c "
import json, sys
t = 0
for p in sys.argv[1:]:
    s = open(p).read(); t += json.loads(s[s.index('{'):])['jobs'][0]['read']['io_bytes']
print(f'{t / 1e9:.1f}')" "$out"/reads-*.json)
echo "probe: drive done $(( $(date +%s) - t0 )) s, $gb GB read (reader ${PROBE_READER:-ioreplay})"

# CPU: candidate sets from the labels; which placement is faster is measured inside the engine (bootstrap)
python3 "$here/coresets.py" "$out/cpus.txt" > "$out/coresets.txt"

# the grouped-prefill lanes (B4): one expert's time on the GPU's batched engine (from device memory, and in place where
# the GPU reads host memory so), on the CPU's grouped path on the sink's threads, and one copy from page-locked
# memory, at the model's expert shape and types, each busy beside the others as in a pass (test-kimi-vram-group
# --time, with the tier DSO preloaded). The engine starts from these lines and replaces them with its own.
if [[ -n ${GROUP_TIMER:-} && -x $GROUP_TIMER ]]; then
  bin=$(dirname "$GROUP_TIMER")
  threads=$(awk '$1 == "all" { print split($2, c, ",") }' "$out/coresets.txt")
  [[ -n $threads ]] || threads=$(wc -l < "$out/cpus.txt")
  args=$(python3 "$here/expert_source.py" lane-args "$src") &&
  env LD_LIBRARY_PATH="$bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" LD_PRELOAD="$bin/libkimi-vram-tier.so" \
      "$GROUP_TIMER" --time $args --threads "$threads" > "$out/lanes.txt" 2>&1 < /dev/null ||
    echo "probe: the lane timing failed (lanes.txt): the engine starts from its built-in values" >&2
  echo "probe: lanes done $(( $(date +%s) - t0 )) s"
fi

python3 "$here/device_profile.py" "$out" "$src" > "$out/device.json"
echo "probe: total $(( $(date +%s) - t0 )) s -> $out/device.json"
