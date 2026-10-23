#!/usr/bin/env bash
# Golden replica test of the launchers: run-deepseek-v41-flash-mxfp4.sh, now a wrapper of run-ssd-llama.sh, must
# start the server exactly as the DeepSeek launcher of 2026-10-20 did (the fixture from commit 1f174b4) for
# the same inputs: the same environment, the same argv, the same scope.  Both run against stubs (systemd-run records
# what it would start; systemctl, nvidia-smi and ss answer as an idle machine with a 32 GB GPU would) and dummy
# assets (a GGUF header with MXFP4 experts, an alias index, empty libraries), so the test needs no GPU and starts
# nothing.  The wrapper's LAUNCH_DRY_RUN=1 output must equal what it started.  The test goes with the legacy profile
# of run-ssd-llama.sh: once a value of it moves, the old launcher is no longer the reference.  A variable the new
# launcher no longer writes because the engine's default is the old value (engine_defaults below) is not a change.
# usage: tools/adaptive/tests/test_launcher_replica.sh   (prints PASS/FAIL per case, exit 1 on any failure)
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../../.." && pwd)
T=$(mktemp -d)
[[ ${KEEP:-0} == 1 ]] && echo "keeping $T" || trap 'rm -rf "$T"' EXIT
mkdir -p "$T/stub" "$T/bin" "$T/model" "$T/cap" "$T/home"

# stubs: systemd-run records its argv and its environment (NUL-separated) and exits
cat > "$T/stub/systemd-run" << EOF
#!/bin/bash
printf '%s\0' "\$@" > "$T/cap/argv"
env -0 > "$T/cap/env"
EOF
printf '#!/bin/bash\nexit 0\n' > "$T/stub/systemctl"
cat > "$T/stub/nvidia-smi" << 'EOF'
#!/bin/bash
case " $* " in
    *memory.total*) echo "${STUB_GPU_MIB:-32607}" ;;
    *) printf '%s' "${STUB_GPU_APPS:-}" ;;
esac
EOF
printf '#!/bin/bash\nexit 0\n' > "$T/stub/ss"
chmod +x "$T/stub/"*

# dummy assets: the engine's files next to a stub server, libggml-base.so older than the preload libraries
printf '#!/bin/bash\nexit 0\n' > "$T/bin/llama-server"
chmod +x "$T/bin/llama-server"
touch -d '2026-01-01' "$T/bin/libggml-base.so"
touch "$T/bin/libkimi-no-mmap-prefetch.so" "$T/bin/libkimi-mxfp4-parity.so" "$T/bin/libkimi-vram-tier.so"
MODEL=$T/model/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf
python3 - "$MODEL" << 'PY'
# a GGUF v3 header: no metadata but the architecture, the three expert tensors of one layer as MXFP4
import struct, sys
s = lambda b: struct.pack('<Q', len(b)) + b
out = b'GGUF' + struct.pack('<IQQ', 3, 3, 1) + s(b'general.architecture') + struct.pack('<I', 8) + s(b'deepseek41')
off = 0
for role in (b'gate', b'up', b'down'):
    out += s(b'blk.0.ffn_' + role + b'_exps.weight') + struct.pack('<I3QIQ', 3, 64, 4, 2, 39, off)
    off += 64 * 4 * 2 // 32 * 17
open(sys.argv[1], 'wb').write(out)
PY
printf 'tensor\texpert_idx\tnbytes\tsource_path\toffset\nblk.0.ffn_gate_exps.weight\t0\t136\t%s\t1056\n' "$MODEL" > "$T/model/experts.tsv"
# a release whose GGUF names part of the model's sampling, as converters copy it from generation_config.json
mkdir -p "$T/model/sampled"
python3 - "$MODEL" "$T/model/sampled/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf" << 'PY'
import struct, sys
s = lambda b: struct.pack('<Q', len(b)) + b
src = open(sys.argv[1], 'rb').read()
kv = s(b'general.sampling.temp') + struct.pack('<If', 6, 0.6) + s(b'general.sampling.top_p') + struct.pack('<If', 6, 0.9)
open(sys.argv[2], 'wb').write(src[:16] + struct.pack('<Q', struct.unpack_from('<Q', src, 16)[0] + 2) + kv + src[24:])
PY
head -c 8192 /dev/zero > "$T/model/experts.pack"

# the golden launcher, at the repository root as it was (its SCRIPT_DIR names the template and the server)
GOLDEN="$HERE/fixtures/run-deepseek-v41-flash-mxfp4.golden-1f174b4.sh"
if [[ ! -f $GOLDEN ]]; then
    echo "FAIL: golden launcher fixture is missing: $GOLDEN"; exit 1
fi
cp "$GOLDEN" "$T/golden.in"
awk -v root="$ROOT" '/^SCRIPT_DIR=/ { printf "SCRIPT_DIR=\"%s\"\n", root; next } { print }' "$T/golden.in" > "$T/golden.sh"

BASE=(HOME="$T/home" USER=tester PATH="$T/stub:/usr/bin:/bin" SERVER_BIN="$T/bin/llama-server" MODEL="$MODEL")
start() {   # SCRIPT CAPTURE_DIR [VAR=VALUE ...] -- [ARGS]: run SCRIPT under the stubs, keep what systemd-run saw
    local script=$1 into=$2 vars=() ; shift 2
    while [[ $# -gt 0 && $1 != -- ]]; do vars+=("$1"); shift; done
    [[ ${1:-} == -- ]] && shift
    rm -f "$T/cap/argv" "$T/cap/env"
    (cd "$T" && env -i "${BASE[@]}" "${vars[@]}" bash "$script" "$@" > "$T/out" 2>&1)
    mkdir -p "$into"; cp "$T/out" "$into/out"
    [[ -f $T/cap/argv ]] && mv "$T/cap/argv" "$T/cap/env" "$into/"
}
compare() {   # GOLDEN_DIR NEW_DIR DRY_RUN_OUTPUT [MEMORY_MAX_EXACT]
    python3 - "$@" << 'PY'
import sys
gold, new, dry = sys.argv[1:4]
exact = len(sys.argv) > 4

def capture(d):
    argv = open(f'{d}/argv', 'rb').read().decode().split('\0')[:-1]
    env = dict(x.split('=', 1) for x in open(f'{d}/env', 'rb').read().decode().split('\0')[:-1])
    i = argv.index('env')
    scope, assigned, j = argv[:i], [], i + 1
    while '=' in argv[j] and not argv[j].startswith('-'):
        assigned.append(argv[j]); j += 1
    effective = dict(env)
    effective.update(x.split('=', 1) for x in assigned)
    effective.pop('_', None)
    return scope, assigned, effective, argv[j:]

def norm_scope(scope):
    out = []
    for a in scope:
        if a.startswith('--unit='):
            a = '--unit=' + '-'.join(a[7:].split('-')[:-3]) + '-<date>-<time>-<pid>'
        out.append(a)
    return out

gs, ga, genv, gargv = capture(gold)
ns, na, nenv, nargv = capture(new)
bad = []
mm = lambda s: int(next(a for a in s if a.startswith('MemoryMax=')).split('=')[1])
if exact:
    if norm_scope(gs) != norm_scope(ns):
        bad.append(f'scope {norm_scope(gs)} != {norm_scope(ns)}')
else:
    strip = lambda s: [a for a in norm_scope(s) if not a.startswith('MemoryMax=')]
    if strip(gs) != strip(ns):
        bad.append(f'scope {strip(gs)} != {strip(ns)}')
    if abs(mm(gs) - mm(ns)) > 256 << 20:   # MemAvailable moves between the two starts
        bad.append(f'MemoryMax {mm(gs)} vs {mm(ns)}')
# inputs the old launcher read as ${X:-default} without writing the value back: given empty, the server inherits
# them empty from it and with the value used from the new one; llama-server reads none of them
read_only_inputs = {'UBATCH', 'LLAMA_SERVER_HOST', 'LLAMA_SERVER_PORT'}
# the engine reads these with the old launcher's value as its default
engine_defaults = {'GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY': '1'}
# values that left the legacy profile for what they follow from: the new launcher's value is that one's
derived_now = {'GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB': 'GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB'}
notes, defaults, derived = [], [], []
for k in sorted(set(genv) | set(nenv)):
    if genv.get(k) == nenv.get(k):
        continue
    if k in read_only_inputs and genv.get(k) == '':
        notes.append(k)
    elif k in engine_defaults and genv.get(k) == engine_defaults[k] and k not in nenv:
        defaults.append(k)
    elif k in derived_now and k in nenv and nenv[k] == nenv.get(derived_now[k]):
        derived.append(f'{k}={nenv[k]}')
    elif k == 'MEMORY_MAX' and not exact and abs(int(genv[k]) - int(nenv[k])) <= 256 << 20:
        pass   # exported empty: each launcher writes back what it computed from MemAvailable
    else:
        bad.append(f'env {k}: golden {genv.get(k)!r} new {nenv.get(k)!r}')
if gargv != nargv:
    bad.append(f'argv golden {gargv} new {nargv}')
# the dry run prints exactly what the wrapper started
lines = open(dry).read().splitlines()
d_scope = [x[6:] for x in lines if x.startswith('SCOPE ')][1:]   # without systemd-run itself
d_env = [x[4:] for x in lines if x.startswith('ENV ')]
d_argv = [x[4:] for x in lines if x.startswith('ARG ')]
if d_env != na:
    bad.append(f'dry-run env differs from the started env: {sorted(set(d_env) ^ set(na))}')
if d_argv != nargv:
    bad.append(f'dry-run argv {d_argv} != {nargv}')
if [a for a in norm_scope(d_scope) if not a.startswith('MemoryMax=')] != \
   [a for a in norm_scope(ns) if not a.startswith('MemoryMax=')]:
    bad.append(f'dry-run scope {d_scope} != {ns}')
print('\n'.join(bad) if bad else f'{len(nenv)} environment variables ({len(na)} set by the launcher), {len(nargv)} arguments equal'
      + (f' (inherited empty from the old launcher, filled by the new: {" ".join(notes)})' if notes else '')
      + (f' (left to the engine default: {" ".join(defaults)})' if defaults else '')
      + (f' (derived now: {" ".join(derived)})' if derived else ''))
sys.exit(1 if bad else 0)
PY
}

fail=0
case_() {   # NAME [VAR=VALUE ...] [-- ARGS]
    local name=$1; shift
    local exact=()
    [[ " $* " =~ \ MEMORY_MAX=[^\ ] ]] && exact=(exact)
    start "$T/golden.sh" "$T/r/$name/golden" "$@"
    start "$ROOT/run-deepseek-v41-flash-mxfp4.sh" "$T/r/$name/new" "$@"
    start "$ROOT/run-deepseek-v41-flash-mxfp4.sh" "$T/r/$name/dry" LAUNCH_DRY_RUN=1 "$@"
    if [[ ! -f $T/r/$name/golden/argv || ! -f $T/r/$name/new/argv ]]; then
        echo "FAIL $name: a launcher did not start the server"; sed 's/^/  golden: /' "$T/r/$name/golden/out"
        sed 's/^/  new: /' "$T/r/$name/new/out"; fail=1; return
    fi
    if r=$(compare "$T/r/$name/golden" "$T/r/$name/new" "$T/r/$name/dry/out" "${exact[@]}"); then
        echo "PASS $name: $r"
    else
        echo "FAIL $name:"; echo "$r" | sed 's/^/  /'; fail=1
    fi
}

case_ alias-defaults EXPERT_ALIAS="$T/model/experts.tsv"
case_ pack-every-input EXPERT_PACK="$T/model/experts.pack" CONTEXT=65536 UBATCH=16384 THREADS=10 \
    MEMORY_MAX=15032385536 MEMORY_HEADROOM_MIB=1024 MEMORY_RESERVE_MIB=256 VRAM_RESERVE_MIB=576 \
    VRAM_ELASTIC_SAFETY_MIB=700 PROMPT_CACHE_DIR="$T/pcache" PROMPT_CACHE_MIB=4096 PRELOAD_LIB="$T/bin/libkimi-no-mmap-prefetch.so" \
    LLAMA_SERVER_HOST=0.0.0.0 LLAMA_SERVER_PORT=18999 GGML_KIMI_CPU_SINK_VRAM_BYTES=12000000000 \
    STUB_GPU_APPS="123, monitor" -- --reasoning off --temp 0 --top-k 1
case_ both-sources-no-cache EXPERT_ALIAS="$T/model/experts.tsv" EXPERT_PACK="$T/model/experts.pack" PROMPT_CACHE_DIR= \
    MEMORY_MAX=8589934592 XDG_CACHE_HOME="$T/xdg"
# the framework's knobs (adaptive.sh arm.sh): traces, a placement's thread pools (placement.py split on a 20-CPU
# device.json), a larger ubatch
case_ gguf-names-some-sampling MODEL="$T/model/sampled/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf" \
    EXPERT_ALIAS="$T/model/experts.tsv"
# empty inputs: the defaults, as the old launcher's ${X:-default}
case_ empty-means-default EXPERT_ALIAS="$T/model/experts.tsv" EXPERT_PACK= THREADS= CONTEXT= UBATCH= LLAMA_SERVER_HOST= \
    LLAMA_SERVER_PORT= PROMPT_CACHE_MIB= VRAM_RESERVE_MIB= VRAM_ELASTIC_SAFETY_MIB= MEMORY_HEADROOM_MIB= \
    MEMORY_RESERVE_MIB= PRELOAD_LIB= GGML_KIMI_CPU_SINK_VRAM_BYTES= MEMORY_MAX= CPUSET=
case_ framework-split-arm EXPERT_ALIAS="$T/model/experts.tsv" GGML_KIMI_CPU_SINK_TRACE=1 GGML_KIMI_CPU_SINK_PROFILE=1 \
    GGML_KIMI_CPU_SINK_MRC=1 THREADS=20 GGML_KIMI_CPU_SINK_THREADS=20 \
    GGML_KIMI_CPU_SINK_COMPUTE_CPUS=19,15,16,17,18,5,6,7,8,10,11,12,13,14,0,1,2,3,4 GGML_KIMI_CPU_SINK_IO_CPUS=9 \
    GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS=9 UBATCH=16384 LLAMA_SERVER_PORT=18250 \
    -- --reasoning off --temp 0 --top-k 1
exit $fail
