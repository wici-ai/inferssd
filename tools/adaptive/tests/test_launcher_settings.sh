#!/usr/bin/env bash
# run-ssd-llama.sh's own rules, on dry runs against stubs and dummy assets (no GPU, nothing started): the model found
# by the conf's glob, the expert types choosing the preloads, the conf's sampling, the prefill
# threshold and what follows from it, SETTINGS and the caller's environment and their precedence, the expert index.
# usage: tools/adaptive/tests/test_launcher_settings.sh   (prints PASS/FAIL per check, exit 1 on any failure)
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/stub" "$T/bin" "$T/models/Some-GGUF/Q4" "$T/home"
printf '#!/bin/bash\nexit 0\n' | tee "$T/stub/systemctl" "$T/stub/ss" "$T/stub/nvidia-smi" "$T/bin/llama-server" > /dev/null
chmod +x "$T/stub/"* "$T/bin/llama-server"
touch "$T/bin/libkimi-no-mmap-prefetch.so" "$T/bin/libkimi-mxfp4-parity.so" "$T/bin/libkimi-vram-tier.so"

gguf() {   # PATH TYPE SAMPLING(0|1): a GGUF v3 header with one layer of experts of ggml type TYPE, at offsets off 4 KiB
    python3 - "$@" << 'PY'
import struct, sys
path, t, sampling = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
s = lambda b: struct.pack('<Q', len(b)) + b
kv = [(b'general.architecture', 8, s(b'test'))] + ([(b'general.sampling.temp', 6, struct.pack('<f', 0.6))] if sampling else [])
out = b'GGUF' + struct.pack('<IQQ', 3, 3, len(kv)) + b''.join(s(k) + struct.pack('<I', vt) + v for k, vt, v in kv)
for i, role in enumerate((b'gate', b'up', b'down')):
    out += s(b'blk.0.ffn_' + role + b'_exps.weight') + struct.pack('<I3QIQ', 3, 256, 4, 2, t, 96 + i * 8192)
open(path, 'wb').write(out)
PY
}
gguf "$T/models/Some-GGUF/Q4/Some-Q4_K-00001-of-00002.gguf" 12 1      # Q4_K experts, sampling in the GGUF
gguf "$T/models/Other-MXFP4.gguf" 39 0                                # MXFP4 experts, no sampling
cat > "$T/test.conf" << 'EOF'
# a test model
NAME=testmodel
MODEL_GLOB=Some-Q4_K-00001-of-*.gguf
TEMPLATE_KWARGS='{"a": 1}'   # quoted, with a comment
SAMPLING=--temp 1.0 --top-k 9
ENGINE_ENV=LLAMA_TEST_SWITCH=1 GGML_TEST_GLOB=*
EOF

fail=0
stops() {   # LABEL DESCRIPTION MESSAGE [VAR=VALUE ...]: a dry run that must stop (or report so) with MESSAGE on stderr
    local label=$1 what=$2 message=$3; shift 3
    run "$label" "$@"
    if grep -qF -- "$message" "$T/$label.err"; then echo "PASS $label: $what"; else echo "FAIL $label: $what"; fail=1; fi
}
run() {   # LABEL [VAR=VALUE ...]: a dry run of the generic launcher -> $T/LABEL.out
    local label=$1; shift
    (cd "$T" && env -i HOME="$T/home" USER=tester PATH="$T/stub:/usr/bin:/bin" SERVER_BIN="$T/bin/llama-server" \
        MODEL_CONF="$T/test.conf" MODEL_DIR="$T/models" LAUNCH_DRY_RUN=1 "$@" bash "$ROOT/run-ssd-llama.sh" \
        > "$T/$label.out" 2> "$T/$label.err")
}
check() {   # LABEL DESCRIPTION GREP_ARGS...: the output of LABEL has (or with -v lacks) a line
    local label=$1 what=$2; shift 2
    if grep -q "$@" "$T/$label.out"; then echo "PASS $label: $what"; else echo "FAIL $label: $what"; fail=1; fi
}

run base
check base "-m is the shard under MODEL_DIR" -x "ARG $T/models/Some-GGUF/Q4/Some-Q4_K-00001-of-00002.gguf"
check base "Q4_K experts: no parity library" -x "ENV LD_PRELOAD=$T/bin/libkimi-no-mmap-prefetch.so:$T/bin/libkimi-vram-tier.so"
check base "Q4_K experts: no MXFP4 GPU path" -v -e "^ENV GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN"
check base "the conf's SAMPLING is passed, whatever the GGUF names" -x "ARG --top-k"
check base "a quoted conf value with a comment" -x 'ARG {"a": 1}'
check base "ENGINE_ENV taken literally, unexpanded" -xF "ENV GGML_TEST_GLOB=*"
check base "ENGINE_ENV's source is the model" -e "name=LLAMA_TEST_SWITCH value=1 source=model"
check base "default threshold: every ubatch on the sink (the legacy spelling)" -x "ENV GGML_KIMI_CPU_PREFILL=1"
check base "graphs sized to each request" -x "ENV GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE=1"
check base "which follows from the sink" -e "name=GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE value=1 source=derived input=GGML_KIMI_CPU_SINK"
check base "the engine's granularity" -v -e "^ENV GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY"
check base "no widened alias reads without the graph route" -v -e "^ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH"
check base "the GGUF's ranges indexed into EXPERT_INDEX" -x "ENV GGML_MOE_EXPERT_GGUF_ALIAS_TSV=$T/home/.cache/ssd-llama/experts/Some-Q4_K-00001-of-00002/experts.tsv"
check base "the scope is named after the conf" -e "^SCOPE --unit=testmodel-"
run owned-scope LAUNCH_SCOPE=inferssd-arm-test.scope
check owned-scope "the benchmark controls its own scope" -x "SCOPE --unit=inferssd-arm-test.scope"
stops invalid-scope "scope names cannot inject options or patterns" "Invalid LAUNCH_SCOPE" LAUNCH_SCOPE='*.scope'
# the index is made once and kept while it is newer than the model's files
run again
if grep -q "indexing the experts" "$T/base.err" && ! grep -q "indexing the experts" "$T/again.err"; then
    echo "PASS again: an up-to-date expert index is kept"; else echo "FAIL again: the index was not made once"; fail=1; fi
touch "$T/models/Some-GGUF/Q4/Some-Q4_K-00001-of-00002.gguf"
run touched
if grep -q "indexing the experts" "$T/touched.err"; then
    echo "PASS touched: a model file newer than the index is indexed again"; else echo "FAIL touched"; fail=1; fi

printf 'GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0\nCONTEXT=8192  # a comment\nGGML_KIMI_CPU_SINK_PREFETCH=4\nGGML_NEW_KNOB=7\nARGS=--jinja --seed 1\n' > "$T/gpu.settings"
run gpu SETTINGS="$T/gpu.settings"
check gpu "threshold 0: GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0" -x "ENV GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0"
check gpu "threshold 0: no legacy spelling" -v -e "^ENV GGML_KIMI_CPU_PREFILL="
check gpu "threshold 0: the banner names the graph route" -e "on the GPU graph above)"
check gpu "threshold 0: graphs sized to each request" -x "ENV GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE=1"
check gpu "threshold 0: the engine's granularity" -v -e "^ENV GGML_KIMI_CPU_PREFILL_GRAPH_GRANULARITY"
check gpu "graph route reachable and an unaligned alias: widened batched reads" -x "ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1"
check gpu "SETTINGS replaces a profile value" -e "name=GGML_KIMI_CPU_SINK_PREFETCH value=4 source=settings"
check gpu "SETTINGS adds an engine variable" -x "ENV GGML_NEW_KNOB=7"
check gpu "SETTINGS sets a launcher setting" -x "ARG 8192"
check gpu "SETTINGS ARGS after the launcher's" -x "ARG --seed"

run env-wins SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_SINK_PREFETCH=3 CONTEXT=4096
check env-wins "the caller's engine variable wins over SETTINGS" -e "name=GGML_KIMI_CPU_SINK_PREFETCH value=3 source=override"
check env-wins "the caller's setting wins over SETTINGS" -x "ARG 4096"

run explicit-threshold SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_PREFILL_MAX_TOKENS=8192
check explicit-threshold "the caller's threshold over SETTINGS'" -e "name=GGML_KIMI_CPU_PREFILL_MAX_TOKENS value=8192 source=override"
check explicit-threshold "a threshold of UBATCH: every ubatch on the sink" -e "CPU sink)"
check explicit-threshold "and no widened alias reads" -v -e "^ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH"
run sink-off GGML_KIMI_CPU_SINK=0 GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0
check sink-off "the sink off: no graphs sized to the request (their transitions are the sink's)" -x "ENV GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE=0"
run legacy-switch GGML_KIMI_CPU_PREFILL=0
check legacy-switch "the legacy spelling 0: the graph route reachable" -x "ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1"
check legacy-switch "and passed as given" -e "name=GGML_KIMI_CPU_PREFILL value=0 source=override"
stops both-spellings "the threshold has one input" "are both given: the threshold is one input" \
    SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_PREFILL=1
stops threshold-number "a threshold is a number of tokens" "GGML_KIMI_CPU_PREFILL_MAX_TOKENS must be a number of tokens" \
    GGML_KIMI_CPU_PREFILL_MAX_TOKENS=all
stops threshold-range "a threshold is at most 2^32 - 1" "GGML_KIMI_CPU_PREFILL_MAX_TOKENS must be a number of tokens" \
    GGML_KIMI_CPU_PREFILL_MAX_TOKENS=4294967296
# base 10, as the engine reads it: 0500 is 500 and 08 is 8 (bash arithmetic would read octal)
run threshold-zeros SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0500
check threshold-zeros "a threshold with a leading zero is read in base 10" -e "CPU sink up to 500 tokens"
run threshold-08 SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_PREFILL_MAX_TOKENS=08
check threshold-08 "08 is 8 tokens, the graph route reachable" -x "ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1"
run threshold-all SETTINGS="$T/gpu.settings" GGML_KIMI_CPU_PREFILL_MAX_TOKENS=4294967295
check threshold-all "the caller's every-ubatch threshold over SETTINGS' 0" -e "name=GGML_KIMI_CPU_PREFILL_MAX_TOKENS value=4294967295 source=override"
check threshold-all "and every ubatch on the sink" -v -e "^ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH"
printf 'PREFILL_ROUTE=gpu\n' > "$T/route.settings"
stops one-route-input "the route has one input, the threshold" "PREFILL_ROUTE is neither a launcher setting" \
    SETTINGS="$T/route.settings"

run mxfp4 MODEL="$T/models/Other-MXFP4.gguf" EXPERT_ALIAS="$T/home/.cache/ssd-llama/experts/Some-Q4_K-00001-of-00002/experts.tsv"
check mxfp4 "MXFP4 experts: the parity library preloaded" -x "ENV LD_PRELOAD=$T/bin/libkimi-no-mmap-prefetch.so:$T/bin/libkimi-mxfp4-parity.so:$T/bin/libkimi-vram-tier.so"
check mxfp4 "MXFP4 experts: the MXFP4 GPU path" -x "ENV GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN=1"
check mxfp4 "no sampling in the GGUF: the conf's" -x "ARG --top-k"

# the alias is the launcher's EXPERT_ALIAS, whatever its file name: its unaligned entries widen the GPU route's reads
cp "$T/home/.cache/ssd-llama/experts/Some-Q4_K-00001-of-00002/experts.tsv" "$T/ranges.idx"
run alias-name EXPERT_ALIAS="$T/ranges.idx" GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0
check alias-name "an alias not named .tsv: widened batched reads on the GPU route" -x "ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1"

printf 'CONTXT=8192\n' > "$T/typo.settings"
stops typo "an unknown SETTINGS name stops the launch" "CONTXT is neither a launcher setting nor an engine variable" \
    SETTINGS="$T/typo.settings"
printf '# shell-style lines are not data\nexport GGML_KIMI_CPU_SINK_PREFETCH=4\n' > "$T/export.settings"
stops export "an 'export' line stops the launch, with its line number" "export.settings:2: not a KEY=VALUE line" \
    SETTINGS="$T/export.settings"
printf 'CONTEXT = 8192\n' > "$T/blanks.settings"
stops blanks "blanks around '=' stop the launch" "blanks.settings:1: not a KEY=VALUE line" SETTINGS="$T/blanks.settings"

# a settings.env as tools/adaptive/settings.py writes it: every value with its receipt
python3 - "$ROOT/tools/adaptive" > "$T/written.settings" << 'PY'
import sys
sys.path.insert(0, sys.argv[1])
import settings
S = settings.Setting
print(settings.render([S('GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS', 84, 'decided', 'a=b; c: 336 # d'),
                       S('UBATCH', 4096, 'decided', 'passes'),
                       S('GGML_KIMI_CPU_PREFILL_MAX_TOKENS', 300, 'derived', 'n*', True)], ['a header']), end='')
PY
run written SETTINGS="$T/written.settings"
check written "settings.py's receipt is a comment: an engine variable gets the value alone" \
    -x "ENV GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS=84"
check written "settings.py's receipt is a comment: a launcher setting too" -x "ARG 4096"
check written "a pending value is a comment: the launcher keeps its own threshold" -x "ENV GGML_KIMI_CPU_PREFILL=1"
check written "and passes no other" -v -e "^ENV GGML_KIMI_CPU_PREFILL_MAX_TOKENS"

# empty values: no value, the next source decides; except where empty means none
printf 'CONTEXT=\nGGML_KIMI_CPU_SINK_PREFETCH=\n' > "$T/empty.settings"
run empty SETTINGS="$T/empty.settings" THREADS= UBATCH= LLAMA_SERVER_HOST= GGML_KIMI_CPU_SINK_VRAM_BYTES= PROMPT_CACHE_DIR=
check empty "an empty CONTEXT in SETTINGS: the default" -x "ARG 131072"
check empty "an empty UBATCH: the default" -x "ARG 8192"
check empty "an empty host: the default" -x "ARG 127.0.0.1"
check empty "an empty THREADS: the default count" -E -x "ENV GGML_KIMI_CPU_SINK_THREADS=[1-9][0-9]*"
check empty "an empty engine variable in SETTINGS: the profile's" -x "ENV GGML_KIMI_CPU_SINK_PREFETCH=2"
check empty "an empty engine variable from the caller: the profile's" -x "ENV GGML_KIMI_CPU_SINK_VRAM_BYTES=21000000000"
check empty "PROMPT_CACHE_DIR empty: no disk prompt cache" -x "ARG --cache-ram"

# the engine variables the server inherits are logged and listed too
run inherited GGML_KIMI_CPU_SINK_COMPUTE_CPUS=1,2 LLAMA_ARG_SEED=5
check inherited "an inherited engine variable is an ENV line" -x "ENV GGML_KIMI_CPU_SINK_COMPUTE_CPUS=1,2"
check inherited "and has a receipt" -e "name=LLAMA_ARG_SEED value=5 source=override input=environment"
check inherited "launcher settings are not engine variables" -v -e "^ENV LLAMA_SERVER_"

# what the launcher derives is not replaced
stops pack-contradicts "an engine expert source against the launcher's stops the launch" \
    "GGML_MOE_EXPERT_PACK=/x contradicts GGML_MOE_EXPERT_PACK=, which the launcher derives from EXPERT_PACK" \
    GGML_MOE_EXPERT_PACK=/x
stops threads-contradict "sink threads against THREADS stop the launch" "set THREADS instead" \
    THREADS=4 GGML_KIMI_CPU_SINK_THREADS=3
run threads-agree THREADS=4 GGML_KIMI_CPU_SINK_THREADS=4 GGML_MOE_EXPERT_PACK=
check threads-agree "sink threads equal to THREADS (placement.py) are accepted" -e "name=GGML_KIMI_CPU_SINK_THREADS value=4 source=derived"

# a discrete GPU (the stub reports nothing): the legacy profile's tier base and margin
check base "a discrete GPU: the tier's base budget" -e "name=GGML_KIMI_CPU_SINK_VRAM_BYTES value=21000000000 source=profile"
check base "and the margin of MEMORY_HEADROOM_MIB" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=512 source=profile input=MEMORY_HEADROOM_MIB"
check base "the tier's start-up safety is the elastic margin" -e "name=GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB value=64 source=derived"
run reserve VRAM_RESERVE_MIB=576
check reserve "which follows the reserve" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=640"
run not-elastic GGML_KIMI_CPU_SINK_VRAM_ELASTIC=0
check not-elastic "a tier that does not grow keeps 1 GiB" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024"
for x in GLOBAL_POOL KEEP_ACROSS_REQUESTS; do      # the engine's elastic tier needs both too
    run not-grown-$x GGML_KIMI_CPU_SINK_VRAM_$x=0
    check not-grown-$x "without $x the tier does not grow: 1 GiB" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024"
done
run margin-given GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB=96
check margin-given "the engine's margin, given on its own" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=96"
run margin-blocks GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS=2
check margin-blocks "a margin in blocks (slots the launcher does not know): 1 GiB" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024"
# a GPU without memory of its own: the host-memory ledger arbitrates it, nothing must be given; the tier starts from
# one block and the margin at the 4 GiB a stress run passed
printf '#!/bin/bash\ncase " $* " in *memory.total*) echo "[N/A]" ;; esac\n' > "$T/stub/nvidia-smi"
run uma
if [[ -s $T/uma.err ]]; then echo "FAIL uma: $(cat "$T/uma.err")"; fail=1
else echo "PASS uma: unified memory starts with nothing given"; fi
if [[ $(grep -A1 -x "ARG --ctx-checkpoints" "$T/uma.out" | tail -1) == "ARG 8" ]]; then
    echo "PASS uma: with the legacy profile's checkpoints"; else echo "FAIL uma: checkpoints"; fail=1; fi
check uma "the tier from one block" -e "name=GGML_KIMI_CPU_SINK_VRAM_BYTES value=auto source=derived input=unified memory (memory.total \[N/A\])"
check uma "the margin at 4 GiB" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=4096 source=derived"
check uma "the tier's start-up safety as before" -x "ENV GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB=1024"
if (( $(grep -c "^ENV GGML_KIMI_CPU_SINK_VRAM_\(BYTES\|HOST_HEADROOM_MIB\)=" "$T/uma.out") == 2 )); then
    echo "PASS uma: each once"; else echo "FAIL uma: each once"; fail=1; fi
run uma-headroom MEMORY_HEADROOM_MIB=8192
check uma-headroom "a larger MEMORY_HEADROOM_MIB raises the margin" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=8192 source=derived"
run uma-given CTX_CHECKPOINTS=0 GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB=16384 GGML_KIMI_CPU_SINK_VRAM_BYTES=21000000000
if [[ -s $T/uma-given.err ]]; then echo "FAIL uma-given: $(cat "$T/uma-given.err")"; fail=1
else echo "PASS uma-given: unified memory with both given starts"; fi
check uma-given "with the given headroom" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=16384 source=override"
check uma-given "and the given tier base" -e "name=GGML_KIMI_CPU_SINK_VRAM_BYTES value=21000000000 source=override"
# what settings.py derives on unified memory: its values, and the launcher's margin unless a run learned a larger one
python3 - "$ROOT/tools/adaptive" > "$T/uma.settings" << 'PY'
import sys
sys.path.insert(0, sys.argv[1])
import settings
S = settings.Setting
print(settings.render([S('CONTEXT', 4608, 'derived', 'w'), S('GGML_KIMI_CPU_PREFILL_MAX_TOKENS', 314, 'derived', 'n*'),
                       S('UBATCH', 4608, 'derived', 'u'), S('CTX_CHECKPOINTS', 1, 'derived', 'c')]), end='')
PY
run uma-settings SETTINGS="$T/uma.settings"
if grep -q "would stop here" "$T/uma-settings.err"; then echo "FAIL uma-settings: $(cat "$T/uma-settings.err")"; fail=1
else echo "PASS uma-settings: settings.py's values on unified memory start"; fi
check uma-settings "with its context, ubatch and checkpoints" -x "ARG 4608"
check uma-settings "and the launcher's margin" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=4096 source=derived"
check uma-settings "and the threshold, the graph route reachable" -x "ENV GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1"
echo 'GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB=6145  # source=self-tuned; input=a learned margin' >> "$T/uma.settings"
run uma-learned SETTINGS="$T/uma.settings"
check uma-learned "a learned margin replaces the launcher's" -e "name=GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB value=6145 source=settings"
printf '#!/bin/bash\nexit 0\n' > "$T/stub/nvidia-smi"

# dense weights homed off the GPU: a model with dense matrices next to its experts (Q8_0 rows of 544 B: attn_q 1024
# rows, split into 4 blocks of at most the 147456 B role slice, the engine's stage 2 x 4 slots of 143360 B; attn_odd
# 997 rows, which no equal block of whole rows can split)
python3 - "$T/models/Dense-Q8_0.gguf" << 'PY'
import struct, sys
s = lambda b: struct.pack('<Q', len(b)) + b
ts = [(b'blk.0.ffn_gate_exps.weight', (512, 256, 8), 12), (b'blk.0.ffn_up_exps.weight', (512, 256, 8), 12),
      (b'blk.0.ffn_down_exps.weight', (256, 1024, 8), 12), (b'blk.0.attn_q.weight', (512, 1024), 8),
      (b'blk.0.attn_odd.weight', (512, 997), 8), (b'output.weight', (512, 100), 8)]
out = b'GGUF' + struct.pack('<IQQ', 3, len(ts), 1) + s(b'general.architecture') + struct.pack('<I', 8) + s(b'test')
off = 0
for name, shape, t in ts:
    out += s(name) + struct.pack(f'<I{len(shape)}QIQ', len(shape), *shape, t, off)
    off += 1 << 21
open(sys.argv[1], 'wb').write(out)
PY
printf 'blk.0.attn_q.weight\tdrive  # a comment\noutput.weight\tram\n' > "$T/homes.tsv"
stops dense-no-uring "drive rows without liburing stop the launch" "but the CUDA backend does not call liburing" \
    MODEL="$T/models/Dense-Q8_0.gguf" DENSE_HOMES="$T/homes.tsv"
printf 'io_uring_queue_init' > "$T/bin/libggml-cuda.so"
run dense MODEL="$T/models/Dense-Q8_0.gguf" DENSE_HOMES="$T/homes.tsv"
check dense "the drive's alias rows after the experts'" -x "ENV GGML_MOE_EXPERT_GGUF_ALIAS_TSV=$T/home/.cache/ssd-llama/experts/Dense-Q8_0/experts.tsv:$T/homes.tsv.alias.tsv"
check dense "the drive's weights stay in the mapped GGUF" -xF 'ARG ^(blk\.0\.attn_q\.weight)$=CPU'
check dense "the RAM's in pinned host memory" -xF 'ARG ^(output\.weight)$=CUDA_Host'
check dense "a receipt of what DENSE_HOMES holds" -e "name=DENSE_HOMES value=$T/homes.tsv source=environment input=drive=1 drive_bytes=557056 ram=1 ram_bytes=54400 stage_bytes=1146880"
if [[ $(grep -c '^blk\.0\.attn_q\.weight' "$T/homes.tsv.alias.tsv") == 4 ]]; then
    echo "PASS dense: the matrix in its 4 blocks"; else echo "FAIL dense: the matrix in its 4 blocks"; fail=1; fi
printf 'output.weight\tram\n' > "$T/ram.tsv"
run dense-ram MODEL="$T/models/Dense-Q8_0.gguf" DENSE_HOMES="$T/ram.tsv"
check dense-ram "RAM homes alone: the experts' alias alone" -x "ENV GGML_MOE_EXPERT_GGUF_ALIAS_TSV=$T/home/.cache/ssd-llama/experts/Dense-Q8_0/experts.tsv"
printf 'blk.0.attn_odd.weight\tdrive\n' > "$T/odd.tsv"
stops dense-no-split "a drive row without a block split stops the launch" "has no split into equal blocks" \
    MODEL="$T/models/Dense-Q8_0.gguf" DENSE_HOMES="$T/odd.tsv"
rm "$T/bin/libggml-cuda.so"
check base "no DENSE_HOMES: no buffer override" -v -e "^ARG -ot"

cp "$T/models/Some-GGUF/Q4/Some-Q4_K-00001-of-00002.gguf" "$T/models/Some-Q4_K-00001-of-00009.gguf"
run ambiguous
if grep -q "matches 2 files" "$T/ambiguous.err"; then
    echo "PASS ambiguous: two shards matching the glob stop the launch"; else echo "FAIL ambiguous"; fail=1; fi
exit $fail
