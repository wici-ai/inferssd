#!/usr/bin/env bash
# llama.cpp baseline with PR #28696 (vcruz305 @ 2076412) on one Linux device: build it unmodified (CUDA, upstream
# default architectures; HTTPS off), then plain and tuned (first TUNED_LAYERS expert layers on the GPU) arms.
# env: ROOT, MODEL, ENGRAM_SHARDS, EXPERT_PACK (fio drive probe only), PR_SRC, THREADS, TUNED_LAYERS, PROMPTS,
#      EXTRA, COOL_BELOW_MC, JOBS
set -uo pipefail
P=$ROOT/bench/prompts
export ARM_DEADLINE_S=${ARM_DEADLINE_S:-2700} WATCHDOG_MIN_AVAIL_MB=${WATCHDOG_MIN_AVAIL_MB:-300}
L=$ROOT/bench/run-llamacpp-pr.sh
BIN=$PR_SRC/build-cuda/bin/llama-server
if [[ ! -x $BIN ]]; then
  ( cd $PR_SRC && export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH &&
    cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release > $ROOT/logs/pr-cmake.log 2>&1 &&
    cmake --build build-cuda -j${JOBS:-$(nproc)} --target llama-server > $ROOT/logs/pr-build.log 2>&1 ) ||
    { echo PR_BUILD_FAILED; exit 1; }
fi
echo "pr28696 build: $(grep -m1 -o 'CMAKE_CUDA_ARCHITECTURES:[A-Z]*=.*' $PR_SRC/build-cuda/CMakeCache.txt)"
# short coherence check through the same launcher before any timed arm
SERVER_BIN=$BIN SRC_DIR=$PR_SRC THREADS=$THREADS CONTEXT=4096 LLAMA_SERVER_PORT=18233 bash $L --reasoning off --temp 0 --top-k 1 \
  > $ROOT/logs/pr-qcheck-server.log 2>&1 &
qpid=$!
for _ in $(seq 1 900); do curl -fsS -m 2 http://127.0.0.1:18233/health 2>/dev/null | grep -q '"ok"' && break; kill -0 $qpid 2>/dev/null || break; sleep 2; done
reply=$(curl -s -m 1200 http://127.0.0.1:18233/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+3? Reply with just the number, then name the capital of France."}],"max_tokens":24,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>&1)
pkill -f "[l]lama-server.*--port 18233"; sleep 5; pkill -9 -f "[l]lama-server.*--port 18233"
echo "pr28696 qcheck reply: $reply"
echo "$reply" | grep -q 5 && echo "$reply" | grep -q -i paris || { echo PR_QCHECK_FAILED; exit 1; }
cool() {
  [[ -n "${COOL_BELOW_MC:-}" ]] || return 0
  for _ in $(seq 1 60); do t=$(sed -n 2p <(cat /sys/class/nvme/nvme0/hwmon*/temp*_input)); (( t < COOL_BELOW_MC )) && return 0; sleep 10; done
}
for s in $PROMPTS; do
  cool; SERVER_BIN=$BIN SRC_DIR=$PR_SRC LAUNCH=$L ARM_ENV="THREADS=$THREADS ${EXTRA:-}" bash $ROOT/bench/arm.sh pr28696-plain-$s $P/replay-$s.json
done
if (( ${TUNED_LAYERS:-0} > 0 )); then
  for s in $PROMPTS; do
    cool; SERVER_BIN=$BIN SRC_DIR=$PR_SRC LAUNCH=$L ARM_ENV="THREADS=$THREADS GPU_EXPERT_LAYERS=$TUNED_LAYERS ${EXTRA:-}" \
      bash $ROOT/bench/arm.sh pr28696-tuned$TUNED_LAYERS-$s $P/replay-$s.json
  done
fi
echo PR_BASELINES_DONE
