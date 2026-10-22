#!/usr/bin/env bash
# Build the SSD-streaming engine for the machine this runs on.  Nothing here names a machine: every flag follows
# from what the machine reports.
#   CUDA arch   nvidia-smi compute_cap of each GPU, as the arch-specific target (12.0 -> 120a, 12.1 -> 121a; the
#               "a" targets exist from compute capability 9.0 on)
#   CPU         x86: GGML_NATIVE (the compiler's -march=native).  aarch64: GGML_NATIVE=OFF and an explicit
#               GGML_CPU_ARM_ARCH from the /proc/cpuinfo features ggml's kernels dispatch on (asimddp -> dotprod,
#               i8mm, bf16, fphp+asimdhp -> fp16), because gcc 13.3 -mcpu=native resolves to a plain armv8 on GB10
#               (X925/A725) and enables none of them.  SVE is left out: on GB10 its vectors are 128 bits, no wider
#               than NEON's, and the SVE2 build prefilled 4% slower (2026-10-20).  The compiler's macro dump must
#               show every selected feature before anything is built (a wrong flag is a SIGILL later).
#   liburing    the CUDA expert copy reads with io_uring through liburing; without it every GPU-side expert read
#               falls back to synchronous preads (1.5 GB/s on the Spark instead of the drive's rate).  System
#               liburing-dev, or LIBURING_PREFIX (a prefix with include/liburing.h and lib/liburing.a, linked
#               statically; its include directory goes into the cached compiler flags, so that a later
#               `cmake --build` still finds the header).  The build checks the library it made: one that does not
#               call liburing is reported loudly.
#   flags       -ffp-contract=off, GGML_CUDA_MOE_STREAM_BATCH=ON (the GGUF expert-range resolver needs it),
#               GGML_CUDA_KIMI_VRAM_TIER=ON
#   targets     llama-server kimi-vram-tier kimi-mxfp4-parity, the no-mmap-prefetch preload library, and with
#               BUILD_TESTS=1 the unit tests test-kimi-cpu-sink, test-kimi-vram-q2, test-kimi-prefill-cost,
#               test-kimi-host-lock, test-kimi-vram-mm and test-kimi-vram-group (whose --time mode times one expert
#               on each lane)
# usage: tools/adaptive/build.sh [BUILD_DIR]      (default: build-cuda130 in this checkout, the launchers' default)
# env:   CUDA_HOME (default /usr/local/cuda, else the nvcc on PATH), LIBURING_PREFIX, BUILD_TESTS=1, JOBS (nproc),
#        CMAKE_EXTRA (more cmake arguments, appended last), DRY_RUN=1 (print the configuration and stop)
set -Eeuo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
B=${1:-$SRC/build-cuda130}
die() { echo "build.sh: $*" >&2; exit 1; }

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [[ ! -x $CUDA_HOME/bin/nvcc ]]; then
    nvcc=$(command -v nvcc) || die "no nvcc in $CUDA_HOME/bin or on PATH (set CUDA_HOME)"
    CUDA_HOME=$(dirname -- "$(dirname -- "$(readlink -f "$nvcc")")")
fi
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH

# CUDA architectures: one per distinct GPU kind, arch-specific ("a") from 9.0 on
command -v nvidia-smi > /dev/null || die "nvidia-smi not found: the CUDA architecture comes from the GPU"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u | awk -F. '
    NF == 2 { printf "%s%s%s", sep, $1 $2, ($1 >= 9 ? "a" : ""); sep = ";" }')
[[ -n $archs ]] || die "nvidia-smi reports no GPU compute capability"

CPU_ARGS=()
case $(uname -m) in
    aarch64|arm64)
        features=" $(sed -n '/^Features/{s/^Features[[:space:]]*:\(.*\)/\1/p;q}' /proc/cpuinfo) "
        has() { local f; for f; do [[ $features == *" $f "* ]] || return 1; done; }
        arm=armv8.2-a                       # the oldest base that admits all four extensions below
        macros=()
        has asimddp      && { arm+=+dotprod; macros+=(__ARM_FEATURE_DOTPROD); }
        has i8mm         && { arm+=+i8mm;    macros+=(__ARM_FEATURE_MATMUL_INT8); }
        has fphp asimdhp && { arm+=+fp16;    macros+=(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC); }
        has bf16         && { arm+=+bf16;    macros+=(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC); }
        dump=$("${CC:-cc}" -march="$arm" -dM -E -x c /dev/null) || die "the compiler rejects -march=$arm"
        for m in "${macros[@]}"; do
            grep -q "^#define $m 1" <<< "$dump" || die "-march=$arm does not define $m with ${CC:-cc}"
        done
        CPU_ARGS=(-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$arm")
        cpu_note="GGML_CPU_ARM_ARCH=$arm (checked: ${macros[*]:-no extensions})"
        ;;
    *)
        cpu_note="GGML_NATIVE (compiler -march=native)"
        ;;
esac

URING_ARGS=(-UGGML_CUDA_LIBURING)           # system liburing: CMake's find_library(uring) searches afresh
uring_note="system (find_library)"
include=
if [[ -n ${LIBURING_PREFIX:-} ]]; then
    [[ -f $LIBURING_PREFIX/include/liburing.h && -f $LIBURING_PREFIX/lib/liburing.a ]] ||
        die "LIBURING_PREFIX=$LIBURING_PREFIX has no include/liburing.h and lib/liburing.a"
    URING_ARGS=(-DGGML_CUDA_LIBURING="$LIBURING_PREFIX/lib/liburing.a")
    include=" -I$LIBURING_PREFIX/include"
    uring_note="$LIBURING_PREFIX (static)"
fi
# the CUDA backend calls liburing: io_uring_queue_init is linked into its library (or the server) or imported
uring_linked() { LC_ALL=C grep -qs io_uring_queue_init "$1"/libggml-cuda.so "$1"/llama-server; }

TARGETS=(llama-server kimi-vram-tier kimi-mxfp4-parity)
[[ ${BUILD_TESTS:-0} == 1 ]] && TARGETS+=(test-kimi-cpu-sink test-kimi-vram-q2 test-kimi-prefill-cost test-kimi-host-lock
    test-kimi-vram-mm test-kimi-vram-group)
GEN=()
command -v ninja > /dev/null && [[ ! -f $B/CMakeCache.txt || -f $B/build.ninja ]] && GEN=(-G Ninja)

CONFIGURE=(cmake -S "$SRC" -B "$B" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_FLAGS="-ffp-contract=off$include" -DCMAKE_CXX_FLAGS="-ffp-contract=off$include"
    -DCMAKE_CUDA_FLAGS="${include# }"
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$archs"
    -DGGML_CUDA_MOE_STREAM_BATCH=ON -DGGML_CUDA_KIMI_VRAM_TIER=ON -DLLAMA_BUILD_TESTS=ON
    "${CPU_ARGS[@]}" "${URING_ARGS[@]}" ${CMAKE_EXTRA:-})
echo "build.sh: $B"
echo "  CUDA $CUDA_HOME, architectures $archs"
echo "  CPU  $cpu_note"
echo "  liburing $uring_note"
echo "  targets ${TARGETS[*]} + libkimi-no-mmap-prefetch.so"
printf '  %q' "${CONFIGURE[@]}"; echo
[[ ${DRY_RUN:-0} == 1 ]] && exit 0

"${CONFIGURE[@]}"
uring=$(sed -n 's/^GGML_CUDA_LIBURING:[A-Z]*=//p' "$B/CMakeCache.txt")
nice -n 5 cmake --build "$B" -j"${JOBS:-$(nproc)}" --target "${TARGETS[@]}"
cc -O2 -fPIC -shared -o "$B/bin/libkimi-no-mmap-prefetch.so" "$SRC/scripts/kimi-no-mmap-prefetch.c" -ldl

echo "build.sh: BUILD_OK $B/bin ($cpu_note; CUDA $archs)"
if ! uring_linked "$B/bin"; then
    cat >&2 << EOF
build.sh: WARNING: the CUDA backend does not call liburing (GGML_CUDA_LIBURING=${uring:-unset}; without the header it
build.sh: WARNING: compiles the fallback).  Its expert copy (GPU-graph prefill, prompt tails) then reads every expert
build.sh: WARNING: with synchronous preads, far below the drive's rate.  Install liburing-dev, or build liburing into
build.sh: WARNING: a prefix (./configure --prefix=P && make install) and rerun with LIBURING_PREFIX=P.
EOF
else
    echo "build.sh: liburing $uring"
fi
