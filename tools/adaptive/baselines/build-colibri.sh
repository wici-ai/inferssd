#!/usr/bin/env bash
# Build the pinned colibri DeepSeek-V4.1 CPU engine as the paper's baseline did (ARCH=native, CUDA=0),
# and a python venv for its engram sidecar / resource planner.
# usage: build-colibri.sh <workdir> <jobs>
set -euo pipefail
w=$1; j=${2:-8}
cd "$w"
mkdir -p colibri && tar xzf colibri-ce370e87.tar.gz -C colibri --strip-components=1
make -C colibri/c -B -j"$j" deepseek_v41 ARCH=native CUDA=0 > colibri-build.log 2>&1 || { tail -30 colibri-build.log; exit 1; }
ls -la colibri/c/deepseek_v41 && sha256sum colibri/c/deepseek_v41
python3 -m venv colibri-venv
colibri-venv/bin/pip install -q ${PIP_INDEX:+-i $PIP_INDEX} numpy sympy tokenizers transformers safetensors > colibri-pip.log 2>&1 || { tail -20 colibri-pip.log; exit 1; }
colibri-venv/bin/python -c "import numpy, sympy, tokenizers, transformers; print('venv ok', transformers.__version__)"
echo COLIBRI_BUILD_OK
