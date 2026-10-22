#!/usr/bin/env bash
# After the official download: write colibri's engram sidecar and ask colibri's own planner for the cache cap
# this machine's available memory allows (the standalone engine takes the cap as argv[1]). Run it on an idle
# machine with the page cache dropped: the planner sizes the cache from the memory it sees free.
# usage: colibri-prepare.sh <workdir> <snap>
set -euo pipefail
w=$1; snap=$2
n=$(ls "$snap"/*.safetensors.ok | wc -l); [[ $n == 48 ]] || { echo "only $n of 48 shards verified"; exit 1; }
cd "$w"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 colibri-venv/bin/python colibri/c/tools/prepare_dsv41.py --model "$snap" --out "$snap/dsv41_engram.json"
ls -la "$snap/dsv41_engram.json"
cd colibri/c
../../colibri-venv/bin/python coli plan --model "$snap" --ctx 131072 --json > "$w/colibri-plan.json" 2> "$w/colibri-plan.err" || { cat "$w/colibri-plan.err"; exit 1; }
# the engine's per-layer cache cap is the planner's RAM tier slot count (at least 1)
python3 - "$w/colibri-plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("cap:", max(1, int(p["tiers"]["ram"]["cache_slots_per_layer"])), "warnings:", p.get("warnings"))
PY
