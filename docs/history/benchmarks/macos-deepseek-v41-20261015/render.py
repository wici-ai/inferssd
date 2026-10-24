#!/usr/bin/env python3
"""Render the saved five-length Mac comparison; this does not run inference."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHORT = json.loads((HERE / "summary-short.json").read_text())
LONG = json.loads((HERE / "summary-long.json").read_text())
LENGTHS = (32, 128, 512, 1024, 2048)
ENGINES = ("fork", "original", "pack", "colibri")
HEADERS = ("llama.cpp V4.1 fork (baseline)", "Original ssd-llama", "Pack + cache", "Colibri")


def rate(length, engine, phase):
    if length <= 512:
        return SHORT[str(length)][engine][f"{phase}_tok_s"]
    return LONG[str(length)][engine][f"{phase}_tps"]


def table(phase):
    rows = [
        "| Input tokens | " + " | ".join(HEADERS) + " |",
        "| ---: | " + " | ".join("---:" for _ in ENGINES) + " |",
    ]
    for length in LENGTHS:
        baseline = rate(length, "fork", phase)
        fastest = max(ENGINES, key=lambda engine: rate(length, engine, phase))
        cells = []
        for engine in ENGINES:
            value = rate(length, engine, phase)
            delta = (value / baseline - 1) * 100
            cell = f"{value:.3f} (0%)" if engine == "fork" else f"{value:.3f} ({delta:+.1f}%)"
            cell = cell.replace("-", "−")
            if engine == fastest:
                cell = f"🟩 **{cell}**"
            cells.append(cell)
        rows.append(f"| {length} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


for phase in ("prefill", "decode"):
    print(f"**{phase.capitalize()}**\n")
    print(table(phase))
    print()
