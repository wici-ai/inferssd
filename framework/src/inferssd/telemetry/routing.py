# SPDX-License-Identifier: Apache-2.0
"""Effective cache hits from the last single-stream request in a native layer trace."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from inferssd.telemetry.jsonl import read_events
from inferssd.telemetry.observations import NativeObservations


def last_request_hits(
    lines: Iterable[str], routed_layers: set[int], experts: int, top_k: int
) -> tuple[float, float]:
    """Count VRAM hits and misses of both tiers, using complete decode tokens only."""
    parser = NativeObservations()
    previous: Optional[tuple[int, int, int]] = None
    seen: set[int] = set()
    selected = gpu = missed = 0
    token_gpu = token_missed = 0
    for event in read_events(lines):
        parser.feed(event)
        if event.name == "ssd.prefill.graph" or (
            event.name == "ssd.place.layer" and event.payload.get("decode") is False
        ):
            # Epoch and token positions can repeat across requests, or the new prompt can be
            # longer. A new prefill invalidates every hit counted for the preceding decode.
            selected = gpu = missed = 0
            previous = None
        # Consume layer observations incrementally, including for long native trace files.
        parser.graphs.clear()
        if not parser.trace:
            continue
        (request, token), layer_data = parser.trace.popitem()
        parser.prof.clear()
        current = (*request, token)
        if previous != current:
            if previous is None or previous[:2] != request:
                selected = gpu = missed = 0
            seen.clear()
            token_gpu = token_missed = 0
            previous = current
        layer, (ids, vram, cache) = layer_data.popitem()
        if (
            layer not in routed_layers
            or len(ids) != top_k
            or len(set(ids)) != top_k
            or max(ids) >= experts
        ):
            raise ValueError("decode trace does not match the target's expert topology")
        seen.add(layer)
        token_gpu += sum(vram)
        token_missed += sum(not (v or h) for v, h in zip(vram, cache))
        if len(seen) == len(routed_layers):
            selected += len(routed_layers) * top_k
            gpu += token_gpu
            missed += token_missed
    if not selected:
        raise ValueError("no complete decode tokens in the last request; enable --layer-events")
    return gpu / selected, missed / selected
