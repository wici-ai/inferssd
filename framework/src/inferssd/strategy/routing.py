# SPDX-License-Identifier: Apache-2.0
"""Capacity and rank-based routing assumptions shared by all MoE models."""

from __future__ import annotations

import math

from inferssd.strategy.inputs import Experts


def rank_hit_curve(experts: int, top_k: int) -> list[float]:
    """Cumulative hit mass when the warmed tier holds the most popular ranks.

    A rank's share is min(1/k, a/r) where a normalizes the shares to sum to one.
    A token's k distinct choices can include any one rank at most once. Saturate
    ranks in descending popularity, redistributing the remaining mass among the tail.
    """
    if not 1 <= top_k <= experts:
        raise ValueError("each layer must select between one and all of its experts")
    tail_weight = math.fsum(1 / rank for rank in range(1, experts + 1))
    cap = 1 / top_k
    hits = [0.0]
    for rank in range(1, experts):
        weight = 1 / rank
        share = min(cap, (1 - hits[-1]) * weight / tail_weight)
        hits.append(hits[-1] + share)
        tail_weight -= weight
    hits.append(1.0)
    return hits


def hit_at_slots(hits: list[float], quota: float) -> float:
    """Fractional slots per layer correspond to a mix of adjacent integer quotas."""
    if quota <= 0:
        return 0.0
    if quota >= len(hits) - 1:
        return 1.0
    lower = int(quota)
    return hits[lower] + (quota - lower) * (hits[lower + 1] - hits[lower])


def cache_slot_bytes(experts: Experts) -> tuple[int, int]:
    """Conservative host and GPU slot sizes when only aggregate GGUF model facts are available.

    A host slot fits one direct-I/O, page-covering read of each role at any layer.
    A GPU slot fits the largest expert, with each of its three payloads separately
    page-rounded. Without the per-layer type cross-product, bound the combined
    padding by the maximum observed for each role, then round down to a whole page.
    """
    page = 4096
    host = 0
    padding = 0
    for role in experts.roles:
        sizes = [size for _, _, size in role.formats]
        host += ((max(sizes) + 2 * (page - 1)) // page) * page
        padding += max(-size % page for size in sizes)
    return host, ((experts.expert_bytes + padding) // page) * page
