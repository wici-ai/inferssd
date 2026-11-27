# SPDX-License-Identifier: Apache-2.0
"""Whether a launch drafts with the MTP head published with its model.

A step that verifies n drafted tokens runs the model on n + 1 rows. The rows share the step's
dense work, but each row routes to experts of its own in every layer, and each draft runs the
head's layers. So drafts, even all of them accepted, save only the shared dense work of a step:
every accepted token still pays its own routed experts. At full acceptance a token then costs

    cost(n) = (D(n + 1) + n Dd + (n + 1) Lm e + n d e) / (n + 1)

milliseconds with n tokens drafted a step. D(r) is the measured dense time of a target step of r
rows and Dd that of a draft step; Lm is the model's routed layers and d the head's; e is one
token's routed experts in one layer with the head loaded, Eh / (Lm + d), where Eh prices them with
the head's routed layer sharing the expert caches and its tensors taking memory from them.
Without the head a token costs D(1) + Ep. The head is worth loading where some n costs less: full
acceptance is the best case, and the engine drafts the length that pays as it runs, down to none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from inferssd.strategy.estimate import Estimate


@dataclass(frozen=True)
class Drafting:
    """A token's cost without the head and, all drafts accepted, drafting 1, 2, ... a step."""

    plain_ms: float
    costs_ms: tuple[float, ...]

    @property
    def length(self) -> Optional[int]:
        """The draft length that costs least, None when none costs less than the plain launch."""
        if not self.costs_ms:
            return None
        best = min(range(len(self.costs_ms)), key=lambda index: self.costs_ms[index])
        return best + 1 if self.costs_ms[best] < self.plain_ms else None


def routed_ms(estimate: Estimate) -> Optional[float]:
    """A token's routed experts in a decode estimate: their CPU and GPU work, their copies to the
    GPU and their reads from the drive; None when the estimate has no rate."""
    if estimate.tokens_per_second is None:
        return None
    parts = estimate.components
    return parts["cpu_experts_ms"] + parts["gpu_experts_ms"] + parts["h2d_ms"] + parts["storage_ms"]


def drafting(
    dense_ms: Sequence[float],
    draft_ms: float,
    plain_experts_ms: Optional[float],
    head_experts_ms: Optional[float],
    routed_layers: int,
    draft_layers: int,
) -> Drafting:
    """The costs of the module docstring.

    DENSE_MS are D(1), D(2), ...; DRAFT_MS is Dd; PLAIN_EXPERTS_MS and HEAD_EXPERTS_MS are Ep and
    Eh, None where they cannot be estimated: then neither side counts routed experts, and the
    comparison is of the dense work alone. ROUTED_LAYERS are the routed layers with the head,
    DRAFT_LAYERS of them the head's.
    """
    if plain_experts_ms is None or head_experts_ms is None:
        plain_experts_ms = head_experts_ms = 0.0
    layer_ms = head_experts_ms / routed_layers if routed_layers else 0.0
    model_layers = routed_layers - draft_layers
    costs = tuple(
        (
            dense_ms[n]
            + n * draft_ms
            + (n + 1) * model_layers * layer_ms
            + n * draft_layers * layer_ms
        )
        / (n + 1)
        for n in range(1, len(dense_ms))
    )
    return Drafting(dense_ms[0] + plain_experts_ms, costs)
