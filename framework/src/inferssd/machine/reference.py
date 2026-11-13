# SPDX-License-Identifier: Apache-2.0
"""Recognize a calibrated GGUF variant from headers without loading its weights."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inferssd.app import catalog
from inferssd.contracts import ContractError
from inferssd.source.header import read_model
from inferssd.strategy.inputs import facts_from_headers
from inferssd.strategy.reference import match


def estimate_models(paths: Sequence[Path]) -> tuple[catalog.Model, ...]:
    """Keep catalog estimates and include compatible local variants, once per model.

    ``--model`` also accepts directories or arbitrary files for drive profiling.
    Invalid/non-GGUF files therefore do not prevent that existing use case.
    Only headers are read; no tensor payload, generation, server or download.
    """
    result = list(catalog.MODELS)
    selected: set[str] = set()
    seen: set[str] = set()
    for path in paths:
        candidates = sorted(path.glob("*.gguf")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() != ".gguf" or str(candidate.resolve()) in seen:
                continue
            try:
                headers = read_model(str(candidate))
                seen.update(str(Path(item.path).resolve()) for item in headers)
                facts = facts_from_headers(headers)
            except (OSError, ValueError, ContractError):
                continue
            dense = facts.dense_bytes - sum(
                t.nbytes for h in headers for t in h.tensors if t.name == "token_embd.weight"
            )
            reference = match(facts, dense)
            if reference is None or reference.identifier in selected:
                continue
            selected.add(reference.identifier)
            if any(
                m.facts is not None and match(m.facts, m.dense_read_bytes) == reference
                for m in result
            ):
                continue
            # This is a recognized local model, not a claimed Hugging Face repository/revision.
            result.append(
                catalog.Model(
                    reference.label, reference.quantization, "", facts, facts.dense_bytes - dense
                )
            )
    return tuple(result)
