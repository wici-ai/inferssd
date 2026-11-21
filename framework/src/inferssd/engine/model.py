# SPDX-License-Identifier: Apache-2.0
"""The facts of a model as the engine would load it: ``inferssd-server --ssd-model-facts``.

The engine reads the model's GGUF headers the way its load does (the NextN blocks
dropped, the routed experts laid out over their source index) and prints one JSON
object (``llama_ssd_model_facts`` in engine/include/llama-ssd.h). Nothing in the
framework reads a GGUF file or keeps a copy of ggml's types.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from inferssd.contracts import ContractError
from inferssd.machine.characteristics import (
    COUNT,
    POSITIVE,
    TEXT,
    Check,
    array,
    integer,
    mapping,
    nullable,
    record,
)

VERSION = 1
# Header reads and the routed experts' layout; seconds even for many shards on a cold drive.
TIMEOUT_S = 120.0
ROLES = ("up", "gate", "down")


class Completed(Protocol):
    @property
    def returncode(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...


Run = Callable[[Sequence[str]], Completed]


def _anything(value: object, field: str) -> None:
    del value, field


def _pair(value: object, field: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError(f"{field} must be [a layer, a format]")
    for index, item in enumerate(value):
        COUNT(item, f"{field}[{index}]")


ROLE = record({"type": TEXT, "bytes": POSITIVE})
EXPERTS = record(
    {
        "count": POSITIVE,
        "used": COUNT,
        "latent": POSITIVE,
        "hidden": POSITIVE,
        "formats": array(record({role: ROLE for role in ROLES}), minimum=1),
        "layers": array(_pair, minimum=1),
        "read_bytes": record({role: POSITIVE for role in ROLES}),
        "expert_bytes": POSITIVE,
        "expert_read_bytes": POSITIVE,
    }
)
TENSOR = record(
    {
        "name": TEXT,
        "type": TEXT,
        "shape": array(POSITIVE, minimum=1),
        "file": COUNT,
        "offset": COUNT,
        "bytes": COUNT,
    }
)
MODEL: Check = record(
    {
        "version": integer(VERSION),
        "architecture": TEXT,
        "layers": COUNT,
        "files": array(TEXT, minimum=1),
        "metadata": mapping(_anything),
        "tensors": array(TENSOR),
        "experts": nullable(EXPERTS),
    }
)


def validate(value: object) -> dict[str, Any]:
    """VALUE as the engine's model facts; errors name the offending field."""
    MODEL(value, "model")
    model = cast(dict[str, Any], value)
    if model["version"] != VERSION:
        raise ContractError(f"model.version must be {VERSION}")
    files = len(model["files"])
    for index, tensor in enumerate(model["tensors"]):
        if tensor["file"] >= files:
            raise ContractError(f"model.tensors[{index}].file names no file")
    experts = model["experts"]
    if experts is not None:
        for index, (_, format) in enumerate(experts["layers"]):
            if format >= len(experts["formats"]):
                raise ContractError(f"model.experts.layers[{index}] names no format")
    return model


def query(binary: str, path: str, run: Run, shards: Sequence[str] = ()) -> dict[str, Any]:
    """The facts of the model PATH (a GGUF file or a split model's first shard) from BINARY.

    SHARDS are the model's --shard/--ple values, as the server takes them: the facts name
    the files the engine loads the model from, theirs among them."""
    argv = [binary, "--ssd-model-facts", path]
    for shard in shards:
        argv += ["--shard", shard]
    try:
        done = run(argv)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot read the model {path}: {error}") from error
    if done.returncode != 0:
        said = [line for line in done.stderr.splitlines() if line.startswith("error: ")]
        reason = said[-1][len("error: ") :] if said else (done.stderr.strip() or "no reason given")
        raise ValueError(f"the engine cannot read the model {path}: {reason}")
    try:
        return validate(json.loads(done.stdout))
    except ValueError as error:
        raise ValueError(f"the engine's facts of the model {path}: {error}") from error
