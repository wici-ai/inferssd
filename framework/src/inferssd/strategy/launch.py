# SPDX-License-Identifier: Apache-2.0
"""The strategy step of a typed launch's resolution.

The resolver asks four facts: the machine characteristics, the engine's facts
of the model, the key of the drive holding the model and the runtime facts. A live
launch reads them on this host; a recorded dry-run takes them from its host
facts. The strategy fills only settings and thread pool fields the user left
unset, and its receipt records what it chose, what the user's values replaced
and what it left out to agree with them.

Where the model is published with an MTP head, the strategy also derives the
launch that drafts with it, from the engine's facts of the model with the head
loaded, has the engine measure that launch's decode steps once, and launches
with the head only where drafting can be faster on this machine
(inferssd.strategy.drafting).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional, Union, cast

from inferssd.configuration.settings import Snapshot
from inferssd.launch.types import State, Threads
from inferssd.machine.characteristics import loads
from inferssd.strategy.derive import Strategy, derive, tier_backends
from inferssd.strategy.drafting import drafting, routed_ms
from inferssd.strategy.estimate import Estimate, decode_estimate
from inferssd.strategy.inputs import ModelFacts, cuda_device_order, runtime_facts_from, visible_gpus
from inferssd.strategy.merge import apply, server_arguments


@dataclass(frozen=True)
class Planned:
    """The settings and thread pools to launch with, extra server arguments and the receipt."""

    settings: Snapshot
    threads: Threads
    server_args: tuple[str, ...]
    receipt: dict[str, object]
    # A derived strategy prefills each prompt in one pass (-b and -ub as large as the context);
    # a skipped one leaves the launch's sizes.
    one_pass: bool = False
    # the MTP head the server drafts with, which the expert sources list too
    draft_head: Optional[str] = None


@dataclass(frozen=True)
class DraftHead:
    """The MTP head published with a launch's model: how the strategy finds it, reads the model's
    facts with it loaded and has the engine measure a launch that drafts with it."""

    find: Callable[[], Optional[str]]
    # the model's facts with the head H loaded, or why the engine read none
    facts: Callable[[str], Union[ModelFacts, str]]
    # the engine's decode probe of a launch that drafts with its head
    # (inferssd.engine.decode_steps), or why it measured nothing
    measure: Callable[[Planned], Union[Mapping[str, Any], str]]


# The draft tokens a step verifies at most with an MTP head; the server drafts the length that pays.
DRAFT_MAX = 3
# The server's speculative decoding options: a user who sets one decides it all.
SPEC_OPTIONS = ("--spec-", "-md", "--model-draft", "--draft")


def user_speculates(extra: Sequence[str], environment: Mapping[str, str]) -> bool:
    return any(argument.split("=", 1)[0].startswith(SPEC_OPTIONS) for argument in extra) or any(
        key.startswith("LLAMA_ARG_SPEC") for key in environment
    )


def plan(
    s: State,
    platform: str,
    memory: int,
    paths: Sequence[str],
    model_facts: Callable[[], ModelFacts],
    settings: Snapshot,
    threads: Threads,
    gpu_backends: Optional[tuple[str, ...]] = None,
    placement: Sequence[str] = (),
    *,
    fit_layers: bool = False,
    draft_head: Optional[DraftHead] = None,
) -> Planned:
    """SETTINGS and the user's THREADS completed by the strategy for this launch.

    MODEL_FACTS reads the model's facts when a strategy needs them. MEMORY is
    the launch's memory budget: it is the strategy's budget and bounds the
    memory available to it. GPU_BACKENDS are the engine's, None when unknown.
    Without characteristics, or when no strategy can be derived from them, the
    launch keeps its inputs and the receipt names why. DRAFT_HEAD finds the MTP
    head published with the model. Unless the user configures speculative
    decoding, the launch drafts with it where that can be faster on this
    machine, the head's memory counted; the server then drafts the length that
    pays as it runs, down to none.
    """

    def skipped(reason: str) -> Planned:
        s.emit("inferssd launch: no launch strategy: " + reason + "\n", "stderr")
        return Planned(settings, threads, (), {"skipped": reason})

    text = s.value("machine_characteristics")
    if text is None:
        return skipped("no machine characteristics")
    characteristics = loads(cast(str, text))
    model = model_facts()
    key = cast(Optional[str], s.value("model_drive_key", *paths))
    backends = tier_backends(gpu_backends, characteristics) if model.experts is not None else []
    runtime = runtime_facts_from(s.value("runtime_facts", *backends))
    if runtime.cuda_uuids is None:
        # Legacy recordings kept memory readings and the unmasked native profile.
        order = cuda_device_order(characteristics, s.env.get("CUDA_DEVICE_ORDER"))
        visible = visible_gpus(list(enumerate(order)), s.env.get("CUDA_VISIBLE_DEVICES"))
        runtime = replace(runtime, cuda_uuids=tuple(uuid for _, uuid in visible))
    available = runtime.available_bytes
    runtime = replace(
        runtime,
        available_bytes=None if available is None else min(available, memory),
        memory_budget_bytes=memory,
        gpu_backends=gpu_backends,
    )
    # Placement must use the reserve the native context will actually receive.
    reserve = settings.at("vram.reserve_mib")

    def derived(facts: ModelFacts) -> Strategy:
        return derive(
            characteristics,
            facts,
            runtime,
            platform,
            model_drive_key=key,
            vram_reserve_mib=cast(int, reserve.value) if reserve.origin != "default" else None,
            force_unified="GGML_CUDA_ENABLE_UNIFIED_MEMORY" in s.env,
            fit_layers=fit_layers,
        )

    try:
        strategy = derived(model)
    except ValueError as error:
        return skipped(str(error))

    def planned(
        strategy: Strategy, note: Optional[str] = None, head: Optional[str] = None
    ) -> Planned:
        if note is not None:
            strategy = replace(strategy, notes=strategy.notes + (note,))
        applied = apply(strategy, settings, threads)
        arguments = server_arguments(strategy, (*placement, *s.inputs.extra), s.env)
        pairs = strategy.server_args
        yielded = [option for option in pairs[::2] if option not in arguments[::2]]
        kept = list(applied.kept) + ["server " + option for option in yielded]
        left = set(kept) | set(applied.dropped)
        pools = {
            name: getattr(applied.threads, name)
            for name in ("decode", "batch", "cpus", "decode_cpus")
            if getattr(strategy.threads, name) and "threads." + name not in left
        }
        receipt: dict[str, object] = {
            "inputs": {
                "characteristics_sha256": strategy.inputs.characteristics_sha256,
                "model_paths": list(strategy.inputs.model_paths),
                "model_drive": key,
                "platform": strategy.inputs.platform,
            },
            "chosen": {
                "settings": {
                    name: {"value": choice.value_text, "reason": choice.reason}
                    for name, choice in strategy.settings.items()
                    if name not in left
                },
                "threads": {
                    name: list(value) if name.endswith("cpus") else value
                    for name, value in pools.items()
                },
                "server_args": list(arguments),
            },
            "kept": kept,
            "dropped": list(applied.dropped),
            "notes": list(strategy.notes),
        }
        # One pass a prompt saves the weight reads a pass makes: a model with routed experts reads
        # the experts each pass routes to; one without them reads none of its weights a pass.
        return Planned(
            applied.settings,
            applied.threads,
            arguments,
            receipt,
            one_pass=model.experts is not None,
            draft_head=head,
        )

    head = draft_head.find() if draft_head is not None else None
    if draft_head is None or head is None:
        return planned(strategy)
    if user_speculates(s.inputs.extra, s.env):
        return planned(strategy, "draft head: the user configures speculative decoding")
    facts = draft_head.facts(head)
    if isinstance(facts, str):
        return planned(strategy, f"draft head: {head} left out; the engine read no facts: {facts}")
    # The launch with the head is derived as any launch is, from the model's facts with the head:
    # its tensors take memory, its routed layer shares the expert caches.
    drafted = derived(facts)
    drafting_args: tuple[str, ...] = ("--spec-type", "draft-mtp", "--spec-draft-model", head)
    drafting_args += ("--spec-draft-n-max", str(DRAFT_MAX))
    drafted = replace(
        drafted,
        server_args=drafted.server_args + drafting_args,
        notes=drafted.notes + (f"draft head: {head}, drafts of up to {DRAFT_MAX} tokens",),
    )
    measured = draft_head.measure(planned(drafted, head=head))
    if isinstance(measured, str):
        return planned(
            strategy, f"draft head: {head} left out; its decode probe failed: {measured}"
        )
    if measured["draft"] is None:
        return planned(strategy, f"draft head: {head} left out; its decode probe timed no draft")

    def estimated(priced: ModelFacts) -> Estimate:
        # Facts recorded without what a token reads whole keep every dense byte resident.
        read = priced.dense_bytes if priced.dense_read_bytes is None else priced.dense_read_bytes
        return decode_estimate(characteristics, priced, read, runtime, platform, key)

    # Both launches' routed experts are priced alike, before inference; the dense work is measured.
    plain, loaded = estimated(model), estimated(facts)
    plain_ms, head_ms = routed_ms(plain), routed_ms(loaded)
    layers = facts.experts.layers if facts.experts is not None else 0
    rows = measured["rows"][: DRAFT_MAX + 1]
    choice = drafting(
        [row["dense_ms"] for row in rows],
        measured["draft"]["dense_ms"],
        plain_ms,
        head_ms,
        layers,
        facts.draft_layers,
    )
    if plain_ms is None or head_ms is None:
        reason = plain.detail if plain_ms is None else loaded.detail
        experts = f"routed experts unpriced ({reason}), so the dense work alone is compared"
    else:
        experts = (
            f"routed experts {plain_ms:.2f} ms a token without the head, {head_ms:.2f} ms with it "
            f"over {layers} routed layers, {facts.draft_layers} of them the head's"
        )
    numbers = (
        f"dense ms of a step of 1 to {len(rows)} rows "
        + ", ".join(f"{row['dense_ms']:.2f}" for row in rows)
        + f" and of a draft {measured['draft']['dense_ms']:.2f}, measured with the head loaded "
        "(its one-row step stands for the launch without the head, which would save only the "
        f"cache writes the head adds to it); {experts}"
    )
    if choice.length is not None:
        cost = choice.costs_ms[choice.length - 1]
        return planned(
            drafted,
            f"draft head: loaded; drafting {choice.length} a step, all accepted, costs {cost:.2f} "
            f"ms a token against {choice.plain_ms:.2f} ms without it; " + numbers,
            head,
        )
    least = min(choice.costs_ms, default=None)
    return planned(
        strategy,
        f"draft head: {head} left out; no draft length costs less than the "
        f"{choice.plain_ms:.2f} ms a token without it"
        + (f" (the least, all accepted: {least:.2f} ms)" if least is not None else "")
        + "; "
        + numbers,
    )
