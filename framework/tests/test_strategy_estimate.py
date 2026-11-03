# SPDX-License-Identifier: Apache-2.0
"""Decode speed estimates of the supported models from a machine profile."""

from __future__ import annotations

import copy
from pathlib import Path

from inferssd.app import catalog, hub
from inferssd.machine.store import load
from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import RuntimeFacts

MAC = load(Path(__file__).parent / "fixtures/machine/macos-m5.json")  # M5, 16 GiB, Apple SSD
DRIVE = MAC["storage"][0]["drive_key"]
GIB = 1 << 30


def runtime(available):
    return RuntimeFacts(tuple(row["id"] for row in MAC["cpu"]["logical"]), (), available)


def estimate(model, characteristics=MAC, available=9 * GIB, drive=DRIVE):
    return decode_estimate(
        characteristics, model.facts, model.dense_read_bytes, runtime(available), "darwin", drive
    )


def test_the_catalog_is_the_download_list_and_its_facts_add_up():
    assert hub.MODELS == {model.repo: model.quantization for model in catalog.MODELS}
    for model in catalog.MODELS:
        facts = model.facts
        assert 0 < model.dense_read_bytes < facts.dense_bytes
        routed = facts.layers * facts.experts * facts.expert_bytes
        assert sum(size for _, size in facts.format_bytes) == routed
        assert facts.expert_read_bytes >= facts.expert_bytes


def test_every_supported_model_is_priced_on_a_mac():
    for model in catalog.MODELS:
        value = estimate(model)
        assert value.tokens_per_second is not None and 0.05 < value.tokens_per_second < 100
        assert " ms a token: dense " in value.detail


def test_more_memory_and_a_faster_drive_are_faster():
    faster = copy.deepcopy(MAC)
    for row in faster["storage"][0]["random_read"]:
        row["Bps"] *= 2
    for model in catalog.MODELS:
        base = estimate(model).tokens_per_second
        assert estimate(model, available=40 * GIB).tokens_per_second > base
        assert estimate(model, characteristics=faster).tokens_per_second > base


def test_no_estimate_without_the_models_drive():
    value = estimate(catalog.MODELS[0], drive=None)
    assert value.tokens_per_second is None
    assert value.detail == "the drive holding the model is not known"
