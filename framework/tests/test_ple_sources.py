# SPDX-License-Identifier: Apache-2.0
"""Explicit PLE sources must agree across alias indexing, metadata, and owned launch."""

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_gguf_alias import EXPERT, fixture
from test_typed_launch import resolve, setup

from inferssd.launch.recorded import alias_key
from inferssd.launch.types import Fact
from inferssd.source import gguf_alias as m

PLE = (m.PLE_TENSOR, (32, 8), 1, 0)
PLE_METADATA = [("general.architecture", 8, "deepseek41"), ("deepseek41.ple.layers", 9, (4, [0]))]


def split(number, count, tensors):
    return [("split.no", 2, number), ("split.count", 2, count), ("split.tensors.count", 5, tensors)]


def table(tmp_path, **kwargs):
    path = tmp_path / "different directory" / "chosen table.gguf"
    path.parent.mkdir(exist_ok=True)
    fixture(path, tensors=[PLE], **kwargs)
    return path


@pytest.mark.parametrize("layout", ["complete", "missing", "override"])
def test_explicit_table_preserves_expert_payloads_without_links(tmp_path, layout):
    first = tmp_path / "backbone-00001-of-00002.gguf"
    second = tmp_path / "backbone-00002-of-00002.gguf"
    fixture(first, kv=split(0, 2, 2) + PLE_METADATA)
    extras = table(tmp_path)
    paths = [first]
    if layout != "missing":
        fixture(
            second,
            tensors=[PLE] if layout == "override" else [("dense", (32,), 0, 0)],
            kv=split(1, 2, 2),
        )
        paths.append(second)
    rows, summary = m.build_rows(paths, {"up"}, 256, extra_paths=[extras])
    assert len(rows) == 256
    assert {row["source_path"] for row in rows} == {str(first)}
    assert rows[0]["offset"] == m.read_header(first).data_offset
    assert rows[-1]["offset"] == rows[0]["offset"] + 255 * 68
    assert summary["extra_paths"] == [str(extras)]
    loaded = [tensor for header in m.load_headers(paths, [extras]) for tensor in header.tensors]
    assert [tensor.name for tensor in loaded].count(m.PLE_TENSOR) == 1
    assert all(not path.is_symlink() for path in paths + [extras])


@pytest.mark.parametrize(
    "count,total,declares_ple,inline",
    [
        (2, 3, True, False),
        (3, 2, True, False),
        (2, 2, False, False),
        (2, 3, True, True),
    ],
)
def test_extra_table_cannot_hide_missing_backbone(tmp_path, count, total, declares_ple, inline):
    first = tmp_path / f"backbone-00001-of-{count:05d}.gguf"
    tensors = [(EXPERT, (32, 2, 256), 8, 0)]
    if inline:
        tensors.append((m.PLE_TENSOR, (32, 8), 1, 17408))
    fixture(
        first, tensors=tensors, kv=split(0, count, total) + (PLE_METADATA if declares_ple else [])
    )
    with pytest.raises(RuntimeError, match="complete numbered|not proven"):
        m.build_rows([first], {"up"}, 256, extra_paths=[table(tmp_path)])


@pytest.mark.parametrize("metadata", [[], split(0, 1, 1), split(1, 2, 2)])
def test_explicit_table_metadata_can_be_standalone_or_original_split(tmp_path, metadata):
    first = tmp_path / "backbone-00001-of-00002.gguf"
    fixture(first, kv=split(0, 2, 2) + PLE_METADATA)
    assert m.build_rows([first], {"up"}, 256, extra_paths=[table(tmp_path, kv=metadata)])[0]


@pytest.mark.parametrize(
    "metadata",
    [
        [("split.no", 2, 1)],
        split(1, 1, 1),
        split(0, 1, 2),
        split(0, 2, 2),
        split(1, 2, 3),
        split(1, 3, 2),
        [("split.no", 4, 0), ("split.count", 2, 1), ("split.tensors.count", 5, 1)],
    ],
)
def test_inconsistent_explicit_table_split_metadata_is_rejected(tmp_path, metadata):
    first = tmp_path / "backbone-00001-of-00002.gguf"
    fixture(first, kv=split(0, 2, 2) + PLE_METADATA)
    with pytest.raises((RuntimeError, ValueError), match="split"):
        m.build_rows([first], {"up"}, 256, extra_paths=[table(tmp_path, kv=metadata)])


@pytest.mark.parametrize(
    "tensors", [[(EXPERT, (32, 2, 256), 8, 0)], [PLE, ("dense", (1,), 0, 512)]]
)
def test_explicit_extra_must_contain_only_the_ple_tensor(tmp_path, tensors):
    model = tmp_path / "model.gguf"
    extra = tmp_path / "anything.gguf"
    fixture(model)
    fixture(extra, tensors=tensors)
    with pytest.raises(RuntimeError, match="must contain only"):
        m.build_rows([model], {"up"}, 256, extra_paths=[extra])


def test_duplicate_table_and_physical_alias_are_rejected(tmp_path):
    model = tmp_path / "model.gguf"
    fixture(model)
    extra = table(tmp_path)
    with pytest.raises(RuntimeError, match="at most one"):
        m.build_rows([model], {"up"}, 256, extra_paths=[extra, extra])
    alias = tmp_path / "hardlink.gguf"
    os.link(model, alias)
    with pytest.raises(RuntimeError, match="duplicate GGUF source"):
        m.build_rows([model], {"up"}, 256, extra_paths=[alias])


@pytest.mark.parametrize("option", ["--shard", "--ple", "environment"])
def test_alias_cli_respects_explicit_table_inputs(tmp_path, monkeypatch, option):
    model = tmp_path / "model.gguf"
    fixture(model)
    extra = table(tmp_path)
    out, summary = tmp_path / "index.tsv", tmp_path / "summary.json"
    command = ["--model-glob", str(model), "--out", str(out), "--summary-json", str(summary)]
    monkeypatch.delenv("LLAMA_ARG_SHARD", raising=False)
    if option == "environment":
        monkeypatch.setenv("LLAMA_ARG_SHARD", str(extra))
    else:
        command.extend([option, str(extra)])
    assert m.main(command) == 0
    assert json.loads(summary.read_text())["extra_paths"] == [str(extra)]
    assert len(list(csv.DictReader(out.open(), delimiter="\t"))) == 256


def test_repeat_and_comma_paths_do_not_silently_drop_inputs():
    assert m.shard_paths(["first,second", "third"], "env") == list(
        map(Path, ["env", "first", "second", "third"])
    )
    for empty in ("", "first,", ",first"):
        with pytest.raises(ValueError, match="nonempty"):
            m.shard_paths([empty])


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("environment", [False, True])
@pytest.mark.parametrize("comma", [False, True])
def test_owned_launch_indexes_and_forwards_table_once(tmp_path, platform, environment, comma):
    inputs, spec, _, adapter, out, calls = setup(tmp_path, platform, automatic=True)
    extra = table(tmp_path)
    if comma:
        extra = extra.rename(extra.with_name("chosen, table.gguf"))
    argument = m.shard_argument(str(extra))
    if environment:
        inputs = replace(inputs, environment=inputs.environment + (("LLAMA_ARG_SHARD", argument),))
        adapter.inputs = inputs
    else:
        spec = replace(spec, shards=(argument,))
    result = resolve(inputs, spec, adapter, out, platform)
    assert result.code == 0
    assert result.argv.count("--shard") == 1
    assert result.argv[result.argv.index("--shard") + 1] == argument
    assert "LLAMA_ARG_SHARD" not in dict(result.effective_environment)
    assert calls[0][-2:] == ["--shard", argument]
    assert alias_key((spec.model,), None, "", (str(extra),)) != alias_key((spec.model,), None, "")
    facts = adapter(Fact("model_facts", (spec.model, "--shard", str(extra)))).value
    assert facts["dense_bytes"] == 512
    assert facts["paths"] == [spec.model, str(extra)]


def test_model_facts_count_replacement_table_once(tmp_path):
    _, spec, _, adapter, _, _ = setup(tmp_path)
    fixture(
        Path(spec.model),
        tensors=[
            (f"blk.0.ffn_{role}_exps.weight", (32, 2, 256), 8, i * 17408)
            for i, role in enumerate(("up", "gate", "down"))
        ]
        + [(m.PLE_TENSOR, (32, 8), 0, 52224)],
    )
    extra = table(tmp_path)
    facts = adapter(Fact("model_facts", (spec.model, "--shard", str(extra)))).value
    assert facts["dense_bytes"] == 512  # F16 replacement, not original F32 plus replacement


@pytest.mark.parametrize(
    "metadata",
    [
        ("deepseek41.ple.layers", 4, 1),
        ("deepseek41.ple.layers", 9, (8, ["not a layer"])),
        ("deepseek41.embedding_length_per_layer_input", 2, 32),
    ],
)
def test_malformed_ple_declaration_cannot_justify_missing_shard(tmp_path, metadata):
    model = tmp_path / "model-00001-of-00002.gguf"
    fixture(model, kv=split(0, 2, 2) + [("general.architecture", 8, "deepseek41"), metadata])
    with pytest.raises(ValueError, match="Bad type"):
        m.build_rows([model], {"up"}, 256, extra_paths=[table(tmp_path)])


def test_owned_launch_accepts_proven_missing_ple_slot(tmp_path):
    inputs, spec, _, adapter, out, calls = setup(tmp_path, automatic=True)
    model = Path(spec.model).with_name("backbone-00001-of-00002.gguf")
    tensors = [
        (f"blk.0.ffn_{role}_exps.weight", (32, 2, 256), 8, i * 17408)
        for i, role in enumerate(("up", "gate", "down"))
    ]
    fixture(model, tensors=tensors, kv=split(0, 2, 4) + PLE_METADATA)
    extra = table(tmp_path)
    spec = replace(spec, model=str(model), shards=(str(extra),))
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0 and calls
    assert calls[0][-4:] == ["--model-glob", str(model), "--shard", str(extra)]
    assert not model.with_name("backbone-00002-of-00002.gguf").exists()


def test_explicit_cli_and_environment_sources_are_both_validated(tmp_path):
    inputs, spec, _, adapter, out, calls = setup(tmp_path, automatic=True)
    extra = table(tmp_path)
    inputs = replace(inputs, environment=inputs.environment + (("LLAMA_ARG_SHARD", str(extra)),))
    adapter.inputs = inputs
    with pytest.raises(RuntimeError, match="at most one"):
        resolve(inputs, replace(spec, shards=(str(extra),)), adapter, out)
    assert not calls
