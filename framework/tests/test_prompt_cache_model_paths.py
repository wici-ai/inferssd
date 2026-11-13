# SPDX-License-Identifier: Apache-2.0
"""Disk prompt caches belong to model paths, including equally named models."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from test_typed_launch import resolve, setup


def cache_path(result):
    assert result.code == 0
    return Path(result.argv[result.argv.index("--cache-disk") + 1])


def test_equal_model_names_in_different_directories_have_separate_caches(tmp_path):
    platform = "windows" if sys.platform == "win32" else "linux"
    root = tmp_path / "prompts"
    chosen = []
    for directory in ("original", "fine-tuned"):
        inputs, spec, _, adapter, out, _ = setup(tmp_path / directory, platform)
        result = resolve(inputs, replace(spec, prompt_cache=str(root)), adapter, out, platform)
        chosen.append(cache_path(result))
    assert chosen[0] != chosen[1]
    assert all(root in path.parents for path in chosen)


def test_one_model_reuses_its_cache_across_run_directories(tmp_path):
    platform = "windows" if sys.platform == "win32" else "linux"
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    spec = replace(spec, prompt_cache=str(tmp_path / "prompts"))
    first = cache_path(resolve(inputs, spec, adapter, out, platform))
    second_run = tmp_path / "second-run"
    second_run.mkdir()
    second = cache_path(resolve(inputs, spec, adapter, second_run, platform))
    assert first == second


def test_resolved_paths_keep_equivalent_spellings_together(tmp_path):
    platform = "windows" if sys.platform == "win32" else "linux"
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    spec = replace(spec, prompt_cache=str(tmp_path / "prompts"))
    first = cache_path(resolve(inputs, spec, adapter, out, platform))
    second_run = tmp_path / "second-run"
    second_run.mkdir()
    spelling = str(Path(spec.model).parent / ".." / "repo" / "model.gguf")
    second = cache_path(
        resolve(inputs, replace(spec, model=spelling), adapter, second_run, platform)
    )
    assert first == second
