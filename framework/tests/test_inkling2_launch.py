"""Inkling 2 Mini launch resolution: profile mapping, Metal residency, and DSpark discovery."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from test_strategy_inputs import dense
from test_typed_launch import model_file, setup

from inferssd.launch.prepare import prepare
from inferssd.launch.typed import resolve_typed


def _launch_inkling(
    tmp_path: Path, platform: str, backends: list[str], architecture: str = "inkling2_loop"
):
    inputs, spec, host, adapter, out = setup(tmp_path, platform, automatic=True)
    # Name the model like the native GGUF so sibling DSpark discovery can be exercised cleanly.
    native_model = tmp_path / "Inkling-2-Mini-native.gguf"
    spec = replace(spec, model=str(native_model))
    model_file(native_model, host, *dense(architecture))
    host["engine_gpu_backends"] = backends
    return replace(inputs, profile=""), spec, host, adapter, out


def _resolve(inputs, spec, adapter, out, platform: str, memory: int = 48 << 30):
    return prepare(
        inputs,
        resolver=partial(
            resolve_typed, spec=spec, platform=platform, directory=str(out), memory=memory
        ),
        observe=adapter,
        emit=lambda _: None,
    )


@pytest.mark.parametrize("architecture", ["inkling2", "inkling2_loop"])
def test_inkling2_architectures_map_to_inkling2_profile(tmp_path: Path, architecture: str):
    inputs, spec, _, adapter, out = _launch_inkling(
        tmp_path, "darwin", ["Metal"], architecture=architecture
    )
    result = _resolve(inputs, spec, adapter, out, "darwin")
    assert result.code == 0
    argv = list(result.argv)
    # Fits in unified memory on Darwin Metal -> resident Metal (--no-mmap) and 4 checkpoints.
    assert "--no-mmap" in argv and "--mmap" not in argv
    assert argv[argv.index("--ctx-checkpoints") + 1] == "4"
    assert "--chat-template-file" not in argv
    assert argv[argv.index("-c") + 1] == "8192"
    receipt = json.loads(result.outputs[-1].text)
    assert receipt["profile"] == "inkling2"
    assert "draft_model" not in receipt["launch"]
    assert "--inkling-ssd-layers" not in argv


def test_inkling2_falls_back_to_mmap_when_memory_is_tight_or_non_darwin(tmp_path: Path):
    inputs, spec, _, adapter, out = _launch_inkling(tmp_path, "darwin", ["Metal"])
    tight = _resolve(inputs, spec, adapter, out, "darwin", memory=2 << 30)
    assert tight.code == 0
    assert "--mmap" in tight.argv and "--no-mmap" not in tight.argv

    linux_dir = tmp_path / "linux"
    linux_dir.mkdir()
    l_inputs, l_spec, _, l_adapter, l_out = _launch_inkling(linux_dir, "linux", ["CUDA"])
    linux_res = _resolve(l_inputs, l_spec, l_adapter, l_out, "linux")
    assert linux_res.code == 0
    assert "--mmap" in linux_res.argv and "--no-mmap" not in linux_res.argv


def test_inkling2_autodiscovers_sibling_dspark_and_honors_off(tmp_path: Path):
    inputs, spec, _, adapter, out = _launch_inkling(tmp_path, "darwin", ["Metal"])
    dspark = tmp_path / "Inkling-2-Mini-dspark-native.gguf"
    dspark.write_bytes(b"GGUF")

    result = _resolve(inputs, spec, adapter, out, "darwin")
    assert result.code == 0
    argv = list(result.argv)
    assert argv[argv.index("-md") + 1] == str(dspark.resolve())
    assert argv[argv.index("-ngld") + 1] == "99"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "16"
    receipt = json.loads(result.outputs[-1].text)
    assert receipt["launch"]["draft_model"] == str(dspark.resolve())

    # Explicit opt-out via draft_model="off"
    out_off = tmp_path / "out_off"
    out_off.mkdir()
    off_res = _resolve(inputs, replace(spec, draft_model="off"), adapter, out_off, "darwin")
    assert off_res.code == 0
    assert "-md" not in off_res.argv


def test_inkling_ssd_layers_preserves_draft_selection(tmp_path: Path):
    inputs, spec, _, adapter, out = _launch_inkling(tmp_path, "darwin", ["Metal"])
    drafter = tmp_path / "Inkling-2-Mini-dspark-native.gguf"
    drafter.write_bytes(b"GGUF")
    result = _resolve(inputs, replace(spec, inkling_ssd_layers=24), adapter, out, "darwin")
    assert result.code == 0
    argv = list(result.argv)
    assert argv[argv.index("--inkling-ssd-layers") + 1] == "24"
    assert argv[argv.index("-md") + 1] == str(drafter.resolve())
    assert "--no-mmap" in argv
    assert "--ctx-checkpoints" not in argv
    assert json.loads(result.outputs[-1].text)["launch"]["inkling_ssd_layers"] == 24


@pytest.mark.parametrize(
    ("platform", "backends", "layers", "message"),
    [
        ("darwin", ["Metal"], -1, "nonnegative count"),
        ("linux", ["CUDA"], 1, "macOS Metal"),
    ],
)
def test_inkling_ssd_layers_rejects_unsupported_launch(
    tmp_path, platform, backends, layers, message
):
    inputs, spec, _, adapter, out = _launch_inkling(tmp_path, platform, backends)
    with pytest.raises(ValueError, match=message):
        _resolve(inputs, replace(spec, inkling_ssd_layers=layers), adapter, out, platform)
