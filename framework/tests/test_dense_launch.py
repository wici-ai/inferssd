"""A model without routed experts launches as a model with them does, without a model download."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from test_strategy_inputs import dense
from test_typed_launch import model_file, setup

from inferssd.configuration.settings import parse
from inferssd.launch.prepare import prepare
from inferssd.launch.typed import resolve_typed


def launch(tmp_path, platform, backends, architecture="gemma4"):
    inputs, spec, host, adapter, out = setup(tmp_path, platform, automatic=True)
    model_file(Path(spec.model), host, *dense(architecture))
    host["engine_gpu_backends"] = backends
    return replace(inputs, profile=""), spec, host, adapter, out


def with_strategy(tmp_path, platform="linux"):
    """A dense model's launch with the RTX 5090 machine's characteristics."""
    inputs, spec, host, adapter, out = launch(tmp_path, platform, ["CUDA"])
    machine = Path(__file__).parent / "fixtures/machine/linux-x86-rtx5090.json"
    host["machine_characteristics"] = machine.read_text()
    host["model_drive_key"] = None
    host["runtime_facts"] = {"cpus": list(range(61)), "gpus": [], "available_bytes": 64 << 30}
    return inputs, replace(spec, strategy=True), host, adapter, out


def resolved(inputs, spec, adapter, out, platform, memory=8 << 30):
    return prepare(
        inputs,
        resolver=partial(
            resolve_typed, spec=spec, platform=platform, directory=str(out), memory=memory
        ),
        observe=adapter,
        emit=lambda _: None,
    )


@pytest.mark.parametrize(
    "platform,backends",
    [("linux", ["CUDA"]), ("darwin", ["Metal"]), ("windows", ["Vulkan"]), ("windows", [])],
)
def test_a_dense_model_takes_the_ssd_launch(tmp_path, platform, backends):
    inputs, spec, _, adapter, out = launch(tmp_path, platform, backends)
    result = resolved(inputs, spec, adapter, out, platform)
    assert result.code == 0
    argv = list(result.argv)
    # The engine maps the model and puts its layers on the GPU; nothing locks or copies it.
    assert "--mmap" in argv and "--no-mmap" not in argv and "--mlock" not in argv
    assert argv[argv.index("-ngl") + 1] == "99"
    # The engine fits the context to the memory it measures.
    assert argv[argv.index("-c") + 1] == "0"
    # A pass reads none of the model's weights: no prompt in one pass.
    assert argv[argv.index("-b") + 1] == argv[argv.index("-ub") + 1] == "512"
    settings = Path(argv[argv.index("--ssd-settings") + 1])
    assert settings.name.endswith("settings.ssd") and "--ssd-telemetry" in argv
    # No routed experts: no expert source, and no mapped experts to leave unread.
    assert parse(settings.read_text(), platform=platform).at("expert.source").value == ()
    assert "--defer-experts" not in argv
    receipt = json.loads(result.outputs[-1].text)
    assert receipt["profile"] == "gemma4" and "mode" not in receipt
    assert receipt["effective"] is not None


def test_a_dense_model_needs_no_profile(tmp_path):
    inputs, spec, _, adapter, out = launch(tmp_path, "linux", ["CUDA"], architecture="qwen35")
    result = resolved(inputs, spec, adapter, out, "linux")
    assert result.code == 0
    # Without a profile the GGUF's own chat template stays.
    assert "--chat-template-file" not in result.argv
    assert json.loads(result.outputs[-1].text)["profile"] is None


def test_a_dense_model_takes_no_expert_source(tmp_path):
    inputs, spec, _, adapter, out = launch(tmp_path, "linux", ["CUDA"])
    source = tmp_path / "experts.tsv"
    source.write_text("an expert index")
    with Path(spec.settings_file).open("a") as stream:
        stream.write("expert.source=" + json.dumps([str(source)]) + "\n")
    with pytest.raises(ValueError, match="names routed experts; the model has none"):
        resolved(inputs, spec, adapter, out, "linux")


def test_a_dense_model_keeps_its_batch_sizes_with_a_strategy(tmp_path):
    inputs, spec, _, adapter, out = with_strategy(tmp_path)
    result = resolved(inputs, spec, adapter, out, "linux")
    assert result.code == 0
    argv = list(result.argv)
    # The strategy ran (its server arguments are there) and kept the batch sizes.
    assert argv[-2:] == ["--ctx-checkpoints", "1"]
    assert argv[argv.index("-b") + 1] == argv[argv.index("-ub") + 1] == "512"
