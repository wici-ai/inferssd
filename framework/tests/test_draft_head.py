"""A launch drafts with the MTP head published with its model, where the engine drafts with one."""

import posixpath
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from test_launch_strategy import native, resolved, strategic
from test_strategy_inputs import Q8_0, moe, recorded_facts

from inferssd.app import hub
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.typed import draft_head, head_model
from inferssd.launch.types import Fact, LaunchInput

FLASH = "Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf"


class Files:
    """A launch state whose draft_heads fact lists this machine's directories."""

    def __init__(self):
        self.artifacts = RecordedArtifacts(LaunchInput("/repo", ()), {})

    def value(self, name, *arguments):
        return self.artifacts(Fact(name, tuple(arguments))).value


def touch(path, size=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"G" * size)
    return str(path)


def test_a_head_names_its_model():
    assert head_model("mtp-Qwen3.8-Flash-Next-Q4_0.gguf") == "qwen3.8-flash-next"
    assert head_model("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf") == "qwen3.8-flash-next"


def test_heads_beside_above_and_in_an_mtp_directory(tmp_path):
    # bartowski: the heads at the repository's root, a quantization in a directory of its own
    model = touch(tmp_path / "bart" / "Qwen3.8-Flash-Next-IQ4_XS" / FLASH)
    small = touch(tmp_path / "bart" / "mtp-Qwen3.8-Flash-Next-Q4_0.gguf", 4)
    touch(tmp_path / "bart" / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", 8)
    assert draft_head(Files(), model, posixpath) == small
    # unsloth: the heads in an MTP directory; a head borrowing the model's matrices comes first
    model = touch(
        tmp_path / "uns" / "UD-Q4_K_XL" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    )
    touch(tmp_path / "uns" / "MTP" / "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf", 4)
    shared = touch(tmp_path / "uns" / "MTP" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 8)
    assert draft_head(Files(), model, posixpath) == shared
    # a head of another model, or none
    model = touch(tmp_path / "other" / "Muse-Spark-Q4_0.gguf")
    touch(tmp_path / "other" / "mtp-Qwen3.8-Flash-Next-Q4_0.gguf")
    assert draft_head(Files(), model, posixpath) is None
    assert draft_head(Files(), touch(tmp_path / "alone" / FLASH), posixpath) is None


def flash_launch(tmp_path, head=True):
    inputs, spec, host, adapter, out = strategic(tmp_path)
    directory = tmp_path / "models" / "Qwen3.8-Flash-Next-IQ4_XS"
    model = touch(directory / "Qwen3.8-Flash-Next-IQ4_XS.gguf")
    tensors, kv = moe(
        "qwen4exp",
        layers=2,
        experts=64,
        used=4,
        latent=256,
        hidden=128,
        gate=Q8_0,
        up=Q8_0,
        down=Q8_0,
    )
    host["model_facts"] = recorded_facts(tensors, kv, path=model)
    heads = touch(tmp_path / "models" / "mtp-Qwen3.8-Flash-Next-Q4_0.gguf") if head else None
    template = tmp_path / "repo" / "engine/models/templates/muse-spark.jinja"
    spec = replace(spec, model=model, template=str(template), settings_file=None)
    return inputs, spec, adapter, out, heads


def test_the_strategy_drafts_with_the_head_and_reads_its_experts(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path)
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0
    argv = list(result.argv)
    start = argv.index("--spec-type")
    assert argv[start : start + 6] == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-model",
        head,
        "--spec-draft-n-max",
        "3",
    ]
    assert native(out, spec).at("expert.source").value == (spec.model, head)
    assert any("draft head" in note for note in line["strategy"]["notes"])


def test_without_a_head_nothing_drafts(tmp_path):
    inputs, spec, adapter, out, _ = flash_launch(tmp_path, head=False)
    result, _, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "--spec-type" not in result.argv
    assert native(out, spec).at("expert.source").value == (spec.model,)


def test_a_users_speculative_decoding_wins(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path)
    inputs = replace(inputs, extra=("--spec-type", "ngram-simple"))
    adapter = RecordedArtifacts(inputs, adapter.host)
    result, _, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0
    assert "draft-mtp" not in result.argv and head not in result.argv
    assert list(result.argv[-2:]) == ["--spec-type", "ngram-simple"]


def test_the_hub_fetches_the_catalog_head(tmp_path, monkeypatch):
    repo = "bartowski/Qwen3.8-Flash-Next-GGUF"
    shards = [
        f"Qwen3.8-Flash-Next-IQ4_XS/Qwen3.8-Flash-Next-IQ4_XS-{n:05}-of-00003.gguf"
        for n in (1, 2, 3)
    ]
    names = shards + [hub.HEADS[repo], "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"]
    info = SimpleNamespace(
        sha="a" * 40, siblings=[SimpleNamespace(rfilename=n, size=4) for n in names]
    )
    api = Mock()
    api.model_info.return_value = info

    def fetch(**kwargs):
        for name in kwargs["allow_patterns"]:
            (kwargs["local_dir"] / name).parent.mkdir(parents=True, exist_ok=True)
            (kwargs["local_dir"] / name).write_bytes(b"GGUF")

    snapshot = Mock(side_effect=fetch)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=snapshot),
    )
    monkeypatch.setattr(hub.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    first = hub.download(repo, "main", tmp_path)
    assert first.name == shards[0].split("/")[1]
    assert snapshot.call_args.kwargs["allow_patterns"] == shards + [hub.HEADS[repo]]
    assert (tmp_path / repo.replace("/", "--") / hub.HEADS[repo]).is_file()
