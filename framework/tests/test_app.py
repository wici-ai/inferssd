# SPDX-License-Identifier: Apache-2.0
"""Download, CLI-to-owned-runtime and terminal lifecycle contracts without weights."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inferssd.app import chat, command, hub
from inferssd.cli import main
from inferssd.engine import command as engine_command


def test_tested_quantizations_and_shards():
    assert hub.selection(next(iter(hub.MODELS)))[1] == "Q8_0"
    assert hub.selection(list(hub.MODELS)[1])[1] == "UD-Q4_K_XL"
    files = [f"UD-Q4_K_XL/Muse-UD-Q4_K_XL-{n:05}-of-00002.gguf" for n in (1, 2)]
    assert hub.select_files(files + ["Q8_0/Muse-Q8_0.gguf"], "UD-Q4_K_XL") == files
    with pytest.raises(ValueError, match="incomplete"):
        hub.select_files(files[:1], "UD-Q4_K_XL")
    with pytest.raises(ValueError, match="ambiguous"):
        hub.select_files(files + ["Other-UD-Q4_K_XL.gguf"], "UD-Q4_K_XL")
    with pytest.raises(ValueError, match="tested quantization"):
        hub.selection(next(iter(hub.MODELS)) + ":Q4_K_M")
    with pytest.raises(ValueError, match="README models"):
        hub.selection("other/model")
    with pytest.raises(ValueError, match="unsafe"):
        hub.select_files(["../model-Q8_0.gguf"], "Q8_0")


@pytest.fixture
def fake_hub(monkeypatch):
    names = [f"Model-Q8_0-{n:05}-of-00002.gguf" for n in (1, 2)]
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[SimpleNamespace(rfilename=name, size=4) for name in names],
    )
    api = Mock()
    api.model_info.return_value = info

    def fetch(**kwargs):
        for name in kwargs["allow_patterns"]:
            (kwargs["local_dir"] / name).write_bytes(b"GGUF")

    snapshot = Mock(side_effect=fetch)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: api,
            snapshot_download=snapshot,
        ),
    )
    monkeypatch.setattr(hub.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    return api, snapshot


def test_download_pins_revision_and_reuses_local_files(tmp_path, fake_hub, monkeypatch):
    api, snapshot = fake_hub
    repo = next(iter(hub.MODELS))
    first = hub.download(repo, "main", tmp_path)
    assert first.name.endswith("00001-of-00002.gguf")
    assert first.read_bytes() == b"GGUF"
    assert snapshot.call_args.kwargs["revision"] == "a" * 40
    assert len(snapshot.call_args.kwargs["allow_patterns"]) == 2
    monkeypatch.setattr(hub.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert hub.download(repo, "main", tmp_path) == first  # no extra disk needed
    first.unlink()
    with pytest.raises(ValueError, match="not enough SSD"):
        hub.download(repo, "main", tmp_path)
    assert snapshot.call_count == 2
    api.model_info.assert_called_with(repo, revision="main", files_metadata=True)


@pytest.mark.parametrize("commit", [None, 123, "main", "../invalid"])
def test_download_requires_a_commit_string(tmp_path, fake_hub, commit):
    api, snapshot = fake_hub
    api.model_info.return_value.sha = commit
    with pytest.raises(ValueError, match="pinned commit"):
        hub.download(next(iter(hub.MODELS)), "main", tmp_path)
    snapshot.assert_not_called()


def test_download_never_launches_with_partial_shards(tmp_path, fake_hub):
    _, snapshot = fake_hub
    snapshot.side_effect = lambda **_: None
    with pytest.raises(ValueError, match="download incomplete"):
        hub.download(next(iter(hub.MODELS)), "main", tmp_path)


@pytest.fixture
def ready_app(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERSSD_HOME", str(tmp_path))
    resources = tmp_path / "engine.json"
    resources.write_text("{}")
    monkeypatch.setenv("INFERSSD_RESOURCES", str(resources))
    binary = tmp_path / "server"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setattr(
        command,
        "read_resources",
        lambda _: SimpleNamespace(
            binary=str(binary),
            templates=(),
            platform="darwin",
        ),
    )
    monkeypatch.setattr(command, "physical_memory", lambda: 48 * command.GIB)
    monkeypatch.setattr(sys, "platform", "darwin")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return model


def test_serve_reuses_owned_launch_and_policy(ready_app, monkeypatch):
    launch = Mock(return_value=0)
    monkeypatch.setattr(command.typed_command, "run", launch)
    assert main(["serve", "-m", str(ready_app), "--port", "8181", "--memory-gib", "24"]) == 0
    args = launch.call_args.args[0]
    assert args.execute and args.model == ready_app
    assert args.port == 8181
    policy = json.loads(args.runtime_policy.read_text())
    assert policy["memory_budget_bytes"] == 24 * command.GIB
    assert policy["host_reserve_bytes"] >= 2 * command.GIB
    assert not args.output_dir.exists()  # owned runtime reserves it


def test_hf_and_chat_share_owned_launch(ready_app, monkeypatch):
    download = Mock(return_value=ready_app)
    monkeypatch.setattr(command, "download", download)
    monkeypatch.setattr(chat, "chat", Mock(return_value=0))

    def launch(args, inputs, *, session):
        assert args.model == ready_app
        assert inputs.mode == "server"
        return session()

    monkeypatch.setattr(engine_command, "run_native", launch)
    repo = next(iter(hub.MODELS))
    assert main(["cli", "-hf", repo, "-p", "Hello", "-n", "20"]) == 0
    download.assert_called_once()
    chat.chat.assert_called_once_with(8080, "Hello", 20)


def test_missing_engine_fails_before_download(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERSSD_RESOURCES", str(tmp_path / "missing.json"))
    download = Mock()
    monkeypatch.setattr(command, "download", download)
    with pytest.raises(SystemExit):
        main(["serve", "-hf", next(iter(hub.MODELS))])
    download.assert_not_called()


def test_invalid_budget_fails_before_download(ready_app, monkeypatch):
    download = Mock()
    monkeypatch.setattr(command, "download", download)
    with pytest.raises(SystemExit):
        main(["serve", "-hf", next(iter(hub.MODELS)), "--memory-gib", "48"])
    download.assert_not_called()


@pytest.mark.parametrize("outcome", [0, RuntimeError("request failed"), KeyboardInterrupt()])
def test_session_always_stops_owned_server(tmp_path, monkeypatch, outcome):
    policy = command.default_policy(48 * command.GIB, 24 * command.GIB, tmp_path, 86400)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(policy)))
    args = SimpleNamespace(
        result=None,
        host_facts=None,
        runtime_policy=path,
        output_dir=tmp_path / "run",
        port=8080,
        platform=policy.platform,
        machine=None,
        reprofile=False,
    )
    engine = Mock(ok=True)
    runtime = Mock()
    runtime.engine.return_value = engine
    monkeypatch.setattr(engine_command, "Runtime", Mock(return_value=runtime))
    monkeypatch.setattr(engine_command, "specification", lambda *a, **kw: None)
    monkeypatch.setattr(engine_command.signal, "signal", Mock())
    session = (
        Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=0)
    )
    inputs = SimpleNamespace(mode="server", extra=())
    if isinstance(outcome, RuntimeError):
        with pytest.raises(RuntimeError, match="request failed"):
            engine_command.run_native(args, inputs, session=session)
    else:
        assert engine_command.run_native(args, inputs, session=session) == (
            130 if isinstance(outcome, KeyboardInterrupt) else 0
        )
    engine.stop.assert_called_once()


def test_terminal_stream_and_clear(monkeypatch, capsys):
    stream = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n'
    opener = Mock()
    opener.open.side_effect = lambda *a, **kw: io.BytesIO(stream)
    monkeypatch.setattr(chat.urllib.request, "build_opener", lambda _: opener)
    prompts = iter(["one", "/clear", "two", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    assert chat.chat(8080, None, 16) == 0
    requests = [json.loads(call.args[0].data) for call in opener.open.call_args_list]
    assert requests[1]["messages"] == [{"role": "user", "content": "two"}]
    assert "hello" in capsys.readouterr().out


def test_truncated_stream_is_an_error(monkeypatch):
    opener = Mock()
    opener.open.return_value = io.BytesIO(b'data: {"choices":[]}\n')
    monkeypatch.setattr(chat.urllib.request, "build_opener", lambda _: opener)
    with pytest.raises(RuntimeError, match="before completing"):
        chat.completion(8080, [], 16)
