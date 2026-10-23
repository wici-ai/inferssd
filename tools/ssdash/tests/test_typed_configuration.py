"""Cross-language settings/JSONL contracts, including actual C++ writer output."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ssdash.configuration import parse, render, schema
from ssdash.telemetry.jsonl import effective_settings, read_events


@pytest.fixture(scope="session")
def native_contract(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path_factory.mktemp("settings-native") / "contract"
    compiler = shutil.which("c++")
    assert compiler, "a C++17 compiler is required for the settings contract"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "ggml/src/ggml-ssd"),
            str(root / "ggml/src/ggml-ssd/settings/settings.cpp"),
            str(root / "ggml/src/ggml-ssd/telemetry/telemetry.cpp"),
            str(root / "ggml/src/ggml-ssd/telemetry/debug.cpp"),
            str(root / "tests/ssd/test-settings-telemetry.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    return binary


def test_registry_generated_schema_exact(native_contract):
    actual = json.loads(subprocess.check_output([str(native_contract), "--schema"], text=True))
    assert actual == schema()
    assert len(actual["properties"]) == 14


@pytest.mark.parametrize(
    "platform,text",
    [
        ("linux", "ssdash-settings 1\n"),
        (
            "linux",
            "ssdash-settings 1\ncpu.threads=4 # source=measured\ncpu.compute_cpus=2-4\n"
            'expert.source=["/path # 1", "unicode-🐈"]\n'
            "place.costs=cpu_fixed_us=204,link_gbps=12.3\n",
        ),
        ("darwin", "ssdash-settings 1\nhost.expert_cache_mib=12288\ncpu.threads=8\n"),
    ],
)
def test_cpp_python_snapshot_exact(native_contract, tmp_path, platform, text):
    source = tmp_path / "input.ssd"
    source.write_text(text)
    actual = json.loads(
        subprocess.check_output([str(native_contract), "--parse", str(source), platform], text=True)
    )
    expected = parse(text, str(source), platform=platform)
    assert actual == expected.document()
    rendered = tmp_path / "rendered.ssd"
    rendered.write_text(render(expected))
    round_trip = json.loads(
        subprocess.check_output(
            [str(native_contract), "--parse", str(rendered), platform], text=True
        )
    )
    assert round_trip["values"] == actual["values"]


@pytest.mark.parametrize(
    "body",
    [
        "cpu.threads=0",
        'cpu.threads="4"',
        "cpu.threads=18446744073709551616",
        "cpu.compute_cpus=1,1",
        "cpu.compute_cpus=2-0",
        "cpu.decode_participants=2",
        "unknown=0",
        "vram.reserve_mib=64",
        "place.costs=cpu_fixed_us=nan",
        "place.costs=link_gbps=0",
        "place.costs=unknown=1",
        "place.costs=cpu_fixed_us=1,cpu_fixed_us=2",
        'vram.budget="123"',
        "expert.source=[1]",
        "host.expert_cache_mib=0",
        "prefetch.depth=65",
    ],
)
def test_invalid_contract_refused_by_both(native_contract, tmp_path, body):
    text = "ssdash-settings 1\n" + body + "\n"
    source = tmp_path / "invalid.ssd"
    source.write_text(text)
    with pytest.raises(ValueError):
        parse(text, str(source))
    result = subprocess.run(
        [str(native_contract), "--parse", str(source), "linux"], capture_output=True, text=True
    )
    assert result.returncode == 1 and str(source) in result.stderr


def test_cli_precedence_and_snapshot_owned():
    value = parse(
        "ssdash-settings 1\ncpu.threads=4 # source=derived\n",
        "input.ssd",
        overrides=[("cpu.threads", "6")],
    )
    assert value.at("cpu.threads").value == 6
    assert value.at("cpu.threads").origin == "cli"
    assert parse().at("cpu.threads").value == 2


def test_real_cpp_jsonl_consumed(native_contract, tmp_path):
    path = tmp_path / "events.jsonl"
    subprocess.run([str(native_contract), "--events", str(path)], check=True)
    events = list(read_events(path.read_text().splitlines()))
    assert len(events) == 2 and events[-1].payload["read_bytes"] == 12345
    assert effective_settings(events) == parse().document()


def test_effective_settings_requires_registry_and_typed_receipts():
    from ssdash.telemetry.jsonl import Event

    with pytest.raises(ValueError, match="14 registered"):
        effective_settings([Event(1, "ssd.settings", 0, {"values": {}, "sources": {}})])
    payload = parse().document()
    payload["values"]["cpu.threads"] = True
    with pytest.raises(ValueError, match="expected integer"):
        effective_settings([Event(1, "ssd.settings", 0, payload)])


@pytest.mark.parametrize(
    "line",
    [
        '{"v":2,"seq":0,"ev":"ssd.settings"}',
        '{"v":1,"seq":1,"ev":"ssd.settings"}',
        '{"v":1,"seq":0,"ev":"ssd.settings","v":1}',
        '{"v":1,"seq":0,"ev":"ssd.settings","number":NaN}',
        '{"v":1,"seq":0,"ev":"ssd.settings","number":1e999}',
        '{"v":true,"seq":0,"ev":"ssd.settings"}',
    ],
)
def test_telemetry_protocol_rejects_invalid(line):
    with pytest.raises(ValueError):
        list(read_events([line]))
