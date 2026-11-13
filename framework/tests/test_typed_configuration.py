"""Cross-language settings/JSONL contracts, including actual C++ writer output."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from inferssd.configuration import parse, render, schema
from inferssd.telemetry.jsonl import effective_settings, read_events


@pytest.fixture(scope="session")
def native_contract(tmp_path_factory):
    if sys.platform == "win32":
        pytest.skip(
            "compiled here with a POSIX c++ driver; the Windows engine build "
            "runs it as ssd_settings_telemetry"
        )
    root = Path(__file__).resolve().parents[2]
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
            str(root / "engine/ggml/src/ggml-ssd"),
            str(root / "engine/ggml/src/ggml-ssd/settings/settings.cpp"),
            str(root / "engine/ggml/src/ggml-ssd/telemetry/telemetry.cpp"),
            str(root / "engine/ggml/src/ggml-ssd/telemetry/debug.cpp"),
            str(root / "engine/tests/ssd/test-settings-telemetry.cpp"),
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
    registry = (
        Path(__file__).resolve().parents[2] / "engine/ggml/src/ggml-ssd/settings/registry.def"
    )
    declared = re.findall(r'^SSD_SETTING\("([^"]+)"', registry.read_text(), re.M)
    assert list(actual["properties"]) == declared


@pytest.mark.parametrize(
    "platform,text",
    [
        ("linux", "inferssd-settings 1\n"),
        (
            "linux",
            "inferssd-settings 1\ncpu.io_cpus=2-4 # source=measured\ncpu.decode_participants=3\n"
            'expert.source=["/path # 1", "unicode-🐈"]\n'
            "place.costs=cpu_fixed_us=204,link_gbps=12.3\n",
        ),
        ("darwin", "inferssd-settings 1\nprefetch.depth=8\n"),
        (
            "darwin",
            "inferssd-settings 1\ncpu.decode_participants=4\ncpu.io_cpus=0-2\n"
            "host.budget_mib=8192\nprefetch.depth=4\nhost.headroom_mib=1024\n",
        ),
        (
            "windows",
            "inferssd-settings 1\ncpu.decode_participants=3\ncpu.io_cpus=3\n"
            "place.costs=cpu_fixed_us=204\nhost.budget_mib=8192\nmodel.hc_fused=0\n",
        ),
        ("linux", 'inferssd-settings 1\nexpert.source=["/path-\u2028-file"]\n'),
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
        "cpu.threads=4",
        "cpu.compute_cpus=0-2",
        'cpu.decode_participants="4"',
        "cpu.decode_participants=256",
        "vram.max_slots=18446744073709551616",
        "cpu.io_cpus=1,1",
        "cpu.io_cpus=2-0",
        "unknown=0",
        "place.costs=cpu_fixed_us=nan",
        "place.costs=link_gbps=0",
        "place.costs=unknown=1",
        "place.costs=cpu_fixed_us=1,cpu_fixed_us=2",
        'vram.budget="123"',
        "expert.source=[1]",
        "host.expert_cache_mib=0",
        "host.budget_mib=1",
        "prefetch.depth=65",
    ],
)
def test_invalid_contract_refused_by_both(native_contract, tmp_path, body):
    text = "inferssd-settings 1\n" + body + "\n"
    source = tmp_path / "invalid.ssd"
    source.write_text(text)
    with pytest.raises(ValueError):
        parse(text, str(source))
    result = subprocess.run(
        [str(native_contract), "--parse", str(source), "linux"], capture_output=True, text=True
    )
    assert result.returncode == 1 and str(source) in result.stderr


@pytest.mark.parametrize("platform", ["linux", "darwin", "windows"])
def test_the_cuda_tiers_keys_parse_on_every_platform_by_both(native_contract, tmp_path, platform):
    # Which engine has the tier decides; the platform does not (see the launch's backend check).
    text = "inferssd-settings 1\nvram.max_slots=1\nvram.reserve_mib=64\n"
    source = tmp_path / "tier.ssd"
    source.write_text(text)
    parse(text, str(source), platform=platform)
    result = subprocess.run(
        [str(native_contract), "--parse", str(source), platform], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "platform,body",
    [
        ("linux", "host.budget_mib=8192"),
    ],
)
def test_platform_scope_refused_by_both(native_contract, tmp_path, platform, body):
    text = "inferssd-settings 1\n" + body + "\n"
    source = tmp_path / "platform.ssd"
    source.write_text(text)
    with pytest.raises(ValueError):
        parse(text, str(source), platform=platform)
    result = subprocess.run(
        [str(native_contract), "--parse", str(source), platform], capture_output=True, text=True
    )
    assert result.returncode == 1 and str(source) in result.stderr


def test_unicode_header_whitespace_refused_by_both(native_contract, tmp_path):
    text = "\u3000inferssd-settings 1\u3000\n"
    source = tmp_path / "header.ssd"
    source.write_text(text)
    with pytest.raises(ValueError):
        parse(text, str(source))
    result = subprocess.run(
        [str(native_contract), "--parse", str(source), "linux"], capture_output=True, text=True
    )
    assert result.returncode == 1


def test_cli_precedence_and_snapshot_owned():
    value = parse(
        "inferssd-settings 1\nprefetch.depth=4 # source=derived\n",
        "input.ssd",
        overrides=[("prefetch.depth", "6")],
    )
    assert value.at("prefetch.depth").value == 6
    assert value.at("prefetch.depth").origin == "cli"
    assert parse().at("prefetch.depth").value == 64


def test_real_cpp_jsonl_consumed(native_contract, tmp_path):
    path = tmp_path / "events.jsonl"
    subprocess.run([str(native_contract), "--events", str(path)], check=True)
    events = list(read_events(path.read_text().splitlines()))
    assert len(events) == 2 and events[-1].payload["read_bytes"] == 12345
    assert effective_settings(events) == parse().document()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('{\n"answer": 42\n}', id="lf"),
        pytest.param('{\r\n"nested": [1,\r\n{"label": "two words"}]\r\n}', id="crlf"),
        pytest.param('{\r"answer": 42\r}', id="cr"),
        pytest.param(
            json.dumps(
                {"escaped": 'line\nnext\r\t"quote"\\path', "unicode": "🐈\u2028\u2029\u0085"},
                indent=2,
                ensure_ascii=False,
            ),
            id="formatted-escaped-strings",
        ),
        pytest.param("{\n}", id="empty-object"),
        pytest.param('{\t"label": "two words"\t}', id="tabs"),
        pytest.param(
            '{"number":1.2300e+02,"integer":9007199254740993,"escaped":"\\n\\r\\t"}',
            id="compact-lexemes",
        ),
        pytest.param(' \r\n {"label": "🐈\u2028\u2029\u0085"} \r\n ', id="outer-whitespace"),
    ],
)
def test_native_payload_stays_one_jsonl_record(native_contract, tmp_path, payload):
    path = tmp_path / "payload.jsonl"
    subprocess.run([str(native_contract), "--payload", str(path), payload], check=True)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r" not in raw
    events = list(read_events(raw.decode("utf-8").split("\n")[:-1]))
    assert [(event.sequence, event.name) for event in events] == [
        (0, "ssd.payload"),
        (1, "ssd.after"),
    ]
    assert events[0].payload == json.loads(payload)
    assert events[1].payload == {}
    if "1.2300e+02" in payload:
        assert b"1.2300e+02" in raw and b"9007199254740993" in raw


def test_effective_settings_requires_registry_and_typed_receipts():
    from inferssd.telemetry.jsonl import Event

    with pytest.raises(ValueError, match="registered settings"):
        effective_settings([Event(1, "ssd.settings", 0, {"values": {}, "sources": {}})])
    payload = parse().document()
    payload["values"]["prefetch.depth"] = True
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


@pytest.mark.parametrize(
    "text,source",
    [
        ("inferssd-settings 1\nprefetch.depth=2 # \ud800\n", "input.ssd"),
        ("inferssd-settings 1\n", "input-\udfff.ssd"),
    ],
)
def test_unpaired_surrogate_rejected_even_in_receipts(text, source):
    with pytest.raises(ValueError):
        parse(text, source)


def test_unicode_source_and_receipt_preserved():
    value = parse("inferssd-settings 1\nprefetch.depth=2 # 来源=测量🐈\n", "设置🐈.ssd")
    assert value.at("prefetch.depth").location == "设置🐈.ssd:2"
    assert value.at("prefetch.depth").receipt == " 来源=测量🐈"
