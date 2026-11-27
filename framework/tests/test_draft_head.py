# SPDX-License-Identifier: Apache-2.0
"""A launch drafts with the MTP head published with its model, where the engine drafts with one
and drafting can be faster on the machine."""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_decode_profiling import setup as gpu_machine
from test_launch_strategy import native, resolved, strategic
from test_machine_command import execute, machine
from test_machine_probes import FakeRunner
from test_strategy_estimate import MAC
from test_strategy_inputs import Q8_0, EngineHost, engine_facts, moe, recorded_facts

from inferssd.app import catalog, hub
from inferssd.engine import decode_steps
from inferssd.launch.live import LiveArtifacts
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.typed import draft_head, head_model
from inferssd.launch.types import Fact, LaunchInput
from inferssd.machine.characteristics import dumps
from inferssd.machine.identity import Command
from inferssd.machine.store import save
from inferssd.strategy.drafting import drafting, routed_ms
from inferssd.strategy.estimate import decode_estimate
from inferssd.strategy.inputs import (
    RuntimeFacts,
    facts_from_engine,
    model_facts_from,
    model_facts_json,
)
from inferssd.strategy.weights import dense_weights

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
    assert draft_head(Files(), model, os.path) == small
    # unsloth: the heads in an MTP directory; a head borrowing the model's matrices comes first
    model = touch(
        tmp_path / "uns" / "UD-Q4_K_XL" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    )
    touch(tmp_path / "uns" / "MTP" / "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf", 4)
    shared = touch(tmp_path / "uns" / "MTP" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 8)
    assert draft_head(Files(), model, os.path) == shared
    # a head of another model, or none
    model = touch(tmp_path / "other" / "Muse-Spark-Q4_0.gguf")
    touch(tmp_path / "other" / "mtp-Qwen3.8-Flash-Next-Q4_0.gguf")
    assert draft_head(Files(), model, os.path) is None
    assert draft_head(Files(), touch(tmp_path / "alone" / FLASH), os.path) is None


def probe(dense, draft=0.5):
    """What the engine's decode probe prints: dense ms of target steps of 1, 2, ... rows."""
    rows = [
        {"rows": index + 1, "ms": ms + 2, "sink_ms": 2.0, "dense_ms": ms}
        for index, ms in enumerate(dense)
    ]
    step = None if draft is None else {"ms": draft + 0.25, "sink_ms": 0.25, "dense_ms": draft}
    return {"version": 1, "steps": 8, "rows": rows, "draft": step}


# Dense work a step shares across its rows, as a GPU's: drafting pays.
PAYS = probe([10.0, 10.2, 10.4, 10.6])
# Every row as dear as a step of its own: no draft pays.
NEVER = probe([10.0, 20.0, 30.0, 40.0], 5.0)
DRAFT_FACTS = ("draft_model_facts", "decode_steps", "decode_probe")
FLASH_SHAPE = dict(experts=64, used=4, latent=256, hidden=128, gate=Q8_0, up=Q8_0, down=Q8_0)


# A Linux launch names its files with POSIX paths, which a Windows tmp_path is not.
linux_launch = pytest.mark.skipif(sys.platform == "win32", reason="a Linux launch: POSIX paths")


def flash_launch(tmp_path, head=True, measured=PAYS):
    inputs, spec, host, adapter, out = strategic(tmp_path)
    directory = tmp_path / "models" / "Qwen3.8-Flash-Next-IQ4_XS"
    model = touch(directory / "Qwen3.8-Flash-Next-IQ4_XS.gguf")
    shape = FLASH_SHAPE
    host["model_facts"] = recorded_facts(*moe("qwen4exp", layers=2, **shape), path=model)
    heads = touch(tmp_path / "models" / "mtp-Qwen3.8-Flash-Next-Q4_0.gguf") if head else None
    # With the head loaded its routed layer follows the model's.
    loaded = recorded_facts(*moe("qwen4exp", layers=3, **shape), path=model)
    host["draft_model_facts"] = dict(loaded, draft_layers=1)
    if measured is not None:
        host["decode_probe"] = measured
    template = tmp_path / "repo" / "engine/models/templates/muse-spark.jinja"
    spec = replace(spec, model=model, template=str(template), settings_file=None)
    return inputs, spec, adapter, out, heads


def asked(result, name):
    return [fact for fact in result.observations if fact.name == name]


@linux_launch
def test_the_strategy_drafts_with_the_head_and_reads_its_experts(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path)
    result, line, stderr = resolved(inputs, spec, adapter, out)
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
    notes = line["strategy"]["notes"]
    assert f"draft head: {head}, drafts of up to 3 tokens" in notes
    assert notes[-1].startswith("draft head: loaded; drafting 3 a step, all accepted, costs ")
    assert "dense ms of a step of 1 to 4 rows 10.00, 10.20, 10.40, 10.60" in notes[-1]
    # The engine measured the launch it starts, once: the same command, its own run files.
    assert "measuring the model's decode steps with its draft head" in stderr
    (fact,) = asked(result, "decode_probe")
    request = decode_steps.Request.parse(fact.arguments)
    own = {
        str(out / "decode-probe.ssd"): str(out / "settings.ssd"),
        str(out / "decode-probe-events.jsonl"): str(out / "native-events.jsonl"),
    }
    assert [own.get(item, item) for item in request.argv] == argv + ["--ssd-decode-probe"]
    probed = (out / "decode-probe.ssd").read_text()
    assert probed == (out / "settings.ssd").read_text()
    assert request.settings_sha256 == hashlib.sha256(probed.encode()).hexdigest()
    assert (request.model, request.head) == ((spec.model,), head)
    assert request.characteristics_sha256 == line["strategy"]["inputs"]["characteristics_sha256"]


@linux_launch
def test_a_head_that_cannot_pay_is_left_out(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path, measured=NEVER)
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "--spec-type" not in result.argv and head not in result.argv
    assert native(out, spec).at("expert.source").value == (spec.model,)
    note = line["strategy"]["notes"][-1]
    assert note.startswith(f"draft head: {head} left out; no draft length costs less than the ")
    # The launch is the one derived without the head.
    plain_inputs, plain_spec, plain_adapter, plain_out, _ = flash_launch(
        tmp_path / "plain", head=False
    )
    plain, plain_line, _ = resolved(plain_inputs, plain_spec, plain_adapter, plain_out)
    rebased = [item.replace(str(tmp_path / "plain"), str(tmp_path)) for item in plain.argv]
    assert list(result.argv) == rebased
    assert line["strategy"]["notes"][:-1] == plain_line["strategy"]["notes"]


@linux_launch
def test_a_failed_probe_or_facts_leave_the_head_out(tmp_path):
    reason = "the server exited with status 1: unknown argument: --ssd-decode-probe"
    inputs, spec, adapter, out, head = flash_launch(tmp_path / "probe", measured=reason)
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "--spec-type" not in result.argv
    assert line["strategy"]["notes"][-1] == (
        f"draft head: {head} left out; its decode probe failed: " + reason
    )
    inputs, spec, adapter, out, head = flash_launch(tmp_path / "facts", measured=None)
    adapter.host["draft_model_facts"] = "the engine cannot read the model m.gguf: no head"
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "--spec-type" not in result.argv
    assert line["strategy"]["notes"][-1] == (
        f"draft head: {head} left out; the engine read no facts: "
        "the engine cannot read the model m.gguf: no head"
    )
    assert not asked(result, "decode_steps") and not asked(result, "decode_probe")


@linux_launch
def test_a_kept_measurement_is_reused_without_the_engine(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path, measured=None)
    adapter.host["decode_steps"] = PAYS
    result, line, stderr = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and head in result.argv
    assert line["strategy"]["notes"][-1].startswith("draft head: loaded; ")
    assert len(asked(result, "decode_steps")) == 1 and not asked(result, "decode_probe")
    assert "measuring" not in stderr and not (out / "decode-probe.ssd").exists()


def test_without_a_head_nothing_drafts(tmp_path):
    inputs, spec, adapter, out, _ = flash_launch(tmp_path, head=False)
    result, _, stderr = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and "--spec-type" not in result.argv
    assert native(out, spec).at("expert.source").value == (spec.model,)
    assert not any(asked(result, name) for name in DRAFT_FACTS) and stderr == ""


@linux_launch
def test_a_users_speculative_decoding_wins(tmp_path):
    inputs, spec, adapter, out, head = flash_launch(tmp_path)
    inputs = replace(inputs, extra=("--spec-type", "ngram-simple"))
    adapter = RecordedArtifacts(inputs, adapter.host)
    result, line, _ = resolved(inputs, spec, adapter, out)
    assert result.code == 0
    assert "draft-mtp" not in result.argv and head not in result.argv
    assert list(result.argv[-2:]) == ["--spec-type", "ngram-simple"]
    assert not any(asked(result, name) for name in DRAFT_FACTS)
    assert line["strategy"]["notes"][-1] == "draft head: the user configures speculative decoding"


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


QWEN = next(model for model in catalog.MODELS if model.head)
# Its Q4_0 head: one routed layer of 640 x 2560 experts at 4.5 bits a weight, beside its own
# attention, embedding and output.
Q4_0_EXPERT = 921600
HEAD_DENSE = 600 << 20


def qwen_facts():
    """Qwen3.8-Flash-Next's facts without and with its Q4_0 head, as the engine reads them."""
    plain = replace(QWEN.facts, dense_read_bytes=QWEN.dense_read_bytes)
    loaded = model_facts_json(plain)
    experts = loaded["experts"]
    experts["layers"] += 1
    for role in ("gate", "up", "down"):
        experts["roles"][role]["formats"].append(["Q4_0", 1, Q4_0_EXPERT])
    experts["expert_bytes"] = 3 * Q4_0_EXPERT
    experts["expert_read_bytes"] = 3 * (Q4_0_EXPERT + 4096)
    experts["format_bytes"].append(["Q4_0", experts["count"] * 3 * Q4_0_EXPERT])
    loaded["dense_bytes"] += HEAD_DENSE
    loaded["dense_read_bytes"] += HEAD_DENSE
    return plain, model_facts_from(dict(loaded, draft_layers=1))


def routed(characteristics, runtime, platform, facts):
    drive = characteristics["storage"][0]["drive_key"]
    return routed_ms(
        decode_estimate(characteristics, facts, facts.dense_read_bytes, runtime, platform, drive)
    )


def test_a_draft_saves_only_the_shared_dense_work():
    choice = drafting([10.0, 11.0, 12.0], 1.0, 20.0, 21.0, 21, 1)
    # e = 21 / 21 = 1 ms a layer: cost(n) = (D(n + 1) + n Dd + (n + 1) 20 e + n e) / (n + 1)
    assert choice.plain_ms == 30.0
    assert choice.costs_ms == ((11 + 1 + 40 + 1) / 2, (12 + 2 + 60 + 2) / 3)
    assert choice.length == 2
    assert drafting([10.0, 20.0], 5.0, 20.0, 21.0, 21, 1).length is None
    assert drafting([10.0], 1.0, 20.0, 21.0, 21, 1).length is None  # no draft length measured


def test_on_a_drive_bound_machine_the_heads_memory_can_cost_more_than_drafts_save():
    # An M5 with 16 GiB: the experts stream from the drive. The head's tensors and its routed
    # layer take expert cache: with 5 GiB each token's experts take more than the dense work the
    # drafts share, with 9 GiB less.
    plain, loaded = qwen_facts()
    runtime = RuntimeFacts(tuple(row["id"] for row in MAC["cpu"]["logical"]), (), 0)
    dense = [24.0, 25.0, 26.0, 27.5]
    for gib, length in ((5, None), (9, 3)):
        memory = replace(runtime, available_bytes=gib << 30)
        without, with_head = (routed(MAC, memory, "darwin", facts) for facts in (plain, loaded))
        # one more routed layer, and fewer cache slots of the head's larger experts for each
        assert with_head / loaded.experts.layers > without / plain.experts.layers
        choice = drafting(dense, 2.0, without, with_head, loaded.experts.layers, 1)
        assert choice.length == length


def test_on_a_gpu_whose_cache_holds_the_experts_drafting_pays():
    characteristics, runtime = gpu_machine()
    plain, loaded = qwen_facts()
    without, with_head = (routed(characteristics, runtime, "linux", f) for f in (plain, loaded))
    assert 0 < without < with_head < 2 * without
    choice = drafting([6.0, 6.3, 6.6, 7.0], 0.7, without, with_head, loaded.experts.layers, 1)
    # Every accepted token still pays its own routed experts.
    assert choice.length == 3 and with_head < choice.costs_ms[2] < choice.plain_ms


def test_without_an_expert_estimate_only_the_dense_work_is_compared():
    for unknown in ((None, 21.0), (20.0, None)):
        choice = drafting([10.0, 11.0], 1.0, *unknown, 21, 1)
        assert (choice.plain_ms, choice.costs_ms) == (10.0, ((11 + 1) / 2,))
    gone = decode_estimate(
        MAC, replace(QWEN.facts, experts=None), 1, RuntimeFacts((0,), (), 0), "darwin", None
    )
    assert routed_ms(gone) is None


def test_a_measured_dense_time_replaces_the_priced_one():
    characteristics, runtime = gpu_machine()
    plain, _ = qwen_facts()
    drive = characteristics["storage"][0]["drive_key"]

    def estimate(**options):
        return decode_estimate(
            characteristics, plain, plain.dense_read_bytes, runtime, "linux", drive, **options
        )

    priced, measured = estimate(), estimate(measured_dense_ms=4.5)
    assert measured.components["dense_ms"] == 4.5 != priced.components["dense_ms"]
    assert routed_ms(measured) == routed_ms(priced)
    assert measured.tokens_per_second == pytest.approx(1e3 / (4.5 + routed_ms(priced)))
    added = set(measured.assumptions) - set(priced.assumptions)
    assert len(added) == 1 and "engine's measurement" in added.pop()
    assert estimate(measured_dense_ms=None) == priced


def test_the_engines_facts_with_the_head(tmp_path):
    model, head = str(tmp_path / "m.gguf"), str(tmp_path / "mtp-m.gguf")
    shape = dict(experts=8, used=2, latent=64, hidden=32, gate=Q8_0, up=Q8_0, down=Q8_0)
    raw = dict(engine_facts(*moe("qwen4exp", layers=3, **shape), path=model), draft_layers=1)
    host = EngineHost(raw)
    adapter = LiveArtifacts(LaunchInput("", ()), host=host)
    value = adapter(Fact("draft_model_facts", ("/bin/server", model, head, "/t/table.gguf"))).value
    assert host.asked == [
        ["/bin/server", "--ssd-model-facts", model, "--shard", "/t/table.gguf"]
        + ["--spec-type", "draft-mtp", "--spec-draft-model", head]
    ]
    facts = model_facts_from(value)
    assert (facts.draft_layers, facts.experts.layers) == (1, 3)
    assert facts.dense_read_bytes == dense_weights(raw).active_bytes
    # Facts without a head say so, or say nothing.
    assert facts_from_engine(dict(raw, draft_layers=0)).draft_layers == 0
    assert "draft_layers" not in model_facts_json(facts_from_engine(dict(raw, draft_layers=0)))

    class Refusing:
        def command(self, argv, timeout=10):
            return Command(1, "", "error: unknown argument: --spec-type\n")

    refused = LiveArtifacts(LaunchInput("", ()), host=Refusing())
    reason = refused(Fact("draft_model_facts", ("/bin/server", model, head))).value
    assert reason == f"the engine cannot read the model {model}: unknown argument: --spec-type"
    recorded = RecordedArtifacts(LaunchInput("", ()), {"draft_model_facts": reason})
    assert recorded(Fact("draft_model_facts", ())).value == reason


def probe_files(tmp_path):
    """A server, a split model and its head on disk, and a request to probe them."""
    (tmp_path / "bin").mkdir()
    server = tmp_path / "bin" / "server"
    server.write_bytes(b"server build 1")
    model = [touch(tmp_path / f"m-0000{n}-of-00002.gguf", 8) for n in (1, 2)]
    head = touch(tmp_path / "mtp-m-Q4_0.gguf", 4)
    argv = (str(server), "-m", model[0], "--ssd-settings", "/run/a/settings.ssd")
    argv += ("--ssd-telemetry", "/run/a/events.jsonl", "--host", "127.0.0.1", "--port", "8080")
    argv += ("--spec-draft-model", head, decode_steps.FLAG)
    return decode_steps.Request("c" * 64, "s" * 64, tuple(model), head, argv), server


class Engine:
    """A server that prints its log, then the decode probe's measurement."""

    def __init__(self, printed=PAYS):
        self.printed, self.asked = printed, []

    def command(self, argv, timeout=10):
        self.asked.append((list(argv), timeout))
        if isinstance(self.printed, Command):
            return self.printed
        return Command(0, "loading the model\n" + json.dumps(self.printed) + "\n")


def test_a_probe_is_measured_once_and_kept_for_what_it_depends_on(tmp_path):
    request, server = probe_files(tmp_path)
    kept = tmp_path / "state" / "decode-steps.json"
    engine = Engine()
    adapter = LiveArtifacts(LaunchInput("", ()), host=engine, measurements=kept)

    def ask(name, request=request):
        return adapter(Fact(name, request.arguments())).value

    assert ask("decode_steps") is None and not engine.asked
    assert ask("decode_probe") == PAYS
    assert engine.asked == [(list(request.argv), decode_steps.TIMEOUT_S)]
    (entry,) = json.loads(kept.read_text())["entries"]
    assert entry["arguments"] == [
        str(server),
        "-m",
        request.model[0],
        "--ssd-settings",
        "sha256:" + "s" * 64,
        "--host",
        "127.0.0.1",
        "--spec-draft-model",
        request.head,
        decode_steps.FLAG,
    ]
    assert entry["model"] == [[path, 8] for path in request.model]
    assert entry["head"] == [request.head, 4]
    # Another run: its own port, telemetry and settings path; the same settings and files.
    again = replace(
        request,
        argv=tuple(
            {"8080": "9090", "/run/a/settings.ssd": "/run/b/s.ssd"}.get(item, item)
            for item in request.argv
        ),
    )
    assert ask("decode_steps", again) == PAYS and len(engine.asked) == 1
    # Other settings, another server build or another model file are measured again.
    assert ask("decode_steps", replace(request, settings_sha256="t" * 64)) is None
    assert ask("decode_steps", replace(request, characteristics_sha256="d" * 64)) is None
    Path(request.model[1]).write_bytes(b"G" * 9)
    assert ask("decode_steps") is None
    Path(request.model[1]).write_bytes(b"G" * 8)
    server.write_bytes(b"server build 2")
    assert ask("decode_steps") is None


def test_a_failed_probe_says_why_and_keeps_nothing(tmp_path):
    request, _ = probe_files(tmp_path)
    kept = tmp_path / "decode-steps.json"
    for printed, reason in (
        (
            Command(1, "", "load\nerror: unknown argument: --ssd-decode-probe\n"),
            "status 1: unknown",
        ),
        (Command(134, "", "GGML_ASSERT failed\n"), "status 134: GGML_ASSERT failed"),
        ({"version": 1, "steps": 8, "rows": [], "draft": None}, "printed no measurement"),
        (dict(PAYS, rows=PAYS["rows"][1:]), "rows[0].rows must be 1"),
    ):
        adapter = LiveArtifacts(LaunchInput("", ()), host=Engine(printed), measurements=kept)
        value = adapter(Fact("decode_probe", request.arguments())).value
        assert isinstance(value, str) and reason in value
        assert not kept.exists()


def test_recorded_decode_probes():
    def recorded(host, name):
        return RecordedArtifacts(LaunchInput("", ()), host)(Fact(name, ("x",))).value

    assert recorded({}, "decode_steps") is None
    assert recorded({"decode_steps": PAYS}, "decode_steps") == PAYS
    assert recorded({"decode_probe": "it failed"}, "decode_probe") == "it failed"
    with pytest.raises(ValueError, match="Missing recorded host fact: decode_probe"):
        recorded({}, "decode_probe")
    with pytest.raises(ValueError, match="Missing recorded host fact: draft_model_facts"):
        recorded({}, "draft_model_facts")
    with pytest.raises(ValueError, match="decode_probe.steps"):
        recorded({"decode_steps": dict(PAYS, steps=0)}, "decode_steps")
    with pytest.raises(ValueError, match="decode_steps"):
        recorded({"decode_steps": "not a measurement"}, "decode_steps")


def test_the_estimates_take_a_kept_measurement_of_the_model(tmp_path):
    host, paths, build = machine(tmp_path)
    profile, runtime = gpu_machine()
    saved = tmp_path / "machine.json"
    save(profile, saved)
    server = build / "bin" / "inferssd-server"
    server.write_bytes(b"server build")
    raw = engine_facts(*moe("qwen4exp", layers=2, **FLASH_SHAPE), path=paths[0])
    exported = tmp_path / "facts.json"
    exported.write_text(json.dumps(raw))
    args = ["--show", "--output", str(saved), "--json", "--build", str(build)]
    args += ["--estimate-facts", str(exported)]

    def dense_ms():
        (row,) = json.loads(execute(args, host, FakeRunner({}), tmp_path, runtime=runtime)[1][0])[
            "estimates"
        ]
        return row["components"]["dense_ms"]

    priced = dense_ms()
    # A launch on this machine and engine measured the model with any arguments.
    kept = tmp_path / "state" / "inferssd" / decode_steps.FILENAME
    entry = {
        "key": "k",
        "server_sha256": decode_steps.file_sha256(str(server)),
        "characteristics_sha256": hashlib.sha256(dumps(profile).encode()).hexdigest(),
        "model": [["/elsewhere/" + Path(paths[0]).name, 1]],
        "head": ["/elsewhere/mtp-head.gguf", 1],
        "arguments": [str(server)],
        "measured_utc": "2026-11-27T00:00:00Z",
        "measurement": probe([3.25, 3.5]),
    }
    kept.parent.mkdir(parents=True)
    kept.write_text(json.dumps({"schema": decode_steps.SCHEMA, "entries": [entry]}))
    assert priced != 3.25 and dense_ms() == 3.25
    # Not another engine's, nor another model's.
    kept.write_text(
        json.dumps({"schema": decode_steps.SCHEMA, "entries": [dict(entry, server_sha256="x")]})
    )
    assert dense_ms() == priced
    entry["model"] = [["/elsewhere/other.gguf", 1]]
    kept.write_text(json.dumps({"schema": decode_steps.SCHEMA, "entries": [entry]}))
    assert dense_ms() == priced
