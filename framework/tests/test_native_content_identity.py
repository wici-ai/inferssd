"""Tiny GGUF loader receipts; no real weights or benchmark measurements."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from test_mac_alias import fixture


@pytest.fixture
def native():
    path = os.environ.get("INFERSSD_CONTENT_TEST_BINARY")
    if not path:
        pytest.skip("set INFERSSD_CONTENT_TEST_BINARY to built test-ssd-content-identity")
    assert Path(path).is_file()
    return path


@pytest.mark.parametrize("arch,name,quant", [("deepseek41", "Src", 38), ("muse-spark", "Muse Spark Open", 14)])
def test_missing_uuid_multishard_loaded_content(native, tmp_path, arch, name, quant):
    paths = [tmp_path / f"model-{i+1:05d}-of-00002.gguf" for i in range(2)]
    for i, path in enumerate(paths):
        fixture(path, tensors=[(f"weight.{i}", (32,), 0, 0)], payload=1024 * 1024 + 128,
                kv=[("general.architecture", 8, arch), ("general.name", 8, name),
                    ("general.file_type", 4, quant), ("general.quantization_version", 4, 2),
                    ("split.no", 2, i), ("split.count", 2, 2), ("split.tensors.count", 5, 2)])
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    identity = "sha256:" + hashlib.sha256("".join(x + "\n" for x in hashes).encode()).hexdigest()
    settings = tmp_path / "settings.ssd"

    def run(claims=hashes, rule_identity=identity, file_type=quant):
        rule = f"tiny|on|{rule_identity}|{arch}|{file_type}|2|Test GPU|1|0"
        settings.write_text("inferssd-settings 1\n" +
                            ("model.content_manifest=" + json.dumps(claims) + "\n" if claims is not None else "") +
                            "model.optimizations=" + json.dumps([rule]) + "\n")
        result = subprocess.run([native, str(settings), str(paths[0])], check=True,
                                capture_output=True, text=True)
        return json.loads(result.stdout)

    actual = run()
    assert actual["uuid"] == "" and actual["content"] == identity
    assert actual["matched_rule"] == "tiny" and actual["engram_prefetch"] and not actual["hc_fused"]
    assert run(file_type=quant+1)["matched_rule"] == ""
    assert run(rule_identity="sha256:" + "f" * 64)["matched_rule"] == ""
    for claims in (None, [], hashes[:1], list(reversed(hashes)), ["0" * 64, hashes[1]]):
        fallback = run(claims)
        assert fallback["matched_rule"] == "" and fallback["content"] == ""
    # Same path and metadata, different weights: stale claims do not match.
    with paths[1].open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    assert run()["matched_rule"] == ""


def test_production_launcher_file_reaches_native_owned_context(native, tmp_path):
    import sys
    from test_typed_launch import setup, resolve

    if sys.platform != "darwin":
        pytest.skip("this unbound Mac source fixture exercises the Darwin fallback")
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform="darwin")
    model = Path(spec.model)
    fixture(model, tensors=[("weight", (32,), 0, 0)], payload=128,
            kv=[("general.architecture", 8, "deepseek41"), ("general.name", 8, "Src"),
                ("general.file_type", 4, 38), ("general.quantization_version", 4, 2)])
    hashes = [hashlib.sha256(model.read_bytes()).hexdigest()]
    identity = "sha256:" + hashlib.sha256((hashes[0] + "\n").encode()).hexdigest()
    with Path(spec.settings_file).open("a") as stream:
        stream.write("model.content_manifest=" + json.dumps(hashes) + "\n")
        stream.write("model.optimizations=" + json.dumps([
            f"production-fixture|on|{identity}|deepseek41|38|2|Test GPU|1|0"]) + "\n")
    result = resolve(inputs, spec, adapter, out, platform="darwin")
    assert result.code == 0
    settings = result.argv[result.argv.index("--ssd-settings") + 1]
    observed = subprocess.run([native, settings, spec.model], check=True, capture_output=True, text=True)
    receipt = json.loads(observed.stdout)
    # The native model verifies the content; Mac's unbound external source must
    # still prevent the context from applying the rule.
    assert receipt["loader_content"] == identity
    assert receipt["content"] == "" and receipt["matched_rule"] == ""
