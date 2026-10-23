"""The reference bridge is explicit, source-specific and observation-bound."""

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_ci_performance
import test_device_ci

from ci import compare, reference

SETTINGS_PACKAGE = "inferssd" if importlib.util.find_spec("inferssd") is not None else "ssdash"
settings_registry = importlib.import_module(SETTINGS_PACKAGE + ".configuration.settings")

pair = test_ci_performance.pair
policy = test_device_ci.policy
synthetic = test_device_ci.synthetic


@pytest.fixture
def binding():
    values = settings_registry.parse(SETTINGS_PACKAGE + '-settings 1\nexpert.source=["/models/experts.tsv"]\n').document()["values"]
    values = {key: value for key, value in values.items() if key in reference.SETTING_KEYS}
    policy = {
        "args": ["-t", "2"], "environment": {"THREADS": "2", "ALIAS": "/models/experts.tsv"},
        "template_sha256": "a" * 64, "alias_sha256": ["b" * 64],
    }
    default = {"kind": "source-default", "revision": reference.LEGACY_REVISION,
               "path": "defaults.cpp", "sha256": hashlib.sha256(b"reviewed defaults\n").hexdigest()}
    result = {
        "schema_version": 1, "kind": "legacy-to-typed", "semantic_settings": values,
        "legacy_mapping": {key: {"value": value, "inputs": [copy.deepcopy(default)],
                                 "explanation": "Synthetic reviewed default; not device evidence."}
                           for key, value in values.items()},
        "sides": {},
    }
    result["legacy_mapping"]["cpu.threads"]["inputs"] = [{"kind": "environment", "name": "THREADS", "value": "2"}]
    result["legacy_mapping"]["expert.source"]["inputs"] = [{"kind": "environment", "name": "ALIAS", "value": "/models/experts.tsv"}]
    for side, revision, target, library, mark in (
        ("base", reference.LEGACY_REVISION, "llama-server", "libllama.so", "c"),
        ("candidate", "d" * 40, "inferssd", "libinferssd.so", "e"),
    ):
        launch = copy.deepcopy(policy)
        if side == "candidate":
            launch.update(environment={}, typed_values=copy.deepcopy(values))
        result["sides"][side] = {
            "revision": revision, "profile_sha256": mark * 64, "launch_policy": launch,
            "images": [{"path": "/build/" + side + "/" + library, "sha256": mark * 64, "role": "engine"}],
            "build": {"server_target": target, "binary_relative": "bin/" + target,
                      "image_relatives": ["lib/" + library],
                      "template_relatives": {model: "models/" + model + ".jinja" for model in ("tiny", "deepseek", "muse")}},
        }
    return result


@pytest.fixture
def bound_pair(pair, binding):
    a, b, ar, br = pair
    for side, receipt in (("base", ar), ("candidate", br)):
        spec = binding["sides"][side]
        receipt["revision"] = spec["revision"]
        receipt["build"].update(revision=spec["revision"], loaded_libraries=[
            {key: image[key] for key in ("path", "sha256")} for image in spec["images"]])
        receipt["host"]["effective_settings"].update(
            native_profile_sha256=spec["profile_sha256"],
            native_launch_policy=copy.deepcopy(spec["launch_policy"]),
            source_mode="direct-gguf", expert_cache_budget_bytes=4096,
            native_resource_policy={"memory_budget_bytes": 1024},
        )
    return a, b, ar, br


def test_reviewed_rename_and_actual_legacy_launch_compare_without_faking_typed_values(bound_pair, binding, policy):
    result = compare.compare_pair(*bound_pair, policy, comparison_binding=binding)
    assert result["status"] == "passed"
    assert "typed_values" not in bound_pair[2]["host"]["effective_settings"]["native_launch_policy"]
    with pytest.raises(ValueError, match="typed configuration"):
        compare.compare_pair(*bound_pair, policy)


@pytest.mark.parametrize("change", [
    lambda r: r.update(revision="f" * 40),
    lambda r: r["host"]["effective_settings"].update(native_profile_sha256="f" * 64),
    lambda r: r["host"]["effective_settings"]["native_launch_policy"]["environment"].update(OTHER="1"),
    lambda r: r["build"]["loaded_libraries"][0].update(sha256="f" * 64),
    lambda r: r["build"]["loaded_libraries"][0].update(path="/other/libinferssd.so"),
    lambda r: r["build"]["loaded_libraries"].append({"path": "/foreign.so", "sha256": "f" * 64}),
    lambda r: r["host"].update(memory_budget_bytes=2048),
    lambda r: r["host"].update(hardware="another-device"),
    lambda r: r["host"]["effective_settings"].update(expert_cache_budget_bytes=8192),
    lambda r: r["host"]["effective_settings"]["native_resource_policy"].update(memory_budget_bytes=2048),
    lambda r: r["model_identity"]["shards"][0].update(sha256="f" * 64),
])
def test_reviewed_bridge_does_not_allow_identity_or_configuration_drift(bound_pair, binding, policy, change):
    change(bound_pair[3])
    with pytest.raises(ValueError):
        compare.compare_pair(*bound_pair, policy, comparison_binding=binding)


@pytest.mark.parametrize("change", [
    lambda b: b["sides"]["base"].update(revision="f" * 40),
    lambda b: b["sides"]["base"]["launch_policy"].update(typed_values=b["semantic_settings"]),
    lambda b: b["sides"]["candidate"]["launch_policy"]["typed_values"].update(**{"cpu.threads": 3}),
    lambda b: b["semantic_settings"].pop("host.expert_cache_mib"),
    lambda b: b["legacy_mapping"].pop("cpu.threads"),
    lambda b: b["legacy_mapping"]["cpu.threads"]["inputs"][0].update(value="3"),
    lambda b: b["sides"]["candidate"]["images"][0].update(role="unreviewed-other-role"),
    lambda b: b["sides"]["candidate"]["build"].update(binary_relative="../escape"),
    lambda b: b["sides"]["candidate"]["build"]["template_relatives"].update(tiny="/external/template"),
])
def test_reference_binding_cannot_be_generalized_or_weakened(binding, change):
    change(binding)
    with pytest.raises(ValueError):
        reference.validate_binding(binding)


def test_build_layout_is_revision_bound_and_copied(binding):
    spec = reference.build_spec(binding, "candidate", "d" * 40)
    assert spec["server_target"] == "inferssd" and spec["binary_relative"] == "bin/inferssd"
    spec["template_relatives"]["tiny"] = "altered"
    assert binding["sides"]["candidate"]["build"]["template_relatives"]["tiny"] != "altered"
    with pytest.raises(ValueError, match="another source"):
        reference.build_spec(binding, "candidate", "e" * 40)


def test_explicit_sixteen_key_muse_binding_preserves_historical_fourteen(binding, bound_pair, policy):
    if not reference.MODEL_SETTING_KEYS <= settings_registry.schema()["properties"].keys():
        pytest.skip("sixteen-key test needs the reviewed InferSSD registry")
    old = copy.deepcopy(binding)
    values = {"model.engram_prefetch": 0, "model.hc_fused": 1}
    for key, value in values.items():
        binding["semantic_settings"][key] = value
        mapping = copy.deepcopy(binding["legacy_mapping"]["host.expert_cache_mib"])
        binding["legacy_mapping"][key] = dict(mapping, value=value)
    binding["sides"]["candidate"]["launch_policy"]["typed_values"].update(values)
    for side, receipt in (("base", bound_pair[2]), ("candidate", bound_pair[3])):
        spec = binding["sides"][side]
        spec["build"]["template_relatives"] = {"muse": "models/muse.jinja"}
        receipt["host"]["effective_settings"]["native_launch_policy"] = copy.deepcopy(spec["launch_policy"])
    assert compare.compare_pair(*bound_pair, policy, comparison_binding=binding)["passed"]
    assert reference.validate_binding(old) is old
    assert set(old["semantic_settings"]) == reference.SETTING_KEYS
    binding["sides"]["candidate"]["launch_policy"]["typed_values"].pop("model.hc_fused")
    with pytest.raises(ValueError, match="complete reviewed"):
        reference.validate_binding(binding)


@pytest.mark.parametrize("change", [
    lambda b: b["semantic_settings"].update({"model.hc_fused": 1}),
    lambda b: b["sides"]["base"]["build"].update(template_relatives={}),
    lambda b: b["sides"]["base"]["build"].update(template_relatives={"unknown": "models/muse.jinja"}),
])
def test_binding_generation_and_checkpoint_template_scope_are_explicit(binding, change):
    change(binding)
    with pytest.raises(ValueError):
        reference.validate_binding(binding)


def test_exact_source_defaults_are_checked_without_running_a_candidate(binding, tmp_path, monkeypatch):
    source_bytes = b"reviewed defaults\n"
    (tmp_path / "defaults.cpp").write_bytes(source_bytes)
    calls = []

    def git(argv, **kwargs):
        calls.append(argv)
        assert argv[:3] == ["git", "-C", str(tmp_path)]
        if argv[3:] == ["rev-parse", "HEAD"]:
            return reference.LEGACY_REVISION + "\n"
        assert argv[3:] == ["show", reference.LEGACY_REVISION + ":defaults.cpp"]
        return source_bytes

    monkeypatch.setattr(reference.subprocess, "check_output", git)
    result = reference.verify_source_defaults(binding, tmp_path)
    assert result["files"] == {"defaults.cpp": hashlib.sha256(source_bytes).hexdigest()}
    changed = b"modified defaults\n"
    (tmp_path / "defaults.cpp").write_bytes(changed)
    for item in binding["legacy_mapping"].values():
        for entry in item["inputs"]:
            if entry["kind"] == "source-default":
                entry["sha256"] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="source bytes"):
        reference.verify_source_defaults(binding, tmp_path)


def test_explicit_argv_and_observed_setting_inputs_are_verified(binding, bound_pair):
    mapping = binding["legacy_mapping"]["cpu.threads"]
    mapping["inputs"] = [{"kind": "argument", "index": 1, "value": "2"},
                         {"kind": "effective-setting", "name": "expert_cache_budget_bytes", "value": 4096}]
    reference.comparable(bound_pair[2], binding, "base")
    mapping["inputs"][1]["value"] = 8192
    with pytest.raises(ValueError, match="observed setting"):
        reference.comparable(bound_pair[2], binding, "base")


def test_private_config_binding_must_name_merge_candidate_not_pr_head(binding, synthetic):
    roots, plan, policy = synthetic
    root = roots[0]
    plan["base"] = reference.LEGACY_REVISION
    plan["head"] = binding["sides"]["candidate"]["revision"]
    assert plan["candidate"] != plan["head"]
    path = root / "runner-config.json"
    config = test_device_ci.load(path)
    config["comparison_bindings"] = {name: binding for name in ("tiny", "deepseek", "muse")}
    path.write_text(json.dumps(config))
    plan["device_config_sha256"][config["device"]] = test_device_ci.digest(path)
    path = root / "ci-manifest.json"
    manifest = test_device_ci.load(path)
    manifest["plan"] = plan
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact base/merge candidate"):
        compare.device_report(root, plan, policy)
