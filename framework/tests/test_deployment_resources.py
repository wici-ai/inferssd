"""Installed deployments resolve resources without a checkout or cwd discovery."""

import json
from dataclasses import replace

import pytest
from test_typed_launch import resolve, setup

from inferssd.launch.resources import read_resources


def descriptor(tmp_path, spec, profile="muse"):
    path = tmp_path / "engine.json"
    path.write_text(
        json.dumps(
            {
                "schema": "inferssd-engine 1",
                "binary": "repo/bin/server",
                "templates": {profile: "repo/engine/models/templates/muse-spark.jinja"},
                "capabilities": {"settings": "inferssd-settings 1", "telemetry": 1},
                "probes": {"io": "bin/inferssd-ioreplay"},
                "revision": "fixture",
            }
        )
    )
    return path


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_deployed_resources_do_not_need_checkout(tmp_path, monkeypatch, platform):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    resources = read_resources(descriptor(tmp_path, spec))
    foreign = tmp_path / "unrelated-cwd"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    inputs = replace(inputs, repo="")
    adapter.inputs = inputs
    result = resolve(
        inputs, replace(spec, binary=None, resources=resources), adapter, out, platform
    )
    assert result.code == 0
    assert result.argv[0] == spec.binary
    assert result.argv[result.argv.index("--chat-template-file") + 1] == resources.template("muse")
    assert all("unrelated-cwd" not in argument for argument in result.argv)


def test_explicit_paths_override_descriptor(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    resources = replace(read_resources(descriptor(tmp_path, spec)), binary="/missing")
    explicit = tmp_path / "custom.jinja"
    explicit.write_text("custom")
    result = resolve(
        replace(inputs, repo=""),
        replace(spec, template=str(explicit), resources=resources),
        adapter,
        out,
    )
    assert result.code == 0 and result.argv[0] == spec.binary
    assert str(explicit) in result.argv


@pytest.mark.parametrize(
    "capabilities",
    [{}, {"settings": "old", "telemetry": 1}, {"settings": "inferssd-settings 1", "telemetry": 2}],
)
def test_incompatible_capabilities_refuse_before_preparation(tmp_path, capabilities):
    inputs, spec, _, _, out, _ = setup(tmp_path)
    path = descriptor(tmp_path, spec)
    value = json.loads(path.read_text())
    value["capabilities"] = capabilities
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_resources(path)
    assert not list(out.iterdir())
