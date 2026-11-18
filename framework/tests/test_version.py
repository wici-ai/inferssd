import json

from inferssd.launch.typed import PROFILES
from inferssd.version import report


def test_supported_launch_profiles_do_not_depend_on_descriptor_build(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"build": 224, "branch": "windows"}))
    value = report({"INFERSSD_RESOURCES": str(path)})
    assert value["build"] == 224
    assert value["capabilities"]["model_profiles"] == list(PROFILES)
    assert "gemma4" in value["capabilities"]["model_profiles"]
    assert "qwen4exp" in value["capabilities"]["model_profiles"]


def test_supported_profiles_are_reported_without_an_installed_descriptor(tmp_path):
    value = report({"INFERSSD_RESOURCES": str(tmp_path / "missing.json")})
    assert value["build"] is None
    assert value["capabilities"]["model_profiles"] == list(PROFILES)


def test_gpu_backends_come_from_the_descriptor(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"build": 225, "capabilities": {"gpu_backends": ["CUDA"]}}))
    assert report({"INFERSSD_RESOURCES": str(path)})["capabilities"]["gpu_backends"] == ["CUDA"]
    path.write_text(json.dumps({"build": 225, "capabilities": {"gpu_backends": []}}))
    assert report({"INFERSSD_RESOURCES": str(path)})["capabilities"]["gpu_backends"] == []


def test_gpu_backends_are_left_out_when_not_recorded_or_not_known(tmp_path):
    path = tmp_path / "resources.json"
    for capabilities in (
        {},
        {"gpu_backends": ["OpenCL"]},
        {"gpu_backends": [["CUDA"]]},
        {"gpu_backends": "CUDA"},
    ):
        path.write_text(json.dumps({"build": 225, "capabilities": capabilities}))
        assert "gpu_backends" not in report({"INFERSSD_RESOURCES": str(path)})["capabilities"]
    assert (
        "gpu_backends"
        not in report({"INFERSSD_RESOURCES": str(tmp_path / "missing.json")})["capabilities"]
    )


def test_a_release_reports_its_recorded_version_and_tag(tmp_path):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"version": "0.1.5", "tag": "v0.1.5", "build": 289}))
    value = report({"INFERSSD_RESOURCES": str(path)})
    assert (value["version"], value["tag"]) == ("0.1.5", "v0.1.5")


def test_a_development_build_has_no_tag(tmp_path):
    path = tmp_path / "resources.json"
    for recorded in (
        {"version": "0.1.6.dev3", "tag": None},
        {"version": "0.1.6.dev3", "tag": "v0.1.6"},
        {"version": "0.1.5", "tag": "0.1.5"},
    ):
        path.write_text(json.dumps({**recorded, "build": 292}))
        value = report({"INFERSSD_RESOURCES": str(path)})
        assert value["version"] == recorded["version"] and value["tag"] is None


def test_without_a_recorded_version_the_framework_version_stands(tmp_path):
    from inferssd import __version__

    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"version": None, "build": 266}))
    assert report({"INFERSSD_RESOURCES": str(path)})["version"] == __version__
    assert report({"INFERSSD_RESOURCES": str(tmp_path / "missing.json")})["tag"] is None
