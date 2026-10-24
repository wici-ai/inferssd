"""Exercise real Request construction; no credentials or API traffic required."""

import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import private_controller as controller, selected_controller as selected

ROOT = "https://api.github.com/repos/wici-ai/inferssd-device-ci"


@pytest.mark.parametrize(
    "path",
    [
        "",
        "branches/main",
        "collaborators/maintainer/permission",
        "actions/runs?per_page=100&page=2",
    ],
)
def test_repository_root_has_no_extra_slash_and_subpaths_remain_exact(
    monkeypatch, path
):
    requests = []

    def open_request(request, *, timeout):
        requests.append(request)
        assert timeout == 7
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    assert controller.API("wici-ai/inferssd-device-ci", "synthetic", timeout=7).call(
        path
    ) == {"ok": True}
    request = requests[0]
    assert request.full_url == ROOT + ("/" + path if path else "")
    assert request.get_method() == "GET" and request.data is None
    assert request.get_header("Authorization") == "Bearer synthetic"
    assert len(requests) == 1


@pytest.mark.parametrize(
    "method,data",
    [(None, {"ref": "main"}), ("PATCH", {"status": "done"}), ("DELETE", None)],
)
def test_write_request_method_and_body_unchanged(monkeypatch, method, data):
    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b"")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    assert (
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call(
            "actions/workflows/1/dispatches", data, method
        )
        is None
    )
    request = requests[0]
    assert request.full_url == ROOT + "/actions/workflows/1/dispatches"
    assert request.get_method() == (method or "POST")
    assert request.data == (None if data is None else json.dumps(data).encode())
    assert len(requests) == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permission_failure_propagates_once_without_path_fallback(monkeypatch, status):
    requests = []

    def refuse(request, **kwargs):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, status, "refused", {}, None)

    monkeypatch.setattr(controller.urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.HTTPError) as error:
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert error.value.code == status
    assert len(requests) == 1 and requests[0].full_url == ROOT


def test_operator_access_uses_real_canonical_metadata_and_permission_urls(monkeypatch):
    seen = []
    sha = "a" * 40

    def open_request(request, **kwargs):
        url = request.full_url
        seen.append(url)
        if url == ROOT:
            value = dict(private=True, default_branch="main")
        elif url == ROOT + "/branches/main":
            value = dict(protected=True, commit=dict(sha=sha))
        elif url in (
            ROOT + "/collaborators/maintainer/permission",
            ROOT + "/collaborators/service/permission",
        ):
            value = dict(permission="write")
        else:
            raise urllib.error.HTTPError(url, 404, "wrong URL", {}, None)
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    result = selected.operator_access(
        dict(controller_repository="wici-ai/inferssd-device-ci"),
        dict(sender=dict(id=17)),
        dict(run_attempt=1, actor="maintainer", workflow_sha=sha),
        dict(
            GITHUB_TOKEN="synthetic",
            GITHUB_TRIGGERING_ACTOR="service",
            INFERSSD_CI_ENABLED="true",
        ),
    )
    assert result == dict(login="maintainer", id=17, triggering_actor="service")
    assert seen == [
        ROOT,
        ROOT + "/branches/main",
        ROOT + "/collaborators/maintainer/permission",
        ROOT + "/collaborators/service/permission",
    ]
