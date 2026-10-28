"""Exercise real Request construction; no credentials or API traffic required."""

import errno
import io
import http.client
import json
from pathlib import Path
import socket
import ssl
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
        assert 0 < timeout <= 7
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


class Clock:
    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(controller.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(controller.time, "sleep", value.sleep)
    return value


def http_error(status):
    return urllib.error.HTTPError(ROOT + "/secret-path?token=never-log", status,
                                  "never-log-private-body", {}, io.BytesIO(b"private"))


def diagnostics(capsys):
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "never-log" not in captured.err and ROOT not in captured.err
    return [json.loads(line) for line in captured.err.splitlines()]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_get_transient_http_is_retried_once_with_only_fixed_metadata(
    monkeypatch, clock, capsys, status
):
    calls = []
    failure = http_error(status)

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise failure
        return io.BytesIO(b'{"permission":"write"}')

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    api = controller.API("wici-ai/inferssd-device-ci", "never-log-secret")
    assert api.call("collaborators/never-log-user/permission") == {"permission": "write"}
    assert len(calls) == 2 and clock.sleeps == [.25] and failure.fp.closed
    assert all(0 < timeout <= 10 for _, timeout in calls)
    rows = diagnostics(capsys)
    assert [row["decision"] for row in rows] == ["retry", "return"]
    assert rows[0]["http_status"] == status and rows[1]["category"] == "success"
    assert [row["attempt"] for row in rows] == [1, 2]
    assert all(set(row) == {"schema_version", "method", "endpoint_class", "attempt",
                            "category", "http_status", "decision", "elapsed_ms"} for row in rows)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 422, 429, 501, 505])
def test_non_allowlisted_http_always_fails_once(monkeypatch, clock, capsys, status):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        raise http_error(status)

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(urllib.error.HTTPError) as error:
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert error.value.code == status and len(calls) == 1 and not clock.sleeps
    assert diagnostics(capsys)[0]["decision"] == "raise"


def test_denial_after_transient_is_immediate_and_permissions_are_never_cached(monkeypatch, clock):
    replies = [io.BytesIO(b'{"permission":"write"}'), http_error(500), http_error(403)]
    calls = []

    def open_request(request, **_):
        calls.append(request)
        value = replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    api = controller.API("wici-ai/inferssd-device-ci", "synthetic")
    assert api.call("collaborators/operator/permission") == {"permission": "write"}
    with pytest.raises(urllib.error.HTTPError) as error:
        api.call("collaborators/operator/permission")
    assert error.value.code == 403 and len(calls) == 3 and clock.sleeps == [.25]


@pytest.mark.parametrize("failure", [
    TimeoutError("never-log-private-timeout"), ConnectionResetError(errno.ECONNRESET, "never-log-reset"),
    http.client.RemoteDisconnected("never-log-eof"),
    urllib.error.URLError(TimeoutError("never-log-wrapped-timeout")),
    socket.gaierror(socket.EAI_AGAIN, "never-log-temporary-dns"),
])
def test_known_transport_gets_are_bounded_and_can_recover(monkeypatch, clock, capsys, failure):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        if len(calls) == 1:
            raise failure
        return io.BytesIO(b"{}")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    assert controller.API("wici-ai/inferssd-device-ci", "synthetic").call("actions/runs/73") == {}
    assert len(calls) == 2 and clock.sleeps == [.25]
    assert [row["decision"] for row in diagnostics(capsys)] == ["retry", "return"]


@pytest.mark.parametrize("failure", [
    ssl.SSLCertVerificationError("never-log-cert"),
    urllib.error.URLError(ssl.SSLCertVerificationError("never-log-wrapped-cert")),
    socket.gaierror(socket.EAI_NONAME, "never-log-permanent-dns"),
    OSError("never-log-unknown"), ValueError("never-log-unknown-value"),
])
def test_unknown_or_permanent_transport_errors_never_retry(monkeypatch, clock, capsys, failure):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        raise failure

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(type(failure)):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert len(calls) == 1 and not clock.sleeps
    assert diagnostics(capsys)[0]["decision"] == "raise"


def test_bad_json_does_not_retry(monkeypatch, clock, capsys):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        return io.BytesIO(b"not-json-never-log")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(json.JSONDecodeError):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert len(calls) == 1 and not clock.sleeps
    assert diagnostics(capsys)[0]["decision"] == "raise"


def test_three_attempt_limit_and_total_budget_include_backoffs(monkeypatch, clock, capsys):
    timeouts = []

    def open_request(request, *, timeout):
        timeouts.append(timeout)
        clock.now += timeout
        raise TimeoutError("never-log-timeout")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(TimeoutError):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert timeouts == [10, 10, 9] and clock.sleeps == [.25, .75]
    assert clock.now == 130
    assert [row["decision"] for row in diagnostics(capsys)] == ["retry", "retry", "raise"]


def test_fast_errors_still_stop_after_three_attempts(monkeypatch, clock):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        raise http_error(503)

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(urllib.error.HTTPError):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("")
    assert len(calls) == 3 and clock.sleeps == [.25, .75]


@pytest.mark.parametrize("phase", ["open", "read"])
def test_late_success_is_rejected_without_extra_request(monkeypatch, clock, phase):
    calls = []

    class Response(io.BytesIO):
        def read(self):
            if phase == "read":
                clock.now += 2
            return super().read()

    def open_request(request, *, timeout):
        calls.append(timeout)
        if phase == "open":
            clock.now += 2
        return Response(b"{}")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises(TimeoutError, match="after " + phase):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call("", deadline=101)
    assert calls == [1] and not clock.sleeps


def test_expired_deadline_never_opens_and_overslept_backoff_never_reopens(monkeypatch, clock):
    calls = []

    def open_request(request, **_):
        calls.append(request)
        raise http_error(500)

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    api = controller.API("wici-ai/inferssd-device-ci", "synthetic")
    with pytest.raises(TimeoutError, match="budget"):
        api.call("", deadline=100)
    assert not calls
    monkeypatch.setattr(controller.time, "sleep", lambda _: setattr(clock, "now", 102))
    with pytest.raises(urllib.error.HTTPError):
        api.call("", deadline=101)
    assert len(calls) == 1


@pytest.mark.parametrize("path,method,data", [
    ("actions/workflows/1/dispatches", "POST", {"ref": "main"}),
    ("actions/runs/73/cancel", "POST", {}),
    ("check-runs/1", "PATCH", {"status": "done"}),
    ("unknown", "DELETE", None), ("unknown", "HEAD", None),
    ("unknown", "GET", {"body": "not-a-read-only-get"}),
])
@pytest.mark.parametrize("failure", [500, "eof"])
def test_writes_and_non_plain_get_never_retry_or_emit_read_diagnostics(
    monkeypatch, clock, capsys, path, method, data, failure
):
    calls = []

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        raise http_error(500) if failure == 500 else http.client.RemoteDisconnected("ambiguous write")

    monkeypatch.setattr(controller.urllib.request, "urlopen", open_request)
    with pytest.raises((urllib.error.HTTPError, http.client.RemoteDisconnected)):
        controller.API("wici-ai/inferssd-device-ci", "synthetic").call(path, data, method)
    assert len(calls) == 1 and calls[0][1] == 30 and not clock.sleeps
    assert capsys.readouterr().err == ""
