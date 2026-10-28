"""Observed failures and terminal denials use the same read-only policy."""

import errno
import http.client
from pathlib import Path
import socket
import ssl
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import read_retry


@pytest.mark.parametrize("message", [
    b'Get "https://api.github.com/repos/example/project/pulls/1/reviews": dial tcp example.invalid:443: connect: operation timed out\n',
    b'gh: Get "https://api.github.com/": dial tcp6 [2001:db8::1]:443: i/o timeout\n',
])
def test_observed_go_dial_timeout_is_retryable_only_as_error_line(message):
    assert read_retry.classify(b"", message) == ("transport-dial-timeout", None, True)
    assert read_retry.classify(message, b"") == ("unknown-failure", None, False)


@pytest.mark.parametrize("message", [
    b"HTTP 500\nHTTP 403\n",
    b"HTTP 503\nForbidden\n",
    b"HTTP 500\nx509: certificate has expired\n",
])
def test_terminal_denial_or_certificate_error_has_priority(message):
    assert read_retry.classify(b"", message)[2] is False


def test_unstructured_eof_and_unrecognized_failure_are_not_retried():
    assert read_retry.classify(b'{"message":"EOF"}', b"opaque failure")[2] is False
    assert read_retry.classify(b"", b"unknown failure")[2] is False


@pytest.mark.parametrize("status, retry", [
    (500, True), (502, True), (503, True), (504, True),
    (400, False), (401, False), (403, False), (404, False),
    (408, False), (429, False), (501, False),
])
def test_typed_http_status_is_a_whitelist(status, retry):
    error = urllib.error.HTTPError("https://api.github.com/", status, "hidden", {}, None)
    assert read_retry.classify_exception(error) == (
        "http-retryable" if retry else "http-terminal", status, retry)


@pytest.mark.parametrize("error", [
    TimeoutError("hidden"),
    urllib.error.URLError(TimeoutError("hidden")),
    ConnectionResetError(errno.ECONNRESET, "hidden"),
    urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "hidden")),
    http.client.RemoteDisconnected("hidden"),
])
def test_only_known_typed_transport_errors_are_retryable(error):
    assert read_retry.classify_exception(error)[2] is True


@pytest.mark.parametrize("wrapped", [False, True])
def test_python39_distinct_socket_timeout_is_retryable(monkeypatch, wrapped):
    class LegacySocketTimeout(OSError):
        pass

    monkeypatch.setattr(read_retry.socket, "timeout", LegacySocketTimeout)
    error = LegacySocketTimeout("hidden")
    assert not isinstance(error, TimeoutError) and error.errno is None
    if wrapped:
        error = urllib.error.URLError(error)
    assert read_retry.classify_exception(error) == ("transport-timeout", None, True)


@pytest.mark.parametrize("error", [
    ssl.SSLCertVerificationError("hidden"),
    urllib.error.URLError(ssl.SSLCertVerificationError("hidden")),
    socket.gaierror(socket.EAI_NONAME, "hidden"),
    urllib.error.URLError("operation timed out"),
    OSError(errno.ENOSPC, "hidden"),
    ValueError("HTTP 500 and connection timed out are untrusted text"),
])
def test_permanent_or_unknown_errors_do_not_use_exception_text(error):
    assert read_retry.classify_exception(error)[2] is False


def test_attempt_and_backoff_limits_consume_one_absolute_budget():
    assert read_retry.attempt_timeout(30, 0) == 10
    assert read_retry.retry_delay(1, 30, 10) == .25
    assert read_retry.retry_delay(2, 30, 20.25) == .75
    assert read_retry.attempt_timeout(30, 21) == 9
    assert read_retry.retry_delay(3, 30, 22) is None
    assert read_retry.retry_delay(1, 30, 29.75) is None
    assert read_retry.attempt_timeout(30, 31) == 0


@pytest.mark.parametrize("path, label", [
    ("actions/runs/123", "actions-run"),
    ("collaborators/private-person/permission", "repository-permission"),
    ("pulls/64/reviews?per_page=100&page=1", "pull-reviews"),
    ("unknown/private-person/secret", "other-get"),
])
def test_diagnostic_endpoint_is_a_fixed_label(path, label):
    assert read_retry.endpoint_label(path) == label
