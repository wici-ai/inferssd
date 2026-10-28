"""Small read-only transport policy shared by hosted API and local gh adapters.

This module performs no requests. Callers retain their existing identity,
authorization, audit and single-shot write rules.
"""

import errno
import http.client
import re
import socket
import ssl
import urllib.error

READ_TIMEOUT = 10
MAX_ATTEMPTS = 3
BACKOFF = (0.25, 0.75)
RETRY_CODES = frozenset((500, 502, 503, 504))


def attempt_timeout(deadline, now):
    return max(0, min(READ_TIMEOUT, deadline - now))


def retry_delay(attempt, deadline, now):
    if not 1 <= attempt < MAX_ATTEMPTS:
        return None
    delay = BACKOFF[attempt - 1]
    return delay if deadline - now > delay else None


def classify(stdout, stderr, *, timed_out=False):
    """Classify a failed gh GET; never treat arbitrary stdout EOF as transport."""
    raw = (stderr + b"\n" + stdout).lower()
    codes = [int(value) for value in re.findall(
        rb"(?:\bhttp(?:/[0-9.]+)?\s+|\bstatus(?:\s+code)?[:=]?\s+)([0-9]{3})\b", raw)]
    terminal = [code for code in codes if code not in RETRY_CODES]
    if terminal:
        return "http-terminal", terminal[0], False
    if any(text in raw for text in (b"unauthorized", b"forbidden", b"not found", b"bad credentials")):
        return "authorization-terminal", None, False
    if any(text in stderr.lower() for text in (
        b"certificate verify failed", b"certificate has expired", b"x509:",
    )):
        return "tls-terminal", None, False
    if codes:
        return "http-retryable", codes[0], True
    if timed_out:
        return "local-timeout", None, True
    if re.search(rb'(?m)^(?:gh:\s*)?(?:(?:get|head) "[^\r\n]+":\s*(?:unexpected )?eof|(?:error:\s*)?unexpected eof)[ \t]*\r?$', stderr.lower()):
        return "transport-eof", None, True
    # Go's actual dial error uses "operation timed out", not "connection timed
    # out". Match only an explicit stderr request/dial line; body text is data.
    if re.search(rb'(?m)^(?:gh:\s*)?(?:get|head) "[^\r\n]+":\s*dial tcp[46]?\s+[^\r\n]+:\s*(?:connect:\s*)?(?:operation timed out|i/o timeout)[ \t]*\r?$', stderr.lower()):
        return "transport-dial-timeout", None, True
    if any(text in stderr.lower() for text in (
        b"error connecting to", b"connection refused", b"connection reset by peer",
        b"connection timed out", b"i/o timeout", b"tls handshake timeout",
        b"context deadline exceeded", b"client.timeout exceeded",
        b"temporary failure in name resolution", b"network is unreachable",
    )):
        return "transport-temporary", None, True
    return "unknown-failure", None, False


def classify_exception(error):
    """Use typed urllib/socket failures; never inspect secret-bearing text."""
    if isinstance(error, urllib.error.HTTPError):
        retry = error.code in RETRY_CODES
        return "http-retryable" if retry else "http-terminal", error.code, retry
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLError):
        return "tls-terminal", None, False
    if isinstance(reason, socket.gaierror):
        retry = reason.errno == socket.EAI_AGAIN
        return "transport-dns-temporary" if retry else "transport-dns-terminal", None, retry
    # Python 3.9 still has a distinct socket.timeout class (with errno=None).
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "transport-timeout", None, True
    if isinstance(reason, (http.client.RemoteDisconnected, http.client.IncompleteRead, EOFError)):
        return "transport-eof", None, True
    if isinstance(reason, OSError) and reason.errno in (
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
        errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE,
    ):
        return "transport-temporary", None, True
    return "unknown-failure", None, False


def endpoint_label(path):
    """Return fixed labels only; URLs, actors, PR bodies and arguments stay out."""
    patterns = (
        (r"actions/runs/[0-9]+(?:\?.*)?", "actions-run"),
        (r"collaborators/[^/]+/permission(?:\?.*)?", "repository-permission"),
        (r"pulls/[0-9]+/reviews(?:\?.*)?", "pull-reviews"),
        (r"branches/[^/]+(?:\?.*)?", "repository-branch"),
    )
    for pattern, label in patterns:
        if re.fullmatch(pattern, path):
            return label
    return "other-get"
