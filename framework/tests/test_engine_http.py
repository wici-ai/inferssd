# SPDX-License-Identifier: Apache-2.0
"""Owned-server HTTP stays local even when the environment configures a proxy."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from inferssd.engine import core


@contextmanager
def local_server(status):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.command, self.path, body, self.headers.get("Content-Type")))
            payload = b'{"status":"ok"}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
        assert not worker.is_alive()


@pytest.mark.parametrize("proxy_variable", ["HTTP_PROXY", "http_proxy"])
@pytest.mark.parametrize("operation", ["health", "post"])
def test_owned_server_requests_bypass_environment_proxy(monkeypatch, proxy_variable, operation):
    for name in list(os.environ):
        if name.lower().endswith("_proxy") or name == "REQUEST_METHOD":
            monkeypatch.delenv(name)
    # urllib caches its default opener; make the environment observable in this test.
    monkeypatch.setattr(urllib.request, "_opener", None)
    with local_server(200) as (port, received), local_server(502) as (proxy_port, proxied):
        proxy = f"http://127.0.0.1:{proxy_port}"
        monkeypatch.setenv(proxy_variable, proxy)
        assert urllib.request.getproxies().get("http") == proxy
        if operation == "health":
            assert core.NativeEffects("127.0.0.1").health(port) == b'{"status":"ok"}'
            assert received == [("GET", "/health", b"", None)]
        else:
            assert core.post(port, "/completion", {"prompt": "你好"}, timeout=2) == {"status": "ok"}
            assert received == [
                (
                    "POST",
                    "/completion",
                    json.dumps({"prompt": "你好"}, ensure_ascii=False).encode(),
                    "application/json",
                )
            ]
        assert proxied == []
