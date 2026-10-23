"""Protocol tests use synthetic traffic; no GPU or native performance evidence."""

import base64
import json
import socket
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.stage_session import BrokerError, StageSession
from ci import staged_chat
from test_final import records
import run_final


def test_stream_replay_preserves_client_timestamps_and_fixed_protocol(tmp_path):
    rows = records(input_count=600)
    chunks = []
    for row in rows:
        wire = b"data: " + json.dumps(row["event"]).encode() + b"\n\n"
        chunks.append(dict(elapsed_ns=row["monotonic_ns"],
                           data_b64=base64.b64encode(wire).decode()))

    class HTTP:
        port = 18080

        def json(self, method, path, body, timeout):
            if path == "/apply-template":
                return {"prompt": "rendered", "tokens": [7] * 600}
            return {"total_slots": 1, "default_generation_settings": {"n_ctx": 8192}}

        def request(self, method, path, body, timeout):
            assert path == "/completion" and body["n_predict"] == 256
            return dict(status=200, started_monotonic_ns=100,
                        ended_monotonic_ns=257_000_000_100, chunks=chunks)

    path = tmp_path / "result.json"
    result = staged_chat.record(HTTP(), "512", path, 300)
    assert result["status"] == "passed", result
    assert result["metrics"]["decode_tps"] == 1
    assert result["metrics"]["decode_first128_tps"] == 1
    assert result["metrics"]["decode_second128_tps"] == 1
    assert result["metrics"]["ttft_s"] == 1
    assert run_final.read_result(path, "512", 300) == json.loads(path.read_text())


def test_worker_cannot_substitute_http_clock_or_foreign_loopback():
    response = dict(status=200, started_monotonic_ns=100, ended_monotonic_ns=200,
                    chunks=[dict(elapsed_ns=101, data_b64="Cg==")])
    with pytest.raises(ValueError, match="receive times"):
        staged_chat.validate_response(response)

    class Session:
        def call(self, *args, **kwargs):
            return {"url": "http://localhost:9999"}

    http = staged_chat.NativeHTTP(Session(), None, 18080)
    with pytest.raises(ValueError, match="unexpected loopback"):
        http.request("POST", "/completion", {}, 1)
    with pytest.raises(ValueError, match="no retries"):
        http.request("POST", "/completion", {}, 1)


def test_session_binds_worker_and_closes_connection_on_ambiguous_response():
    short_directory = tempfile.TemporaryDirectory(prefix="ci-", dir="/tmp")
    path = Path(short_directory.name) / "broker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    observed = []

    def broker():
        peer, _ = listener.accept()
        with peer, peer.makefile("rb") as lines:
            observed.append(json.loads(lines.readline()))
            peer.sendall(b'{"worker_pid":42,"worker_uid":999,"handle":"h",'
                         b'"birth_identity":"birth","scope_identity":"scope"}\n')
            observed.append(json.loads(lines.readline()))
            peer.sendall(b'{"running":false,"running":true}\n')
            observed.append(lines.readline())

    thread = threading.Thread(target=broker)
    thread.start()
    try:
        session = StageSession(path, "a" * 32, controller_uid=1000)
        stage = session.start("build-base", "build", ["/usr/bin/cmake"], {}, "/scratch", 30)
        with pytest.raises(BrokerError, match="duplicate"):
            session.call(stage, "stage.poll")
        assert session.closed
    finally:
        thread.join(timeout=2)
        listener.close()
        short_directory.cleanup()
    assert not thread.is_alive()
    assert observed[1]["handle"] == "h" and observed[1]["job_id"] == "a" * 32
    assert observed[-1] == b""  # independent broker sees EOF and owns cleanup


def test_broker_timeout_retains_type_operation_stage_and_closes_session():
    class Connection:
        closed = False
        def settimeout(self, value): pass
        def connect(self, address): pass
        def sendall(self, value): pass
        def recv(self, size): raise socket.timeout("timed out")
        def close(self): self.closed = True
    connection = Connection()
    session = StageSession('/unused', 'a' * 32, controller_uid=1000,
                           connect=lambda *args: connection)
    with pytest.raises(socket.timeout, match='stage.probe stage=candidate-smoke'):
        session.request('stage.probe', stage_id='candidate-smoke', handle='h')
    assert session.closed and connection.closed
