"""Fixed socket wire contracts without SSH, root privileges or model access.

SO_PEERCRED is mocked on this portable test host; Linux enforces it at runtime.
"""

import io
import json
from pathlib import Path
import socket
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import operator_stage as stage

HASH = "a" * 64


class Connection:
    family = socket.AF_UNIX

    def __init__(self, incoming, chunk=None):
        self.incoming = incoming
        self.chunk = chunk
        self.sent = bytearray()
        self.timeouts = []
        self.destination = None
        self.shutdown_mode = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, destination):
        self.destination = destination

    def recv(self, size):
        size = min(size, self.chunk) if self.chunk else size
        result, self.incoming = self.incoming[:size], self.incoming[size:]
        return result

    def sendall(self, raw):
        self.sent.extend(raw)

    def shutdown(self, mode):
        self.shutdown_mode = mode


def proof():
    return dict(status="passed", root_uid=0, root_input_sha256=HASH)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    broker = tmp_path / "broker.json"
    broker.write_text(json.dumps(dict(device="dgx-spark", coordination_owner=dict(uid=1001))))
    monkeypatch.setattr(stage, "BROKER_CONFIG", broker)
    monkeypatch.setattr(stage, "protected", lambda path, **kw: Path(path))
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)
    monkeypatch.setattr(stage, "peer_uid", lambda connection: 1001)
    calls = []

    def receive(expected, incoming):
        calls.append((expected, incoming.read()))
        return proof()

    monkeypatch.setattr(stage, "receive", receive)
    return calls


def test_bridge_preserves_complete_wire_before_root_receive(bridge):
    body = b"plain tar bytes" + b"\0" * 1024
    connection = Connection((HASH + "\n").encode() + body, chunk=7)
    stage.root_bridge(1001, connection)
    assert bridge == [(HASH, body)]
    assert json.loads(connection.sent) == proof()
    assert all(0 < value <= 300 for value in connection.timeouts)


@pytest.mark.parametrize("header", [b"b\n", b"A" * 64 + b"\n", b"a" * 64 + b"x"])
def test_bridge_rejects_truncated_or_malformed_header(bridge, header):
    connection = Connection(header)
    stage.root_bridge(1001, connection)
    assert not bridge
    assert json.loads(connection.sent)["status"] == "failed"


def test_bridge_rejects_other_peer_before_receive(bridge, monkeypatch):
    monkeypatch.setattr(stage, "peer_uid", lambda connection: 995)
    connection = Connection((HASH + "\n").encode() + b"tar")
    stage.root_bridge(1001, connection)
    assert not bridge
    assert "peer UID refused" in json.loads(connection.sent)["error"]


def test_bridge_requires_root_and_exact_protected_coordination_owner(bridge, monkeypatch):
    with pytest.raises(ValueError, match="coordination owner"):
        stage.root_bridge(1002, Connection(b""))
    monkeypatch.setattr(stage.os, "geteuid", lambda: 1001)
    with pytest.raises(PermissionError, match="requires root"):
        stage.root_bridge(1001, Connection(b""))


def test_bridge_counts_padding_before_delegation(bridge, monkeypatch):
    monkeypatch.setattr(stage, "WIRE_LIMIT", 4)
    connection = Connection((HASH + "\n").encode() + b"tar\0\0")
    stage.root_bridge(1001, connection)
    assert not bridge
    assert "wire exceeds bound" in json.loads(connection.sent)["error"]


def test_bridge_header_deadline_does_not_reset_for_each_byte(bridge, monkeypatch):
    now = iter(range(100))
    monkeypatch.setattr(stage.time, "monotonic", lambda: next(now))
    connection = Connection((HASH + "\n").encode(), chunk=1)
    stage.root_bridge(1001, connection)
    assert not bridge
    assert "deadline exceeded" in json.loads(connection.sent)["error"]


def test_bridge_body_has_absolute_total_deadline(bridge, monkeypatch):
    now = iter([0, 0, 0, 299, 301, 302])
    monkeypatch.setattr(stage.time, "monotonic", lambda: next(now))
    connection = Connection((HASH + "\n").encode() + b"tar")
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        stage.root_bridge(1001, connection)
    assert not bridge


def test_bridge_bounds_root_response(bridge, monkeypatch):
    monkeypatch.setattr(stage, "RESPONSE_LIMIT", 4096)
    monkeypatch.setattr(stage, "receive", lambda *args: dict(payload="x" * 4096))
    connection = Connection((HASH + "\n").encode() + b"tar")
    stage.root_bridge(1001, connection)
    assert len(connection.sent) < 4096
    assert "response exceeds bound" in json.loads(connection.sent)["error"]


@pytest.fixture
def client(monkeypatch):
    info = SimpleNamespace(st_mode=stat.S_IFSOCK | 0o660, st_uid=0)
    path = SimpleNamespace(parent=Path("/run"), lstat=lambda: info)
    monkeypatch.setattr(stage, "BRIDGE_SOCKET", path)
    monkeypatch.setattr(stage, "protected", lambda path, **kw: path)
    monkeypatch.setattr(stage, "peer_uid", lambda connection: 0)

    def configure(response=None):
        connection = Connection(json.dumps(response if response is not None else proof()).encode())
        monkeypatch.setattr(stage.socket, "socket", lambda *args, **kw: connection)
        return connection

    return info, configure


def test_client_sends_fixed_hash_and_raw_wire(client):
    _, configure = client
    connection = configure()
    assert stage.client(HASH, io.BytesIO(b"tar\0")) == proof()
    assert connection.sent == (HASH + "\n").encode() + b"tar\0"
    assert connection.shutdown_mode == socket.SHUT_WR


@pytest.mark.parametrize("field,value", [("st_uid", 1001), ("st_mode", stat.S_IFLNK | 0o660), ("st_mode", stat.S_IFSOCK | 0o666)])
def test_client_rejects_unprotected_socket_leaf(client, field, value):
    info, _ = client
    setattr(info, field, value)
    with pytest.raises(ValueError, match="socket owner/type/mode"):
        stage.client(HASH, io.BytesIO())


def test_client_rejects_nonroot_server_peer(client, monkeypatch):
    _, configure = client
    connection = configure()
    monkeypatch.setattr(stage, "peer_uid", lambda connection: 1001)
    with pytest.raises(ValueError, match="peer is not root"):
        stage.client(HASH, io.BytesIO())
    assert not connection.sent


@pytest.mark.parametrize("change", [dict(root_uid=1001), dict(root_input_sha256="b" * 64), dict(status="failed", error="bounded refusal")])
def test_client_binds_root_receipt(client, change):
    _, configure = client
    configure(proof() | change)
    with pytest.raises(ValueError, match="bridge refused"):
        stage.client(HASH, io.BytesIO(b"tar"))


def test_client_bounds_wire_and_response(client, monkeypatch):
    _, configure = client
    configure()
    monkeypatch.setattr(stage, "WIRE_LIMIT", 4)
    with pytest.raises(ValueError, match="wire exceeds bound"):
        stage.client(HASH, io.BytesIO(b"tar\0\0"))
    configure(proof() | dict(extra="x" * 4096))
    monkeypatch.setattr(stage, "RESPONSE_LIMIT", 4096)
    with pytest.raises(ValueError, match="response exceeds bound"):
        stage.client(HASH, io.BytesIO(b"tar"))


def test_units_preserve_model_host_vfs_and_bound_root_connection():
    root = Path(stage.__file__).parent / "linux-sandbox"
    service = (root / "inferssd-ci-stage@.service").read_text()
    listener = (root / "inferssd-ci-stage.socket").read_text()
    for setting in ("Accept=yes", "MaxConnections=2", "SocketMode=0660", "SocketUser=root", "SocketGroup=sparker", "ListenStream=/run/inferssd-ci-stage.sock"):
        assert setting in listener.splitlines()
    for setting in ("RuntimeMaxSec=300", "KillMode=control-group", "ProtectSystem=full", "ProtectHome=no"):
        assert setting in service.splitlines()
    assert "--root-bridge --peer-uid 1001" in service
    assert "ProtectSystem=strict" not in service
    assert "ProtectHome=read-only" not in service
    # No root-service RO remount may masquerade as host model admission.
    # The existing current-worker before/after write-refusal probes remain required.
    assert "ReadOnlyPaths=" not in service
