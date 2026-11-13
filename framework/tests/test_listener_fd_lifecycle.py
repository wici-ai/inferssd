# SPDX-License-Identifier: Apache-2.0
"""A disappearing descriptor must not invalidate another owned listener."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest
from test_engine_runtime import policy

from inferssd.platform.resources import Resources

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="reads live Linux procfs sockets")


@pytest.mark.parametrize("close_before_read", [True, False], ids=["before-read", "after-read"])
@pytest.mark.parametrize("listening", [True, False], ids=["listener", "bound-only"])
def test_transient_socket_does_not_hide_listener(
    tmp_path, monkeypatch, close_before_read, listening
):
    runtime = Resources(policy(tmp_path, "linux"))
    with socket.socket() as listener, socket.socket() as transient:
        listener.bind(("127.0.0.1", 0))
        if listening:
            listener.listen()
        port = listener.getsockname()[1]
        transient_path = Path(f"/proc/{os.getpid()}/fd/{transient.fileno()}")
        readlink = Path.readlink
        closed = False

        def close_during_read(path):
            nonlocal closed
            if path != transient_path or closed:
                return readlink(path)
            closed = True
            if close_before_read:
                transient.close()
                return readlink(path)
            target = readlink(path)
            transient.close()
            return target

        monkeypatch.setattr(Path, "readlink", close_during_read)
        if listening:
            runtime.confirm_listener(os.getpid(), "127.0.0.1", port)
        else:
            with pytest.raises(RuntimeError, match="exact server PID"):
                runtime.confirm_listener(os.getpid(), "127.0.0.1", port)
        assert closed
        assert transient.fileno() == -1


def test_descriptor_permission_error_still_refuses_confirmation(tmp_path, monkeypatch):
    runtime = Resources(policy(tmp_path, "linux"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        target = Path(f"/proc/{os.getpid()}/fd/{listener.fileno()}")
        readlink = Path.readlink

        def denied(path):
            if path == target:
                raise PermissionError("descriptor access denied")
            return readlink(path)

        monkeypatch.setattr(Path, "readlink", denied)
        with pytest.raises(PermissionError, match="descriptor access denied"):
            runtime.confirm_listener(os.getpid(), "127.0.0.1", listener.getsockname()[1])
