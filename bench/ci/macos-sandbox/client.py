#!/usr/bin/python3
"""JSONL transport for the trusted coordinator; no candidate command runs here."""
import json
from pathlib import Path
import socket
import sys

SOCKET = Path('/Library/InferSSD-CI/control/executor.sock')
MAX_FRAME = 1024 * 1024


def main():
    with socket.socket(socket.AF_UNIX) as peer:
        peer.connect(str(SOCKET))
        stream = peer.makefile('rb')
        for line in sys.stdin.buffer:
            if len(line) > MAX_FRAME or not line.endswith(b'\n'):
                raise ValueError('invalid coordinator frame')
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be an object')
            peer.sendall(line)
            response = stream.readline(MAX_FRAME + 1)
            if not response or len(response) > MAX_FRAME or not response.endswith(b'\n'):
                raise RuntimeError('broker disconnected or returned an oversized frame')
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
    # stdin EOF closes the authenticated peer; guardian performs UID cleanup
    # before it releases the device lease, even if the controller was killed.


if __name__ == '__main__':
    main()
