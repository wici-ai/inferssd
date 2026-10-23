#!/usr/bin/python3
"""Persistent JSONL bridge for the trusted controller; EOF disconnects the job."""

import json
import socket
import sys

SOCKET = '/run/inferssd-ci-broker/broker.sock'
MAX = 1024 * 1024


def main():
    if len(sys.argv) != 1:
        raise ValueError('broker client takes no command arguments')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(43205)
        connection.connect(SOCKET)
        with connection.makefile('rb') as reader:
            while body := sys.stdin.buffer.readline(MAX + 1):
                if len(body) > MAX:
                    raise ValueError('oversized broker request')
                json.loads(body)
                connection.sendall(body.rstrip(b'\n') + b'\n')
                response = reader.readline(MAX + 1)
                if not response or len(response) > MAX:
                    raise RuntimeError('broker disconnected or returned oversized response')
                value = json.loads(response)
                sys.stdout.buffer.write(response)
                sys.stdout.buffer.flush()
                if value.get('error'):
                    return 2
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print('broker refused: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
