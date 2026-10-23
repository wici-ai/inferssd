#!/usr/bin/python3
"""Trusted loopback client. Runs under a separate unprivileged client UID."""

import base64
import http.client
import json
import sys
import time


def main():
    spec = json.load(sys.stdin)
    body = spec.get('body')
    body = None if body is None else json.dumps(body).encode()
    connection = http.client.HTTPConnection('127.0.0.1', spec['port'], timeout=spec['timeout_s'])
    start = time.monotonic_ns()
    total, chunks = 0, []
    value = {'status': 0, 'headers': [], 'started_monotonic_ns': start, 'chunks': chunks}
    try:
        connection.request(spec['method'], spec['path'], body=body,
                           headers={'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        while True:
            # A timestamp accompanies each received SSE line; no worker clock or
            # worker-generated performance summary is treated as trusted evidence.
            line = response.readline(65536)
            if not line:
                if response.length:
                    raise http.client.IncompleteRead(b'', response.length)
                break
            total += len(line)
            if total > spec['response_max']:
                raise RuntimeError('HTTP response exceeded trusted byte bound')
            if time.monotonic_ns() - start > spec['timeout_s'] * 10**9:
                raise TimeoutError('HTTP response exceeded trusted deadline')
            chunks.append({'elapsed_ns': time.monotonic_ns() - start,
                           'data_b64': base64.b64encode(line).decode()})
        value.update(status=response.status, headers=response.getheaders())
    except (OSError, http.client.HTTPException) as error:
        # Transport failures are unsuccessful HTTP results, not broken broker
        # framing. Keep the job alive so its controller can collect logs/cleanup.
        value['transport_error'] = (type(error).__name__ + ': ' + str(error))[:1024]
    finally:
        connection.close()
    value['ended_monotonic_ns'] = time.monotonic_ns()
    json.dump(value, sys.stdout)


if __name__ == '__main__':
    main()
