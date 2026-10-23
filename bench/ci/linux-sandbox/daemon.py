#!/usr/bin/python3
"""Root-owned persistent JSONL stage broker; EOF cancels and drains a whole job."""

import argparse
import json
import os
from pathlib import Path
import re
import select
import socket
import stat
import struct
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import GATES, LIMIT, digest, load, protected
from stages import JobSession

SOCKET = '/run/inferssd-ci-broker/broker.sock'
FILES = ('contract.py', 'executor.py', 'guardian.py', 'daemon.py', 'client.py',
         'worker_probe.py', 'probe.py', 'stages.py', 'worker_entry.py', 'http_client.py', 'stage_probe.py')


def describe(config, nonce):
    proof = protected(Path(config['install_root']) / 'isolation-proof.json')
    value = json.loads(proof.read_text())
    if (value.get('config_sha256') != config['_sha256'] or
            any(value.get('gates', {}).get(gate) is not True for gate in GATES) or
            value.get('async_stage_gate') is not True):
        raise RuntimeError('real namespace and async stage isolation proof is incomplete or stale')
    if set(value.get('installed_sha256', {})) != set(FILES):
        raise RuntimeError('incomplete installed broker identity')
    for relative, expected in value['installed_sha256'].items():
        if digest(protected(Path(config['install_root']) / relative)) != expected:
            raise RuntimeError('installed broker differs from real isolation proof')
    lock = os.stat(config['lock_path'], follow_symlinks=False)
    if [lock.st_dev, lock.st_ino] != config['lock_identity']:
        raise RuntimeError('original lock inode differs')
    activation = Path(config['install_root']) / 'ACTIVE'
    activated = activation.exists() and protected(activation).read_text().strip() == digest(proof)
    return {'schema_version': 1, 'nonce': nonce, 'broker_uid': 0,
            'controller_uid': config['controller_uid'], 'execution_uid': config['worker_uid'],
            'network': 'loopback-only', 'candidate_access': 'sandboxed-stages',
            'proof_sha256': digest(proof), 'gates': value['gates'],
            'production_adapter_ready': activated,
            'socket_path': SOCKET, 'protocol': 'persistent-jsonl-v1'}


def serve(config, connection, peer_uid):
    if peer_uid != config['controller_uid']:
        raise PermissionError('dedicated controller UID required')
    session = None
    finished, disconnected = threading.Event(), threading.Event()
    def monitor():
        while not finished.wait(.1):
            try:
                ready, _, _ = select.select([connection], [], [], 0)
                if ready and connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b'':
                    disconnected.set()
                    if session is not None:
                        session.cancel.set()
                    return
            except BlockingIOError:
                continue
            except OSError:
                disconnected.set()
                if session is not None:
                    session.cancel.set()
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with connection.makefile('rb') as reader:
            while not disconnected.is_set():
                line = reader.readline(LIMIT + 1)
                if not line:
                    break
                if len(line) > LIMIT:
                    raise ValueError('oversized stage request')
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('object request required')
                operation = value.get('operation')
                if operation == 'describe':
                    if set(value) != {'operation', 'nonce'} or not re.fullmatch(r'[0-9a-f]{64}', str(value.get('nonce', ''))):
                        raise ValueError('fresh describe nonce required')
                    result = describe(config, value['nonce'])
                else:
                    if session is None:
                        if operation not in ('job.open', 'stage.start'):
                            raise ValueError('open a job before controlling a stage')
                        identity = describe(config, '0' * 64)
                        if not identity['production_adapter_ready']:
                            raise RuntimeError('broker activation is disabled')
                        session = JobSession(config, value.get('job_id', ''))
                        if disconnected.is_set():
                            session.cancel.set()
                            raise InterruptedError('controller disconnected during admission')
                    result = session.request(value)
                encoded = json.dumps(result).encode() + b'\n'
                if len(encoded) > LIMIT:
                    raise RuntimeError('response exceeded protocol bound')
                connection.sendall(encoded)
                if operation == 'job.finish':
                    return
    except Exception as error:
        try:
            connection.sendall(json.dumps({'error': type(error).__name__ + ': ' + str(error)}).encode() + b'\n')
        except OSError:
            pass
    finally:
        finished.set()
        watcher.join(timeout=1)
        if session is not None and not session.closed:
            session.cancel.set()
            session.close()


def main(args):
    if os.geteuid() != 0:
        raise PermissionError('root broker required')
    config = load(args.config)
    config['_sha256'] = digest(args.config)
    location = Path(SOCKET)
    protected(location.parent, directory=True)
    if location.exists() or location.is_symlink():
        info = location.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != 0:
            raise RuntimeError('unsafe previous broker socket')
        location.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(SOCKET)
        os.chown(location, 0, config['controller_gid'])
        os.chmod(location, 0o660)
        server.listen(4)
        while True:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(120)
                uid = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
                try:
                    serve(config, connection, uid)
                except Exception as error:
                    # A cleanup failure already wrote STOP. Do not silently
                    # restart a second job in the same connection.
                    print(type(error).__name__ + ': ' + str(error), file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args())
