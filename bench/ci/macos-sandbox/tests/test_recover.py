"""Portable recovery contracts; no broker, root process, GPU or TCC probe runs."""
import errno
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import executor
import recover


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'private').mkdir()
        (self.root / 'control').mkdir()
        (self.root / 'control/config.json').write_bytes(b'root-config')
        lock = self.root / 'original.lock'
        lock.write_bytes(b'original')
        info = lock.stat()
        self.config = {'lock': str(lock), 'stop': str(self.root / 'STOP'),
                       'lock_dev_inode': [info.st_dev, info.st_ino],
                       'runner_uid': 62001, 'runner_gid': 62001, 'worker_uid': 62002,
                       'executor': {'enabled': True}, 'installed_files': {'control/executor.py': 'frozen'}}
        self.base_patch = patch.object(recover.core, 'BASE', self.root)
        self.base_patch.start()
        self.birth = {'pid': 123, 'uid': 0, 'start_sec': 100, 'start_usec': 2}
        self.response = {'schema_version': 1, 'status': 'ready', 'protocol': 'mac-executor-v1',
                         'broker_birth': self.birth,
                         'config_sha256': recover.core.sha(self.root / 'control/config.json'),
                         'installed_files': self.config['installed_files'],
                         'lock_dev_inode': self.config['lock_dev_inode'],
                         'runner_uid': 62001, 'worker_uid': 62002}

    def tearDown(self):
        self.base_patch.stop()
        self.temp.cleanup()

    def peer(self, raw=None, connect_error=None):
        stream = io.BytesIO(raw if raw is not None else json.dumps(self.response).encode() + b'\n')
        class Peer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def settimeout(self, value): pass
            def connect(self, path):
                if connect_error: raise connect_error
            def sendall(self, data):
                if data != b'{"schema_version":1,"operation":"broker.status"}\n':
                    raise AssertionError('unexpected protocol')
            def makefile(self, mode): return stream
        return Peer()

    def status(self, **kwargs):
        with patch.object(recover, 'socket_identity'), \
             patch.object(recover.socket, 'socket', return_value=self.peer(**kwargs)), \
             patch.object(executor, 'peer_identity', return_value=(0, 0)), \
             patch.object(executor, 'birth', return_value=self.birth):
            return recover.status(self.config)

    def test_exact_managed_broker_identity(self):
        self.assertEqual(self.status(), self.response)

    def test_wrong_config_files_or_birth_cannot_be_reused(self):
        for key in ('config_sha256', 'installed_files', 'lock_dev_inode'):
            bad = dict(self.response, **{key: 'wrong'})
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                self.status(raw=json.dumps(bad).encode() + b'\n')
        bad = dict(self.response, broker_birth=dict(self.birth, start_sec=999))
        with self.assertRaisesRegex(RuntimeError, 'PID/birth'):
            self.status(raw=json.dumps(bad).encode() + b'\n')

    def test_legacy_error_duplicate_or_truncated_response_refused(self):
        for raw in (b'{"status":"failed","error":"job.open required"}\n',
                    b'{"status":"ready","status":"ready"}\n', b'{"status":"ready"}'):
            with self.assertRaises(RuntimeError):
                self.status(raw=raw)

    def test_only_refused_socket_is_stale_not_busy_timeout(self):
        self.assertIsNone(self.status(connect_error=OSError(errno.ECONNREFUSED, 'stale')))
        with self.assertRaises(TimeoutError):
            self.status(connect_error=TimeoutError('busy'))

    def test_original_lock_replacement_and_stop_symlink_refused(self):
        recover.original_lock(self.config)
        original = Path(self.config['lock'])
        original.rename(original.with_suffix('.saved'))
        original.write_bytes(b'not the same inode')
        with self.assertRaisesRegex(RuntimeError, 'original lock inode'):
            recover.original_lock(self.config)
        Path(self.config['stop']).symlink_to(self.root / 'absent')
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            recover.original_lock(self.config)

    def test_live_broker_reuse_never_spawns_or_takes_device_lease(self):
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', return_value=self.response), \
             patch.object(recover, 'spawn') as spawn, patch.object(recover.core, 'lease') as lease:
            self.assertEqual(recover.ensure()['action'], 'reused')
            spawn.assert_not_called()
            lease.assert_not_called()

    def test_unknown_broker_error_never_unlinks_or_spawns(self):
        path = self.root / 'control/executor.sock'
        path.write_bytes(b'preserved')
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', side_effect=TimeoutError('busy')), \
             patch.object(recover, 'spawn') as spawn:
            with self.assertRaises(TimeoutError): recover.ensure()
            spawn.assert_not_called()
            self.assertEqual(path.read_bytes(), b'preserved')

    def test_stop_appearing_during_status_prevents_ready(self):
        def reply(config):
            Path(config['stop']).touch()
            return self.response
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', side_effect=reply), \
             patch.object(recover, 'spawn') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'STOP'): recover.ensure()
            spawn.assert_not_called()

    def test_nonempty_worker_refuses_recovery_without_cleanup_or_signal(self):
        with patch.object(recover.core, 'members', return_value=[{'pid': 99}]), \
             patch.object(recover.core, 'cleanup') as cleanup:
            with self.assertRaisesRegex(RuntimeError, 'worker UID'):
                recover.validate_idle(self.config)
            cleanup.assert_not_called()
        # The real original lease is released on this failure.
        fd = recover.core.lease(self.config)
        os.close(fd)

    def test_live_guardian_management_lock_refuses_before_device_lease(self):
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', return_value=None), \
             patch.object(recover, 'management_lock', side_effect=BlockingIOError('guardian alive')), \
             patch.object(recover, 'validate_idle') as idle, patch.object(recover, 'spawn') as spawn:
            with self.assertRaises(BlockingIOError): recover.ensure()
            idle.assert_not_called()
            spawn.assert_not_called()

    def test_changed_stale_socket_is_not_unlinked(self):
        path = self.root / 'control/executor.sock'
        path.write_bytes(b'preserve changed socket')
        manager = os.open(self.root / 'private/test-manager', os.O_CREAT | os.O_RDWR, 0o600)
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', return_value=None), \
             patch.object(recover, 'management_lock', return_value=manager), \
             patch.object(recover, 'socket_identity', side_effect=[(7, 8), (7, 9)]), \
             patch.object(recover.core, 'members', return_value=[]), \
             patch.object(recover, 'spawn') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'socket changed'): recover.ensure()
            spawn.assert_not_called()
        self.assertEqual(path.read_bytes(), b'preserve changed socket')

    def test_stale_restart_requires_manager_idle_lease_and_rechecks_socket(self):
        path = self.root / 'control/executor.sock'
        path.write_bytes(b'fixture stale socket')
        manager = os.open(self.root / 'private/test-manager', os.O_CREAT | os.O_RDWR, 0o600)
        checks = [None, None, None, self.response]
        with patch.object(recover.core, 'config', return_value=self.config), \
             patch.object(recover, 'status', side_effect=checks), \
             patch.object(recover, 'management_lock', return_value=manager), \
             patch.object(recover, 'socket_identity', return_value=(7, 8)), \
             patch.object(recover.core, 'members', return_value=[]), \
             patch.object(recover, 'spawn', return_value=(123, 'root-log')) as spawn, \
             patch.object(recover.os, 'waitpid', return_value=(0, 0)):
            self.assertEqual(recover.ensure()['action'], 'started')
            self.assertFalse(path.exists())
            spawn.assert_called_once()
        self.assertEqual(Path(self.config['lock']).read_bytes(), b'original')

    def test_manager_file_root_regular_and_nonshared_required(self):
        with patch.object(recover.os, 'fstat', return_value=SimpleNamespace(
                st_mode=stat.S_IFREG | 0o600, st_uid=0, st_nlink=1)):
            fd = recover.management_lock()
            try:
                with self.assertRaises(BlockingIOError): recover.management_lock()
            finally:
                os.close(fd)
        with patch.object(recover.os, 'fstat', return_value=SimpleNamespace(
                st_mode=stat.S_IFREG | 0o666, st_uid=0, st_nlink=1)):
            with self.assertRaisesRegex(RuntimeError, 'unprotected'):
                recover.management_lock()


class StatusAuthorizationTests(unittest.TestCase):
    def connection(self, request, peer_uid):
        class Peer:
            def __init__(self): self.response = b''
            def recv(self, size): return json.dumps(request).encode() + b'\n'
            def sendall(self, data): self.response += data
        peer = Peer()
        config = {'runner_uid': 62001, 'runner_gid': 62001}
        with patch.object(executor, 'peer_identity', return_value=peer_uid), \
             patch.object(executor.select, 'select', return_value=([peer], [], [])), \
             patch.object(executor, 'Session') as session:
            executor.serve_connection(config, peer, broker_identity={'status': 'ready'})
            session.assert_not_called()
        return json.loads(peer.response)

    def test_root_status_does_not_open_job_or_device_lease(self):
        self.assertEqual(self.connection({'schema_version': 1, 'operation': 'broker.status'}, (0, 0)),
                         {'status': 'ready'})

    def test_runner_cannot_get_privileged_status(self):
        with self.assertRaisesRegex(ValueError, 'root managed'):
            self.connection({'schema_version': 1, 'operation': 'broker.status'}, (62001, 62001))

    def test_root_cannot_open_candidate_jobs(self):
        with self.assertRaisesRegex(ValueError, 'root cannot'):
            self.connection({'schema_version': 1, 'operation': 'job.open'}, (0, 0))

    def test_isolated_cli_help_and_noninstalled_entry_refusal(self):
        entry = str(Path(recover.__file__))
        help_result = subprocess.run([sys.executable, '-I', '-B', entry, '--help'], capture_output=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        result = subprocess.run([sys.executable, '-I', '-B', entry, '--ensure'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'fixed installed entry', result.stderr)
        self.assertNotIn(b'No module named', result.stderr)


if __name__ == '__main__':
    unittest.main()
