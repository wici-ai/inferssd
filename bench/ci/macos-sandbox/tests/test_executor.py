"""Portable contract/file-sealing checks; never claimed as native isolation proof."""
import os
from pathlib import Path
import tempfile
import unittest
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import executor


class SealingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'build'
        self.source.mkdir()
        self.target = self.root / 'sealed'

    def tearDown(self):
        for directory, _, files in os.walk(self.root):
            Path(directory).chmod(0o700)
            for file in files:
                p = Path(directory) / file
                if not p.is_symlink():
                    p.chmod(0o600)
        self.temp.cleanup()

    def test_seals_distinct_regular_bytes(self):
        original = self.source / 'binary'
        original.write_bytes(b'actual build bytes')
        original.chmod(0o755)
        result = executor.copy_frozen(self.source, self.target, 100)
        self.assertEqual((self.target / 'binary').read_bytes(), original.read_bytes())
        self.assertNotEqual(original.stat().st_ino, (self.target / 'binary').stat().st_ino)
        self.assertEqual(result['binary']['mode'], 0o555)
        self.assertEqual(original.stat().st_mode & 0o777, 0o755)
        self.assertEqual((self.target / 'binary').stat().st_mode & 0o777, 0o555)

    def test_normalizes_only_internal_library_symlink(self):
        (self.source / 'lib.1.dylib').write_bytes(b'dylib bytes')
        (self.source / 'lib.dylib').symlink_to('lib.1.dylib')
        result = executor.copy_frozen(self.source, self.target, 100)
        self.assertFalse((self.target / 'lib.dylib').is_symlink())
        self.assertEqual(result['lib.dylib']['original_symlink'], 'lib.1.dylib')
        self.assertEqual(result['lib.dylib']['resolved_relative'], 'lib.1.dylib')

    def test_rejects_symlink_escape(self):
        (self.root / 'private').write_bytes(b'secret')
        (self.source / 'escape').symlink_to('../private')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            executor.copy_frozen(self.source, self.target, 100)

    def test_rejects_directory_symlink(self):
        (self.source / 'escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'non-directory'):
            executor.copy_frozen(self.source, self.target, 100)

    def test_rejects_hardlink(self):
        original = self.root / 'private'
        original.write_bytes(b'secret')
        os.link(original, self.source / 'hardlink')
        with self.assertRaisesRegex(ValueError, 'hardlinks'):
            executor.copy_frozen(self.source, self.target, 100)
        self.assertEqual(original.stat().st_mode & 0o777, 0o644)

    def test_rejects_fifo_without_blocking(self):
        os.mkfifo(self.source / 'pipe')
        with self.assertRaisesRegex(ValueError, 'special'):
            executor.copy_frozen(self.source, self.target, 100)

    def test_rejects_oversize_and_existing_destination(self):
        (self.source / 'file').write_bytes(b'too big')
        with self.assertRaisesRegex(ValueError, 'budget'):
            executor.copy_frozen(self.source, self.target, 2)
        with self.assertRaises(FileExistsError):
            executor.copy_frozen(self.source, self.target, 100)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.job = Path(self.temp.name) / 'candidate'
        self.job.mkdir()
        self.config = {'executor': {'max_stage_seconds': 60}}
        self.request = {'schema_version': 1, 'job_id': 'candidate', 'stage_id': 'compile',
                        'kind': 'build', 'argv': ['/usr/bin/clang', '--version'],
                        'environment': {}, 'cwd': str(self.job), 'timeout_s': 30}

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_request(self):
        self.assertEqual(executor.request_stage(self.request, self.job, self.config)[0][0], '/usr/bin/clang')

    def test_rejects_loader_environment(self):
        self.request['environment'] = {'DYLD_INSERT_LIBRARIES': '/private/evidence/attack.dylib'}
        with self.assertRaisesRegex(ValueError, 'environment'):
            executor.request_stage(self.request, self.job, self.config)

    def test_exact_legacy_environment_only(self):
        self.request['environment'] = {'GGML_MOE_CACHE_BYTES': '123'}
        executor.request_stage(self.request, self.job, self.config, {'GGML_MOE_CACHE_BYTES': '123'})
        with self.assertRaisesRegex(ValueError, 'unapproved'):
            executor.request_stage(self.request, self.job, self.config, {'GGML_MOE_CACHE_BYTES': '124'})
        self.request['environment'] = {'DYLD_INSERT_LIBRARIES': '/tmp/bad'}
        with self.assertRaisesRegex(ValueError, 'loader'):
            executor.request_stage(self.request, self.job, self.config, self.request['environment'])

    def test_dotdot_escape_rejected(self):
        self.request['cwd'] = str(self.job / '..')
        with self.assertRaisesRegex(ValueError, 'normalized'):
            executor.request_stage(self.request, self.job, self.config)

    def test_rejects_unbounded_deadline(self):
        for timeout in (float('inf'), float('nan'), True, -1, 61):
            self.request['timeout_s'] = timeout
            with self.assertRaises(ValueError):
                executor.request_stage(self.request, self.job, self.config)

    def test_rejects_cross_job_cwd_and_id(self):
        self.request['cwd'] = str(self.job.parent)
        with self.assertRaisesRegex(ValueError, 'outside'):
            executor.request_stage(self.request, self.job, self.config)
        self.request['cwd'] = str(self.job)
        self.request['job_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'wrong job'):
            executor.request_stage(self.request, self.job, self.config)

    def test_rejects_cwd_symlink(self):
        (self.job / 'link').symlink_to(self.job)
        self.request['cwd'] = str(self.job / 'link')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            executor.request_stage(self.request, self.job, self.config)

    def test_handle_binds_stage_and_job(self):
        session = object.__new__(executor.Session)
        session.job = self.job
        session.records = {'token': {'stage_id': 'compile'}}
        request = {'job_id': 'candidate', 'stage_id': 'compile', 'handle': 'token'}
        self.assertEqual(session.check_handle(request)['stage_id'], 'compile')
        request['stage_id'] = 'server'
        with self.assertRaisesRegex(ValueError, 'another job/stage'):
            session.check_handle(request)


if __name__ == '__main__':
    unittest.main()


class ListenerTests(unittest.TestCase):
    def test_startup_without_listener_is_not_a_broker_failure(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        session = object.__new__(executor.Session)
        session.config = {'worker_uid': 62002, 'executor': {'server_port': 18080}}
        with patch.object(executor.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')):
            self.assertEqual(session.listener([{'pid': 123}]), {'ready': False, 'listener_pids': []})
        with patch.object(executor.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='p123\nn127.0.0.1:18080\n')):
            self.assertEqual(session.listener([{'pid': 123}]), {'ready': True, 'listener_pids': [123]})
            with self.assertRaisesRegex(ValueError, 'outside owned'):
                session.listener([{'pid': 124}])
        with patch.object(executor.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='p123\nn*:18080\n')):
            with self.assertRaisesRegex(ValueError, 'fixed IPv4'):
                session.listener([{'pid': 123}])


class LightProbeTests(unittest.TestCase):
    def test_light_client_keeps_birth_and_listener_checks_without_full_lsof(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        session = object.__new__(executor.Session)
        session.config = {'worker_uid': 62002, 'executor': {'server_port': 18080}}
        record = dict(state='running', kind='server', handle='h', worker_pid=123,
                      worker_uid=62002, birth_identity='birth', scope_identity='scope')
        def run(argv, **kwargs):
            if argv[-1] == 'pcuftn':
                raise OSError('full file inventory unavailable')
            return SimpleNamespace(returncode=0, stdout='p123\nn127.0.0.1:18080\n')
        with patch.object(executor, 'birth', return_value='birth'), \
             patch.object(executor.core, 'members', return_value=[dict(pid=123, rss_bytes=1)]), \
             patch.object(executor.subprocess, 'run', side_effect=run):
            with self.assertRaisesRegex(OSError, 'full file inventory'):
                session.probe(record)
            self.assertTrue(session.probe(record, light=True)['ready'])
            self.assertEqual(session.client(record)['listener_pids'], [123])
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                session.probe(dict(record, birth_identity='reused'), light=True)
            with self.assertRaisesRegex(ValueError, 'boolean'):
                session.probe(record, light=1)
