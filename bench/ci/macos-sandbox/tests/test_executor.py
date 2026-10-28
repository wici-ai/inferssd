"""Portable contract/file-sealing checks; never claimed as native isolation proof."""
from contextlib import ExitStack
import ctypes
import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import Mock, patch

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


class FreshSideWorkspaceTests(unittest.TestCase):
    """Exercise the real job initialization before its first alias/build RPC."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('jobs', 'evidence', 'public'):
            (self.root / name).mkdir()
        for name in ('worker.sb', 'server.sb'):
            (self.root / 'public' / name).write_text('(version 1)\n(deny default)\n')
        self.config = {'worker_uid': 62002, 'worker_gid': 62002,
                       'stop': str(self.root / 'STOP'),
                       'lock_dev_inode': [1, 2],
                       'tools': {'clt': '/Library/Developer/CommandLineTools'},
                       'executor': {'max_stage_seconds': 60, 'max_processes': 128}}
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.chown = stack.enter_context(patch.object(executor.os, 'chown'))
        stack.enter_context(patch.object(executor.core, 'BASE', self.root))
        stack.enter_context(patch.object(executor.core, 'members', return_value=[]))
        stack.enter_context(patch.object(executor, 'job_manifest', return_value={
            'input_roots': [], 'model_roots': [], 'manifest_sha256': 'a' * 64}))
        lease = self.root / 'original.lock'
        lease.touch()
        stack.enter_context(patch.object(executor.core, 'lease', side_effect=lambda _: os.open(lease, os.O_RDONLY)))
        self.session = executor.Session(self.config, None, '1' * 32)
        self.addCleanup(self.session.close)

    def request(self, side):
        return {'schema_version': 1, 'job_id': self.session.job.name,
                'operation': 'stage.start', 'stage_id': side + '-alias',
                'kind': 'build', 'argv': ['/usr/bin/true'], 'environment': {},
                'cwd': str(self.session.job / 'work' / side), 'timeout_s': 30}

    def test_first_alias_reaches_spawn_in_both_fresh_side_workspaces(self):
        for side in ('base', 'candidate'):
            with self.subTest(side=side):
                path = self.session.job / 'work' / side
                # Reaching Popen proves the actual first stage request passed
                # cwd validation; no candidate or native process is executed.
                with patch.object(executor.subprocess, 'Popen', side_effect=RuntimeError('spawn reached')) as spawn:
                    with self.assertRaisesRegex(RuntimeError, '^spawn reached$'):
                        self.session.dispatch(self.request(side))
                    spawn.assert_called_once()
                    self.assertEqual(spawn.call_args.kwargs['preexec_fn'].__name__, 'drop')
                self.assertTrue(path.is_dir())
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                self.chown.assert_any_call(path, 62002, 62002)
                self.assertIsNone(self.session.stage)
                self.assertEqual(self.session.records, {})

    def test_missing_unreviewed_cwd_is_not_created(self):
        request = self.request('other')
        with patch.object(executor.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'missing controlled path'):
                self.session.dispatch(request)
            spawn.assert_not_called()
        self.assertFalse(Path(request['cwd']).exists())

    def test_missing_side_reproduces_retained_refusal_without_starting_worker(self):
        path = self.session.job / 'work' / 'base'
        path.rmdir()
        with patch.object(executor.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, '^missing controlled path$'):
                self.session.dispatch(self.request('base'))
            spawn.assert_not_called()
        self.assertFalse(path.exists())

    def test_side_symlink_and_cross_job_cwd_still_rejected(self):
        path = self.session.job / 'work' / 'base'
        path.rmdir()
        path.symlink_to(self.root, target_is_directory=True)
        with patch.object(executor.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.session.dispatch(self.request('base'))
            request = self.request('candidate')
            request['cwd'] = str(self.root / 'jobs' / ('2' * 32))
            with self.assertRaisesRegex(ValueError, 'outside this job'):
                self.session.dispatch(request)
            spawn.assert_not_called()


class ResourceViolationTests(unittest.TestCase):
    """Retain the root guard's actual sample, without changing its thresholds."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = dict(max_processes=2, memory_budget_bytes=40 << 30,
                           host_reserve_bytes=3 << 30)
        self.session = object.__new__(executor.Session)
        self.session.config = dict(worker_uid=62002, stop=str(self.root / 'STOP'),
                                   executor=self.policy)
        self.session.peer = self.session.broker_lifeline = None
        self.record = dict(started_monotonic=0, timeout_s=30,
                           evidence=str(self.root), running=True)
        self.process = Mock(stdout=None, stderr=None, returncode=-15)
        self.process.poll.return_value = None
        self.session.stage = (self.process, self.record, {}, {})
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(executor.time, 'monotonic', return_value=10))
        stack.enter_context(patch.object(executor.core, 'disconnected', return_value=False))
        stack.enter_context(patch.object(executor.core, 'drain_logs'))
        stack.enter_context(patch.object(executor.core, 'cleanup', return_value={
            'confirmed': True, 'remaining': [], 'signals': []}))
        self.members = stack.enter_context(patch.object(executor.core, 'members'))
        self.available = stack.enter_context(patch.object(executor.core, 'free_memory'))
        self.memory = stack.enter_context(patch.object(executor, 'worker_memory', side_effect=lambda members, uid: dict(
            memory_metric=executor.MEMORY_METRIC, physical_footprint_bytes=sum(
                p.get('physical_footprint_bytes', p['rss_bytes']) for p in members),
            memory_members=[], memory_exited_pids=[])))

    def tick(self, members, available):
        self.members.return_value = members
        self.available.return_value = available
        self.session.tick()
        self.members.assert_called_once_with(62002)
        self.available.assert_called_once_with()

    def test_distinguishes_and_persists_each_resource_violation(self):
        cases = (
            ([dict(pid=1, rss_bytes=1), dict(pid=2, rss_bytes=1), dict(pid=3, rss_bytes=1)],
             4 << 30, 'process count budget'),
            ([dict(pid=1, rss_bytes=1, physical_footprint_bytes=(40 << 30) + 1)],
             4 << 30, 'worker physical footprint budget'),
            ([dict(pid=1, rss_bytes=1)], (3 << 30) - 1, 'host memory reserve'),
        )
        for members, available, reason in cases:
            with self.subTest(reason=reason):
                folder = self.root / reason
                folder.mkdir()
                self.record['evidence'] = str(folder)
                self.session.stage = (self.process, self.record, {}, {})
                self.members.reset_mock()
                self.available.reset_mock()
                self.tick(members, available)
                result = json.loads((folder / 'result.json').read_text())
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['resource_violation'], dict(
                    reason=reason, monotonic_s=10, members=members,
                    process_count=len(members), rss_bytes=sum(p['rss_bytes'] for p in members),
                    memory_metric=executor.MEMORY_METRIC, physical_footprint_bytes=sum(
                        p.get('physical_footprint_bytes', p['rss_bytes']) for p in members),
                    memory_members=[], memory_exited_pids=[],
                    free_memory_bytes=available, resource_policy=self.policy))
                self.assertEqual(result['returncode'], -15)
                self.assertTrue(result['cleanup_confirmed'])

    def test_exact_boundaries_remain_allowed(self):
        self.tick([dict(pid=1, rss_bytes=45 << 30, physical_footprint_bytes=40 << 30),
                   dict(pid=2, rss_bytes=0)], 3 << 30)
        self.assertIsNotNone(self.session.stage)
        self.assertNotIn('resource_violation', self.record)
        self.assertEqual(self.record['resource_observation']['physical_footprint_bytes'], 40 << 30)
        self.assertEqual(self.record['resource_observation']['rss_bytes'], 45 << 30)
        self.assertFalse((self.root / 'result.json').exists())

    def test_first_exceeded_limit_keeps_original_precedence(self):
        self.tick([dict(pid=1, rss_bytes=40 << 30), dict(pid=2, rss_bytes=1),
                   dict(pid=3, rss_bytes=1)], 0)
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['reason'], 'process count budget')
        self.assertEqual(result['resource_violation']['rss_bytes'], (40 << 30) + 2)
        self.assertEqual(result['resource_violation']['free_memory_bytes'], 0)

    def test_peak_survives_an_empty_final_inventory(self):
        self.tick([dict(pid=1, rss_bytes=45 << 30, physical_footprint_bytes=20 << 30)], 6 << 30)
        self.members.return_value = []
        self.session.tick()
        self.assertEqual(self.record['resource_observation']['physical_footprint_bytes'], 0)
        self.assertEqual(self.record['resource_peak'], dict(
            memory_metric=executor.MEMORY_METRIC, physical_footprint_bytes=20 << 30, rss_bytes=45 << 30))

    def test_diagnostic_failure_does_not_mask_existing_footprint_violation(self):
        self.available.side_effect = OSError('vm_stat unavailable')
        self.tick([dict(pid=1, rss_bytes=(40 << 30) + 1)], None)
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['reason'], 'worker physical footprint budget')
        self.assertIsNone(result['resource_violation']['free_memory_bytes'])
        self.assertEqual(result['resource_violation']['free_memory_error'],
                         'OSError: vm_stat unavailable')
        self.assertTrue(result['cleanup_confirmed'])

    def test_unavailable_guard_sample_still_fails_closed(self):
        self.members.return_value = [dict(pid=1, rss_bytes=1)]
        self.available.side_effect = OSError('vm_stat unavailable')
        self.session.tick()
        self.available.assert_called_once_with()
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['reason'], 'host memory accounting unavailable')
        self.assertIn('vm_stat unavailable', result['resource_violation']['free_memory_error'])
        self.assertTrue(result['cleanup_confirmed'])

    def test_unknown_footprint_fails_closed_and_retains_error(self):
        self.memory.side_effect = PermissionError('proc_pid_rusage refused')
        self.tick([dict(pid=1, rss_bytes=1)], 12 << 30)
        result = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(result['reason'], 'worker memory accounting unavailable')
        self.assertIsNone(result['resource_violation']['physical_footprint_bytes'])
        self.assertEqual(result['resource_violation']['memory_accounting_error'],
                         'PermissionError: proc_pid_rusage refused')
        self.assertEqual(result['resource_violation']['rss_bytes'], 1)
        self.assertTrue(result['cleanup_confirmed'])


class PhysicalFootprintTests(unittest.TestCase):
    def identity(self, pid, **kwargs):
        self.assertEqual(kwargs, dict(include_exited=True))
        return dict(pid=pid, uid=62002, start_sec=100, start_usec=pid)

    def memory(self, pid):
        return dict(physical_footprint_bytes=pid * 10, resident_bytes=pid * 100,
                    start_abstime=pid, exit_abstime=0)

    def test_v0_sdk_layout_and_kernel_bytes_are_preserved(self):
        self.assertEqual(ctypes.sizeof(executor.RusageInfoV0), 96)
        self.assertEqual(executor.RusageInfoV0.ri_phys_footprint.offset, 72)
        function = Mock()
        def sample(pid, flavor, pointer):
            self.assertEqual((pid, flavor), (123, 0))
            usage = ctypes.cast(pointer, ctypes.POINTER(executor.RusageInfoV0)).contents
            usage.ri_proc_start_abstime = 456
            usage.ri_phys_footprint = 3 << 30
            usage.ri_resident_size = 1 << 30  # Compressed/GPU charges can exceed RSS.
            return 0
        function.side_effect = sample
        with patch.object(executor.ctypes, 'CDLL', return_value=Mock(proc_pid_rusage=function)):
            self.assertEqual(executor.process_memory(123), dict(
                physical_footprint_bytes=3 << 30, resident_bytes=1 << 30,
                start_abstime=456, exit_abstime=0))

    def test_api_permission_failure_is_not_zero_usage_or_process_exit(self):
        def fail(*args):
            ctypes.set_errno(errno.EPERM)
            return -1
        with patch.object(executor.ctypes, 'CDLL', return_value=Mock(proc_pid_rusage=Mock(side_effect=fail))):
            with self.assertRaises(PermissionError):
                executor.process_memory(123)
        with patch.object(executor.ctypes, 'CDLL', return_value=Mock(proc_pidinfo=Mock(side_effect=fail))):
            with self.assertRaises(PermissionError):
                executor.birth(123)

    def test_sums_each_live_worker_and_retains_separate_rss(self):
        active = [dict(pid=1, rss_bytes=100), dict(pid=2, rss_bytes=200)]
        with patch.object(executor, 'birth', side_effect=self.identity), \
             patch.object(executor, 'process_memory', side_effect=self.memory):
            result = executor.worker_memory(active, 62002)
        self.assertEqual(result['physical_footprint_bytes'], 30)
        self.assertEqual(result['memory_metric'], 'physical-footprint')
        self.assertEqual([row['rss_bytes'] for row in result['memory_members']], [100, 200])
        self.assertEqual(result['memory_exited_pids'], [])

    def test_cached_zombie_usage_is_not_charged_as_live_memory(self):
        with patch.object(executor, 'birth', side_effect=self.identity), \
             patch.object(executor, 'process_memory', return_value=dict(self.memory(1), exit_abstime=99)):
            result = executor.worker_memory([dict(pid=1, rss_bytes=0)], 62002)
        self.assertEqual(result['physical_footprint_bytes'], 0)
        self.assertEqual(result['memory_exited_pids'], [1])

    def test_disappeared_pid_requires_fresh_uid_inventory(self):
        for inventory in ([], [dict(pid=1, rss_bytes=1)]):
            with self.subTest(inventory=inventory), \
                 patch.object(executor, 'birth', side_effect=ProcessLookupError(errno.ESRCH, 'exited')), \
                 patch.object(executor.core, 'members', return_value=inventory) as members:
                if inventory:
                    with self.assertRaisesRegex(ValueError, 'remains inventoried'):
                        executor.worker_memory(inventory, 62002)
                else:
                    result = executor.worker_memory([dict(pid=1, rss_bytes=1)], 62002)
                    self.assertEqual(result['memory_exited_pids'], [1])
                members.assert_called_once_with(62002)

    def test_uid_change_or_pid_reuse_fails_closed(self):
        original = dict(pid=1, uid=62002, start_sec=100, start_usec=1)
        for identities in ([dict(original, uid=501)],
                           [original, dict(original, start_usec=99)]):
            with self.subTest(identities=identities), \
                 patch.object(executor, 'birth', side_effect=identities), \
                 patch.object(executor, 'process_memory', side_effect=self.memory):
                with self.assertRaisesRegex(ValueError, 'UID differs|PID reused'):
                    executor.worker_memory([dict(pid=1, rss_bytes=1)], 62002)


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
        session.config = {'worker_uid': 62002, 'executor': {
            'server_port': 18080, 'max_processes': 128,
            'memory_budget_bytes': 40 << 30, 'host_reserve_bytes': 3 << 30}}
        record = dict(state='running', kind='server', handle='h', worker_pid=123,
                      worker_uid=62002, birth_identity='birth', scope_identity='scope')
        def run(argv, **kwargs):
            if argv[-1] == 'pcuftn':
                raise OSError('full file inventory unavailable')
            return SimpleNamespace(returncode=0, stdout='p123\nn127.0.0.1:18080\n')
        with patch.object(executor, 'birth', return_value='birth'), \
             patch.object(executor.core, 'members', return_value=[dict(pid=123, rss_bytes=1)]), \
             patch.object(executor, 'worker_memory', return_value=dict(
                 memory_metric=executor.MEMORY_METRIC, physical_footprint_bytes=1)), \
             patch.object(executor.subprocess, 'run', side_effect=run):
            with self.assertRaisesRegex(OSError, 'full file inventory'):
                session.probe(record)
            self.assertTrue(session.probe(record, light=True)['ready'])
            self.assertEqual(session.client(record)['listener_pids'], [123])
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                session.probe(dict(record, birth_identity='reused'), light=True)
            with self.assertRaisesRegex(ValueError, 'boolean'):
                session.probe(record, light=1)
