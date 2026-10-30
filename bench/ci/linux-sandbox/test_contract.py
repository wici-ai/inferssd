"""Focused local checks; these do not claim real Linux isolation."""

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import stages
from stages import JobSession, RunningStage, copy_sealed_tree, freeze_tree, prune_cache


class PublishTests(unittest.TestCase):
    def test_release_appears_whole_and_readable_in_one_step(self):
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / 'release'
            seen = []
            real = os.replace

            def replace(source, destination):
                # Until the rename, a reader polling the name finds nothing.
                seen.append((target.exists(), Path(source).read_text(), stat.S_IMODE(os.stat(source).st_mode)))
                real(source, destination)

            old = os.umask(0o077)
            try:
                with patch.object(stages.os, 'replace', replace):
                    stages.publish(target, 'token')
            finally:
                os.umask(old)
            self.assertEqual(seen, [(False, 'token', 0o644)])
            self.assertEqual(target.read_text(), 'token')
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
            self.assertEqual([p.name for p in Path(name).iterdir()], ['release'])


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'build'
        self.source.mkdir()
        self.output = self.root / 'sealed'

    def tearDown(self):
        # The sealed artifact mode is intentional. Restore directories solely
        # so this local test can remove its own temporary files.
        for directory in self.root.rglob('*'):
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o700)
        self.temp.cleanup()

    def test_normalizes_only_verified_internal_library_link(self):
        library = self.source / 'libengine.so.1'
        library.write_bytes(b'compiled-library')
        library.chmod(0o755)
        (self.source / 'libengine.so').symlink_to('libengine.so.1')
        (self.source / 'libengine.so.0').symlink_to('libengine.so')
        (self.source / '.ninja_log').write_bytes(b'ninja build log')
        result = freeze_tree(self.source, self.output, 1024)
        normalized = self.output / 'libengine.so'
        self.assertFalse(normalized.is_symlink())
        self.assertEqual(normalized.read_bytes(), library.read_bytes())
        self.assertEqual(normalized.stat().st_mode & 0o777, 0o555)
        row = next(row for row in result['files'] if row['path'] == 'libengine.so')
        self.assertEqual(row['normalized_symlink'], {'link': 'libengine.so.1', 'resolved_target': 'libengine.so.1'})
        self.assertNotIn('alias_of', row)
        for alias in ('libengine.so.0', 'libengine.so.1'):
            sealed = self.output / alias
            self.assertFalse(sealed.is_symlink())
            self.assertEqual(sealed.stat().st_ino, normalized.stat().st_ino)
            alias_row = next(row for row in result['files'] if row['path'] == alias)
            self.assertEqual(alias_row['alias_of'], 'libengine.so')
            self.assertEqual(alias_row['sha256'], row['sha256'])
        self.assertEqual(normalized.stat().st_nlink, 3)
        self.assertEqual(result['manifest_sha256'], hashlib.sha256(
            json.dumps(result['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        library.write_bytes(b'worker later changed original')
        self.assertEqual(normalized.read_bytes(), b'compiled-library')

    def test_outside_symlink_is_not_read(self):
        outside = self.root / 'secret'
        outside.write_bytes(b'not-an-artifact')
        (self.source / 'leak').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            freeze_tree(self.source, self.output, 1024)
        self.assertFalse((self.output / 'leak').exists())

    def test_multilink_and_fifo_are_rejected(self):
        item = self.source / 'file'
        item.write_bytes(b'value')
        os.link(item, self.source / 'second')
        with self.assertRaisesRegex(ValueError, 'multiply linked'):
            freeze_tree(self.source, self.output, 1024)
        self.output.rmdir()
        item.unlink()
        (self.source / 'second').unlink()
        os.mkfifo(self.source / 'fifo')
        with self.assertRaisesRegex(ValueError, 'special'):
            freeze_tree(self.source, self.output, 1024)

    def test_size_bound_rejects_before_copy(self):
        (self.source / 'huge').write_bytes(b'0123456789')
        with self.assertRaisesRegex(ValueError, 'bound'):
            freeze_tree(self.source, self.output, 9)
        self.assertFalse((self.output / 'huge').exists())

    def test_symlink_loop_is_rejected(self):
        (self.source / 'first').symlink_to('second')
        (self.source / 'second').symlink_to('first')
        with self.assertRaises((RuntimeError, OSError)):
            freeze_tree(self.source, self.output, 1024)


class BuildCacheTests(unittest.TestCase):
    """The base build cache: a verified copy in, a verified copy out, nothing mounted from it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.build = self.root / 'scratch'
        self.build.mkdir()
        library = self.build / 'libengine.so.1'
        library.write_bytes(b'compiled-library')
        library.chmod(0o755)
        (self.build / 'libengine.so').symlink_to('libengine.so.1')
        (self.build / 'bin').mkdir()
        (self.build / 'bin' / 'inferssd-server').write_bytes(b'server')
        (self.build / 'CMakeCache.txt').write_bytes(b'CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-12')
        self.frozen = freeze_tree(self.build, self.root / 'frozen', 1 << 20)
        self.key = 'a' * 64
        # Tests run unprivileged; root ownership checks are the broker's own contract tests.
        patcher = patch.object(stages, 'protected', lambda path, directory=False: Path(path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for directory in self.root.rglob('*'):
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o700)
        self.temp.cleanup()

    def session(self, manifest, side='candidate'):
        session = object.__new__(JobSession)
        session.config = {'state_root': str(self.root / 'state'), 'build_max_bytes': 1 << 20}
        (self.root / 'state').mkdir(exist_ok=True)
        session.job_id = 'b' * 32
        session.root = self.root / 'job'
        (session.root / 'frozen').mkdir(parents=True, exist_ok=True)
        session.manifest = {'builds': {'base': 'base', 'candidate': 'candidate'}, **manifest}
        session.frozen, session.prebuilt, session.prebuilt_refused = {}, {}, {}
        session.lease = type('Lease', (), {'identity': [1, 2]})()
        return session

    def store(self):
        """A baseline job stores its frozen candidate under the cache key."""
        session = self.session({'cache_as': {'candidate': self.key}})
        frozen = session.root / 'frozen' / 'candidate'
        if not frozen.exists():
            copy = copy_sealed_tree(self.root / 'frozen', frozen, self.frozen['files'], 1 << 20)
            self.assertEqual(copy['manifest_sha256'], self.frozen['manifest_sha256'])
        return session.store_cached_build('candidate', self.frozen)

    def test_copy_reproduces_the_sealed_tree_and_its_hash(self):
        result = copy_sealed_tree(self.root / 'frozen', self.root / 'copy', self.frozen['files'], 1 << 20)
        self.assertEqual(result, self.frozen)
        copied = self.root / 'copy'
        self.assertEqual((copied / 'bin' / 'inferssd-server').read_bytes(), b'server')
        self.assertEqual((copied / 'libengine.so').stat().st_ino, (copied / 'libengine.so.1').stat().st_ino)
        self.assertEqual((copied / 'libengine.so.1').stat().st_mode & 0o777, 0o555)

    def test_copy_refuses_changed_missing_or_unsafe_files(self):
        changed = self.root / 'frozen' / 'bin' / 'inferssd-server'
        changed.chmod(0o644)
        changed.write_bytes(b'SERVER')  # Same size, different bytes.
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            copy_sealed_tree(self.root / 'frozen', self.root / 'copy', self.frozen['files'], 1 << 20)
        for index, (rows, message) in enumerate((
                ([{'path': 'missing', 'size': 1, 'sha256': '0' * 64}], 'No such file'),
                ([{'path': '../escape', 'size': 1, 'sha256': '0' * 64}], 'unsafe'),
                ([{'path': '/abs', 'size': 1, 'sha256': '0' * 64}], 'unsafe'))):
            target = self.root / ('copy-' + str(index))
            with self.assertRaisesRegex((ValueError, OSError), message):
                copy_sealed_tree(self.root / 'frozen', target, rows, 1 << 20)

    def test_baseline_job_stores_once_and_a_pr_adopts_a_verified_copy(self):
        self.assertEqual(self.store(), {'status': 'stored', 'key': self.key})
        self.assertEqual(self.store()['status'], 'present')
        entry = self.root / 'state' / 'build-cache' / self.key
        self.assertEqual(json.loads((entry / 'manifest.json').read_text())['manifest_sha256'],
                         self.frozen['manifest_sha256'])
        session = self.session({'prebuilt_offer': {'base': {'key': self.key,
                                                            'manifest_sha256': self.frozen['manifest_sha256']}}})
        session.adopt_cached_builds()
        self.assertEqual(session.prebuilt_refused, {})
        adopted = session.prebuilt['base']
        self.assertEqual((adopted['path'], adopted['key'], adopted['manifest_sha256']),
                         ('/build/base', self.key, self.frozen['manifest_sha256']))
        self.assertEqual(session.frozen['base']['files'], self.frozen['files'])
        self.assertEqual((session.root / 'frozen' / 'base' / 'bin' / 'inferssd-server').read_bytes(), b'server')
        self.assertEqual(json.loads(Path(adopted['manifest_path']).read_text())['manifest_sha256'],
                         self.frozen['manifest_sha256'])

    def test_missing_or_mismatched_cache_entry_leaves_the_side_to_be_built(self):
        offer = {'key': self.key, 'manifest_sha256': self.frozen['manifest_sha256']}
        session = self.session({'prebuilt_offer': {'base': offer}})
        session.adopt_cached_builds()  # miss: nothing stored yet
        self.assertEqual(session.prebuilt, {})
        self.assertIn('base', session.prebuilt_refused)
        self.assertNotIn('base', session.frozen)
        self.store()
        for damage in ('offer-hash', 'tree-bytes'):
            if damage == 'offer-hash':
                current = dict(offer, manifest_sha256='0' * 64)
            else:
                current = offer
                tampered = self.root / 'state' / 'build-cache' / self.key / 'tree' / 'bin' / 'inferssd-server'
                (tampered.parent).chmod(0o755)
                tampered.chmod(0o644)
                tampered.write_bytes(b'SERVER')
            session = self.session({'prebuilt_offer': {'base': current}})
            session.adopt_cached_builds()
            self.assertEqual(session.prebuilt, {}, damage)
            self.assertRegex(session.prebuilt_refused['base'], 'differs', damage)
            self.assertFalse((session.root / 'frozen' / 'base').exists(), damage)

    def test_conflicting_store_is_reported_not_replaced(self):
        self.store()
        other = dict(self.frozen, manifest_sha256='f' * 64)
        session = self.session({'cache_as': {'candidate': self.key}})
        self.assertEqual(session.store_cached_build('candidate', other), {'status': 'conflict', 'key': self.key})
        entry = self.root / 'state' / 'build-cache' / self.key
        self.assertEqual(json.loads((entry / 'manifest.json').read_text())['manifest_sha256'],
                         self.frozen['manifest_sha256'])
        self.assertEqual(self.session({}).store_cached_build('candidate', self.frozen), {'status': 'not-requested'})

    def test_invalid_offer_is_refused_before_any_copy(self):
        for offers in ({'other': {'key': self.key, 'manifest_sha256': '0' * 64}},
                       {'base': {'key': 'short', 'manifest_sha256': '0' * 64}},
                       {'base': {'key': self.key}}):
            with self.assertRaisesRegex(ValueError, 'offer'):
                self.session({'prebuilt_offer': offers}).adopt_cached_builds()

    def test_prune_keeps_the_newest_entries_and_the_protected_key(self):
        cache = self.root / 'cache'
        cache.mkdir()
        for index in range(5):
            entry = cache / (str(index) * 64)
            (entry / 'tree').mkdir(parents=True)
            (entry / 'tree' / 'file').write_bytes(b'x')
            entry.chmod(0o555)
            os.utime(entry, ns=(index * 10**9, index * 10**9))
        (cache / ('.' + 'e' * 64 + '-' + 'f' * 32)).mkdir()
        prune_cache(cache, keep=3, protect=('0' * 64,))
        self.assertEqual(sorted(entry.name[0] for entry in cache.iterdir()), ['0', '2', '3', '4'])


class RequestTests(unittest.TestCase):
    def session(self):
        session = object.__new__(JobSession)
        session.closed = False
        session.cancel = threading.Event()
        session.job_id = 'a' * 32
        session.config = {'stop_path': '/nonexistent-inferssd-test-stop'}
        session.manifest = {'stages': {'build': {'kind': 'build', 'side': 'base',
                            'executables': ['/usr/bin/cmake'], 'cwd': '/scratch/base'}}}
        session.stages = {}
        return session

    def test_arbitrary_executable_refused_before_root_side_effect(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, 'executable'):
            session.request({'schema_version': 1, 'operation': 'stage.start', 'job_id': session.job_id,
                             'stage_id': 'build', 'kind': 'build', 'argv': ['/bin/sh', '-c', 'echo nope'],
                             'environment': {}, 'cwd': '/scratch/base', 'timeout_s': 5})

    def test_handle_required_and_cancelled_job_cannot_start(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, 'handle'):
            session.request({'schema_version': 1, 'operation': 'stage.stop', 'job_id': session.job_id,
                             'stage_id': 'build', 'handle': 'forged'})
        session.cancel.set()
        with self.assertRaisesRegex(ValueError, 'active job'):
            session.request({'schema_version': 1, 'operation': 'job.open', 'job_id': session.job_id})


class ClientTests(unittest.TestCase):
    def test_http_eof_and_timeout_return_failed_results(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/props':
                    time.sleep(.15)
                self.close_connection = True
            def log_message(self, *args):
                pass
        for path in ('/health', '/props'):
            with self.subTest(path=path):
                server = HTTPServer(('127.0.0.1', 0), Handler)
                thread = threading.Thread(target=server.handle_request)
                thread.start()
                try:
                    run = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('http_client.py'))],
                                         input=json.dumps({'port': server.server_port, 'method': 'GET', 'path': path,
                                                           'timeout_s': .05, 'response_max': 1000}),
                                         text=True, capture_output=True, timeout=5, check=True)
                    value = json.loads(run.stdout)
                    self.assertEqual(value['status'], 0)
                    self.assertIn('transport_error', value)
                    self.assertNotIn('error', value)
                    self.assertGreater(value['ended_monotonic_ns'], value['started_monotonic_ns'])
                finally:
                    thread.join(timeout=5)
                    server.server_close()

    def test_light_probe_and_client_avoid_full_maps_but_keep_listener_ownership(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}\n')
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        original_read, original_link = Path.read_text, os.readlink
        owner = [321]
        def read(path, *args, **kwargs):
            if str(path) == '/proc/321/maps':
                raise PermissionError('full mapping evidence unavailable')
            if str(path) in ('/proc/321/net/tcp', '/proc/321/net/tcp6'):
                return 'header\n0: 0100007F:%04X 00000000:0000 0A 0 0 0 0 0 7\n' % server.server_port
            if str(path) == '/sys/fs/cgroup/owned/cgroup.procs':
                return str(owner[0])
            return original_read(path, *args, **kwargs)
        def link(path, *args, **kwargs):
            if str(path).startswith('/proc/'):
                return '/build/server' if Path(path).name == 'exe' else 'socket:[7]'
            return original_link(path, *args, **kwargs)
        stage = object.__new__(RunningStage)
        stage.identity = dict(worker_pid=321, worker_uid=996, birth_identity='birth', scope_identity='/owned')
        stage.config, stage.cgroup = {'worker_uid': 996, 'log_max': 4096}, '/owned'
        stage.done = threading.Event()
        def identity(pid, *_):
            return dict(stage.identity, worker_pid=pid)
        try:
            with tempfile.TemporaryDirectory() as directory, open(os.devnull) as net:
                stage.directory, stage.net_fd = Path(directory), net.fileno()
                with patch('stages.process_identity', side_effect=identity), patch.object(Path, 'read_text', read), \
                     patch.object(Path, 'rglob', return_value=[Path('/sys/fs/cgroup/owned/cgroup.procs')]), \
                     patch.object(Path, 'iterdir', return_value=[Path('/proc/321/fd/4')]), \
                     patch('stages.os.readlink', side_effect=link), patch('stages.digest', return_value='sealed'), \
                     patch('stages.namespace_entry', return_value=[sys.executable, '-I', str(Path(__file__).with_name('http_client.py'))]):
                    with self.assertRaisesRegex(PermissionError, 'mapping evidence'):
                        stage.probe(server.server_port)
                    observed = stage.probe(server.server_port, light=True)
                    self.assertTrue(observed['ready'])
                    self.assertEqual(observed['listener_owners'], [stage.identity])
                    self.assertNotIn('proc_maps', observed)
                    value = stage.client(dict(port=server.server_port, method='GET', path='/health', timeout_s=2))
                    self.assertEqual(value['status'], 200)
                    owner[0] = 322
                    with self.assertRaisesRegex(RuntimeError, 'exact pinned worker'):
                        stage.probe(server.server_port, light=True)
                    with self.assertRaisesRegex(ValueError, 'boolean'):
                        stage.probe(server.server_port, light='yes')
        finally:
            thread.join(timeout=5)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_changed_worker_identity_still_refuses_http(self):
        stage = object.__new__(RunningStage)
        stage.identity = {'worker_pid': 321, 'worker_uid': 996, 'birth_identity': 'old', 'scope_identity': '/owned'}
        stage.config, stage.cgroup = {'worker_uid': 996}, '/owned'
        stage.done = threading.Event()
        with patch('stages.process_identity', return_value=dict(stage.identity, birth_identity='reused')):
            with self.assertRaisesRegex(RuntimeError, 'identity no longer matches'):
                stage.client({'port': 18889, 'method': 'GET', 'path': '/health', 'timeout_s': 1})

    def test_trusted_client_timestamps_actual_response_lines(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'data: first\n\ndata: second\n\n')
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            request = {'port': server.server_port, 'method': 'GET', 'path': '/health',
                       'timeout_s': 5, 'response_max': 1000}
            run = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('http_client.py'))],
                                 input=json.dumps(request), text=True, capture_output=True, timeout=10, check=True)
            result = json.loads(run.stdout)
            self.assertEqual(result['status'], 200)
            self.assertEqual(b''.join(base64.b64decode(item['data_b64']) for item in result['chunks']),
                             b'data: first\n\ndata: second\n\n')
            self.assertGreater(result['ended_monotonic_ns'], result['started_monotonic_ns'])
            self.assertEqual(sorted(item['elapsed_ns'] for item in result['chunks']),
                             [item['elapsed_ns'] for item in result['chunks']])
        finally:
            thread.join(timeout=10)
            server.server_close()


if __name__ == '__main__':
    unittest.main()

class OriginalLeaseTests(unittest.TestCase):
    def test_operator_lease_preserves_identity_and_rejects_replacement_or_worker_owner(self):
        from types import SimpleNamespace
        from contract import coordination_paths
        import stat
        config = dict(lock_path='/srv/host-operator/device.lock', stop_path='/srv/host-operator/STOP',
                      coordination_owner={'uid': 1001, 'gid': 1001}, lock_identity=[30, 42],
                      controller_uid=991, controller_gid=991, worker_uid=992, worker_gid=992,
                      client_uid=993, client_gid=993)
        def metadata(path, **kwargs):
            name = str(path)
            if name.endswith('STOP'):
                raise FileNotFoundError(2, 'missing test path', name)
            return SimpleNamespace(st_mode=(stat.S_IFREG | 0o664) if name.endswith('.lock') else (stat.S_IFDIR | 0o755),
                                   st_uid=1001 if name.startswith('/srv/host-operator') else 0,
                                   st_gid=1001 if name.startswith('/srv/host-operator') else 0,
                                   st_dev=30, st_ino=42)
        with patch.object(Path, 'lstat', metadata), patch.object(Path, 'stat', metadata):
            coordination_paths(config)
            config['lock_identity'] = [30, 43]
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                coordination_paths(config)
            config['lock_identity'] = [30, 42]
            config['coordination_owner']['uid'] = 992
            with self.assertRaisesRegex(ValueError, 'trusted host operator'):
                coordination_paths(config)


class QuarantineTests(unittest.TestCase):
    def test_external_stop_uses_reviewed_coordination_and_preserves_existing(self):
        from executor import quarantine
        with tempfile.TemporaryDirectory() as temporary:
            stop = Path(temporary) / 'STOP'
            config = {'stop_path': str(stop)}
            with patch('executor.coordination_paths') as check:
                quarantine(stop, 'first', config)
                identity = stop.stat().st_ino
                quarantine(stop, 'second', config)
                self.assertEqual(stop.stat().st_ino, identity)
                self.assertEqual(json.loads(stop.read_text())['reason'], 'first')
                self.assertEqual(check.call_count, 3)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    quarantine(stop.with_name('elsewhere'), 'bad', config)
