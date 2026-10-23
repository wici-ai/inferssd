"""Persistent root-owned job session and asynchronous sandbox stages."""

from dataclasses import replace
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import threading
import time
import uuid

from contract import Stage, digest, integer, protected, systemd
from executor import CLEAN_ENV, Guardian, Lease, cleanup, quarantine, show

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z')
JOB = re.compile(r'[0-9a-f]{32}\Z')
START_KEYS = {'schema_version', 'operation', 'job_id', 'stage_id', 'kind',
              'argv', 'environment', 'cwd', 'timeout_s'}


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid, expected_uid, scope):
    directory = Path('/proc') / str(pid)
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
    if any(int(uid) != expected_uid for uid in status['Uid'].split()):
        raise RuntimeError('stage has unexpected execution UID')
    groups = (directory / 'cgroup').read_text().splitlines()
    if not any(line.split(':', 2)[2] == scope or line.split(':', 2)[2].startswith(scope + '/') for line in groups):
        raise RuntimeError('PID is outside the exact stage cgroup')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return {'worker_pid': pid, 'worker_uid': expected_uid,
            'birth_identity': boot + ':' + fields[19], 'scope_identity': scope}


def namespace_entry(config, net_fd):
    return ['/usr/bin/nsenter', '--net=/proc/self/fd/' + str(net_fd), '--',
            '/usr/bin/setpriv', '--reuid=' + str(config['client_uid']),
            '--regid=' + str(config['client_gid']), '--clear-groups',
            '--bounding-set=-all', '--inh-caps=-all', '--ambient-caps=-all',
            '--no-new-privs', '/usr/bin/python3', '-I',
            str(Path(config['install_root']) / 'http_client.py')]


class RunningStage:
    def __init__(self, config, lease, stage, directory, stage_id, cancel=None):
        self.config, self.stage, self.directory = config, stage, directory
        self.stage_id, self.handle = stage_id, uuid.uuid4().hex
        self.identity, self.net_fd = None, -1
        self.stop_event, self.done = threading.Event(), threading.Event()
        self.cancel = cancel or threading.Event()
        self.error, self.cgroup = None, ''
        self.receipt = {'schema_version': 1, 'handle': self.handle, 'stage_id': stage_id,
                        'unit': stage.unit, 'lease': lease.identity, 'cleanup_confirmed': False}
        self.log_path = directory / 'stdout.log'
        launch = directory / 'launch'
        launch.mkdir(mode=0o755)
        launch.chmod(0o755)
        # Loader and Python environment supplied for the candidate must not
        # affect the trusted PID barrier before root releases candidate exec.
        (launch / 'environment.json').write_text(json.dumps(dict(stage.environment)))
        (launch / 'environment.json').chmod(0o644)
        token = uuid.uuid4().hex
        protected(Path(config['install_root']) / 'worker_entry.py')
        wrapped = replace(stage, argv=('/usr/bin/python3', '-I', '/inputs/worker_entry.py', token, *stage.argv), environment=(),
                          readonly=(*stage.readonly, (Path(config['install_root']) / 'worker_entry.py', '/inputs/worker_entry.py'),
                                    (launch, '/inputs/launch')))
        self.guardian = Guardian(config, lease, stage.unit, directory, command=systemd(config, wrapped))
        self.process = None
        try:
            self.process = self.guardian.process
            self.thread = threading.Thread(target=self._watch, name=stage.unit, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 12
            while self.identity is None:
                if self.done.is_set():
                    raise RuntimeError('stage failed before its execution identity was pinned: ' + str(self.error))
                state = show(stage.unit)
                scope = state.get('ControlGroup', '')
                if scope:
                    if scope != '/system.slice/' + stage.unit:
                        raise RuntimeError('unexpected systemd scope')
                    self.cgroup = scope
                    if state.get('ActiveState') == 'active':
                        for key, expected in [('User', str(config['worker_uid'])), ('Group', str(config['worker_gid'])),
                                              ('PrivateNetwork', 'yes'), ('PrivateUsers', 'yes'),
                                              ('NoNewPrivileges', 'yes'), ('MemoryMax', str(config['memory_max'])),
                                              ('MemorySwapMax', '0')]:
                            if state.get(key) != expected:
                                raise RuntimeError('actual isolation property differs: ' + key)
                        self.receipt['active_unit'] = state
                        for entries in (Path('/sys/fs/cgroup') / scope.lstrip('/')).rglob('cgroup.procs'):
                            for item in entries.read_text().split():
                                pid = int(item)
                                try:
                                    args = (Path('/proc') / item / 'cmdline').read_bytes().split(b'\0')
                                    if args[:4] != [b'/usr/bin/python3', b'-I', b'/inputs/worker_entry.py', token.encode()]:
                                        continue
                                    identity = process_identity(pid, config['worker_uid'], scope)
                                    descriptor = os.open('/proc/' + item + '/ns/net', os.O_RDONLY | os.O_CLOEXEC)
                                    if process_identity(pid, config['worker_uid'], scope) != identity:
                                        os.close(descriptor)
                                        raise RuntimeError('stage identity changed while pinning namespace')
                                    self.identity, self.net_fd = identity, descriptor
                                    break
                                except FileNotFoundError:
                                    continue
                            if self.identity:
                                break
                if time.monotonic() > deadline:
                    raise TimeoutError('worker start identity unavailable')
                if self.identity is None:
                    time.sleep(.02)
            self.receipt.update(self.identity)
            self.receipt['argv'] = list(stage.argv)
            # This file is exposed read-only. The trusted shim executes the
            # candidate only after its host PID/birth/cgroup/netns are pinned.
            (launch / 'release').write_text(token)
            (launch / 'release').chmod(0o644)
        except BaseException:
            if self.process is None:
                self.guardian.finish()
            else:
                self.stop()
            raise

    def _watch(self):
        started, count = time.monotonic(), 0
        try:
            with self.log_path.open('xb') as log, (self.directory / 'stderr.log').open('xb') as err, selectors.DefaultSelector() as poller:
                self.log_path.chmod(0o600)
                (self.directory / 'stderr.log').chmod(0o600)
                poller.register(self.process.stdout, selectors.EVENT_READ, log)
                poller.register(self.process.stderr, selectors.EVENT_READ, err)
                while True:
                    if self.stop_event.is_set() or self.cancel.is_set():
                        raise InterruptedError('stage stopped by trusted controller')
                    if Path(self.config['stop_path']).exists():
                        raise InterruptedError('device STOP exists')
                    if time.monotonic() - started > self.stage.timeout + 15:
                        raise TimeoutError('stage exceeded finite deadline')
                    for key, _ in poller.select(.05):
                        data = os.read(key.fd, 65536)
                        if not data:
                            poller.unregister(key.fileobj)
                            continue
                        count += len(data)
                        if count > self.config['log_max']:
                            raise RuntimeError('stage log bound exceeded')
                        key.data.write(data)
                        key.data.flush()
                    if self.process.poll() is not None and not poller.get_map():
                        break
                log.flush()
                os.fsync(log.fileno())
                err.flush()
                os.fsync(err.fileno())
            self.receipt['returncode'] = self.process.wait(timeout=5)
        except BaseException as error:
            self.error = type(error).__name__ + ': ' + str(error)
            self.receipt['error'] = self.error
        finally:
            try:
                self.receipt['guardian'] = self.guardian.finish()
                self.receipt['cleanup'] = self.receipt['guardian']['cleanup']
                self.receipt['returncode'] = self.receipt['guardian'].get('returncode')
                self.receipt['cleanup_confirmed'] = True
            except BaseException as error:
                self.error = self.error or 'cleanup: ' + str(error)
                self.receipt['cleanup_error'] = str(error)
                quarantine(self.config['stop_path'], 'async stage cleanup unconfirmed: ' + self.stage.unit)
            self.receipt['passed'] = self.receipt.get('returncode') == 0 and self.error is None
            try:
                save(self.directory / 'receipt.json', self.receipt)
            finally:
                self.process.stdout.close()
                self.process.stderr.close()
                self.done.set()

    def response(self, offset=0):
        integer(offset, 0, self.config['log_max'])
        body = b''
        try:
            with self.log_path.open('rb') as log:
                log.seek(offset)
                body = log.read(128 * 1024)
        except FileNotFoundError:
            pass
        return {'handle': self.handle, 'stage_id': self.stage_id, **(self.identity or {}),
                'state': 'finished' if self.done.is_set() else 'running',
                'running': not self.done.is_set(), 'returncode': self.receipt.get('returncode'),
                'cleanup_confirmed': self.done.is_set() and self.receipt.get('cleanup_confirmed', False),
                'receipt': self.receipt if self.done.is_set() else None,
                'output_b64': base64.b64encode(body).decode(), 'output_offset': offset,
                'next_output_offset': offset + len(body)}

    def output(self, stream, offset, limit):
        if stream not in ('stdout', 'stderr'):
            raise ValueError('unknown output stream')
        integer(offset, 0, self.config['log_max'])
        integer(limit, 1, 65536)
        path = self.directory / (stream + '.log')
        with path.open('rb') as log:
            log.seek(offset)
            data = log.read(limit)
            end = log.tell()
            size = os.fstat(log.fileno()).st_size
        return {'offset': offset, 'next_offset': end, 'data_b64': base64.b64encode(data).decode(),
                'eof': self.done.is_set() and end == size}

    def verify_alive(self):
        if self.done.is_set() or self.identity is None:
            raise RuntimeError('stage is not alive')
        if process_identity(self.identity['worker_pid'], self.config['worker_uid'], self.cgroup) != self.identity:
            raise RuntimeError('stage execution identity no longer matches')

    def stop(self):
        self.stop_event.set()
        if not self.done.wait(45):
            quarantine(self.config['stop_path'], 'async stage supervisor did not finish: ' + self.stage.unit)
            raise RuntimeError('async stage cleanup deadline exceeded')
        if not self.receipt.get('cleanup_confirmed'):
            raise RuntimeError('stage cleanup not confirmed')
        if self.net_fd >= 0:
            os.close(self.net_fd)
            self.net_fd = -1
        return self.response()

    def probe(self, port=None):
        self.verify_alive()
        result = dict(self.identity)
        process = Path('/proc') / str(self.identity['worker_pid'])
        result['executable'] = os.readlink(process / 'exe')
        result['executable_sha256'] = digest(process / 'exe')
        maps = (process / 'maps').read_text()
        if len(maps.encode()) > 512 * 1024:
            raise RuntimeError('process mapping evidence exceeds protocol bound')
        result['proc_maps'] = maps
        result['image_paths'] = sorted({line.split(None, 5)[5] for line in maps.splitlines()
                                        if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')})
        for name in ('cmdline', 'environ'):
            with (process / name).open('rb') as stream:
                data = stream.read(128 * 1024 + 1)
            if len(data) > 128 * 1024:
                raise RuntimeError('process launch evidence exceeds protocol bound')
            values = [item.decode('utf-8', errors='surrogateescape') for item in data.rstrip(b'\0').split(b'\0') if item]
            if name == 'cmdline':
                result['cmdline'] = values
            else:
                environment = {}
                for item in values:
                    key, separator, value = item.partition('=')
                    if not separator or key in environment:
                        raise RuntimeError('ambiguous worker environment')
                    environment[key] = value
                result['environment'] = environment
        status = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
        result['cpus_allowed_list'] = status['Cpus_allowed_list'].strip()
        cgroup = Path('/sys/fs/cgroup') / self.cgroup.lstrip('/')
        result['cgroup_limits'] = {key: (cgroup / name).read_text().strip() for key, name in (
            ('memory_max', 'memory.max'), ('memory_swap_max', 'memory.swap.max'), ('pids_max', 'pids.max'))}
        result['proc_limits'] = (process / 'limits').read_text()
        self.verify_alive()
        if port is None:
            return result
        integer(port, 1, 65535)
        target = '%04X' % port
        sockets = set()
        for table in ('tcp', 'tcp6'):
            path = Path('/proc') / str(self.identity['worker_pid']) / 'net' / table
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                if fields[1].split(':')[1] == target and fields[3] == '0A':
                    sockets.add('socket:[' + fields[9] + ']')
        owners = []
        for entries in (Path('/sys/fs/cgroup') / self.cgroup.lstrip('/')).rglob('cgroup.procs'):
            for item in entries.read_text().split():
                pid = int(item)
                try:
                    identity = process_identity(pid, self.config['worker_uid'], self.cgroup)
                    if any(os.readlink(path) in sockets for path in (Path('/proc') / item / 'fd').iterdir()):
                        owners.append(identity)
                except (FileNotFoundError, ProcessLookupError):
                    continue
        result.update(port=port, listener_owners=owners)
        result['ready'] = len(owners) == 1 and owners[0] == self.identity
        if owners and not result['ready']:
            raise RuntimeError('listener is not owned by the exact pinned worker PID')
        return result

    def client(self, spec):
        if set(spec) - {'port', 'method', 'path', 'body', 'timeout_s'}:
            raise ValueError('unknown trusted client field')
        if spec.get('method') not in ('GET', 'POST') or spec.get('path') not in ('/health', '/props', '/apply-template', '/completion', '/v1/completions', '/v1/chat/completions'):
            raise ValueError('unapproved loopback endpoint')
        integer(spec['port'], 1, 65535)
        integer(spec.get('timeout_s'), 1, 3600)
        if len(json.dumps(spec).encode()) > 512 * 1024:
            raise ValueError('oversized HTTP request')
        def alive():
            if self.identity is None:
                raise RuntimeError('stage execution identity was not pinned')
            try:
                current = process_identity(self.identity['worker_pid'], self.config['worker_uid'], self.cgroup)
            except (FileNotFoundError, ProcessLookupError):
                if (Path('/proc') / str(self.identity['worker_pid'])).exists():
                    raise  # Ambiguous identity is never an ordinary HTTP failure.
                return False
            if current != self.identity:
                raise RuntimeError('stage execution identity no longer matches')
            return not self.done.is_set()
        def failed(message, started):
            return {'status': 0, 'transport_error': message, 'chunks': [],
                    'started_monotonic_ns': started, 'ended_monotonic_ns': time.monotonic_ns()}
        started = time.monotonic_ns()
        if not alive():
            value = failed('worker exited before HTTP request', started)
            save(self.directory / ('http-' + uuid.uuid4().hex + '.json'), value)
            return value
        try:
            ready = self.probe(spec['port'])['ready']
        except (FileNotFoundError, ProcessLookupError, RuntimeError):
            if alive():
                raise
            value = failed('worker exited before HTTP request', started)
            save(self.directory / ('http-' + uuid.uuid4().hex + '.json'), value)
            return value
        if not ready:
            return {'status': 0, 'not_ready': True, 'chunks': []}
        spec = dict(spec, response_max=min(self.config['log_max'], 512 * 1024))
        try:
            result = subprocess.run(namespace_entry(self.config, self.net_fd), input=json.dumps(spec),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=CLEAN_ENV,
                close_fds=True, pass_fds=(self.net_fd,), timeout=spec['timeout_s'] + 5)
            if result.returncode or len(result.stdout) > 1024 * 1024:
                raise RuntimeError('trusted loopback client failed or exceeded output bound')
            value = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            value = failed('trusted HTTP client exceeded request deadline', started)
        if not alive():
            value = failed('worker exited during HTTP request', started)
        save(self.directory / ('http-' + uuid.uuid4().hex + '.json'), value)
        return value


def freeze_tree(source, destination, max_bytes):
    """Seal regular files, preserving verified in-tree library alias identity."""
    records, total = [], 0
    sealed_by_inode = {}
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('build output is not a real directory')
    destination.mkdir(mode=0o755)
    def walk(current, target, relative):
        nonlocal total
        for entry in sorted(os.scandir(current), key=lambda value: value.name):
            if not re.fullmatch(r'[A-Za-z0-9_.+-]{1,255}', entry.name) or entry.name in ('.', '..'):
                raise ValueError('unsafe build artifact name')
            info = entry.stat(follow_symlinks=False)
            name = relative / entry.name
            output = target / entry.name
            origin = Path(entry.path)
            link = None
            if stat.S_ISLNK(info.st_mode):
                link = {'link': os.readlink(entry.path)}
                origin = origin.resolve(strict=True)
                try:
                    link['resolved_target'] = str(origin.relative_to(source.resolve()))
                except ValueError:
                    raise ValueError('build artifact symlink escapes build tree') from None
                info = origin.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('build artifact symlink target is not regular')
            if stat.S_ISDIR(info.st_mode):
                output.mkdir(mode=0o755)
                walk(Path(entry.path), output, name)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                descriptor = os.open(origin, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, 'rb') as stream:
                    opened = os.fstat(stream.fileno())
                    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink) != (info.st_dev, info.st_ino, info.st_size, 1):
                        raise RuntimeError('build artifact changed while freezing')
                    total += opened.st_size
                    if total > max_bytes or len(records) >= 100000:
                        raise ValueError('frozen output bound exceeded')
                    identity = (opened.st_dev, opened.st_ino)
                    canonical = sealed_by_inode.get(identity)
                    if canonical is None:
                        sha = hashlib.sha256()
                        with output.open('xb') as writer:
                            while block := stream.read(1024 * 1024):
                                sha.update(block)
                                writer.write(block)
                            writer.flush()
                            os.fsync(writer.fileno())
                        output.chmod(0o555 if info.st_mode & 0o111 else 0o444)
                        row = {'path': str(name), 'size': opened.st_size, 'sha256': sha.hexdigest()}
                        sealed_by_inode[identity] = row
                    else:
                        # ELF NEEDED and explicit dlopen may name different aliases.
                        # Root-created sealed aliases must retain one inode so the
                        # loader does not initialize the same DSO twice.
                        if opened.st_size != canonical['size']:
                            raise RuntimeError('build artifact changed while freezing')
                        os.link(destination / canonical['path'], output)
                        row = {'path': str(name), 'size': canonical['size'],
                               'sha256': canonical['sha256'], 'alias_of': canonical['path']}
                    records.append({**row, **({'normalized_symlink': link} if link else {})})
            else:
                raise ValueError('symlink, special or multiply linked build artifact refused')
        target.chmod(0o555)
    walk(source, destination, Path())
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return {'files': records, 'bytes': total, 'manifest_sha256': hashlib.sha256(encoded).hexdigest()}


class JobSession:
    def __init__(self, config, job_id):
        if os.geteuid() != 0 or not JOB.fullmatch(job_id):
            raise ValueError('root supervisor and canonical job identity required')
        self.config, self.job_id = config, job_id
        self.cancel = threading.Event()
        self.root = protected(Path(config['state_root']) / 'jobs' / job_id, directory=True)
        manifest = protected(self.root / 'manifest.json')
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get('schema_version') != 1 or self.manifest.get('job_id') != job_id:
            raise ValueError('root-owned job manifest identity differs')
        self.manifest_sha256 = digest(manifest)
        for mapping in ('sources', 'models', 'builds', 'stages'):
            if not isinstance(self.manifest.get(mapping), dict) or any(not NAME.fullmatch(key) for key in self.manifest[mapping]):
                raise ValueError('invalid manifest mapping')
        for side, relative in self.manifest['builds'].items():
            if side not in self.manifest['sources'] or not NAME.fullmatch(relative) or relative in ('.', '..'):
                raise ValueError('invalid build side')
        self.stages, self.frozen = {}, {}
        self.lease, self.closed = Lease(config), False
        self.lease.__enter__()
        try:
            self.scratch = self.root / 'scratch'
            self.scratch.mkdir(mode=0o700)
            os.chown(self.scratch, config['worker_uid'], config['worker_gid'])
            for relative in self.manifest['builds'].values():
                path = self.scratch / relative
                path.mkdir(mode=0o700)
                os.chown(path, config['worker_uid'], config['worker_gid'])
            (self.root / 'stages').mkdir(mode=0o711)
            (self.root / 'stages').chmod(0o711)
            (self.root / 'frozen').mkdir(mode=0o755)
            (self.root / 'frozen').chmod(0o755)
        except BaseException:
            self.lease.__exit__()
            raise

    def request(self, value):
        if self.closed or self.cancel.is_set() or value.get('schema_version') != 1 or value.get('job_id') != self.job_id:
            raise ValueError('active job identity required')
        if Path(self.config['stop_path']).exists():
            raise RuntimeError('device STOP blocks job')
        operation = value.get('operation')
        if operation == 'job.open':
            if set(value) != {'schema_version', 'operation', 'job_id'}:
                raise ValueError('unknown job.open field')
            return {'job_id': self.job_id, 'manifest_sha256': self.manifest_sha256,
                    'lease': self.lease.identity, 'opened': True}
        if operation == 'job.finish':
            if set(value) != {'schema_version', 'operation', 'job_id'}:
                raise ValueError('unknown finish field')
            return self.close()
        stage_id = value.get('stage_id')
        if not isinstance(stage_id, str) or stage_id not in self.manifest['stages']:
            raise ValueError('stage is not in the trusted manifest')
        plan = self.manifest['stages'][stage_id]
        if operation == 'stage.start':
            if set(value) != START_KEYS or stage_id in self.stages:
                raise ValueError('duplicate stage or unknown start field')
            args, environment = value['argv'], value['environment']
            if (not isinstance(args, list) or not args or args[0] not in plan['executables'] or
                    value['kind'] != plan['kind'] or value['cwd'] != plan['cwd']):
                raise ValueError('executable/kind/cwd differs from trusted stage manifest')
            if not isinstance(environment, dict) or len(environment) > 128:
                raise ValueError('invalid environment')
            if any(not isinstance(name, str) or not re.fullmatch(r'[A-Z_][A-Z_0-9]{0,127}', name)
                   or not isinstance(item, str) or '\0' in item or len(item) > 8192
                   for name, item in environment.items()):
                raise ValueError('invalid candidate environment entry')
            integer(value['timeout_s'], 1, 43200)
            side = plan['side']
            if plan['kind'] != 'build' and side not in self.frozen:
                raise RuntimeError('test/server requires a frozen build')
            if plan['kind'] == 'build' and side in self.frozen:
                raise RuntimeError('frozen build cannot be rebuilt in this job')
            mounts = [(path, '/source/' + name) for name, path in self.manifest['sources'].items()]
            mounts += [(path, '/models/' + name) for name, path in self.manifest['models'].items()]
            mounts += [(self.root / 'frozen' / name, '/build/' + name) for name in self.frozen]
            unit = 'inferssd-stage-' + uuid.uuid4().hex + '.service'
            directory = self.root / 'stages' / stage_id
            directory.mkdir(mode=0o711)
            directory.chmod(0o711)
            stage = Stage(unit, tuple(args), tuple(mounts), self.scratch, value['cwd'],
                          value['timeout_s'], bool(plan.get('gpu', False)), tuple(environment.items()))
            running = RunningStage(self.config, self.lease, stage, directory, stage_id, self.cancel)
            self.stages[stage_id] = running
            return running.response()
        running = self.stages.get(stage_id)
        if running is None or value.get('handle') != running.handle:
            raise ValueError('exact owned stage handle required')
        shared = {'schema_version', 'operation', 'job_id', 'stage_id', 'handle'}
        allowed = {'stage.poll': {'output_offset'}, 'stage.wait': {'output_offset', 'wait_s'},
                   'stage.stop': set(), 'stage.probe': {'port'}, 'stage.client': {'request'},
                   'job.freeze': set(), 'stage.output': {'stream', 'offset', 'limit'}}
        if operation not in allowed or set(value) - shared - allowed[operation]:
            raise ValueError('unknown operation or request field')
        if operation in ('stage.poll', 'stage.wait'):
            if operation == 'stage.wait':
                running.done.wait(integer(value.get('wait_s', 10), 0, 60))
            return running.response(value.get('output_offset', 0))
        if operation == 'stage.stop':
            return running.stop()
        if operation == 'stage.output':
            return running.output(value.get('stream', 'stdout'), value.get('offset', 0), value.get('limit', 65536))
        if operation == 'stage.probe':
            return running.probe(value.get('port'))
        if operation == 'stage.client':
            return running.client(value['request'])
        if operation == 'job.freeze':
            if plan['kind'] != 'build' or not running.done.is_set() or not running.receipt['passed']:
                raise RuntimeError('successful completed build stage required')
            if any(not stage.done.is_set() or not stage.receipt.get('cleanup_confirmed') for stage in self.stages.values()):
                raise RuntimeError('all workers must be drained before freezing output')
            side = plan['side']
            if side in self.frozen:
                raise ValueError('build side already frozen')
            result = freeze_tree(self.scratch / self.manifest['builds'][side], self.root / 'frozen' / side,
                                 self.config.get('build_max_bytes', 32 * 1024**3))
            manifest_path = self.root / ('frozen-' + side + '.json')
            save(manifest_path, result)
            manifest_path.chmod(0o444)
            self.frozen[side] = result
            return {'side': side, 'path': '/build/' + side, 'manifest_sha256': result['manifest_sha256'],
                    'host_path': str(self.root / 'frozen' / side), 'manifest_path': str(manifest_path),
                    'bytes': result['bytes'], 'files': len(result['files']), 'lease': self.lease.identity}
        raise AssertionError('unreachable operation')

    def close(self):
        if self.closed:
            raise RuntimeError('job already closed')
        errors = []
        for stage in self.stages.values():
            try:
                stage.stop()
            except BaseException as error:
                errors.append(str(error))
        receipt = {'schema_version': 1, 'job_id': self.job_id, 'manifest_sha256': self.manifest_sha256,
                   'cleanup_confirmed': not errors, 'errors': errors, 'lease': self.lease.identity,
                   'stages': {key: stage.receipt for key, stage in self.stages.items()},
                   'frozen': {key: value['manifest_sha256'] for key, value in self.frozen.items()}}
        try:
            if errors:
                quarantine(self.config['stop_path'], 'job cleanup unconfirmed: ' + self.job_id)
            save(self.root / 'job-receipt.json', receipt)
        finally:
            self.closed = True
            self.lease.__exit__()
        return receipt
