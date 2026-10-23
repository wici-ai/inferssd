#!/usr/bin/python3
"""One native async build/freeze/server/client/stop gate; no model samples."""

from pathlib import Path
import time
import uuid

from executor import Lease
from stages import JobSession, save


def run(config):
    job_id = uuid.uuid4().hex
    root = Path(config['state_root']) / 'jobs' / job_id
    root.mkdir(mode=0o711)
    source = root / 'source'
    source.mkdir(mode=0o755)
    manifest = {
        'schema_version': 1, 'job_id': job_id, 'sources': {'base': str(source)}, 'models': {},
        'builds': {'base': 'base'},
        'stages': {
            'build': {'kind': 'build', 'side': 'base', 'executables': ['/usr/bin/python3'], 'cwd': '/scratch/base'},
            'server': {'kind': 'server', 'side': 'base', 'executables': ['/usr/bin/python3'], 'cwd': '/scratch/base'},
        },
    }
    save(root / 'manifest.json', manifest)
    worker = """from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path == '/props':
   self.close_connection = True; return
  self.send_response(200); self.end_headers(); self.wfile.write(b'healthy')
HTTPServer(('127.0.0.1', 18889), Handler).serve_forever()
"""
    session = JobSession(config, job_id)
    def request(operation, **values):
        return session.request({'schema_version': 1, 'operation': operation, 'job_id': job_id, **values})
    result = {'job_id': job_id, 'passed': False, 'formal_samples': 0}
    try:
        build = request('stage.start', stage_id='build', kind='build',
                        argv=['/usr/bin/python3', '-I', '-c', "from pathlib import Path; Path('server.py').write_text(" + repr(worker) + ")"],
                        environment={}, cwd='/scratch/base', timeout_s=20)
        handle = {'stage_id': 'build', 'handle': build['handle']}
        done = request('stage.wait', **handle, wait_s=30)
        if done.get('running') or done.get('returncode') != 0 or not done.get('cleanup_confirmed'):
            raise RuntimeError('native build stage failed')
        result['build'] = done
        result['freeze'] = request('job.freeze', **handle)
        # Sealing a side must never relinquish the original job-wide lease.
        try:
            with Lease(config):
                raise RuntimeError('job.freeze released the device lease')
        except BlockingIOError:
            result['freeze_kept_lease'] = True
        server = request('stage.start', stage_id='server', kind='server',
                         argv=['/usr/bin/python3', '-I', '/build/base/server.py'], environment={},
                         cwd='/scratch/base', timeout_s=45)
        server_handle = {'stage_id': 'server', 'handle': server['handle']}
        deadline = time.monotonic() + 10
        while True:
            try:
                result['probe'] = request('stage.probe', **server_handle, port=18889)
                if result['probe']['ready']:
                    break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)
            if time.monotonic() > deadline:
                raise TimeoutError('server listener did not become ready')
            time.sleep(.05)
        result['client'] = request('stage.client', **server_handle,
                                  request={'port': 18889, 'method': 'GET', 'path': '/health', 'timeout_s': 5})
        observed = result['probe']
        if (observed['cmdline'] != ['/usr/bin/python3', '-I', '/build/base/server.py'] or
                observed['environment'].get('HOME') != '/scratch/home' or
                not observed['cpus_allowed_list'] or 'Max locked memory' not in observed['proc_limits'] or
                observed['cgroup_limits'] != {'memory_max': str(config['memory_max']),
                                              'memory_swap_max': '0', 'pids_max': str(config['tasks_max'])}):
            raise RuntimeError('actual stage launch/resource evidence differs')
        if result['client']['status'] != 200:
            raise RuntimeError('trusted separate-UID loopback client failed')
        result['transport_failure'] = request('stage.client', **server_handle,
                                             request={'port': 18889, 'method': 'GET', 'path': '/props', 'timeout_s': 5})
        if result['transport_failure'].get('status') != 0 or not result['transport_failure'].get('transport_error'):
            raise RuntimeError('HTTP EOF was not returned as a recoverable transport failure')
        if not request('stage.probe', **server_handle, port=18889)['ready']:
            raise RuntimeError('HTTP transport failure broke the stage session')
        result['stop'] = request('stage.stop', **server_handle)
        if not result['stop']['cleanup_confirmed']:
            raise RuntimeError('server cleanup was not confirmed')
        result['output'] = request('stage.output', **server_handle, stream='stderr', offset=0, limit=65536)
        result['finish'] = request('job.finish')
        if not result['finish']['cleanup_confirmed']:
            raise RuntimeError('job cleanup was not confirmed')
        with Lease(config):
            result['lease_reclaimable'] = True
        result['passed'] = True
        return result
    finally:
        if not session.closed:
            session.close()
        save(root / 'native-stage-probe.json', result)
