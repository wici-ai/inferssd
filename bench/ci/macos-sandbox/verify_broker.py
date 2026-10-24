#!/usr/bin/python3
"""One fixed native broker-death probe. No persistent activation or model run."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

sys.path.insert(0, '/Library/InferSSD-CI/control')
import executor
import supervisor as core


def main():
    config = core.config()
    assert not config['production_enabled'] and not config['executor']['enabled']
    assert not Path(config['stop']).exists()
    assert not core.members(config['worker_uid'])
    job_id = uuid.uuid4().hex
    root = core.BASE
    output = root / 'evidence' / ('broker-death-' + job_id)
    output.mkdir(mode=0o700)
    inputs = root / 'inputs' / ('broker-proof-' + job_id)
    inputs.mkdir(mode=0o555)
    core.save(root/'private/job-manifests'/(job_id+'.json'), {
        'schema_version': 1, 'job_id': job_id,
        'input_roots': [str(inputs)], 'model_roots': [], 'stage_environments': {},
    })
    path = root / 'control/executor.sock'
    assert not path.exists()
    broker = os.fork()
    if broker == 0:
        # A private copy admits only this finite verification process. The
        # installed config remains disabled and is checked unchanged below.
        config['executor']['enabled'] = True
        try:
            executor.serve(config)
        finally:
            os._exit(0)
    client = None
    receipt = {'job_id':job_id, 'status':'failed', 'production_enabled':False}
    socket_identity = None
    config_hash = core.sha(root/'control/config.json')
    try:
        deadline = time.monotonic()+10
        while not path.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert path.exists(), 'broker did not start'
        info=path.lstat();socket_identity=(info.st_dev,info.st_ino)
        client = subprocess.Popen([config['tools']['python'],'-I',str(root/'control/client.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            close_fds=True,preexec_fn=lambda:core.drop(config['runner_uid'],config['runner_gid'],str(root/'runner')))
        def rpc(request):
            client.stdin.write(json.dumps({'schema_version':1,'job_id':job_id,**request})+'\n')
            client.stdin.flush()
            import select
            assert select.select([client.stdout],[],[],10)[0], 'broker reply timed out'
            response=json.loads(client.stdout.readline())
            assert response.get('status')!='failed',response
            return response
        opened=rpc({'operation':'job.open'})
        stage=rpc({'operation':'stage.start','stage_id':'death-probe','kind':'probe',
            'argv':[config['tools']['python'],'-I','-c','import time; print("BROKER_DEATH_READY",flush=True); time.sleep(120)'],
            'environment':{},'cwd':opened['job_root']+'/work','timeout_s':30})
        assert stage['running'] and stage['worker_uid']==config['worker_uid']
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            reply=rpc({'operation':'stage.output','stage_id':'death-probe','handle':stage['handle'],
                'stream':'stdout','offset':0,'limit':65536})
            import base64
            if b'BROKER_DEATH_READY' in base64.b64decode(reply['data_b64']):break
            time.sleep(.1)
        else:raise RuntimeError('actual sandbox worker did not reach readiness')
        try:
            fd=core.lease(config)
        except BlockingIOError:
            receipt['lease_exclusive_before_kill']=True
        else:
            os.close(fd);raise RuntimeError('original device lease was not held')
        receipt.update(broker_pid=broker,guardian_pid=stage['guardian_pid'],worker_identity=stage['birth_identity'])
        os.kill(broker,signal.SIGKILL);os.waitpid(broker,0);broker=None
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            try:fd=core.lease(config)
            except BlockingIOError:time.sleep(.1);continue
            else:os.close(fd);break
        else:raise RuntimeError('guardian did not release lease after broker death')
        result=json.loads((Path(stage['evidence'])/'result.json').read_text())
        assert result['cleanup_confirmed'] and not core.members(config['worker_uid'])
        assert 'disconnected' in result['reason']
        assert result['state']=='failed'
        assert core.sha(root/'control/config.json')==config_hash
        receipt.update(status='passed',cleanup_confirmed=True,lease_reacquired=True,
            guard_result=str(Path(stage['evidence'])/'result.json'),config_unchanged=True)
    except BaseException as error:
        receipt['error']=repr(error)
    finally:
        if client is not None:
            if client.stdin:client.stdin.close()
            try:client.wait(timeout=10)
            except subprocess.TimeoutExpired:client.kill();client.wait(timeout=5)
        if broker is not None:
            os.kill(broker,signal.SIGTERM);os.waitpid(broker,0)
        clean=core.cleanup(config['worker_uid']);receipt['final_cleanup']=clean
        if not clean['confirmed']:Path(config['stop']).touch(exist_ok=True)
        if path.exists() and socket_identity:
            info=path.lstat()
            assert (info.st_dev,info.st_ino)==socket_identity and stat.S_ISSOCK(info.st_mode)
            path.unlink()
        core.save(output/'proof.json',receipt)
        print(json.dumps({'status':receipt['status'],'proof':str(output/'proof.json'),'details':receipt}))
    return 0 if receipt['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
