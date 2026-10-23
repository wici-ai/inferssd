#!/usr/bin/python3
"""Actual isolation negatives plus positive compiler/Metal; root, no runner enablement."""
import ctypes,fcntl,hashlib,json,os,pathlib,signal,socket,subprocess,sys,time,uuid
sys.path.insert(0,'/Library/InferSSD-CI/control')
import supervisor as s

def require(ok,why):
    if not ok:raise RuntimeError(why)
def peer_identity(peer):
    uid=ctypes.c_uint();gid=ctypes.c_uint();lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
    lib.getpeereid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)];lib.getpeereid.restype=ctypes.c_int
    require(lib.getpeereid(peer.fileno(),ctypes.byref(uid),ctypes.byref(gid))==0,'getpeereid failed')
    return uid.value,gid.value

def main():
    c=s.config();nonce=uuid.uuid4().hex;out=s.BASE/'evidence'/('verification-'+nonce);out.mkdir(mode=0o700)
    result={'schema_version':1,'device':c['device'],'platform':'darwin','status':'failed-disabled','gates':{},'stages':[],'production_enabled':False};runner=None
    tcp=socket.socket();unix=socket.socket(socket.AF_UNIX);path=s.BASE/'control'/('probe-'+nonce+'.sock')
    protected=[s.BASE/'control/canary',s.BASE/'private/canary',s.BASE/'evidence/canary',s.BASE/'runner/isolation-canary']
    before={str(p):s.sha(p)for p in protected}
    try:
        require(not s.members(c['worker_uid']),'execution UID not idle')
        tcp.bind(('127.0.0.1',0));tcp.listen(2);tcp.setblocking(False)
        unix.bind(str(path));unix.listen(2);unix.settimeout(10)
        # This disposable runner process is created by the root verifier; no
        # Actions agent or personal process is signalled during verification.
        runner=subprocess.Popen(['/bin/sleep','120'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,preexec_fn=lambda:s.drop(c['runner_uid'],c['runner_gid'],str(s.BASE/'runner')))
        pc={'nonce':nonce,'worker_uid':c['worker_uid'],'worker_gid':c['worker_gid'],'runner_uid':c['runner_uid'],'personal_home':c['personal_home'],'runner_canary':str(protected[3]),'private_canary':str(protected[1]),'control_canary':str(protected[0]),'evidence_canary':str(protected[2]),'runner_pid':runner.pid,'tcp_port':tcp.getsockname()[1],'unix_socket':str(path)}
        r,where=s.guarded(c,'negative',pc);result['stages'].append({'result':r,'evidence':str(where)})
        require(r['status']=='passed'and r['cleanup']['confirmed'],'actual negative sandbox stage failed')
        p=json.loads((where/'stdout.log').read_text());require(p['nonce']==nonce and p['uid']==c['worker_uid'],'negative probe identity differs')
        needed={'personal_home','runner_canary','private_canary','model_write','control_write','evidence_write','loopback_connect','external_connect','host_socket','root_setuid','runner_signal','sudo','launchctl','osascript','symlink_escape','hardlink_escape'}
        for v in (p,p['inherited_child']):
            require(set(v['negative'])==needed and all(x.get('denied')is True and x['errno']in(1,13)for x in v['negative'].values()),'negative/inherited coverage incomplete')
        require(any(a['pid']==p['detached_child_pid']for a in r['cleanup']['signals']),'detached worker child was not actually reaped by UID cleanup')
        require(runner.poll()is None,'execution UID signalled the runner canary process')
        require(before=={str(p):s.sha(p)for p in protected},'protected control/credential/evidence canary changed')
        try:conn,_=tcp.accept();conn.close();raise RuntimeError('worker reached the actual loopback listener')
        except BlockingIOError:pass
        result['gates'].update(different_execution_uid=True,controller_home_denied=True,runner_credentials_denied=True,control_write_denied=True,evidence_write_denied=True,sudo_denied=True,host_sockets_denied=True,external_network_denied=True,user_service_escape_denied=True,process_signal_denied=True)
        for mode in ('compile','metal'):
            r,where=s.guarded(c,mode);result['stages'].append({'result':r,'evidence':str(where)})
            require(r['status']=='passed'and r['cleanup']['confirmed'],'real compiler/Metal positive failed; no profile widening or runner fallback')
            require(('COMPILE_SANDBOX_PASS'if mode=='compile'else'METAL_SANDBOX_PASS')in(where/'stdout.log').read_text(),'positive computation marker missing')
        # Real AF_UNIX runner peer; root holds the device lease independently of
        # the connection. Closing the authenticated peer must stop all worker UIDs.
        os.chown(path,0,c['runner_gid']);path.chmod(0o660)
        client=subprocess.Popen([c['tools']['python'],'-I',str(s.BASE/'public/disconnect-client.py'),str(path)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,preexec_fn=lambda:s.drop(c['runner_uid'],c['runner_gid'],str(s.BASE/'runner')))
        conn,_=unix.accept()
        with conn:
            require(peer_identity(conn)==(c['runner_uid'],c['runner_gid']),'runner Unix peer identity differs')
            require(conn.recv(64)==b'fixed-disconnect-probe\n','invalid fixed disconnect request');conn.sendall(b'OK')
            r,where=s.guarded(c,'disconnect',peer=conn);result['stages'].append({'result':r,'evidence':str(where)})
        client.wait(timeout=5);require(client.returncode==0 and r.get('controller_disconnected')is True and r['cleanup']['confirmed']and r.get('pid')in[a['pid']for a in r['cleanup']['signals']],'actual controller-loss cleanup not proved')
        lock=s.lease(c);os.close(lock);require(not s.members(c['worker_uid']),'worker UID retained processes after lock reacquisition')
        result['gates'].update(controller_disconnect_cleanup=True,cleanup_and_reset_verified=True,original_lock_inode=True)
        # Own a fresh STOP only after proving there is none; preserve any foreign STOP.
        stop=pathlib.Path(c['stop']);fd=os.open(stop,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);info=os.fstat(fd);os.close(fd)
        try:
            try:s.run(c,'compile')
            except RuntimeError as e:require(str(e)=='STOP is present','wrong STOP refusal')
            else:raise RuntimeError('STOP failed to block a stage')
        finally:
            now=stop.lstat();require((now.st_dev,now.st_ino)==(info.st_dev,info.st_ino),'STOP ownership changed; retain it');stop.unlink()
        result['gates']['stop_quarantine']=True
        # Do not lie about the missing production model/adapter/network gate.
        result['gates']['readonly_model_inputs']=False
        result['status']='isolation-subset-passed-production-off'
        result['pending']=c['pending']
    except BaseException as e:result['error']=repr(e)
    finally:
        clean=s.cleanup(c['worker_uid']);result['cleanup']=clean
        if not clean['confirmed']:pathlib.Path(c['stop']).touch(exist_ok=True)
        if runner is not None:
            try:runner.terminate();runner.wait(timeout=5)
            except ProcessLookupError:pass
            except subprocess.TimeoutExpired:
                runner.kill();runner.wait(timeout=5)
        tcp.close();unix.close()
        if path.exists():path.unlink()
        result['protected_canaries_unchanged']=before=={str(p):s.sha(p)for p in protected}
        s.save(out/'proof.json',result)
        print(json.dumps({'status':result['status'],'proof':str(out/'proof.json'),'proof_sha256':s.sha(out/'proof.json'),'production_enabled':False}))
    return 0 if result['status']=='isolation-subset-passed-production-off'else 1
if __name__=='__main__':raise SystemExit(main())
